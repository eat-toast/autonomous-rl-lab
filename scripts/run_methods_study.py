"""Bounded ten-condition single CUDA worker study. No old study is mutated."""
import copy,csv,hashlib,json,math,os,statistics,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl_lab.config import from_dict
from rl_lab.maze import generate_maze
from rl_lab.runner import run_experiment
from rl_lab.storage import ROOT,utc_now,write_json,runtime_metadata,snapshot_source
from rl_lab.plots import plot_rewards
BASE=ROOT/'runs/20261104T042433Z_2393292b'
STUDY=ROOT/'studies/20261104_methods_batch01'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
    v=[x for x in v if x is not None]
    return {'mean':statistics.mean(v) if v else None,'std':statistics.stdev(v) if len(v)>1 else None,'n':len(v)}

def plan():
    original=read(BASE/'config.json');base=copy.deepcopy(original)
    base['train'].update(device='cuda',method='baseline',max_seconds=300,hard_timeout_seconds=330,max_total_steps=60000,replay_event_step=0,replay_clear=False,replay_pause_steps=0)
    base['hypothesis']='Matched CUDA baseline for independent single-change methods'
    base['change_description']='Original B learning settings, CUDA; no clear/pause'
    variants=[('B_CUDA',{},{}),('success',{'method':'success'},{}),('per',{'method':'per'},{}),('nstep',{'method':'nstep'},{}),('double',{'method':'double'},{}),('mask',{'method':'mask'},{}),('train64',{}, {'train_mazes':64}),('batch64',{'batch_size':64},{}),('every2',{'train_every':2},{}),('target500',{'target_update':500},{})]
    result=[]
    for label,tr,ev in variants:
        raw=copy.deepcopy(base);raw['train'].update(tr);raw['evaluation'].update(ev)
        if label!='B_CUDA':
            raw['hypothesis']=f'Independent {label} intervention vs B_CUDA'
            raw['change_description']=json.dumps({'train':tr,'evaluation':ev})
        result.append((label,from_dict(raw)))
    return result

def extended_manifest(config,original):
    result=copy.deepcopy(original)
    seen={m['hash'] for maps in original['splits'].values() for m in maps}
    seeds={m['seed'] for maps in original['splits'].values() for m in maps}
    for candidate in range(config.evaluation.maze_seed,config.evaluation.maze_seed+10000):
        if len(result['splits']['train'])>=64:break
        if candidate in seeds:continue
        m=generate_maze(config.env,candidate).to_dict()
        if m['hash'] not in seen:
            result['splits']['train'].append(m);seen.add(m['hash'])
    assert len(result['splits']['train'])==64
    return result

def summarize(label,run):
    summaries=read(run/'summary.json')['seeds'];completed=[s for s in summaries if s['status']=='completed']
    val=stats([(s.get('last_validation') or {}).get('success_rate') for s in completed])
    train=stats([(s.get('training_greedy') or {}).get('success_rate') for s in completed])
    eff=stats([(s.get('last_validation') or {}).get('path_efficiency') for s in completed])
    best=[read(run/f"seed_{s['seed']}"/'best_validation.json')['metrics']['success_rate'] for s in completed]
    raw=read(run/'config.json')
    return {'label':label,'run_id':run.name,'status':'completed' if len(completed)==3 else 'incomplete','completed_seeds':len(completed),'requested_seeds':3,
            'device':raw['train'].get('device','cpu'),'method':raw['train'].get('method','baseline'),'train_maze_count':raw['evaluation']['train_mazes'],
            'validation_mazes_per_seed':8,'final_test_used':False,'validation_success_by_seed':[(s.get('last_validation') or {}).get('success_rate') if s['status']=='completed' else None for s in summaries],
            'validation_success_mean':val['mean'],'validation_success_std':val['std'],'validation_success_n':val['n'],
            'training_greedy_success_mean':train['mean'],'training_greedy_success_std':train['std'],'training_greedy_success_n':train['n'],
            'path_efficiency_mean':eff['mean'],'path_efficiency_n':eff['n'],'best_intermediate_validation_success_mean':stats(best)['mean'],
            'optimizer_updates_by_seed':[s.get('optimizer_updates') for s in summaries],
            'processed_samples_by_seed':[s.get('processed_samples',s.get('optimizer_updates',0)*raw['train']['batch_size']) for s in summaries],
            'training_steps_by_seed':[s.get('environment_steps') for s in summaries],
            'evaluation_steps_by_seed':[s.get('evaluation_environment_steps') for s in summaries],
            'wall_seconds_by_seed':[s.get('training_time_seconds') for s in summaries],
            'information_difference':'legal action prior' if label=='mask' else 'none',
            'compute_difference':'batch samples doubled' if label=='batch64' else 'optimizer updates increased' if label=='every2' else 'method overhead' if label in ('per','nstep','success','double','mask') else 'none',
            'seeds':summaries}

def make_report(folder,conditions):
    rows=[summarize(label,run) for label,run in conditions]
    eligible=[r for r in rows if r['label']!='B_CPU_reference' and r['status']=='completed']
    def score(r):return (r['validation_success_mean'],-r['validation_success_std'],r['path_efficiency_mean'] if r['path_efficiency_mean'] is not None else -1)
    winner=max(eligible,key=score)['label'] if eligible else None
    write_json(folder/'comparison.json',{'conditions':rows,'selected_condition':winner,'final_test_used':False,'primary_checkpoint':'last','selection':'mean, lower sample SD, conditional successful path efficiency; exact tie retains baseline'})
    flat=[{k:v for k,v in r.items() if k!='seeds'} for r in rows]
    with (folder/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    seed_rows=[]
    for r in rows:
        for s in r['seeds']:
            v=s.get('last_validation') or {};t=s.get('training_greedy') or {}
            seed_rows.append({'condition':r['label'],'run_id':r['run_id'],'seed':s['seed'],'status':s['status'],
                'validation_success_rate':v.get('success_rate'),'validation_denominator':8,'training_greedy_success_rate':t.get('success_rate'),
                'training_denominator':r['train_maze_count'],'path_efficiency_success_only':v.get('path_efficiency'),
                'training_steps':s.get('environment_steps'),'evaluation_steps':s.get('evaluation_environment_steps'),
                'optimizer_updates':s.get('optimizer_updates'),'processed_samples':s.get('processed_samples'),'wall_seconds':s.get('training_time_seconds')})
    with (folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(seed_rows[0]));w.writeheader();w.writerows(seed_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    x=list(range(len(rows)))
    axes[0].bar(x,[100*(r['validation_success_mean'] or 0) for r in rows],yerr=[100*(r['validation_success_std'] or 0) for r in rows],capsize=3)
    axes[0].set(ylabel='Validation greedy success (%)',ylim=(0,110),title='Last checkpoint: mean and sample SD across 3 training seeds / 8 fixed mazes')
    for i,r in enumerate(rows):
        vals=[v*100 for v in r['validation_success_by_seed'] if v is not None]
        axes[0].scatter([i]*len(vals),vals,color='black',s=12,zorder=4)
    axes[1].bar(x,[100*(r['training_greedy_success_mean'] or 0) for r in rows],color='#3c9a8d')
    axes[1].set(ylabel='Training greedy success (%)',ylim=(0,105),title='Last checkpoint: training pool evaluation (16 mazes; train64 uses 64)')
    for ax in axes:ax.set_xticks(x,[r['label'] for r in rows],rotation=30);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6))
    for r in rows:
        if r['label']=='B_CPU_reference':continue
        hs=[s.get('validation_history',[]) for s in r['seeds'] if s['status']=='completed']
        if len(hs)==3:
            ax.plot([v['step'] for v in hs[0]],[statistics.mean(h[j]['success_rate'] for h in hs)*100 for j in range(len(hs[0]))],label=r['label'])
    ax.set(xlabel='Training environment steps',ylabel='Validation greedy success (%)',ylim=(-2,102));ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'validation_curves.png',dpi=160);plt.close(fig)
    descriptions={'B_CPU_reference':('기존 CPU B 참고','과거 결과를 보존; 장치가 달라 직접 효과 비교 제외'),'B_CUDA':('원래 B를 CUDA에서 새 학습','후보와 장치가 같은 기준 필요'),'success':('2000슬롯을 일반1800+성공200으로 배분','드문 성공 경로를 오래 유지'),'per':('TD 오차 큰 경험을 더 자주 추출','중요한 오차 경험 재사용'),'nstep':('1번 대신 최대3번 이동 보상 누적','목표 보상을 앞선 이동에 전달'),'double':('행동 선택은 online, 가치는 target','max 추정 편향 완화 가능성'),'mask':('벽/경계 행동을 선택과 목표값에서 제외','충돌 줄이기; 합법행동 사전 지식 추가'),'train64':('원래16개 포함 훈련지도를64개로','같은 학습량에서 다양한 지도 경험'),'batch64':('한 갱신의 샘플32→64','더 많은 경험으로 갱신; 처리량2배'),'every2':('4번 이동마다 학습→2번마다','수집 경험당 더 자주 학습'),'target500':('타깃 복사100→500스텝','목표값 변화 속도 완화')}
    lines=['# CUDA 주요 개선 후보 비교 결과','',f'실행 상태와 독립 검증은 별개다. 독립 결과는 review.json / verified_facts.json을 확인한다. 잠정 선택: {winner}.','', '| 조건 | 무엇을 바꿨나 | 왜 했나 | 마지막 검증 평균 ± 표본SD | seed별 성공률 | 완료 |','|---|---|---|---:|---|---|']
    for r in rows:
        what,why=descriptions[r['label']]
        sd=f"{r['validation_success_std']*100:.1f}%p" if r['validation_success_std'] is not None else 'null'
        result=f"{r['validation_success_mean']*100:.1f}% ± {sd}" if r['validation_success_mean'] is not None else '미완료'
        lines.append(f"| {r['label']} | {what} | {why} | {result} | {r['validation_success_by_seed']} | {r['completed_seeds']}/3 |")
    lines+=['','검증 분모는 seed당 같은8개 지도, 3개 학습seed의 평균·표본SD다. 24개의 독립 지도 표본이 아니다. 최종test는 사용하지 않았다.','마지막 checkpoint로 선택했고 중간 최고값은 comparison.json의 별도 필드다. 작은 차이로 우열·일반화·원인을 확정하지 않는다.','훈련 탐색 return 원본/최근20회 평균은 run별 reward_curve(_steps).png, validation greedy는 learning_curve.png, 마지막 train greedy는 final_training_evaluation.json이다.','성공률과 return은 다른 지표이며 성공경로효율은 성공한 사례/seed의 조건부 평균(n 별도)이다.','mask는 합법행동 사전지식이 추가됐다. batch64/every2는 동일 환경스텝에도 실제 처리샘플·연산량이 증가했다. 장치속도 순위는 주장하지 않는다.','success의 known-success 비율은 샘플당시 인식, retrospective-success는 완료성공episode를 생성step별로 사후소급한 실제비율이다. 중복슬롯은 별도 확률 질량이다.','nstep의 goal 샘플은 목표를 포함한 multi-step 항목이다. 전이 삽입/길이/horizon flush는 summary에 기록한다.','','예산·실패·재시도·시간은 ledger.json/protocol.json, 구현/단위테스트 시간은 implementation_log.json을 확인한다.','실행하지 않은 추가 연구와 카드제작/업로드는 없다.','','## 원본 경로']
    lines += [f'- {label}: {run}' for label,run in conditions]
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows,winner

def main():
    import torch
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name(0):raise RuntimeError('Required RTX4060 CUDA unavailable; no fallback')
    STUDY.mkdir(parents=True,exist_ok=False)
    lock=ROOT/'runs/.methods_cuda_worker.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.monotonic();ledger=[];reserved=0;retried=False;conditions=[('B_CPU_reference',BASE)]
    original=read(BASE/'mazes.json');plans=plan()
    protocol={'status':'running','started_at_utc':utc_now(),'source_protocol':'docs/methods_study_protocol_20261104.md','reference_run_id':BASE.name,'reference_manifest_sha256':digest(BASE/'mazes.json'),'planned_training_steps':600000,'maximum_including_recovery':620000,'max_wall_seconds':10800,'seeds':[0,1,2],'conditions':[l for l,_ in plans],'workers':1,'device':'cuda','final_test_used':False,'runtime':runtime_metadata('cuda'),'condition_runs':{}}
    snapshot_source(STUDY)
    freeze={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'rl_lab').glob('*.py')}
    protocol['frozen_training_source_hashes']=freeze
    for label,config in plans:write_json(STUDY/f'config_{label}.json',config.to_dict())
    write_json(STUDY/'protocol.json',protocol)
    def saveledger():
        actual=0;unknown=[]
        for a in ledger:
            if a.get('run_id'):
                rows=read(ROOT/'runs'/a['run_id']/'summary.json')['seeds']
                for s in rows:
                    if a['kind']=='retry' and s['seed']!=a['seed']:continue
                    if 'environment_steps' in s:actual+=s['environment_steps']
                    else:unknown.append({'run_id':a['run_id'],'seed':s['seed']})
        write_json(STUDY/'ledger.json',{'reserved_training_steps':reserved,'actual_training_steps':actual if not unknown else None,'actual_training_steps_lower_bound':actual,'unknown_step_attempts':unknown,'retry_used':retried,'elapsed_seconds':time.monotonic()-started,'attempts':ledger})
    try:
        for label,config in plans:
            assert freeze=={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'rl_lab').glob('*.py')},'Active training source changed'
            if reserved+60000>620000 or time.monotonic()-started+1020>10800:
                protocol.setdefault('not_started',[]).append(label);continue
            manifest=extended_manifest(config,original) if label=='train64' else copy.deepcopy(original)
            assert manifest['splits']['validation']==original['splits']['validation'] and manifest['splits']['test']==original['splits']['test']
            reserved+=60000;a={'condition':label,'kind':'initial','reserved_steps':60000,'started_at_utc':utc_now(),'status':'running'};ledger.append(a);saveledger()
            print('START',label,flush=True)
            run,complete=run_experiment(config,ROOT/'runs',maze_manifest=manifest,workers=1)
            a.update(run_id=run.name,status='completed' if complete else 'incomplete',finished_at_utc=utc_now())
            if not complete:
                failed=[s for s in read(run/'summary.json')['seeds'] if s['status']!='completed']
                if not retried and len(failed)==1 and reserved+20000<=620000 and time.monotonic()-started+360<=10800:
                    retried=True;reserved+=20000
                    retry={'condition':label,'kind':'retry','source_run_id':run.name,'seed':failed[0]['seed'],'reserved_steps':20000,'status':'running','started_at_utc':utc_now()};ledger.append(retry);saveledger()
                    run,complete=run_experiment(config,ROOT/'runs',reuse_completed_from=run,maze_manifest=manifest,workers=1)
                    retry.update(run_id=run.name,status='completed' if complete else 'incomplete',finished_at_utc=utc_now())
            conditions.append((label,run));protocol['condition_runs'][label]=run.name
            write_json(STUDY/'protocol.json',protocol);saveledger()
            r=summarize(label,run);print('DONE',label,r['validation_success_mean'],r['validation_success_std'],flush=True)
        rows,winner=make_report(STUDY,conditions)
        protocol.update(status='completed' if len(conditions)==11 and all(r['status']=='completed' for r in rows) else 'incomplete',finished_at_utc=utc_now(),elapsed_seconds=time.monotonic()-started,selected_condition=winner)
    except Exception as e:
        protocol.update(status='failed',error=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        saveledger();write_json(STUDY/'protocol.json',protocol);lock.unlink()

if __name__=='__main__':main()
