"""Local pixel-game reel pilot, using verified saved evaluation paths only."""
from pathlib import Path
import sys, json, subprocess, hashlib, math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '.tools'))
sys.path.insert(0, str(ROOT))
import imageio_ffmpeg
from scripts.commit_review import validate_review, commit_review

DATA = ROOT / 'studies/MLP-01/results'
OUT = ROOT / 'studies/MLP-01/exports/instagram_MLP-01_EP02/pixel_pilot_v1'
BG, PANEL, CREAM, INK = '#101b23', '#203735', '#f5efd9', '#172925'
COLORS = ['#e37a60', '#65d4b5']
FONT = 'C:/Windows/Fonts/malgunbd.ttf'
def font(n): return ImageFont.truetype(FONT, n)
def center(d, text, y, n=42, fill=CREAM, x=540):
    d.text((x,y), text, font=font(n), fill=fill, anchor='mt')
def box(d, xy, fill, border='#60756a', width=4):
    d.rectangle(xy, fill=fill, outline=border, width=width)

def main():
    validate_review(DATA)
    commit = commit_review(DATA)
    manifest = json.loads((DATA/'maze_manifest.json').read_text('utf-8'))['splits']['validation'][0]
    sources = [DATA.parent/f'runs/{name}_seed_0/validation_20000.json' for name in ['MLP10','MLP14']]
    episodes = [json.loads(p.read_text('utf-8'))['episodes'][0] for p in sources]
    assert [e['steps'] for e in episodes] == [100,30]
    assert all(e['maze_hash']==manifest['hash'] for e in episodes)
    assert episodes[0]['actions']==[1,3]*50
    OUT.mkdir(parents=True, exist_ok=True)
    fps, seconds = 30, 18
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    video = OUT/'pixel_maze_pilot_v1.mp4'
    cmd = [ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1080x1920','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    stills = []
    for frame in range(fps*seconds):
        t = frame/fps
        image = Image.new('RGB',(1080,1920),BG); d = ImageDraw.Draw(image)
        # Sparse pixel decoration, generated without external imagery.
        for i in range(36):
            x,y=(i*193+31)%1080,(i*307+53)%1920
            d.rectangle((x,y,x+3,y+3), fill='#30413f')
        center(d,'미로 실험실  /  실제 평가 기록',105,29,'#a6bbaa')
        center(d,'AI가 같은 두 칸만',180,65)
        center(d,'계속 오간다면?',265,70,'#d3ed8c')
        center(d,'같은 미로 · 다른 입력 · 각각 새로 학습',370,30)
        move = min(100,max(0,(t-3)*10))
        stage = 0 if t<3 else 1 if t<13 else 2
        for panel,e in enumerate(episodes):
            x,y,cell=70+panel*500,585,44
            color=COLORS[panel]
            box(d,(x-12,465,x+452,1145),PANEL,color)
            center(d,['기본 관측','방문 추가'][panel],490,41,color,x+220)
            center(d,['입력 10개','입력 10 + 방문 여부 4개'][panel],548,24,CREAM,x+220)
            for r,row in enumerate(manifest['grid']):
                for c,wall in enumerate(row):
                    a,b=x+c*cell,y+r*cell
                    box(d,(a,b,a+cell-1,b+cell-1),'#38534a' if wall else '#d8ddbd','#263b35',2)
                    if wall: d.rectangle((a+4,b+4,a+cell-6,b+9),fill='#4b6858')
            start=e['path'][0]; goal=(9,9)
            gx,gy=x+goal[1]*cell,y+goal[0]*cell
            d.rectangle((gx+5,gy+5,gx+38,gy+38),fill='#d8ee86')
            d.text((gx+22,gy+22),'G',font=font(24),fill=INK,anchor='mm')
            step=min(move,e['steps']); idx=int(step); fraction=step-idx
            for pos in e['path'][:idx+1]:
                px,py=x+(pos[1]+.5)*cell,y+(pos[0]+.5)*cell
                d.rectangle((px-3,py-3,px+3,py+3),fill=color)
            a=e['path'][idx]; b=e['path'][min(idx+1,e['steps'])]
            r,c=[a[k]+fraction*(b[k]-a[k]) for k in (0,1)]
            cx,cy=int(x+(c+.5)*cell),int(y+(r+.5)*cell)
            # Block character; decorative bounce never changes its recorded cell.
            d.rectangle((cx-14,cy-14,cx+14,cy+14),fill=color,outline=INK,width=2)
            d.rectangle((cx-8,cy-5,cx-4,cy-1),fill=INK)
            d.rectangle((cx+4,cy-5,cx+8,cy-1),fill=INK)
            d.line((cx-4,cy+7,cx+5,cy+7),fill=INK,width=2)
            center(d,f'{idx} / 100 행동',1045,30,CREAM,x+220)
            status = '30행동 도착!' if panel==1 and move>=30 else '100행동 · 도착 실패' if panel==0 and move>=100 else '평가 시작' if move==0 else '두 칸을 왕복 중' if panel==0 else '출구를 향해 이동 중'
            center(d,status,1095,25,color,x+220)
        box(d,(65,1225,1015,1565),CREAM,'#b7ad8c',6)
        lines = (['출구에 도착하면 성공.', '100행동 안에 갈 수 있을까?'] if stage==0 else
                 ['같은 지도에서 재생한 두 모델.', '먼저 도착한 쪽은 멈춰 있습니다.'] if move>=30 and stage==1 else
                 ['기본 관측은 두 칸을 왕복합니다.', '방문 추가는 어떤 길로 갈까요?'] if stage==1 else
                 ['이 지도에서는 방문 추가가 도착.', '전체 평가 결과는 본문에서 봐요.'])
        for k,line in enumerate(lines): center(d,line,1295+k*91,43,INK)
        center(d,'최종 모델의 개별 사례 · 학습 과정 영상이 아닙니다',1605,27,'#c4d0bd')
        center(d,'학습 시드 0 · 개발 지도 01 · 탐색 없이 평가',1650,26,'#c4d0bd')
        center(d,'동일 속도 재생 · 1초 = 10행동 · 이동 생략 없음',1692,26,'#c4d0bd')
        for i,label in enumerate(['질문','실제 이동','결과']):
            xx=95+i*325
            box(d,(xx,1780,xx+235,1790),'#d3ed8c' if i<=stage else '#3b5149',BG,1)
            center(d,label,1810,25,'#d3ed8c' if i==stage else '#a6bbaa',xx+117)
        if frame in [30,150,300,450]:
            path=OUT/f'preview_{frame:03d}.png'; image.save(path); stills.append(path)
        proc.stdin.write(image.tobytes())
    proc.stdin.close(); err=proc.stderr.read().decode(); assert proc.wait()==0,err
    subprocess.run([ff,'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
    sheet=Image.new('RGB',(1080,960),BG)
    for i,p in enumerate(stills): sheet.paste(Image.open(p).resize((270,480)),(i*270,0))
    sheet=sheet.crop((0,0,1080,480)); sheet.save(OUT/'contact_sheet.jpg')
    record={'status':'local_pilot_awaiting_user_feedback','review_commit':commit,'fps':fps,'seconds':seconds,'size':[1080,1920],'audio':False,'map_hash':manifest['hash'],'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'playback':'All 100 / 30 saved actions; 10 actions per second, synchronous start, hold at goal; linear interpolation between recorded positions.'}
    (OUT/'production.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(video)

if __name__=='__main__': main()
