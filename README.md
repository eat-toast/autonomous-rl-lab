# Autonomous RL Lab

[실험 목록](docs/experiment_index.md)에서 다음 계획과 이전 실험의 결과·검증 기록을 찾을 수 있다.

장애물 미로에서 DQN 계열을 비교하고, 이후 예산 내 자율 연구로 확장하는 프로젝트.
현재는 **Phase 1: 새 CNN DQN 기준 모델과 재현 가능한 실행 파이프라인**이다.
기존 PC의 DQN은 참고 자료이며, 기존 성능을 재현했다고 주장하지 않는다.

## 환경과 고정 규칙

- Python 3.12 + PyTorch. 기본 설정은 CPU 1 thread이며 `train.device`로 CUDA를 선택할 수 있다.
- Replay 첫 묶음은 RTX 4060에서 조건별로 순차 실행한다. 기존 CPU 결과와의 비교에는 장치 차이를 기록한다. [실험 프로토콜](docs/replay_study_protocol_20261104.md)을 참고한다.
- N×N 미로, 3채널 전체 지도(벽·에이전트·목표), 상하좌우 4개 행동.
- 일반 이동 -0.01, 충돌 -0.05, 목표 +1.0. 코인 없음.
- CNN 2층 + FC 64 + Q값 4개. 미로 크기마다 별도 모델을 학습한다.
- 검증/최종 테스트는 훈련 미로와 seed·격자 해시가 다르다.
- [고정 실험 계약](docs/design.md)에 생성 규칙, 종료 조건, 지표 정의를 기록했다.

## 실행 (PowerShell)

현재 PC의 Python 경로는 실행 스크립트가 자동으로 찾는다.
라이브러리가 설치되지 않은 새 PC에서는 Python 3.12 환경을 준비하고
`python -m pip install -r requirements.txt`를 먼저 실행한다.
실제 검증한 버전은 `requirements-tested.txt`를 참고한다.

```powershell
# 단위 테스트 + 40스텝 재현성 통합 테스트
.\scripts\lab.ps1 test

# 기본 smoke: 10×10, 1 seed, 512 환경 스텝. 대규모 학습 아님.
.\scripts\lab.ps1 train

# 저장된 체크포인트를 개발 검증 미로에서 독립 평가
.\scripts\lab.ps1 evaluate -Checkpoint 'runs\<run-id>\seed_0\latest.pt'

# Phase 1 검토 후 수동 실행할 수 있는 3-seed 짧은 설정 (자동 실행하지 않음)
.\scripts\lab.ps1 train -Config configs/phase1_three_seeds.json

# 최종 선택이 끝난 뒤에만 실행. 같은 run/seed는 1회만 허용.
.\scripts\lab.ps1 evaluate -Checkpoint 'runs\<run-id>\seed_0\best.pt' -FinalTest
```

다른 Python을 쓰려면 `-PythonPath 'C:\...\python.exe'`를 지정한다.
어떤 환경에서도 프로젝트 루트에서 `python -m rl_lab train --config configs/smoke.json`
또는 `python -m unittest discover -s tests -v`로 실행할 수 있다.

## 결과

각 실행은 `runs/<UTC시각_UUID>/`를 새로 만들고 기존 결과를 덮어쓰지 않는다.

| 파일 | 내용 |
|---|---|
| `config.json`, `metadata.json` | 설정·가설·코드 변경 설명, Git 커밋/변경 상태, 버전, 실행 환경 |
| `source_snapshot.zip` | 해당 실행의 코드·문서·설정 사본 (해시는 metadata에 포함) |
| `mazes.json` | 세 split의 실제 지도, seed, 밀도, A* 경로 |
| `seed_*/training_episodes.csv` | 훈련 에피소드별 결과 |
| `seed_*/validation.jsonl` | 검증 곡선: 스텝, 손실, epsilon, 성공률, 경로 효율 |
| `seed_*/best.pt`, `latest.pt` | 검증 기준 최고/마지막 모델 (학습 resume용은 아님) |
| `seed_*/best_validation.json`, `latest_validation.json` | 평가별 실제 이동 경로와 실패 포함 결과 |
| `summary.json`, `summary.csv` | seed별 지표, 완료 seed의 평균/표본 표준편차, 제외 seed 표시 |
| `status.json`, `seed_*_worker.log` | 완료·실패·제한 도달 상태 및 오류 로그 |
| `learning_curve.png`, `path_comparison.png` | 학습 곡선과 A* 대비 이동 경로 |
| `research_notes.json` | 결론 범위·다음 실험 제안. 현재 자동 AI 가설 생성은 아님 |

`runs/`와 모델 파일은 로컬에 보존하고 Git에 올리지 않는다.
명시적으로 파일을 삭제하지 않는 한 실패 결과도 보존한다.
재현하려면 snapshot을 별도 폴더에 풀고 기록된 패키지 버전과 `config.json`으로 다시 실행한다.
다른 OS/하드웨어/PyTorch 버전의 bit 단위 동일성은 보장하지 않는다.

## 지표 해석과 제한

- 성공률은 평가 미로 에피소드의 성공 비율이다. 게시물의 '학습 seed 중 성공한 정책 비율'과 다르다.
- 성공한 에피소드가 없으면 평균 성공 스텝과 경로 효율은 `null`이다.
- 성공 seed가 아닌 **모든 완료 학습 seed**의 검증 성공률로 평균·표준편차를 낸다.
- sample efficiency 미달은 `null`, 평균은 도달한 seed만 집계하므로 `n`도 확인해야 한다.
- 훈련 시간에는 검증과 체크포인트 저장 시간이 포함된다.
- smoke의 낮은 성공률은 구현 오류나 알고리즘 성능 결론이 아니다. 수렴을 검증하는 학습량이 아니다.
- `best.pt`는 검증 성공률→성공 경로 효율 순서로 선택하며, 동률이면 먼저 얻은 모델을 유지한다.
- 최종 테스트 미로는 train에서 평가하지 않는다. test 반복 사용을 피해야 하며, 1회 제한은
  같은 run/seed 범위의 보조 장치일 뿐 여러 run 간 테스트 누출까지 방지하지 않는다.
- replay 메모리 제한은 replay 배열에만 해당한다. 전체 프로세스 RSS의 OS 강제 제한은 아직 없다.
- 실패 후 새 run으로 재실행 가능. 중간 학습 상태의 완전한 resume는 미구현이다.

## 다음 단계

Phase 2의 Double/Dueling DQN, 공정한 알고리즘 간 벤치마크, 크기 확장,
Codex를 연결한 제한 예산 연구 오케스트레이터는 아직 구현하지 않았다.
GitHub `origin`은 `https://github.com/eat-toast/autonomous-rl-lab.git`이다.
원격 업로드는 별도로 실행하며 이 구현 과정에서 자동 push하지 않는다.

## 인스타용 예비 실험 콘텐츠

3개 seed × 2,000스텝의 짧은 DQN 예비 실험과 실제 평가 기록을 기반으로
카드뉴스 5장(1080×1350), 무음 MP4(1080×1920, 16초), 캡션을 생성했다.
실험 결과는 [예비 실험 보고서](docs/pilot_content_report.md)를 참고한다.
최종 파일은 `exports/20261104_dqn_pilot_final/`에 있다.

재생성하려면 영상 패키지를 프로젝트 전용 폴더에 설치하고 별도 출력 폴더를 지정한다.
아래의 `python`은 실행 스크립트가 사용하는 실제 Python 경로로 바꿀 수 있다.

```powershell
python -m pip install --target .tools --no-deps -r requirements-content.txt
python scripts/export_content.py --run runs/20261104T031358Z_d9a39f92 --output exports/new_export
```

이 exporter는 해당 예비 실험의 편집 템플릿이다. 다른 조건/결과의 게시물을 만들 때에는
문구를 수정해야 한다. 영상은 학습 타임랩스가 아니라 저장된 greedy 평가 경로의 재생이다.
인스타그램 업로드는 수행하지 않는다.

## 에피소드 보상 그래프와 4단계 실험

각 run에 `reward_curve.png`(에피소드 축), `reward_curve_steps.png`(환경 스텝 축)를 저장한다.
에피소드 return은 할인하지 않은 실제 보상의 합이다. 원본과 최근 20개 이동평균을 표시하며,
성공률도 함께 표시한다. 이동평균의 초기 구간은 사용 가능한 과거 에피소드만 평균한다.
마지막 미완료 에피소드는 제외한다. 탐색이 포함된 훈련 return과 greedy 검증 성능은 다르다.

`network.hidden_units`로 FC 너비를 지정할 수 있다. 기존 checkpoint는 기본 64로 복원된다.
`evaluation.diagnose_training_fit=true`이면 마지막 checkpoint를 훈련 미로에서도 greedy 평가한다.

이전 4단계 개발 실험의 조건은 [4단계 프로토콜](docs/study_protocol.md)에 기록되어 있다.
`python scripts/run_study.py`는 4조건 × 3seed를 순서대로 실행하며 결과에 따라 다음 기준을 선택한다.
각 조건의 설정, 선택 근거, 원본 실행 경로, 비교 그림과 표를 `studies/`에 저장한다.

## 세 역할의 에이전트 협업

사용자 요청에 따라 현재 chat의 작업을 실험설계·실행, 독립 결과 검증, 인스타 작성으로 나눴다.
[에이전트 작업 흐름](docs/agent_workflow.md)과 `agents/`에 각 역할의 입력·출력·권한을 기록했다.
실험 담당의 원본을 검증 담당이 확인한 뒤 게시물 담당에게 넘긴다.

4단계 실험의 게시물 생성기는 `scripts/export_study_content.py`다.
`verified_facts.json`의 통과 상태와 comparison/protocol 해시를 확인한 경우만 파일을 생성한다.
결과 검증 뒤에는 `python scripts/commit_review.py --study <study 경로>`로 검증 자료를
`review_snapshots/`에 보존하고 로컬 Git 커밋을 만든다. 4단계 게시물 생성기는 이 훅을
자동 호출하며 커밋 실패 시 제작을 중단한다. 같은 자료는 기존 커밋을 재사용하고 push하지 않는다.
이는 현재 작업의 역할 분리와 게시물 생성 게이트이며, chat 밖에서 상시 실행되는 무인 서비스나
별도 Codex 맞춤 에이전트 등록·스케줄러를 제공하는 것은 아니다.

## 실패 경로 진단

[실패 진단 보고서](docs/failure_diagnosis_20261104.md)에 제자리/왕복 순환, 훈련·검증 격차,
재학습 없는 유효 행동 마스크 진단을 기록했다. 마스크 진단은 평가 정책을 바꾼 별도 확인이며
기존 벤치마크 결과로 대체하지 않는다.

```powershell
python scripts/diagnose_failures.py --output diagnostics/new_diagnosis
```

원본 결과를 변경하거나 최종 테스트를 평가하지 않는다.

## 현재 스펙·실험 후보·인스타 소개 회차

- [현재 실험 상세 스펙](docs/current_experiment_spec_20261104.md): 지도 분리, 모델, Replay, 학습/평가 절차.
- [기법별 실험 후보](docs/experiment_candidates_20261104.md): 버퍼 clear/용량/PER/n-step 등 기대 효과와 trade-off, 대조 설계. 새 학습을 실행한 목록은 아니다.
- `exports/instagram_01_maze_intro/`: 지도·규칙·현재 CNN DQN·향후 알고리즘 계획을 소개하는 카드 6장과 캡션.
- `exports/Instagram_01_maze_intro_20261104.zip`: 1회차 게시용 파일 묶음. 성능 수치는 소개 회차에서 제외했다.

`scripts/export_intro_content.py`는 결과 확인 담당의 `approved_for_intro` 사실 파일과 훈련 예시 지도
해시를 확인해 소개 이미지를 생성한다. 캡션은 게시물 담당이 별도로 검토하며 공개 업로드는 하지 않는다.
