# CUDA DQN 주요 후보 후속 프로토콜 — 2026-11-04

사용자 승인: 10조건 × seed 0·1·2 × 20,000 = 600,000 새 훈련 환경 스텝.
기존 studies/CNN-02/study_protocol.md 및 replay 실험은 완료 기록으로 보존한다. 이 파일이 이번 실행에 적용된다.
기존 B_CPU runs/20261104T042433Z_2393292b는 참고 행이고 선택 대상이 아니다.

고정: B 원본 config/실제 mazes.json, 10×10, reward-v1/observation-v1, CNN16→32 FC64 210212개,
train16/validation8/test16, Adam .001, gamma .99, replay2000, batch32, learning_starts64,
train_every4, target100 환경 스텝, epsilon1→.1/16000, validation500스텝 간격,
episode100행동, gradient_clip10, deterministic float32 CUDA RTX4060 CPU thread1.
각 조건은 원래 CUDA B에서 하나만 변경하고 새 초기화. 앞 조건 설정을 누적하지 않는다.
clear/pause/reset은 모든 조건에서 없으며 최종 test는 평가/선택하지 않는다.

| 순서/조건 | 유일한 주요 개입 | 이유(검증 전 가설) |
|---|---|---|
| B_CUDA | 없음 | 장치가 같은 비교 기준 |
| success | 일반FIFO1800 + 성공FIFO200 | 드문 성공 궤적 유지 |
| per | proportional PER α.6 β.4→1 | 큰 TD 오차 경험 재사용 |
| nstep | 3-step target | 목표 보상 앞쪽 전파 |
| double | online 선택/target 평가 | max 기반 가치 과대평가 완화 가능성 |
| mask | 합법 행동만 탐색/greedy/TD | 충돌 행동 제거, 사전 지식 추가 |
| train64 | 훈련 지도16→64 | 같은 학습량에서 지도 다양성 |
| batch64 | batch32→64 | gradient 변동 완화 가능성, 샘플 연산량2배 |
| every2 | train_every4→2 | 경험당 optimizer 학습 증가 |
| target500 | target100→500 환경 스텝 | 목표값 변화 완화 가능성 |

성공 저장소: 일반에 모든 1-step 전이를 즉시 삽입. 성공 episode 완료 시 전체 전이를 시간순으로
성공FIFO에 추가하고 일반 슬롯에 남은 해당 전이에도 성공 표시. 합친 유효 슬롯에서 균등 비복원추출.
동일 transition의 두 슬롯은 별개 표본으로 함께 뽑힐 수 있다. 배치 할당은 없음.
샘플 당시 known_success와 episode 종료 후 생성step별 샘플 Counter를 소급한 실제 completed-success
샘플 비율을 구분한다. 마지막 미완료 episode는 성공으로 분류하지 않는다. 저장/배치 중복과 점유율 기록.

PER: capacity2000, p=(|δ|+1e-6)^.6/sum. 새 항목 현재 최대 raw priority(빈 경우1).
복원추출하며 중복slot 갱신은 배치 내 최대 |δ|+epsilon. β=.4+.6*min(env_step/20000,1).
w=(N*p)^(-β), 해당 배치 최대값으로 나눔. 개별 Huber loss에 w를 곱한 뒤 mean.
성공 저장과 혼합하지 않는다.

3-step: episode 내 최대3개의 보상을 R=sum γ^j*r_j로 누적하고 실제 k마다 γ^k 사용.
terminated만 bootstrap 차단; truncated는 reset 전 실제 마지막 관측으로 bootstrap.
episode끝 남은 구간을 모두 flush. global20k의 미완료 꼬리도 실제길이 k로 flush하되 추가 optimizer 없음.
따라서 삽입 총수20000, k별 수·horizon flush 수 기록. nstep의 terminated 샘플은 목표를 포함한
다중스텝 항목이며 실제 goal 1-step 전이 개수와 다르다.

Double: online(next).argmax로 고르고 target(next)의 해당 값을 사용.
Mask: 지도 벽/경계 제외. 완전 관측에 사전 합법행동 규칙 추가, 기본과 동등 정보방법이라고 주장하지 않는다.
terminal/no-legal bootstrap0으로 안전 처리. checkpoint 복원시 config.train.method=='mask'이면
model.action_mask=True 설정해야 한다(이 Python attr은 state_dict에 저장되지 않음).

train64는 원래 train16 순서를 유지하고 train seed 구간의 다음 후보부터 reachable-bernoulli-v1으로
48개 고유지도를 추가한다. validation/test 전체 hash 및 seed와 겹침 금지. 기존 validation/test 불변.
최종 test는 geometry hash/seed 겹침 확인을 위한 manifest 복사만 하고 rollout/선택하지 않는다.

실제 optimizer: B/성공/PER/3step/Double/mask/train64/batch64/target500=4985회, every2=9969회.
샘플처리량=실제갱신×batch(기본159520, batch64 319040, every2 319008), 환경스텝과 구분.
에피소드 탐색 return 원본/최근20개 평균, greedy validation곡선, 마지막 train greedy를 별도 기록.
마지막 checkpoint 우선; 중간 best 성능은 보조. validation 평균→낮은 seed 표본SD→성공경로효율,
완전동률 기준 유지. 성공경로효율은 성공한 seed들의 조건부 평균 및 n 명시.
같은8개 validation의 탐색적 선택이며 차이로 우열/일반화/실패원인 확정하지 않는다.

예산: 실패포함 최대620000, 재학습은 전체 seed1개 최대1회 동일설정 처음부터(추가20000).
실행/복구 wall10800초, seed soft300/hard330초, GPU worker1. 다음 전체조건/재시도 상한 시간 없으면 시작 안 함.
실패 원본/로그/source snapshot/commit/config/지도hash 보존. 실패 조건도 표에 남기고 독립후보 진행.
구현/단위검증 시간·작은 테스트 환경스텝은 연구 실행 예산과 별도 기록. CUDA 불가시 CPU대체 없음.
실행 전 Git tag/commit 및 dirty상태/source/config/manifest hash 기록; 활성 학습 코드·설정 수정 금지.
독립 검증 담당은 scripts/verify_methods_study.py와 review/facts 파일만 소유, 실험 원본 수정 금지.
콘텐츠 인계는 검증된 사실 및 comparison/protocol SHA256까지; 카드제작/업로드 없음.
