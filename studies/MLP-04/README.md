# MLP-04-이웃방문횟수 (MLP4)

묶음 ID는 `MLP-04`, 조건 ID는 `MLP14-count`다. 기준 MLP2는 [MLP-01 원본](../MLP-01/README.md)의 `MLP14_seed_0`, `MLP14_seed_1`, `MLP14_seed_2` 최종 체크포인트다. 독립 검증은 7/8, 5/8, 7/8 및 두 칸 왕복 실패 3건을 확인했다. 직전 MLP3의 개발 검증은 19/24였고 MLP2가 유지됐다.

MLP4는 14개 입력 중 마지막 북·동·남·서 네 값을 이번 에피소드의 도착 횟수 `min(count, 3) / 3`로 바꾼다. 시작 칸은 1회로 기록한다. 통로를 통해 실제로 도착한 뒤에만 카운트를 올린다. 경계와 벽 방향은 0이다. 에피소드별로 방문 기록을 새로 만들고 훈련과 평가가 이를 공유하지 않는다. Replay는 이동 전후 관측의 값 복사본을 저장한다. 두 칸 왕복 중 재방문으로 인한 관측 구분이 더 오래 유지되는지 확인한다.

같은 seed에서 새로 초기화한다. MLP2의 설정과 동일한 14→64→64→4 MLP, 보상·DQN·행동 100회 한도 및 원본 훈련 미로 16개와 개발 검증 미로 8개의 순서를 사용한다. 네 행동은 모두 탐색과 greedy의 후보이며 기존 DQN 타깃 최대값도 유지한다. 최종 테스트 사용량은 0이다.

RTX 4060 CUDA worker 1개가 seed 0→1→2를 순차 실행한다. seed당 20,000 환경 스텝, 전체 최대 60,000스텝, seed당 soft 300초·hard 330초, 실행·집계·독립 검증 총 1,800초, 재시도 0회다. 실패 시도도 원장에 기록한다. 단계별 원본을 보존하고 실패가 발생하면 이후 학습을 중단한다.

개발 검증 greedy는 0 및 매 500스텝에 평가한다. 채택 판정에는 20,000스텝 체크포인트만 사용한다. 개발 검증 21/24 이상이고 어느 seed도 MLP2보다 낮지 않으면 MLP4를 채택한다. 기준 미달이면 MLP2를 유지한다. 미완료 seed가 있으면 판정을 보류한다. 같은 seed·미로의 성공 전환, 실패 중 마지막 9개 위치의 두 칸 교대, 충돌, 기타 실패, 성공 경로 효율을 기록한다. 개발 검증 greedy, 훈련 greedy, 탐색 훈련 return과 중간 최고는 구분한다.

추후 별도 후보는 채택 조건에 `sqrt(행 차이² + 열 차이²)`를 입력 하나로 추가하는 실험이다. 이번 범위는 MLP4까지다.

2026-11-07 08:15:02–08:16:36 KST에 RTX 4060에서 순차 실행했다. 세 seed가 각각 20,000스텝을 완료했다. 학습과 집계에는 94.11초를 사용했고 학습 재시도는 0회였다. 복구 지점은 `codex/mlp04-pre-execution` (`290d4a40d7b25997be1506bfb7e0959ca92d810f`)이다. 실행 당시 실제 Git 커밋과 변경 상태는 프로토콜에 기록했다.

- [적용 설정](results/config_MLP14-count.json) · [실행 프로토콜](results/protocol.json) · [지도 원본](results/maze_manifest.json)
- [예산 원장](results/ledger.json) · [실행 로그](results/execution.log) · [worker 로그와 체크포인트](results/runs/) · [실행 전 점검 로그](results/preflight.log)
- [결과 보고서](results/report.md) · [비교 JSON](results/comparison.json) · [seed별 CSV](results/comparison.csv) · [독립 검증](results/review.json) · [검증 사실](results/verified_facts.json)
