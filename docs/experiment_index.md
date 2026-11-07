# 실험 목록

새 세션에서는 **다음 계획 → 직전 결과 → 해당 계획서** 순으로 읽으면 된다.
이 목록은 계획서, 원본 결과, 검증 기록의 위치를 연결한다. 각 실험의 조건과 예산은 개별 문서에서 확인한다.
[실험 이름과 원본 ID](experiment_names.md)에서 모델 계열별 실행 순서와 개별 조건의 이름을 찾을 수 있다.

## 최근 계획 및 실행

| 원본 묶음 ID | 계획 | 실행 | 다음 판단 |
|---|---|---|---|
| MLP-04 | [MLP-04-이웃방문횟수](../studies/MLP-04/README.md) | seed 3/3 완료 · 독립 검증 통과 · [보고서](../studies/MLP-04/results/report.md) · [비교](../studies/MLP-04/results/comparison.json) · [원장](../studies/MLP-04/results/ledger.json) · [독립 검증](../studies/MLP-04/results/review.json) · [검증 사실](../studies/MLP-04/results/verified_facts.json) | MLP4 21/24, MLP2 19/24. 모든 seed가 기준 이상이어서 MLP4 채택. 검증 커밋 `2a1ecde` |
| MLP-03 | [MLP-03-벽방향제외](../studies/MLP-03/README.md) · [계획](../studies/MLP-03/mlp14_action_mask_plan.md) | seed 3/3 완료 · 독립 검증 통과 · [원본 묶음 MLP-02](../studies/MLP-02/README.md) | MLP2와 MLP3 모두 19/24, 충돌 160→0회, 왕복 실패 3→5건. 사전 채택 기준 미달, MLP2 유지 |
| MLP-01 | [MLP-01-기본관측 → MLP-02-이웃방문](../studies/MLP-01/mlp_feelers_study_plan.md) | MLP1·MLP2 각 3/3 완료 · 독립 검증 승인 | 후속 학습은 새 계획과 승인 필요 |

[MLP-01 폴더](../studies/MLP-01/README.md) · [새 세션용 실행 프롬프트](../studies/MLP-01/next_session_mlp_prompt.md)

결과 연결: [프로토콜](../studies/MLP-01/results/protocol.json) · [보고서](../studies/MLP-01/results/report.md) ·
[비교 JSON](../studies/MLP-01/results/comparison.json) · [seed별 CSV](../studies/MLP-01/results/comparison.csv) ·
[독립 검토](../studies/MLP-01/results/review.json) · [검증 사실](../studies/MLP-01/results/verified_facts.json) ·
[체크포인트 복원 감사](../studies/MLP-01/results/checkpoint_restore_audit.json) ·
[경로 분류 보정 기록](../studies/MLP-01/results/classification_correction.json)
· [인스타그램용 검증 사실·재생 근거](../studies/MLP-01/results/instagram_handoff.md)
· [전체 벽시계 기록](../studies/MLP-01/results/workflow_timing.json)

MLP-01의 출발점은 CUDA 방법 비교에서 관찰한 왕복이다. 합법 행동 mask로 충돌은 피했지만,
저장 경로를 대조한 결과 실패 14건 모두 두 칸을 왕복했다. MLP1은 주변 벽과 출구 좌표 차이를
입력으로 사용한다. 기존 CNN과는 모델과 관측 범위가 함께 다르다. MLP2는 MLP1에 이웃 방문 정보를 추가했다.

별도 아이디어는 [2026-11-04 후보 노트](experiment_candidates_20261104.md)에 있다.
후보 노트 작성 이후 실행된 항목은 아래 이력에서 확인한다.

## 실행 이력

2026-11-04에 수행한 작업을 시작 순서로 정리했다. 성공률은 각 행이 사용한 개발 검증 결과다.
훈련 seed와 지도 수가 다르거나 조건·장치가 다른 행은 각 보고서의 비교 범위를 따른다.

| 원본 묶음 ID | 무엇을 했나 | 확인한 결과 | 실행·검증 상태 | 계획·결과 연결 |
|---|---|---|---|---|
| CNN-00 | **CNN-01-실행점검**, 512스텝·seed 1개 | 0/4. 학습과 체크포인트 복원 경로 확인 | 완료 · 복원 평가 대조 | [설계](design.md) · [보고서](../studies/CNN-00/phase1_report.md) · [원본 요약](../studies/CNN-00/runs/20261104T030730Z_0be51831/summary.json) |
| CNN-01 | **CNN-02-예비학습**, 2,000스텝, seed 3개 | seed마다 0/8. 탐색 훈련 중에는 목표 도착 5회 | 완료 · 세 모델의 경로·지표 재평가 대조 | [설정](../studies/CNN-01/configs/phase1_three_seeds.json) · [보고서](../studies/CNN-01/pilot_content_report.md) · [원본 요약](../studies/CNN-01/runs/20261104T031358Z_d9a39f92/summary.json) |
| CNN-02 | **CNN-03~06** · 학습량·탐색 기간·학습률·FC 너비 비교 | CNN-04-탐색연장(B)을 잠정 선택. CPU B 평균 12.5%, 표본 SD 21.7%p. 저장 오류 복구 이력 포함 | 완료 · 독립 검증 통과 | [프로토콜](../studies/CNN-02/study_protocol.md) · [결과 및 인계](../studies/CNN-02/study_results_20261104.md) · [비교 원본](../studies/CNN-02/results/comparison.json) · [Git 검증본](../studies/CNN-02/reviews/snapshots/9a56d0a2a268b934142bb408a4ab4cd26d5a58c35fafdec21ea5cb1cf23a7ee2/review.json) |
| DIAG-01 | CPU B 체크포인트로 실패 경로와 평가 시 mask 진단 | 기본 실패 21/24 중 제자리 15건, 두 칸 왕복 6건. 평가에만 mask를 넣은 뒤 남은 실패 18건은 모두 왕복 | 진단 완료 · 원본 경로 대조 | [진단 보고서](../studies/CNN-02/diagnostics/DIAG-01/failure_diagnosis_20261104.md) · [진단 데이터](../studies/CNN-02/diagnostics/DIAG-01/results/diagnosis.json) |
| CNN-03 | **CNN-07~10** · 단일 지도, 버퍼 용량, pause 및 clear 비교 | 같은 CUDA 장치의 pause 대조에서 유지 8.3%, 초기화 12.5%. 효과는 탐색적 관찰 | 12 seed 완료 · 묶음 독립 승인 기록 미확인 | [프로토콜](../studies/CNN-03/replay_study_protocol_20261104.md) · [보고서](../studies/CNN-03/results/report.md) · [비교 원본](../studies/CNN-03/results/comparison.json) · [원장](../studies/CNN-03/results/ledger.json) |
| CNN-04 | **CNN-11~20** · CUDA 기준과 학습 방법·지도 다양성·설정 9개 후보 비교 | CUDA B 8.3%. CNN-16-벽방향제외(mask) 41.7%로 잠정 선택, seed별 75%·50%·0%. 합법 행동 규칙 추가 | 30 seed 완료 · 독립 검증 통과 | [프로토콜](../studies/CNN-04/methods_study_protocol_20261104.md) · [보고서](../studies/CNN-04/results/report.md) · [seed별 CSV](../studies/CNN-04/results/seed_metrics.csv) · [원장](../studies/CNN-04/results/ledger.json) · [Git 검증본](../studies/CNN-04/reviews/snapshots/47cd56b3f496920f6929f8060c8d96558ebbb619a9f1842ec2343cd251e99c71/review.json) |

보조 자료로 [CPU B 기준 스펙](../studies/CNN-02/current_experiment_spec_20261104.md)과
[장치 측정 기록](../studies/CNN-02/diagnostics/device_benchmark/device_benchmark_20261104.json),
[구간별 장치 측정 기록](../studies/CNN-02/diagnostics/device_benchmark/device_benchmark_split_20261104.json)이 있다.

## 항목을 갱신하는 방법

1. **계획 작성** 시 고유 ID를 붙이고, 계획서 링크와 기준 실험, 변경 변수, 진입 또는 종료 기준을 등록한다.
   실행 승인 전에는 상태를 **승인 대기**로 표시한다.
2. **실행 승인 및 시작** 시 승인 범위와 예산을 개별 프로토콜에 기록하고, 목록에는 진행 중 상태와 study 위치를 연결한다.
3. **실행 종료** 시 같은 ID에 완료·부분 완료·실패·조건 미충족으로 인한 미실행을 기록한다.
   보고서와 비교 원본·원장, run으로 이어지는 링크를 붙이고, 실패 시도와 복구 기록도 원본에 보존한다.
4. **독립 검증** 시 실행 상태와 별개로 검증 상태를 갱신한다. 검증 파일과 사실 파일, Git 보존본이 있으면 연결한다.
   콘텐츠 인계는 [작업 지침](../AGENTS.md)의 검증 및 커밋 절차를 따른다.

실험별 자료의 배치는 [프로젝트 구조 계약](project_structure.md)을 따른다.
계획·전용 설정·스크립트·실행 원본·집계·진단·검증 보존본·콘텐츠는 같은 `studies/<ID>/` 아래에서 찾는다.
실행 원본은 `runs/`, 집계 및 검증 원본은 `results/`, 진단은 `diagnostics/`,
검증 보존본은 `reviews/snapshots/`, 콘텐츠는 `exports/`에 둔다.
과거 기록의 경로는 [이전표](path_migrations.json)와 해당 study README에서 대조한다.
Git에는 계획·설정·코드·보고서와 검증 보존본을 추적하고, 체크포인트와 대용량 산출물은 로컬에 보존한다.
