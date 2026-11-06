# MLP-03-벽방향제외 (MLP3)

표시 ID로 찾으려면 [MLP-03 실험 폴더](../MLP-03/README.md)를 참고한다.

원본 묶음 ID는 `MLP-02`다. MLP2가 이미 받는 벽 정보를 사용해 막힌 방향을 탐색, greedy 선택, DQN target 계산에서 제외했다. [계획](../../docs/mlp14_action_mask_plan.md)에 따라 기존 MLP2(원본 `MLP-01/MLP14`)와 같은 seed·미로의 마지막 체크포인트를 비교한다.

로컬 원본은 [프로토콜](results/protocol.json), [실행 원장](results/ledger.json), [결과 보고서](results/report.md), [비교 데이터](results/comparison.json), [체크포인트 복원 감사](results/checkpoint_restore_audit.json), [검증 기록](results/review.json)에서 볼 수 있다. 각 seed의 학습·평가 원본과 체크포인트는 `results/runs/`에 보존한다.

[Git 검증본](../../review_snapshots/1085d4f524407208dad82fb05d61b0adeea483406daaaa4e83f77c0e53536ec1/verified_facts.json)은 커밋 `cce5a724094348787e9a5272a1415692b5468fc6`에 보존했다.
