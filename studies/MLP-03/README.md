# MLP-03-벽방향제외 (MLP3)

MLP2가 이미 받는 벽 정보를 사용해 막힌 방향을 탐색, greedy 행동 선택, DQN target 계산에서 제외하면 충돌 실패를 줄이고 개발 검증 성공률을 높일 수 있는지 확인했다. 기준 조건 MLP2는 원본 묶음 `MLP-01`의 `MLP14`다.

- [승인된 계획](mlp14_action_mask_plan.md) · [프로토콜](../MLP-02/results/protocol.json) · [실제 설정](../MLP-02/results/config_MLP14.json) · [미로 목록](../MLP-02/results/maze_manifest.json)
- 실행 전 Git 복구 지점은 `0a3df41`이다. 실행 코드와 변경 상태는 [실행 원장](../MLP-02/results/ledger.json) 및 [코드 스냅샷](../MLP-02/results/source_snapshot.zip)에서 확인할 수 있다.
- Seed 0, 1, 2에서 각각 20,000스텝을 완료했다. [실행 로그와 체크포인트](../MLP-02/runs)는 당시 사용한 원본 묶음 ID `MLP-02` 아래에 보존한다.
- [결과 보고서](../MLP-02/results/report.md) · [비교 데이터](../MLP-02/results/comparison.json) · [체크포인트 복원 감사](../MLP-02/results/checkpoint_restore_audit.json) · [독립 검증 기록](../MLP-02/results/review.json) · [검증 사실 Git 보존본](../MLP-02/reviews/snapshots/1085d4f524407208dad82fb05d61b0adeea483406daaaa4e83f77c0e53536ec1/verified_facts.json)

개발 검증은 MLP2와 MLP3 모두 19/24였다. 충돌 행동은 160회에서 0회로 줄었고, 두 칸 왕복 실패는 3건에서 5건으로 늘었다. 사전 채택 기준에 미달해 MLP2를 유지했다.

## 자료 위치 · 2026-11-08 구조 이전


원본 JSON·로그·검증본 내부의 과거 경로는 그대로 보존했다.
[전체 이전표](../../docs/path_migrations.json)와 아래 대응표로 현재 위치를 찾는다.

| 이전 경로 | 현재 위치 |
|---|---|
