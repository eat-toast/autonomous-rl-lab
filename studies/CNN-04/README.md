# CNN-04 · CUDA 학습 방법 비교

이 원본 묶음의 CUDA 기준과 후보는 [새 이름 목록](../../docs/experiment_names.md)의 **CNN-11~20**이다. 벽 방향을 제외한 조건 `mask`는 **CNN-16-벽방향제외**다.

[실험 목록](../../docs/experiment_index.md) · [프로토콜](methods_study_protocol_20261104.md)

로컬 원본은 [results/](results)에 있다. 이전 위치는 `studies/20261104_methods_batch01/`였다.
독립 검증 자료는 Git 커밋 `942847e`의 [검증 스냅샷](reviews/snapshots/47cd56b3f496920f6929f8060c8d96558ebbb619a9f1842ec2343cd251e99c71)에도 보존되어 있다.

## 자료 위치 · 2026-11-08 구조 이전

- [실행 원본·로그·체크포인트](runs/)
- [집계·프로토콜·검증 원본](results/)
- [실험 전용 도구](scripts/)
- [전용 테스트](tests/)
- [검증 보존본](reviews/)

원본 JSON·로그·검증본 내부의 과거 경로는 그대로 보존했다.
[전체 이전표](../../docs/path_migrations.json)와 아래 대응표로 현재 위치를 찾는다.

| 이전 경로 | 현재 위치 |
|---|---|
| `runs/20261104T075746Z_862a1f55` | `runs/20261104T075746Z_862a1f55` |
| `runs/20261104T080359Z_e8d76fbd` | `runs/20261104T080359Z_e8d76fbd` |
| `runs/20261104T081014Z_417d4752` | `runs/20261104T081014Z_417d4752` |
| `runs/20261104T081639Z_844370e8` | `runs/20261104T081639Z_844370e8` |
| `runs/20261104T082257Z_e0aaa23c` | `runs/20261104T082257Z_e0aaa23c` |
| `runs/20261104T082921Z_1787b1f6` | `runs/20261104T082921Z_1787b1f6` |
| `runs/20261104T084344Z_96ef76d9` | `runs/20261104T084344Z_96ef76d9` |
| `runs/20261104T084954Z_bd3b8005` | `runs/20261104T084954Z_bd3b8005` |
| `runs/20261104T085550Z_d964e64b` | `runs/20261104T085550Z_d964e64b` |
| `runs/20261104T090424Z_c1afbc42` | `runs/20261104T090424Z_c1afbc42` |
| `review_snapshots/47cd56b3f496920f6929f8060c8d96558ebbb619a9f1842ec2343cd251e99c71` | `reviews/snapshots/47cd56b3f496920f6929f8060c8d96558ebbb619a9f1842ec2343cd251e99c71` |
| `scripts/run_methods_study.py` | `scripts/run_methods_study.py` |
| `scripts/verify_methods_study.py` | `scripts/verify_methods_study.py` |
| `tests/test_methods.py` | `tests/test_methods.py` |
| `studies/20261104_methods_batch01` | `results` |
