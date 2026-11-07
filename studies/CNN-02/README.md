# CNN-02 · 4단계 개발 실험

이 원본 묶음의 A·B·C·D는 [새 이름 목록](../../docs/experiment_names.md)의 **CNN-03~06**이다.

[실험 목록](../../docs/experiment_index.md) · [프로토콜](study_protocol.md) · [결과 및 인계](study_results_20261104.md)

로컬 원본은 [results/](results)에 있다. 이전 위치는 `studies/20261104_four_steps/`였다.
독립 검증 자료는 Git 커밋 `1f04d0d`의 [검증 스냅샷](reviews/snapshots/9a56d0a2a268b934142bb408a4ab4cd26d5a58c35fafdec21ea5cb1cf23a7ee2)에도 보존되어 있다.

## 자료 위치 · 2026-11-08 구조 이전

- [실행 원본·로그·체크포인트](runs/)
- [집계·프로토콜·검증 원본](results/)
- [실험 전용 도구](scripts/)
- [진단·이전 시도](diagnostics/)
- [검증 보존본](reviews/)
- [콘텐츠·최종 ZIP](exports/)

원본 JSON·로그·검증본 내부의 과거 경로는 그대로 보존했다.
[전체 이전표](../../docs/path_migrations.json)와 아래 대응표로 현재 위치를 찾는다.

| 이전 경로 | 현재 위치 |
|---|---|
| `runs/20261104T041427Z_d0628be7` | `runs/20261104T041427Z_d0628be7` |
| `runs/20261104T041728Z_495ad93c` | `runs/20261104T041728Z_495ad93c` |
| `runs/20261104T042433Z_2393292b` | `runs/20261104T042433Z_2393292b` |
| `runs/20261104T042641Z_28c82317` | `runs/20261104T042641Z_28c82317` |
| `runs/20261104T043140Z_b130954f` | `runs/20261104T043140Z_b130954f` |
| `diagnostics/DIAG-01` | `diagnostics/DIAG-01` |
| `diagnostics/device_benchmark_20261104.json` | `diagnostics/device_benchmark/device_benchmark_20261104.json` |
| `diagnostics/device_benchmark_split_20261104.json` | `diagnostics/device_benchmark/device_benchmark_split_20261104.json` |
| `review_snapshots/9a56d0a2a268b934142bb408a4ab4cd26d5a58c35fafdec21ea5cb1cf23a7ee2` | `reviews/snapshots/9a56d0a2a268b934142bb408a4ab4cd26d5a58c35fafdec21ea5cb1cf23a7ee2` |
| `scripts/benchmark_devices.py` | `scripts/benchmark_devices.py` |
| `scripts/diagnose_failures.py` | `scripts/diagnose_failures.py` |
| `scripts/run_study.py` | `scripts/run_study.py` |
| `scripts/export_study_content.py` | `scripts/export_study_content.py` |
| `exports/export_study_assets.py` | `scripts/export_study_assets.py` |
| `exports/Instagram_DQN_four_steps_20261104.zip` | `exports/Instagram_DQN_four_steps_20261104.zip` |
| `studies/20261104_four_steps` | `results` |
| `diagnostics/20261104_failure_analysis_final` | `diagnostics/DIAG-01/results` |
