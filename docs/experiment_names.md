# 실험 이름과 원본 ID

표시 이름은 **모델 계열-계열 내 실행 순서-조건 특징** 순으로 쓴다. 예를 들어 `MLP-03-벽방향제외`의 짧은 이름은 **MLP3**다. 번호는 각 계열에서 01부터 시작한다. 같은 묶음에서 여러 조건을 실행했으면 실제 조건 순서대로 번호를 매긴다. 같은 조건의 seed와 복구 실행은 같은 이름을 쓰고, 학습 없이 체크포인트를 진단한 작업은 별도 진단 ID로 찾는다.

표시 이름을 실험 목록과 새 문서에 사용한다. 원본 ID 열은 저장된 폴더, 설정 키, 체크포인트, 검증 자료를 찾는 데 사용한다.

## MLP 계열

| 표시 이름 | 짧은 이름 | 원본 묶음 / 조건 | 조건 특징 |
|---|---|---|---|
| **MLP-01-기본관측** | MLP1 | [MLP-01](../studies/MLP-01/README.md) / `MLP10` | 주변 벽 8개와 출구까지 행·열 차이 2개 |
| **MLP-02-이웃방문** | MLP2 | [MLP-01](../studies/MLP-01/README.md) / `MLP14` | MLP1에 이웃 방문 여부 4개 추가 |
| [**MLP-03-벽방향제외**](../studies/MLP-03/README.md) | MLP3 | [MLP-02](../studies/MLP-02/README.md) / `MLP14-mask` | MLP2의 벽 정보를 써서 막힌 방향을 행동 후보에서 제외 |
| **MLP-04-이웃방문횟수** | MLP4 | [MLP-04](../studies/MLP-04/README.md) / `MLP14-count` | MLP2의 이웃 방문 여부를 실제 진입 횟수 min(count, 3) / 3으로 교체 |

MLP2·MLP3·MLP4의 입력은 모두 14개다. MLP4를 후속 기준으로 채택했다. 채택 근거는 [실험 목록](experiment_index.md)에서 확인한다.

## CNN 계열

| 표시 이름 | 원본 묶음 / 조건 | 조건 특징 |
|---|---|---|
| **CNN-01-실행점검** | CNN-00 / [기본 smoke](../studies/CNN-00/phase1_report.md) | 512스텝 실행과 복원 확인 |
| **CNN-02-예비학습** | CNN-01 / [3 seed 예비 학습](../studies/CNN-01/pilot_content_report.md) | seed당 2,000스텝 |
| **CNN-03-학습량확대** | [CNN-02](../studies/CNN-02/README.md) / `A` | 20,000스텝 |
| **CNN-04-탐색연장** | CNN-02 / `B` · `B_CPU_reference` | 탐색 감소 기간 16,000스텝, CPU |
| **CNN-05-학습률감소** | CNN-02 / `C` | 학습률 0.0003 |
| **CNN-06-은닉층확장** | CNN-02 / `D` | FC 너비 128 |
| **CNN-07-단일미로** | [CNN-03](../studies/CNN-03/README.md) / `S_CUDA` | 훈련 지도 1개 |
| **CNN-08-리플레이확장** | CNN-03 / `R_CUDA` | 리플레이 용량 10,000 |
| **CNN-09-갱신일시중지** | CNN-03 / `P_CUDA` | 갱신 일시 중지, 리플레이 유지 |
| **CNN-10-리플레이초기화** | CNN-03 / `F_CUDA` | 같은 일시 중지 조건에 리플레이 초기화 추가 |
| **CNN-11-CUDA기준** | [CNN-04](../studies/CNN-04/README.md) / `B_CUDA` | CNN-04와 같은 설정을 CUDA에서 새로 학습 |
| **CNN-12-성공경험보존** | CNN-04 / `success` | 성공 경험 전용 슬롯 |
| **CNN-13-우선리플레이** | CNN-04 / `per` | TD 오차 기반 샘플링 |
| **CNN-14-3스텝보상** | CNN-04 / `nstep` | 최대 3스텝 보상 누적 |
| **CNN-15-더블DQN** | CNN-04 / `double` | 다음 행동 선택과 가치 추정에 다른 네트워크 사용 |
| **CNN-16-벽방향제외** | CNN-04 / `mask` | 벽·경계 행동 제외 |
| **CNN-17-훈련미로64** | CNN-04 / `train64` | 훈련 지도 16→64개 |
| **CNN-18-배치64** | CNN-04 / `batch64` | 배치 32→64 |
| **CNN-19-갱신주기2** | CNN-04 / `every2` | 4→2스텝마다 학습 갱신 |
| **CNN-20-타깃주기500** | CNN-04 / `target500` | 타깃 복사 주기 100→500스텝 |

CNN 표의 원본 묶음 `CNN-04`는 CUDA 후보 비교 전체를 가리킨다. 표시 이름 `CNN-04-탐색연장`은 그 비교에서 CPU 참고로 사용한 앞선 조건이다. [DIAG-01](../studies/CNN-02/diagnostics/DIAG-01/failure_diagnosis_20261104.md)은 이 CPU 체크포인트의 평가 진단으로, CNN 조건 번호를 부여하지 않는다.
