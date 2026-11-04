# Phase 1: 새 기준 모델의 실험 계약

2026-11-04, 사용자가 기존 DQN을 참고 자료로만 사용하도록 승인했다.
기존 코드를 재현한 결과가 아니라 새 구현이다. 코인 수집은 포함하지 않는다.

## 고정 조건 (maze-v1 / reward-v1 / observation-v1)

- N×N 격자. 시작 (0, 0), 목표 (N-1, N-1). 0=통로, 1=벽.
- 장애물은 각 칸에서 독립 확률로 생성한다. 시작/목표를 비우고 A*로
  도달 가능한 경우만 수락한다. 경로를 사후에 뚫지 않는다.
  요청 밀도와 실제 밀도, 생성 seed, 재시도 횟수, 격자 해시, 최단 경로를 기록한다.
- 입력: float32 (3, N, N). 순서: 벽, 현재 위치, 목표. 전체 지도 관측.
- 행동: 0 위, 1 오른쪽, 2 아래, 3 왼쪽. 벽/경계 충돌은 제자리 이동.
- 보상: 목표 도달 +1.0, 일반 이동 -0.01, 충돌 -0.05. 보상은 서로 더하지 않는다.
- 종료: 목표 도달은 terminated, 이동 횟수 제한은 truncated.
  DQN target은 terminated에만 bootstrap을 막는다. 시간 제한은 task의 종료가 아니다.
- CNN: Conv(3→16, 3×3, padding=1), ReLU, Conv(16→32, 3×3,
  padding=1), ReLU, Flatten, Linear(32N²→64), ReLU, Linear(64→4).
  크기마다 별도 모델. 파라미터 수는 실행마다 기록한다.
- DQN: 균등 replay, epsilon-greedy, Huber loss, Adam, gradient clipping,
  hard target update. Double/Dueling/PER/Rainbow는 Phase 2 이후.

## 실험 분리와 지표

훈련/개발 검증/최종 테스트는 seed 구간과 격자 해시 모두 분리한다.
같은 설정의 알고리즘/학습 seed는 같은 미로 세트를 사용한다.
Phase 1은 고정 유한 훈련 pool을 사용하며, 새 미로 무한 생성 분포 실험은 아직 아니다.
미로 pool 전체를 run에 저장하므로 seed 이외에도 격자 자체로 재현 가능하다.
검증은 epsilon=0이며 훈련 RNG/환경을 공유하지 않는다.
체크포인트 선택은 검증 결과만 사용한다. 최종 test는 train 중 평가하지 않는다.

- success_rate: 평가 미로 중 도달 비율 (정책 성공 seed 비율과 다름).
- average_success_steps, path_efficiency: 성공한 평가만 평균; 성공이 없으면 null.
- failed_episodes, average_failure_steps: 실패를 따로 집계한다.
- sample_efficiency_steps: 검증에서 목표 성공률에 처음 도달한 환경 스텝.
  미달이면 null; 지속적 성공을 보장하는 지표는 아니다.
- 각 seed의 값과 평균/표본 표준편차를 함께 저장한다. seed 1개는 표준편차 null.
- 검증 중 탐색 없이 실패하는 경로도 기록한다. 모든 행동(충돌 포함)을 이동 횟수로 센다.

## 실행 예산과 복구

기본 smoke는 1 seed × 512 스텝이다. 성공률을 결론내리는 연구 실험이 아니다.
각 seed에 스텝/시간/replay 메모리 제한, 실행기에는 자식 프로세스 hard timeout을 둔다.
CPU 1 thread를 기본으로 사용한다. 오류와 시간 제한 결과는 지우지 않는다.
run ID는 UTC 시각+UUID; 기존 디렉터리에 덮어쓰지 않는다.
현재 구현은 실패 후 새 run 재실행 방식이며 중간 replay/RNG를 복구하는 resume는 지원하지 않는다.

## 범위

Phase 1: 환경, DQN, 훈련/평가 분리, JSON/CSV, 체크포인트, 간단한 그림,
핵심 단위 테스트와 소규모 실행 검증. 장시간 학습은 실행하지 않는다.
Phase 2 이후: 알고리즘 비교/다중 크기 벤치마크/애니메이션/예산 내 AI 연구 루프.
현재 반복 코드 수정이나 무인 Codex 호출은 구성되어 있지 않다.

참고: https://www.instagram.com/p/DeBbUk9GPqj/ (전체 격자+CNN),
https://www.instagram.com/p/DeGN4_kGEBY/ (5값 관계형 입력+MLP).
두 실험의 성공률을 이 기준 모델의 재현 목표로 사용하지 않는다.
DQN 구현 참고: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
