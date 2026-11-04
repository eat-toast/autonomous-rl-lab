# Phase 1 실행 보고서

검증일: 2026-11-04 (한국 시간).

## 구현

기존 소스가 없고 참고만 하도록 승인되어 새 CNN DQN을 구현했다.
3채널 전체 지도, 상하좌우 행동, 고정 보상 규칙은 `design.md`에 기록했다.
10×10 모델 파라미터 수는 210,212개다. 이 수치는 기존 게시물 모델과 다르다.
도달 가능한 무작위 장애물 미로, A*, train/validation/test 분리,
bounded worker 실행, CSV/JSON, 체크포인트, 오류 보존, 간단한 그림을 제공한다.

## 실제 검증 결과

- `python -m unittest discover -s tests -v`: 15개 통과.
- A* 결과를 독립 BFS와 5/10/15/20/30/40 크기에서 비교했다.
- 목표 도달/시간 제한의 구분, collision 보상, terminal bootstrap mask를 확인했다.
- 온라인 모델만 optimizer로 갱신되고 target은 동기화 시에만 바뀜을 확인했다.
- 40스텝 동일 seed 실행 두 번의 가중치/검증 결과가 정확히 일치했다.
- 시간 제한 시 상태/오류/체크포인트가 남는 것을 확인했다.
- 실제 CLI smoke: `20261104T030730Z_0be51831`, 10×10, seed 0, 512 환경 스텝.
- train 미로 16개, validation 4개, test 8개. 최종 test 평가를 실행하지 않았다.
- seed worker의 초기화/훈련/검증/저장을 포함한 측정 시간 약 4.31초.
  프로세스 import와 바깥 실행기 전체 시간은 이 수치에 포함하지 않는다.
- 검증 성공률 0/4 (0%), 실패 평균 100스텝. 성공이 없어 경로 효율/성공 평균 스텝은 null.
- 독립 evaluate 명령으로 `latest.pt`를 복원했고 동일한 검증 지표를 확인했다.
- learning_curve.png, path_comparison.png를 생성하고 경로 그림의 표시를 확인했다.

이번 실행은 파이프라인 검증이다. 512스텝과 1 seed 결과에서 알고리즘 성능이나
일반화 능력에 대한 결론을 내리지 않는다. best checkpoint는 검증 동률일 경우
이전 모델을 유지하므로 이 smoke에서는 0스텝 모델이 best일 수 있다.

## 재현과 Git

구현 전 설계 커밋: `3004c12`.
smoke 당시 metadata는 해당 커밋과 dirty 상태를 기록한다.
실행한 구현은 `source_snapshot.zip`과 파일별 SHA-256으로 별도 보존했다.
실행 당시의 Git HEAD가 완성된 구현 커밋이었다고 해석하면 안 된다.
검증 후 전체 구현/문서를 로컬 Git에 커밋한다. GitHub push는 하지 않았다.

## 아직 하지 않은 일

Double/Dueling 및 나머지 알고리즘, 여러 크기의 장시간 학습/벤치마크,
영상 생성, AI 가설 생성/코드 수정 루프, 모델 학습 중단 지점 resume,
프로세스 전체 메모리의 OS 강제 제한은 후속 단계다.
현재는 replay 배열 용량과 CPU thread, 스텝, worker 시간 제한을 적용한다.
