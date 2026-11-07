# MLP1 기본 관측 · MLP2 이웃 방문

원본 묶음 ID는 `MLP-01`이다. 원본 조건 `MLP10`과 `MLP14`는 각각 **MLP-01-기본관측**과 **MLP-02-이웃방문**을 가리킨다.

[실험 목록](../../docs/experiment_index.md) · [계획서](mlp_feelers_study_plan.md) · [새 세션용 프롬프트](next_session_mlp_prompt.md)

계획서는 보존한다. 승인된 MLP-01 실행은 [프로토콜](results/protocol.json)과 [보고서](results/report.md)에 기록했다.
예산은 [원장](results/ledger.json), seed별 원본·체크포인트·재생 자료는 `results/runs/`에 있다.
학습·집계·독립 검증까지의 전체 벽시계는 [workflow timing](results/workflow_timing.json)에 별도 기록했다.
독립 결과 검토와 확인된 수치는 [검토](results/review.json), [검증 사실](results/verified_facts.json)을 따른다.
[체크포인트 복원 감사](results/checkpoint_restore_audit.json)는 여섯 마지막 모델과 48개 개발 경로를 재생 대조한다.
위치 8개와 이동 8회의 혼동을 발견해 보정한 내용은 [분류 보정 기록](results/classification_correction.json)에 남겼다.
[콘텐츠용 검증 사실과 재생 근거](results/instagram_handoff.md)는 승인된 수치와 대표 경로만 연결한다. 게시물은 제작하지 않았다.
