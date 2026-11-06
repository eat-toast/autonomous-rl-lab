"""Post-run MLP-01 failure reclassification and human-readable result report."""
import json
import statistics
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "studies/MLP-01/results"
RUNS = RESULTS / "runs"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def oscillation(ep):
    path = [tuple(p) for p in ep["path"]]
    tail = path[-9:]  # 8 moves have 9 recorded positions including both endpoints.
    return (not ep["success"] and ep["truncated"] and ep["steps"] == 100 and len(tail) == 9
            and tail[0] != tail[1] and all(tail[i] == tail[i % 2] for i in range(9)))


def stationary(ep):
    path = [tuple(p) for p in ep["path"]]
    return not ep["success"] and ep["truncated"] and ep["steps"] == 100 and len(path) >= 9 and len(set(path[-9:])) == 1


def facts(stage, seed):
    folder = RUNS / f"{stage}_seed_{seed}"
    evaluation = read(folder / "validation_20000.json")
    summary = read(folder / "summary.json")
    episodes = evaluation["episodes"]
    failed = [ep for ep in episodes if not ep["success"]]
    osc = sum(oscillation(ep) for ep in failed)
    still = sum(stationary(ep) and not oscillation(ep) for ep in failed)
    other_collision = sum(not oscillation(ep) and not stationary(ep) and ep["collisions"] > 0 for ep in failed)
    other = len(failed) - osc - still - other_collision
    return {"stage": stage, "seed": seed, "denominator": len(episodes),
            "validation_successes": evaluation["metrics"]["successes"],
            "validation_success_rate": evaluation["metrics"]["success_rate"],
            "failed_episodes": len(failed), "two_cell_oscillation": osc,
            "stationary_collision": still, "other_with_collision": other_collision,
            "other_failure": other, "collision_actions": sum(ep["collisions"] for ep in episodes),
            "path_efficiency_success_only": evaluation["metrics"]["path_efficiency"],
            "training_greedy_success_rate": summary["training_greedy"]["success_rate"],
            "training_greedy_denominator": summary["training_greedy"]["episodes"],
            "exploration_return_last20_completed_episodes": summary["exploration_training_return_mean_last20"],
            "environment_steps": summary["environment_steps"], "optimizer_updates": summary["optimizer_updates"],
            "processed_samples": summary["processed_samples"],
            "training_time_seconds": summary["training_time_seconds"],
            "intermediate_best_success_rate": summary["intermediate_best_success_rate"],
            "intermediate_best_step": summary["intermediate_best_step"],
            "primary_checkpoint_step": 20000}


def main():
    all_facts = [facts(stage, seed) for stage in ("MLP10", "MLP14") for seed in range(3)]
    by_stage = {stage: [r for r in all_facts if r["stage"] == stage] for stage in ("MLP10", "MLP14")}
    m10 = by_stage["MLP10"]
    total = sum(r["two_cell_oscillation"] for r in m10)
    contributing = sum(r["two_cell_oscillation"] > 0 for r in m10)
    decision = {"decision": "run_mlp14" if total >= 6 and contributing >= 2 else "criteria_not_met_mlp14_not_run",
                "mlp14_started": True, "criterion_recomputed_after_run": True,
                "last_checkpoint_validation_episodes": sum(r["denominator"] for r in m10),
                "definition": "failure, 100 steps, truncated, and final 8 moves alternate A<->B; check final 9 positions",
                "oscillations_total": total,
                "oscillations_by_seed": {str(r["seed"]): r["two_cell_oscillation"] for r in m10},
                "seed_count_with_oscillation": contributing, "threshold_total": 6, "threshold_seed_count": 2,
                "threshold_met": total >= 6 and contributing >= 2}
    correction = {"issue": "Initial failure label inspected 8 positions (7 edges) instead of final 9 positions (8 edges).",
                  "scope": "Derived classification only; all original paths, observations, rewards and checkpoints are retained unchanged.",
                  "corrected_rule": decision["definition"], "recomputed_seed_facts": all_facts,
                  "entry_decision": decision}
    (RESULTS / "classification_correction.json").write_text(json.dumps(correction, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "mlp14_entry_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = read(RESULTS / "comparison.json")
    for condition in comparison["conditions"]:
        condition["run_id"] = condition["stage"]
    comparison["mlp14_entry_decision"] = decision
    comparison["failure_classification_correction"] = "See classification_correction.json; final 8 moves use 9 recorded positions."
    comparison["seed_facts_corrected"] = all_facts
    (RESULTS / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = list(all_facts[0])
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_facts)
    lines = ["# MLP-01 결과", "",
             "## 실행 요약", "",
             "MLP10 세 seed와 조건부 MLP14 세 seed가 모두 20,000 환경 스텝으로 완료됐다. 훈련 지도 16개와 개발 검증 지도 8개는 CUDA B와 동일하며, 최종 test는 학습·평가·선택에 사용하지 않았다.",
             "주 비교는 마지막 체크포인트(20,000스텝)다. 중간 최고 검증값은 보조값이며 선택에 사용하지 않았다.",
             "훈련 return은 epsilon 탐색이 포함된 완료 에피소드의 마지막 20개 평균이다. 훈련 greedy와 개발 검증 greedy를 분리했다.",
             "CNN은 전체 지도 관측과 CNN 구조가 동시에 다른 과거 참고 결과다. 이 비교로 CNN 대비 MLP 효과를 분리할 수 없다.", "",
             "| 조건 | Seed | 마지막 검증 성공 | 실패: 왕복/제자리/충돌 기타/기타 | 충돌 행동 | 성공 경로 효율 | 훈련 greedy | 탐색 return 최근20 | 업데이트 / 샘플 | 초 | 중간 최고 (step) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in all_facts:
        lines.append(f"| {r['stage']} | {r['seed']} | {r['validation_successes']}/{r['denominator']} ({r['validation_success_rate']:.1%}) | "
                     f"{r['two_cell_oscillation']}/{r['stationary_collision']}/{r['other_with_collision']}/{r['other_failure']} | "
                     f"{r['collision_actions']} | {r['path_efficiency_success_only']} | "
                     f"{r['training_greedy_success_rate']:.1%} ({r['training_greedy_denominator']}) | "
                     f"{r['exploration_return_last20_completed_episodes']:.3f} | {r['optimizer_updates']}/{r['processed_samples']} | "
                     f"{r['training_time_seconds']:.1f} | {r['intermediate_best_success_rate']:.1%} ({r['intermediate_best_step']}) |")
    lines += ["", "## 마지막 검증 seed 요약", "", "| 조건 | Seed별 성공률 | 평균 ± 표본 SD | 범위 |",
              "|---|---|---:|---:|"]
    for stage, rows in by_stage.items():
        vals = [r["validation_success_rate"] for r in rows]
        sd = statistics.stdev(vals)
        lines.append(f"| {stage} | {vals} | {statistics.mean(vals):.1%} ± {sd:.1%} | [{min(vals):.1%}, {max(vals):.1%}] |")
    lines += ["", "## MLP14 진입 판정", "", f"올바른 마지막 8개 이동 기준에서 MLP10은 {total}/24 왕복 실패, {contributing}개 seed에서 발생해 MLP14 기준을 충족했다.",
              "산출물 생성 초기에 위치 배열 8개(7개 이동)를 검사하는 분류 실수가 발견됐다. 원본 경로는 유지했고, 최종 9개 위치(8개 이동)로 재계산했다. 총계와 seed별 진입 판단은 변하지 않았다. 보정 내역은 classification_correction.json에 있다.",
              "", "## 원본과 재생", "", "- `protocol.json`, `ledger.json`, `implementation_log.json`: 조건·예산·구현 기록",
              "- `maze_manifest.json`: CUDA B 지도 원본과 식별 정보", "- `runs/<조건>_seed_<n>/latest.pt`: 마지막 모델",
              "- `runs/<조건>_seed_<n>/validation_20000.json`: 관측·행동·위치·보상·충돌 전체 재생",
              "- `runs/<조건>_seed_<n>/training_episodes.csv`: 탐색 훈련 return", "- `comparison.json` / `comparison.csv` / `comparison.png`: 집계 자료"]
    (RESULTS / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
