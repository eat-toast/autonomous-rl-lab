"""Summarize the paired final-checkpoint comparison without selecting a checkpoint."""
from pathlib import Path
import statistics

from run_mlp02 import BASE, RESULTS, SEEDS, read, write_json


def condition(study, run_id, seeds):
    rows = []
    episodes = {}
    for seed in seeds:
        folder = study / "runs" / f"MLP14_seed_{seed}"
        summary = read(folder / "summary.json")
        rows.append({
            "seed": seed,
            "validation": summary["last_validation"],
            "training_greedy": summary["training_greedy"],
            "exploration_return_last20": summary["exploration_training_return_mean_last20"],
            "intermediate_best_step": summary["intermediate_best_step"],
            "intermediate_best_success_rate": summary["intermediate_best_success_rate"],
        })
        episodes[seed] = read(folder / "validation_20000.json")["episodes"]
    rates = [r["validation"]["success_rate"] for r in rows]
    return {
        "run_id": run_id, "seeds": rows,
        "success_rate_mean": statistics.mean(rates) if rates else None,
        "success_rate_sample_sd": statistics.stdev(rates) if len(rates) > 1 else None,
        "success_rate_range": [min(rates), max(rates)] if rates else None,
    }, episodes


def main():
    ledger = read(RESULTS / "ledger.json")
    completed = [r["seed"] for r in ledger if r["status"] == "completed"]
    baseline, before = condition(BASE, "MLP-01/MLP14", SEEDS)
    candidate, after = condition(RESULTS, "MLP-02/MLP14-mask", completed)
    paired = []
    for seed in completed:
        for old, new in zip(before[seed], after[seed], strict=True):
            if old["maze_hash"] != new["maze_hash"]:
                raise ValueError("Validation maze order differs")
            paired.append({"seed": seed, "maze_hash": old["maze_hash"],
                           "baseline": old["failure_type"], "mask": new["failure_type"]})
    decision = "deferred_incomplete"
    if tuple(completed) == SEEDS:
        previous = [r["validation"]["successes"] for r in baseline["seeds"]]
        current = [r["validation"]["successes"] for r in candidate["seeds"]]
        adopt = sum(current) >= 21 and all(a >= b for a, b in zip(current, previous, strict=True))
        decision = "adopt" if adopt else "retain_baseline"
    comparison = {"conditions": [baseline, candidate], "paired_validation": paired, "decision": decision}
    write_json(RESULTS / "comparison.json", comparison)

    lines = [
        "# MLP-02 결과", "",
        "MLP14에 벽 방향 제외를 적용했다. 기존 MLP-01의 같은 seed와 같은 미로에서 마지막 20,000스텝 greedy 정책을 비교했다.",
        "",
        "| 조건 | Seed | 개발 검증 | 두 칸 왕복 실패 | 제자리 충돌 실패 | 기타 실패 | 충돌 행동 | 훈련 greedy | 탐색 return 최근 20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in zip(("MLP14", "MLP14 + 벽 방향 제외"), comparison["conditions"], strict=True):
        for row in result["seeds"]:
            val = row["validation"]
            train = row["training_greedy"]
            fail = val["failure_types"]
            lines.append(
                f"| {name} | {row['seed']} | {val['successes']}/{val['episodes']} | "
                f"{fail['two_cell_oscillation']} | {fail['stationary_collision']} | "
                f"{fail['other_with_collision'] + fail['other']} | {val['collisions']} | "
                f"{train['successes']}/{train['episodes']} | {row['exploration_return_last20']:.3f} |"
            )
        rates = result["success_rate_range"]
        if rates and result["success_rate_sample_sd"] is not None:
            lines.extend([
                f"| **{name} 요약** | {len(result['seeds'])}개 | "
                f"평균 {result['success_rate_mean']:.1%} ± {result['success_rate_sample_sd'] * 100:.1f}%p "
                f"(표본 SD), {rates[0]:.1%}–{rates[1]:.1%} | | | | | | |",
            ])
    gained = sum(p["baseline"] != "success" and p["mask"] == "success" for p in paired)
    lost = sum(p["baseline"] == "success" and p["mask"] != "success" for p in paired)
    decision_ko = {"adopt": "후속 후보 채택", "retain_baseline": "기존 조건 유지",
                   "deferred_incomplete": "미완료로 판정 보류"}[decision]
    lines.extend(["", f"같은 seed·미로 {len(paired)}쌍에서 실패→성공 {gained}건, 성공→실패 {lost}건이었다.",
                  f"사전 기준에 따른 판정은 **{decision_ko}**다.", "",
                  "개발 검증과 훈련 greedy를 분리했으며 최종 테스트는 사용하지 않았다.",
                  "원본 경로와 seed별 중간 평가 및 경로 효율은 비교 JSON과 각 run에 있다."])
    report = Path(RESULTS / "report.md")
    with report.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
