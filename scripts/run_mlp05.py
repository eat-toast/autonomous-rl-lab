"""Execute the approved MLP5 comparison, with a single sequential CUDA worker."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import os
import statistics
import subprocess
import sys
import time
import traceback

from run_mlp02 import bootstrap, choose, explore
from run_mlp01 import LOCK, ROOT, read, runtime_metadata, sha, snapshot_source, torch, train_worker, utc_now, write_json

BASE = ROOT / "studies/MLP-04/results"
RESULTS = ROOT / "studies/MLP-05/results"
STAGE = "MLP14-count-mask"
SEEDS = (0, 1, 2)
CONFIG = RESULTS / f"config_{STAGE}.json"
MANIFEST = RESULTS / "maze_manifest.json"
RECOVERY_REF = "codex/mlp05-pre-execution"
SOFT_SECONDS = 300
HARD_SECONDS = 330
TOTAL_SECONDS = 1800
POST_RUN_RESERVE = 180
STEPS = 20000


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def summarize(study, stage, seeds):
    rows, paths = [], {}
    for seed in seeds:
        folder = study / "runs" / f"{stage}_seed_{seed}"
        summary = read(folder / "summary.json")
        validation = read(folder / "validation_20000.json")
        if summary["environment_steps"] != STEPS or validation["metrics"]["episodes"] != 8:
            raise ValueError(f"Invalid final summary at {folder}")
        rows.append({
            "seed": seed, "validation": validation["metrics"],
            "training_greedy": summary["training_greedy"],
            "exploration_return_last20": summary["exploration_training_return_mean_last20"],
            "intermediate_best_step": summary["intermediate_best_step"],
            "intermediate_best_success_rate": summary["intermediate_best_success_rate"],
            "checkpoint_sha256": sha(folder / "latest.pt"),
        })
        paths[seed] = validation["episodes"]
    rates = [r["validation"]["success_rate"] for r in rows]
    return {
        "run_id": f"{study.parent.name}/{stage}", "seeds": rows,
        "successes": sum(r["validation"]["successes"] for r in rows),
        "episodes": sum(r["validation"]["episodes"] for r in rows),
        "success_rate_mean": statistics.mean(rates) if rates else None,
        "success_rate_sample_sd": statistics.stdev(rates) if len(rates) > 1 else None,
        "success_rate_range": [min(rates), max(rates)] if rates else None,
    }, paths


def aggregate(ledger, protocol):
    completed = [entry["seed"] for entry in ledger if entry["status"] == "completed"]
    baseline, before = summarize(BASE, "MLP14-count", SEEDS)
    candidate, after = summarize(RESULTS, STAGE, completed)
    paired = []
    for seed in completed:
        for old, new in zip(before[seed], after[seed], strict=True):
            if old["maze_hash"] != new["maze_hash"] or old["maze_seed"] != new["maze_seed"]:
                raise ValueError("Validation maze order differs")
            paired.append({
                "seed": seed, "maze_hash": old["maze_hash"], "maze_seed": old["maze_seed"],
                "baseline": old["failure_type"], "candidate": new["failure_type"],
                "baseline_success": old["success"], "candidate_success": new["success"],
                "baseline_collisions": old["collisions"], "candidate_collisions": new["collisions"],
                "baseline_efficiency": old["path_efficiency"], "candidate_efficiency": new["path_efficiency"],
            })
    decision = "deferred_incomplete"
    if tuple(completed) == SEEDS:
        no_decline = all(a["validation"]["successes"] >= b["validation"]["successes"]
                         for a, b in zip(candidate["seeds"], baseline["seeds"], strict=True))
        collisions = sum(r["validation"]["collisions"] for r in candidate["seeds"])
        oscillations = sum(r["validation"]["failure_types"]["two_cell_oscillation"] for r in candidate["seeds"])
        decision = ("adopt" if candidate["successes"] >= 21 and no_decline
                    and collisions == 0 and oscillations <= 2 else "retain_baseline")
    transitions = {
        f"{old}_{new}": sum(row["baseline_success"] == (old == "success") and
                            row["candidate_success"] == (new == "success") for row in paired)
        for old in ("success", "failure") for new in ("success", "failure")
    }
    write_json(RESULTS / "comparison.json", {
        "study_id": "MLP-05", "conditions": [baseline, candidate], "paired_validation": paired,
        "transitions": transitions, "decision": decision,
    })
    lines = [
        "# MLP5 결과", "",
        f"실행 상태는 `{protocol['status']}`, 완료 seed는 {len(completed)}/3이다.",
        "동일한 seed와 미로의 마지막 20,000스텝 체크포인트를 비교했다.", "",
        "| 조건 | Seed | 개발 검증 | 두 칸 왕복 실패 | 제자리 충돌 실패 | 기타 실패 | 충돌 행동 | 성공 경로 효율 | 훈련 greedy | 탐색 return 최근 20 | 검증 최고 및 스텝 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    csv_rows = []
    for label, result in (("MLP4", baseline), ("MLP5", candidate)):
        for row in result["seeds"]:
            val, train = row["validation"], row["training_greedy"]
            fail = val["failure_types"]
            values = [
                label, row["seed"], f"{val['successes']}/{val['episodes']}",
                fail["two_cell_oscillation"], fail["stationary_collision"],
                fail["other_with_collision"] + fail["other"], val["collisions"],
                val["path_efficiency"], f"{train['successes']}/{train['episodes']}",
                row["exploration_return_last20"],
                f"{row['intermediate_best_success_rate']:.1%} @ {row['intermediate_best_step']}",
            ]
            csv_rows.append(values)
            lines.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in values) + " |")
    lines.append("")
    for label, result in (("MLP4", baseline), ("MLP5", candidate)):
        if not result["episodes"]:
            continue
        low, high = result["success_rate_range"]
        sd = result["success_rate_sample_sd"]
        sd_text = f"{sd * 100:.2f}%p" if sd is not None else "계산 불가"
        lines.append(f"- {label} {result['successes']}/{result['episodes']}, seed 평균 "
                     f"{result['success_rate_mean']:.2%}, 표본 표준편차 {sd_text}, 범위 {low:.2%}–{high:.2%}.")
    lines.extend([
        f"- 동일한 seed·미로 {len(paired)}쌍의 실패→성공 {transitions['failure_success']}건, "
        f"성공→실패 {transitions['success_failure']}건.",
        f"- 사전 채택 판정은 `{decision}`이다. 독립 검증 상태는 별도 기록을 따른다.",
        "- 최종 테스트 사용량은 0이다.",
        "", "[원장](ledger.json) · [비교 JSON](comparison.json) · [독립 검증](review.json)",
    ])
    with (RESULTS / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["condition", "seed", "validation", "two_cell_oscillation", "stationary_collision",
                         "other_failures", "collisions", "success_path_efficiency", "training_greedy",
                         "exploration_return_last20", "best_validation"])
        writer.writerows(csv_rows)
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_worker(entry, runs):
    seed = entry["seed"]
    run_id = f"{STAGE}_seed_{seed}"
    folder = runs / run_id
    seed_started = time.monotonic()
    entry.update(run_id=run_id, status="running", started_at_utc=utc_now(), charged_steps=STEPS)
    write_json(RESULTS / "ledger.json", LEDGER)
    print(f"Starting seed {seed}", flush=True)
    with (runs / f"{run_id}_worker.log").open("x", encoding="utf-8") as log:
        try:
            child = subprocess.run(
                [sys.executable, "-u", __file__, "--worker", str(seed), "--worker-started", str(seed_started)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=HARD_SECONDS - (time.monotonic() - seed_started),
            )
            entry["return_code"] = child.returncode
            entry["status"] = "completed" if child.returncode == 0 and (folder / "summary.json").exists() else "failed"
        except subprocess.TimeoutExpired:
            entry["status"] = "hard_timeout"
    entry.update(elapsed_seconds=time.monotonic() - seed_started, finished_at_utc=utc_now())
    if entry["status"] == "completed" and entry["elapsed_seconds"] >= SOFT_SECONDS:
        entry["status"] = "soft_timeout"
    for name in ("summary.json", "failure.json"):
        if (folder / name).is_file():
            result = read(folder / name)
            entry["environment_steps"] = result["environment_steps"]
            entry["charged_steps"] = entry["environment_steps"]
            if result.get("error", "").startswith("seed soft wall-time"):
                entry["status"] = "soft_timeout"
            break
    if "environment_steps" not in entry and (folder / "progress.json").is_file():
        entry["last_persisted_step"] = read(folder / "progress.json")["step"]
    write_json(RESULTS / "ledger.json", LEDGER)
    print(f"Seed {seed}: {entry['status']}, {entry['elapsed_seconds']:.2f}s", flush=True)


LEDGER = [{"seed": seed, "status": "not_started", "charged_steps": 0} for seed in SEEDS]


def run_study():
    runs = RESULTS / "runs"
    runs.mkdir()
    started_at = datetime.now(timezone.utc)
    protocol = {
        "study_id": "MLP-05", "status": "running", "started_at_utc": started_at.isoformat(),
        "deadline_at_utc": (started_at + timedelta(seconds=TOTAL_SECONDS)).isoformat(),
        "approval": "User approved MLP5 with one RTX 4060 worker and no retries.",
        "seeds": SEEDS, "steps_per_seed": STEPS, "max_total_steps": STEPS * len(SEEDS),
        "max_retries": 0, "max_elapsed_seconds": TOTAL_SECONDS,
        "soft_seconds_per_seed": SOFT_SECONDS, "hard_seconds_per_seed": HARD_SECONDS,
        "primary_checkpoint_step": STEPS, "validation_mazes": 8, "training_mazes": 16,
        "final_test_used": False, "adoption_min_successes": 21,
        "adoption_requires_no_seed_decline": True, "adoption_max_collisions": 0,
        "adoption_max_oscillations": 2,
    }
    write_json(RESULTS / "protocol.json", protocol)
    write_json(RESULTS / "ledger.json", LEDGER)
    started = time.monotonic()
    try:
        protocol["recovery_commit"] = git("rev-parse", RECOVERY_REF)
        protocol["config_sha256"] = sha(CONFIG)
        protocol["baseline_config_sha256"] = sha(BASE / "config_MLP14-count.json")
        protocol["baseline_comparison_sha256"] = sha(BASE / "comparison.json")
        protocol["baseline_checkpoint_sha256"] = {
            str(seed): sha(BASE / "runs" / f"MLP14-count_seed_{seed}" / "latest.pt") for seed in SEEDS
        }
        protocol["maze_manifest_sha256"] = sha(MANIFEST)
        if protocol["maze_manifest_sha256"] != sha(BASE / "maze_manifest.json"):
            raise ValueError("Original maze manifest differs")
        raw = read(CONFIG)
        baseline = read(BASE / "config_MLP14-count.json")
        changed = {k for k in raw.keys() | baseline.keys() if raw.get(k) != baseline.get(k)}
        if (changed != {"train", "hypothesis", "change_description"}
                or raw["train"] != {**baseline["train"], "method": "mask"}):
            raise ValueError("Config differs outside the approved action mask")
        protocol["code_hashes"] = snapshot_source(RESULTS)
        protocol["runtime"] = runtime_metadata("cuda")
        if "RTX 4060" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Approved RTX 4060 CUDA worker unavailable")
        write_json(RESULTS / "protocol.json", protocol)
        for entry in LEDGER:
            if time.monotonic() - started + HARD_SECONDS + POST_RUN_RESERVE > TOTAL_SECONDS:
                entry["reason"] = "insufficient time remaining for a full seed allowance and independent verification"
                break
            run_worker(entry, runs)
            if entry["status"] != "completed":
                break
        protocol["status"] = "completed" if all(e["status"] == "completed" for e in LEDGER) else "incomplete"
    except Exception:
        protocol.update(status="failed", traceback=traceback.format_exc())
    finally:
        write_json(RESULTS / "ledger.json", LEDGER)
        try:
            aggregate(LEDGER, protocol)
        except Exception:
            protocol.update(status="failed", aggregation_traceback=traceback.format_exc())
        protocol["finished_at_utc"] = utc_now()
        protocol["execution_seconds"] = time.monotonic() - started
        write_json(RESULTS / "protocol.json", protocol)
    return int(protocol["status"] != "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=SEEDS)
    parser.add_argument("--worker-started", type=float)
    args = parser.parse_args()
    if args.worker is not None:
        if args.worker_started is None:
            parser.error("--worker-started is required with --worker")
        if not LOCK.is_file() or LOCK.read_text(encoding="utf-8") != str(os.getppid()):
            parser.error("worker must be launched by the locked study runner")
        train_worker(STAGE, args.worker, study=RESULTS, raw=read(CONFIG), worker_started=args.worker_started,
                     choose=choose, explore=explore, bootstrap=bootstrap)
        return 0
    with LOCK.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    try:
        return run_study()
    finally:
        LOCK.unlink()


if __name__ == "__main__":
    sys.exit(main())
