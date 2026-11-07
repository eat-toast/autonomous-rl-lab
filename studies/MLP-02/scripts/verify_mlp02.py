"""Read-only MLP-02 checkpoint and comparison audit; writes only review records."""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from rl_lab.paths import resolve_legacy_path, study_runs, run_path
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import time

import torch

from scripts.study_modules import load as _load_study
_verify_mlp01_checkpoint = _load_study("MLP-01", "verify_mlp01_checkpoint")
MOVES = _verify_mlp01_checkpoint.MOVES
ROOT = _verify_mlp01_checkpoint.ROOT
RestoredPolicy = _verify_mlp01_checkpoint.RestoredPolicy
make_observation = _verify_mlp01_checkpoint.make_observation
oscillation = _verify_mlp01_checkpoint.oscillation
read = _verify_mlp01_checkpoint.read

BASELINE = ROOT / "studies/MLP-01/results"
STUDY = ROOT / "studies/MLP-02/results"
SEEDS = (0, 1, 2)
RUN_IDS = ("MLP-01/MLP14", "MLP-02/MLP14-mask")
CHECKPOINT_STEP = 20000
FAILURE_TYPES = ("two_cell_oscillation", "stationary_collision", "other_with_collision", "other")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, result):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (float, int)):
        return math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-7)
    return a == b


def replay(model, raw, mask):
    grid = raw["grid"]
    pos = (0, 0)
    visited = {pos}
    path, observations, actions, rewards, collisions = [list(pos)], [], [], [], []
    for _ in range(100):
        obs = make_observation(grid, pos, visited, True)
        neighbors = [(pos[0] + dr, pos[1] + dc) for dr, dc in MOVES]
        allowed = [0 <= r < 10 and 0 <= c < 10 and grid[r][c] == 0 for r, c in neighbors]
        scores = model(torch.tensor([obs], dtype=torch.float32))[0].tolist()
        candidates = [a for a in range(4) if not mask or allowed[a]]
        action = max(candidates, key=scores.__getitem__)
        hit = not allowed[action]
        if not hit:
            pos = neighbors[action]
            visited.add(pos)
        success = pos == (9, 9)
        observations.append(obs)
        actions.append(action)
        collisions.append(hit)
        rewards.append(1.0 if success else (-0.05 if hit else -0.01))
        path.append(list(pos))
        if success:
            break
    if success:
        label = "success"
    elif len({tuple(p) for p in path[-9:]}) == 1:
        label = "stationary_collision"
    elif oscillation(path):
        label = "two_cell_oscillation"
    else:
        label = "other_with_collision" if any(collisions) else "other"
    return {
        "maze_seed": raw["seed"], "maze_hash": raw["hash"], "success": success,
        "steps": len(actions), "terminated": success, "truncated": not success,
        "path": path, "actions": actions, "observations": observations, "rewards": rewards,
        "collision_flags": collisions, "collisions": sum(collisions), "return": sum(rewards),
        "shortest_length": raw["shortest_length"], "failure_type": label,
        "path_efficiency": raw["shortest_length"] / len(actions) if success else None,
    }


def metrics(episodes):
    successes = [row for row in episodes if row["success"]]
    failures = Counter(row["failure_type"] for row in episodes if not row["success"])
    return {
        "episodes": len(episodes), "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "failed_episodes": len(episodes) - len(successes),
        "path_efficiency": statistics.mean(r["path_efficiency"] for r in successes) if successes else None,
        "mean_success_steps": statistics.mean(r["steps"] for r in successes) if successes else None,
        "collisions": sum(r["collisions"] for r in episodes),
        "failure_types": {kind: failures[kind] for kind in FAILURE_TYPES},
    }


def exploration_return(folder, summary, train_mazes, mask):
    with (folder / "training_episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    maze_seeds = {m["seed"] for m in train_mazes}
    previous_step = 0
    returns = []
    for i, row in enumerate(entries, 1):
        steps, end, collisions = (int(row[k]) for k in ("steps", "end_step", "collisions"))
        success = row["success"] == "True"
        require(int(row["episode"]) == i and int(row["maze_seed"]) in maze_seeds
                and 1 <= steps <= 100 and end == previous_step + steps and end <= CHECKPOINT_STEP
                and (success or steps == 100) and 0 <= collisions <= steps - success,
                f"training CSV order, maze, or steps mismatch in {folder}, episode {i}")
        require(not mask or collisions == 0, f"masked exploration hit a wall in {folder}, episode {i}")
        previous_step = end
        expected = success - .05 * collisions - .01 * (steps - collisions - success)
        logged = float(row["return"])
        require(close(logged, expected), f"training return mismatch in {folder}, episode {i}")
        returns.append(logged)
    require(len(entries) == summary["completed_training_episodes"] and CHECKPOINT_STEP - previous_step < 100,
            f"training CSV count mismatch in {folder}")
    result = statistics.mean(returns[-20:])
    require(close(result, summary["exploration_training_return_mean_last20"]), f"training return mismatch in {folder}")
    return result


@torch.inference_mode()
def audit_seed(folder, seed, splits, mask):
    checkpoint_path = folder / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require((checkpoint["stage"], checkpoint["seed"], checkpoint["step"], checkpoint["input_size"])
            == ("MLP14", seed, CHECKPOINT_STEP, 14), f"checkpoint identity mismatch in {folder}")
    model = RestoredPolicy(14).eval()
    model.load_state_dict(checkpoint["model_state"])
    summary = read(folder / "summary.json")
    require((summary["status"], summary["seed"], summary["environment_steps"], summary["optimizer_updates"],
             summary["processed_samples"], summary["parameter_count"], summary["final_test_used"])
            == ("completed", seed, CHECKPOINT_STEP, 4985, 159520, 5380, False), f"summary budget mismatch in {folder}")
    history = summary["validation_history"]
    require([item["step"] for item in history] == list(range(0, CHECKPOINT_STEP + 1, 500))
            and sorted(p.name for p in folder.glob("validation_*.json"))
            == [f"validation_{step:05d}.json" for step in range(0, CHECKPOINT_STEP + 1, 500)],
            f"validation schedule mismatch in {folder}")
    best = max(history, key=lambda row: (row["success_rate"],
                                       row["path_efficiency"] if row["path_efficiency"] is not None else -1))
    require((best["step"], best["success_rate"])
            == (summary["intermediate_best_step"], summary["intermediate_best_success_rate"]),
            f"intermediate best mismatch in {folder}")
    audit = {"seed": seed, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
             "checkpoint_sha256": sha(checkpoint_path), "step": CHECKPOINT_STEP}
    audit["exploration_return_last20"] = exploration_return(folder, summary, splits["train"], mask)
    audit["intermediate_best_step"] = best["step"]
    audit["intermediate_best_success_rate"] = best["success_rate"]
    for split, filename, summary_key in (
        ("validation", "validation_20000.json", "last_validation"),
        ("train", "final_training_evaluation.json", "training_greedy"),
    ):
        artifact = read(folder / filename)
        require(len(artifact["episodes"]) == len(splits[split]), f"episode count mismatch in {folder}/{filename}")
        episodes = []
        for maze, saved in zip(splits[split], artifact["episodes"]):
            rebuilt = replay(model, maze, mask)
            mismatches = [key for key, value in rebuilt.items() if not close(value, saved[key])]
            require(not mismatches, f"replay mismatch in {folder}/{filename}, maze {maze['seed']}: {mismatches}")
            episodes.append(rebuilt)
        computed = metrics(episodes)
        require(close(computed, artifact["metrics"]) and close(computed, summary[summary_key]),
                f"aggregate mismatch in {folder}/{filename}")
        require(not mask or computed["collisions"] == 0, f"masked greedy collision in {folder}/{filename}")
        audit[split] = {"metrics": computed, "episodes": [
            {k: row[k] for k in ("maze_seed", "maze_hash", "success", "steps", "failure_type",
                                 "collisions", "path_efficiency")} for row in episodes
        ]}
    return audit


def audit_conditions(protocol):
    for key, location in {
        "maze_manifest_sha256": STUDY / "maze_manifest.json",
        "config_sha256": STUDY / "config_MLP14.json",
        "baseline_config_sha256": BASELINE / "config_MLP14.json",
        "baseline_comparison_sha256": BASELINE / "comparison.json",
    }.items():
        require(sha(location) == protocol[key], f"protocol {key} does not match")
    review = read(BASELINE / "review.json")
    require(review["status"] == "approved_for_dev_report" and "MLP14" in review["approved_run_ids"]
            and review["comparison_sha256"] == protocol["baseline_comparison_sha256"]
            and review["protocol_sha256"] == sha(BASELINE / "protocol.json"), "baseline provenance mismatch")
    require(sha(BASELINE / "maze_manifest.json") == protocol["maze_manifest_sha256"], "manifest changed")
    baseline, candidate = (read(folder / "config_MLP14.json") for folder in (BASELINE, STUDY))
    require(baseline["train"]["method"] == "baseline" and candidate["train"]["method"] == "mask",
            "incorrect action mask control/candidate assignment")
    baseline["train"]["method"] = "mask"
    for key in ("hypothesis", "change_description"):
        baseline.pop(key)
        candidate.pop(key)
    require(baseline == candidate, "more than the approved variable changed in candidate config")
    runtime = protocol["runtime"]
    prior_runtime = read(BASELINE / "protocol.json")["runtime"]
    require(all(runtime[k] == prior_runtime[k] for k in ("python", "packages", "device", "accelerator", "torch_threads")),
            "runtime differs from baseline")
    require(protocol["primary_checkpoint_step"] == CHECKPOINT_STEP
            and protocol["seeds"] == list(SEEDS) and protocol["final_test_used"] is False
            and protocol["adoption_min_successes"] == 21 and protocol["adoption_requires_no_seed_decline"] is True,
            "protocol diverges from the approved plan")
    ledger = read(STUDY / "ledger.json")
    require([row["seed"] for row in ledger] == list(SEEDS) and protocol["max_retries"] == 0,
            "unexpected ledger seed or retry")
    require(all(row["status"] == "completed" and row["environment_steps"] == CHECKPOINT_STEP
                and row["charged_steps"] == CHECKPOINT_STEP and 0 < row["elapsed_seconds"] <= 330 for row in ledger)
            and sum(row["charged_steps"] for row in ledger) == protocol["max_total_steps"] == 60000,
            "training budgets or completed seed count mismatch")


def verify_comparison(comparison, audits):
    require([row["run_id"] for row in comparison["conditions"]] == list(RUN_IDS), "comparison run IDs differ")
    verified = {}
    for condition, runs in zip(comparison["conditions"], audits):
        require([row["seed"] for row in condition["seeds"]] == list(SEEDS), "comparison seed order differs")
        for row, audit in zip(condition["seeds"], runs):
            require(close(row["validation"], audit["validation"]["metrics"])
                    and close(row["training_greedy"], audit["train"]["metrics"])
                    and close(row["exploration_return_last20"], audit["exploration_return_last20"]),
                    f"comparison seed summary mismatch for {condition['run_id']}, seed {row['seed']}")
        validation = [run["validation"]["metrics"] for run in runs]
        rates = [row["success_rate"] for row in validation]
        expected = {
            "success_rate_mean": statistics.mean(rates), "success_rate_sample_sd": statistics.stdev(rates),
            "success_rate_range": [min(rates), max(rates)],
        }
        require(all(close(value, condition[key]) for key, value in expected.items()), "comparison aggregate differs")
        verified[condition["run_id"]] = {
            **expected, "success_rates": rates, "successes_by_seed": [r["successes"] for r in validation],
            "validation_episodes_by_seed": [r["episodes"] for r in validation],
            "failure_types_by_seed": [r["failure_types"] for r in validation],
            "collision_actions_by_seed": [r["collisions"] for r in validation],
            "success_path_efficiency_by_seed": [r["path_efficiency"] for r in validation],
        }
    paired = []
    for base, mask in zip(*audits):
        for before, after in zip(base["validation"]["episodes"], mask["validation"]["episodes"]):
            require(before["maze_hash"] == after["maze_hash"], "pair identity differs")
            paired.append({"seed": base["seed"], "maze_hash": before["maze_hash"],
                           "baseline": before["failure_type"], "mask": after["failure_type"]})
    require(paired == [{k: row[k] for k in paired[0]} for row in comparison["paired_validation"]],
            "paired validation records differ")
    prior, candidate = (verified[run_id]["successes_by_seed"] for run_id in RUN_IDS)
    adopt = sum(candidate) >= 21 and all(a >= b for a, b in zip(candidate, prior))
    decision = "adopt" if adopt else "retain_baseline"
    require(comparison["decision"] == decision, "adoption decision differs")
    return verified, paired, decision


def main():
    torch.set_num_threads(1)
    protocol_path, comparison_path = (STUDY / name for name in ("protocol.json", "comparison.json"))
    protocol = read(protocol_path)
    result = {
        "study_id": "MLP-02", "reviewer": "results_review", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(protocol_path), "comparison_sha256": sha(comparison_path),
    }
    if protocol["status"] != "completed":
        save(STUDY / "review.json", {**result, "status": "incomplete", "detail": "Candidate execution incomplete"})
        return
    try:
        audit_conditions(protocol)
        splits = read(STUDY / "maze_manifest.json")["splits"]
        require(len(splits["validation"]) == 8 and len(splits["train"]) == 16, "maze counts differ")
        audits = [
            [audit_seed(study_runs(folder) / f"MLP14_seed_{seed}", seed, splits, mask) for seed in SEEDS]
            for folder, mask in ((BASELINE, False), (STUDY, True))
        ]
        save(STUDY / "checkpoint_restore_audit.json", {"device": "cpu", "conditions": dict(zip(RUN_IDS, audits))})
        facts, paired, decision = verify_comparison(read(comparison_path), audits)
        elapsed = time.time() - datetime.fromisoformat(protocol["started_at_utc"]).timestamp()
        require(elapsed <= protocol["max_elapsed_seconds"] == 1800, "whole-study elapsed budget exceeded")
    except (FileNotFoundError, TimeoutError) as error:
        save(STUDY / "review.json", {**result, "status": "incomplete", "detail": str(error)})
        raise
    except (ValueError, KeyError) as error:
        save(STUDY / "review.json", {**result, "status": "rejected", "detail": str(error)})
        raise
    result.update(status="approved_for_dev_report", approved_run_ids=list(RUN_IDS),
                  review_elapsed_since_execution_started_seconds=elapsed,
                  checkpoint_step=CHECKPOINT_STEP, final_test_used=False,
                  scope="CPU restore of 6 final checkpoints; 48 validation and 96 train greedy episodes replayed.",
                  checks=["paired config and manifest", "runtime and ledger budgets", "action/observation/path/reward traces",
                          "training exploration return and greedy metrics", "paired validation outcomes and adoption rule"],
                  limit="One device, three paired seeds and eight previously used development validation mazes per seed.")
    save(STUDY / "verified_facts.json", {**result, "conditions": facts, "paired_validation": paired, "decision": decision})
    save(STUDY / "review.json", result)
    print(json.dumps({"status": result["status"], "conditions": facts, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
