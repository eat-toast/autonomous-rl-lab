"""Independent source/data audit and CPU final-checkpoint rollout. Never trains."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime, timezone
import zipfile
import copy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eq(a, b, message):
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        assert abs(a-b) < 1e-7, (message, a, b)
    else:
        assert a == b, (message, a, b)


def independent_rollout(model, env, maps, masked):
    """Batched CPU inference with independently implemented movement/mask/reward."""
    import numpy as np
    import torch
    n, count = env["size"], len(maps)
    grids = np.array([m["grid"] for m in maps], dtype=np.float32)
    positions = [(0, 0) for _ in maps]
    paths = [[[0, 0]] for _ in maps]
    rewards = [0.0]*count
    collisions = [0]*count
    done, success, steps = [False]*count, [False]*count, [0]*count
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    model.eval()
    with torch.inference_mode():
        for _ in range(env["max_steps"]):
            obs = np.zeros((count, 3, n, n), dtype=np.float32)
            obs[:, 0] = grids
            obs[:, 2, -1, -1] = 1
            for i, (r, c) in enumerate(positions):
                obs[i, 1, r, c] = 1
            q = model(torch.from_numpy(obs)).numpy()
            if masked:
                for i, (r, c) in enumerate(positions):
                    for a, (dr, dc) in enumerate(moves):
                        rr, cc = r+dr, c+dc
                        if not (0 <= rr < n and 0 <= cc < n) or grids[i, rr, cc]:
                            q[i, a] = -np.inf
            actions = q.argmax(1)
            for i, action in enumerate(actions):
                if done[i]:
                    continue
                r, c = positions[i]
                dr, dc = moves[action]
                rr, cc = r+dr, c+dc
                collision = not (0 <= rr < n and 0 <= cc < n) or bool(grids[i, rr, cc])
                if not collision:
                    positions[i] = rr, cc
                paths[i].append(list(positions[i]))
                collisions[i] += int(collision)
                steps[i] += 1
                success[i] = positions[i] == (n-1, n-1)
                rewards[i] += env["goal_reward"] if success[i] else env["collision_reward"] if collision else env["step_reward"]
                done[i] = success[i] or steps[i] == env["max_steps"]
            if all(done):
                break
    episodes = [{"maze_hash": maps[i]["hash"], "maze_seed": maps[i]["seed"],
                 "success": success[i], "terminated": success[i], "truncated": not success[i],
                 "steps": steps[i], "return": rewards[i], "collisions": collisions[i],
                 "path": paths[i], "path_efficiency": maps[i]["shortest_length"]/steps[i] if success[i] else None}
                for i in range(count)]
    return {"success_rate": sum(success)/count,
            "path_efficiency": statistics.mean(e["path_efficiency"] for e in episodes if e["success"]) if any(success) else None,
            "episodes": episodes}


def audit_rollout(expected, actual, context):
    eq(expected["metrics"]["success_rate"], actual["success_rate"], context+" success rate")
    eq(expected["metrics"]["path_efficiency"], actual["path_efficiency"], context+" efficiency")
    eq(len(expected["episodes"]), len(actual["episodes"]), context+" denominator")
    for a, b in zip(expected["episodes"], actual["episodes"]):
        for key in ("maze_hash", "maze_seed", "success", "terminated", "truncated", "steps", "return", "collisions", "path", "path_efficiency"):
            eq(a[key], b[key], context+" "+key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT/"studies/CNN-04/results")
    parser.add_argument("--resumed-at-utc", type=datetime.fromisoformat)
    args = parser.parse_args()
    study = args.study.resolve()
    started = time.monotonic()
    resumed = args.resumed_at_utc or datetime.now(timezone.utc)
    protocol, comparison, ledger = [read(study/name) for name in ("protocol.json", "comparison.json", "ledger.json")]
    reference = ROOT/"runs/20261104T042433Z_2393292b"
    original = read(reference/"mazes.json")
    from rl_lab.config import from_dict
    original_config = from_dict(read(reference/"config.json")).to_dict()
    baseline_config = copy.deepcopy(original_config)
    baseline_config["train"].update(device="cuda", method="baseline", max_seconds=300, hard_timeout_seconds=330,
                                    max_total_steps=60000, replay_event_step=0, replay_clear=False, replay_pause_steps=0)
    interventions = {"B_CUDA": ({}, {}), "success": ({"method": "success"}, {}), "per": ({"method": "per"}, {}),
                     "nstep": ({"method": "nstep"}, {}), "double": ({"method": "double"}, {}), "mask": ({"method": "mask"}, {}),
                     "train64": ({}, {"train_mazes": 64}), "batch64": ({"batch_size": 64}, {}),
                     "every2": ({"train_every": 2}, {}), "target500": ({"target_update": 500}, {})}
    facts, errors, checks = [], [], []
    restoration_steps = 0
    import torch
    from rl_lab.dqn import QNetwork
    torch.set_num_threads(1)
    for row in comparison["conditions"]:
        if row.get("device") == "cpu" or row.get("label") in ("B_reference", "B_CPU", "B_CPU_reference"):
            try:
                eq(row["run_id"], reference.name, "historical reference run ID")
                eq(row["device"], "cpu", "historical reference device")
                rates, train_rates = [], []
                for seed in (0, 1, 2):
                    folder = reference/f"seed_{seed}"
                    summary = read(folder/"summary.json")
                    eq(summary["status"], "completed", "historical reference completed")
                    eq(summary["environment_steps"], 20000, "historical reference steps")
                    for split, filename, values in (
                            ("validation", "latest_validation.json", rates),
                            ("train", "final_training_evaluation.json", train_rates)):
                        raw = read(folder/filename)
                        eq([e["maze_hash"] for e in raw["episodes"]],
                           [m["hash"] for m in original["splits"][split]], "historical split")
                        rate = sum(e["success"] for e in raw["episodes"])/len(raw["episodes"])
                        eq(raw["metrics"]["success_rate"], rate, "historical success rate")
                        values.append(rate)
                eq(row["validation_success_by_seed"], rates, "historical seed results")
                eq(row["validation_success_mean"], statistics.mean(rates), "historical mean")
                eq(row["validation_success_std"], statistics.stdev(rates), "historical sample SD")
                eq(row["training_greedy_success_mean"], statistics.mean(train_rates), "historical training mean")
                eq(row["processed_samples_by_seed"], [None]*3, "historical unknown sample counts")
                facts.append({"condition": row["label"], "run_id": reference.name, "reference_only": True,
                              "device": "cpu", "validation_success_by_seed": rates,
                              "validation_success_mean": statistics.mean(rates),
                              "validation_success_sample_sd": statistics.stdev(rates),
                              "n_policy_seeds": 3, "n_validation_mazes": 8, "validation_episode_count": 24,
                              "evidence": [str(reference/f"seed_{seed}"/"latest_validation.json") for seed in (0, 1, 2)]})
                checks.append({"condition": row["label"], "historical_cpu_reference_recomputed": True,
                               "excluded_from_selection": True})
            except Exception as error:
                errors.append({"condition": row.get("label"), "error": repr(error)})
            continue
        try:
            run = ROOT/"runs"/row["run_id"]
            config, manifest, metadata = [read(run/name) for name in ("config.json", "mazes.json", "metadata.json")]
            expected_config = copy.deepcopy(baseline_config)
            train_change, eval_change = interventions[row["label"]]
            expected_config["train"].update(train_change)
            expected_config["evaluation"].update(eval_change)
            for section in ("train", "evaluation", "env", "network", "algorithm"):
                eq(config[section], expected_config[section], "single-change config "+section)
            splits = manifest["splits"]
            eq(config["train"]["device"], "cuda", "requested device")
            eq(metadata["device"], "cuda", "metadata device")
            eq(metadata["worker_concurrency"], 1, "single GPU training worker")
            eq(metadata["git_commit"], protocol["runtime"]["git_commit"], "frozen training commit")
            assert "4060" in metadata["accelerator"]["gpu_name"]
            for name, digest in protocol["frozen_training_source_hashes"].items():
                eq(metadata["source_hashes"][name.replace('\\', '/')], digest, "run training source matches protocol "+name)
            for split in ("validation", "test"):
                eq(splits[split], original["splits"][split], split+" fixed manifest")
            hashes = [m["hash"] for maps in splits.values() for m in maps]
            assert len(set(hashes)) == len(hashes), "geometry overlap"
            maze_seeds = [m["seed"] for maps in splits.values() for m in maps]
            assert len(set(maze_seeds)) == len(maze_seeds), "maze seed overlap"
            for maps in splits.values():
                for maze in maps:
                    import numpy as np
                    eq(hashlib.sha256(np.array(maze["grid"], dtype=np.uint8).tobytes()).hexdigest(), maze["hash"], "map hash")
            eq(splits["train"][:16], original["splits"]["train"], "original train pool prefix")
            assert len(splits["train"]) in (16, 64)
            with zipfile.ZipFile(run/"source_snapshot.zip") as archive:
                for name, digest in metadata["source_hashes"].items():
                    eq(hashlib.sha256(archive.read(name)).hexdigest(), digest, "source archive "+name)
            seeds = []
            for seed in (0, 1, 2):
                folder = run/f"seed_{seed}"
                summary = read(folder/"summary.json")
                eq(summary["status"], "completed", "completion")
                eq(summary["environment_steps"], 20000, "training budget")
                assert summary["training_time_seconds"] <= 300, "completed seed exceeded soft wall budget"
                eq(summary["device"], "cuda", "actual training device")
                eq(summary["parameter_count"], 210212, "architecture")
                expected_updates = sum(s >= config["train"]["learning_starts"] and s % config["train"]["train_every"] == 0 for s in range(1, 20001))
                eq(summary["optimizer_updates"], expected_updates, "actual optimizer updates")
                eq(summary.get("paused_updates", 0), 0, "no pause")
                eq(summary.get("paused_env_steps", 0), 0, "no pause steps")
                expected_samples = expected_updates*config["train"]["batch_size"]
                eq(summary["processed_samples"], expected_samples, "processed samples counter")
                eq(summary["final_replay"]["sampling_totals"]["transitions"], expected_samples, "actual sample count")
                checkpoint = torch.load(folder/"latest.pt", map_location="cpu", weights_only=True)
                eq(checkpoint["step"], 20000, "last checkpoint step")
                eq(checkpoint["seed"], seed, "checkpoint seed")
                eq(checkpoint["config"], config, "checkpoint config")
                model = QNetwork(10, 64)
                model.load_state_dict(checkpoint["model_state"])
                masked = config["train"].get("method") == "mask"
                restored = {}
                for split, filename in (("validation", "latest_validation.json"), ("train", "final_training_evaluation.json")):
                    result = independent_rollout(model, config["env"], splits[split], masked)
                    restoration_steps += sum(e["steps"] for e in result["episodes"])
                    audit_rollout(read(folder/filename), result, row["label"]+f" seed{seed} "+split)
                    restored[split] = result["success_rate"]
                with (folder/"training_episodes.csv").open(encoding="utf-8", newline="") as stream:
                    episodes = list(csv.DictReader(stream))
                assert sum(int(e["steps"]) for e in episodes) <= 20000
                eq([int(e["end_step"]) for e in episodes], list(__import__('itertools').accumulate(int(e["steps"]) for e in episodes)), "episode step accounting")
                if config["train"].get("method") == "nstep":
                    lengths = {int(k): v for k, v in summary["nstep_lengths"].items()}
                    remaining = 20000-sum(int(e["steps"]) for e in episodes)
                    expected_lengths = {k: 0 for k in (1, 2, 3)}
                    for length in [int(e["steps"]) for e in episodes]+([remaining] if remaining else []):
                        for k in range(1, min(length, 3)+1):
                            expected_lengths[k] += 1 if k < 3 else max(0, length-2)
                    eq({k: lengths.get(k, 0) for k in (1, 2, 3)}, expected_lengths, "nstep insertion lengths")
                    eq(sum(lengths.values()), 20000, "nstep total insertions")
                    eq(summary["nstep_horizon_flush"], min(remaining, 2), "nstep unfinished horizon flush")
                    eq(summary["final_replay"]["total_replay_insertions"], 20000, "actual replay inserts")
                history = [readline for readline in (folder/"validation.jsonl").read_text().splitlines() if readline]
                eq([json.loads(line)["step"] for line in history], list(range(0, 20001, 500)), "evaluation schedule")
                histories = [json.loads(line) for line in history]
                validation_steps = sum((h["average_success_steps"] or 0)*(h["episodes"]-h["failed_episodes"])
                                       +(h["average_failure_steps"] or 0)*h["failed_episodes"] for h in histories)
                training_eval_steps = sum(e["steps"] for e in read(folder/"final_training_evaluation.json")["episodes"])
                eq(summary["evaluation_environment_steps"], round(validation_steps)+training_eval_steps, "all 41 validation plus train evaluation steps")
                eq(summary["last_validation"]["success_rate"], restored["validation"], "summary success")
                seeds.append({"seed": seed, "validation_success": restored["validation"], "training_greedy_success": restored["train"],
                              "optimizer_updates": expected_updates, "processed_samples": expected_samples,
                              "training_environment_steps": 20000, "checkpoint_sha256": sha(folder/"latest.pt")})
            rates = [s["validation_success"] for s in seeds]
            eq(row["validation_success_mean"], statistics.mean(rates), "comparison mean")
            eq(row["validation_success_std"], statistics.stdev(rates), "comparison sample SD")
            facts.append({"condition": row["label"], "run_id": row["run_id"], "checkpoint": "latest.pt", "checkpoint_steps": 20000,
                          "n_policy_seeds": 3, "n_validation_mazes": 8, "validation_episode_count": 24,
                          "validation_success_by_seed": rates, "validation_success_mean": statistics.mean(rates),
                          "validation_success_sample_sd": statistics.stdev(rates), "seed_details": seeds,
                          "evidence": [str(run/"config.json"), str(run/"mazes.json"), str(run/"summary.json")],
                          "information_difference": "Wall/boundary action rules applied to exploration, greedy and targets" if masked else None,
                          "interpretation_scope": "Repeated development validation on 8 fixed geometries; 24 episodes are not 24 distinct maps."})
            checks.append({"condition": row["label"], "source_archive": "passed", "final_checkpoints_cpu_restored": 3,
                           "train_validation_paths_returns_and_metrics": "passed", "git_commit": metadata["git_commit"],
                           "recorded_git_status": metadata["git_status"], "worker_concurrency": metadata["worker_concurrency"]})
        except Exception as error:
            errors.append({"condition": row.get("label"), "error": repr(error)})
    status = "approved_for_dev_report" if not errors and len(facts) == 11 else "incomplete"
    try:
        attempts = ledger["attempts"]
        retries = [a for a in attempts if a["kind"] == "retry"]
        assert len(retries) <= 1
        reserved = sum(a["reserved_steps"] for a in attempts)
        eq(ledger["reserved_training_steps"], reserved, "reserved budget ledger")
        assert reserved <= 620000
        actual = 0
        for attempt in attempts:
            for summary in read(ROOT/"runs"/attempt["run_id"]/"summary.json")["seeds"]:
                if attempt["kind"] == "retry" and summary["seed"] != attempt["seed"]:
                    continue
                actual += summary["environment_steps"]
        eq(ledger["actual_training_steps"], actual, "actual budget including failed attempts")
        assert actual <= 620000
        assert protocol["elapsed_seconds"] <= 10800
        completed = datetime.fromisoformat(protocol["finished_at_utc"])
        reviewed = datetime.now(timezone.utc)
        inactive_seconds = (resumed-completed).total_seconds()
        resumed_seconds = (reviewed-resumed).total_seconds()
        active_wall = protocol["elapsed_seconds"] + resumed_seconds
        assert inactive_seconds >= 0 and resumed_seconds >= 0
        assert active_wall <= 10800, "execution plus active resumed work exceeded wall budget"
        checks.append({"budget": "passed", "actual_training_steps_including_failed_attempts": actual,
                       "reserved_steps": reserved, "retries": len(retries), "experiment_wall_seconds": protocol["elapsed_seconds"],
                       "resumed_at_utc": resumed.isoformat(), "inactive_between_execution_and_resumption_seconds": inactive_seconds,
                       "resumed_active_wall_seconds_through_check": resumed_seconds,
                       "execution_plus_resumed_active_wall_seconds_through_check": active_wall})
        eligible = [r for r in comparison["conditions"] if r.get("device") == "cuda" and r["completed_seeds"] == 3]
        def score(r):
            return r["validation_success_mean"], -r["validation_success_std"], r["path_efficiency_mean"] if r["path_efficiency_mean"] is not None else -1
        winner = max(eligible, key=score)["label"]
        eq(comparison["selected_condition"], winner, "predefined final checkpoint selection")
        eq(protocol["selected_condition"], winner, "protocol selected condition")
    except Exception as error:
        errors.append({"scope": "budget and selection", "error": repr(error)})
    if errors:
        status = "incomplete"
    review = {"status": status, "reviewer": "independent_results_review_subagent", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
              "comparison_sha256": sha(study/"comparison.json"), "protocol_sha256": sha(study/"protocol.json"),
              "approved_run_ids": [f["run_id"] for f in facts], "checks": checks, "errors": errors,
              "restoration_device": "cpu", "independent_restoration_environment_steps": restoration_steps,
              "review_wall_seconds": time.monotonic()-started,
              "limitations": ["CPU restoration is an independent rollout; CUDA training equality is checked from original metadata and summaries.",
                              "Development validation is repeatedly reused; small differences do not establish generalization or causation.",
                              "Mask changes information use; batch64 and train_every2 change computation.",
                              "No final-test rollout or model selection was performed by the reviewer."]}
    (study/"review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (study/"verified_facts.json").write_text(json.dumps({**{k: review[k] for k in ("status", "reviewer", "reviewed_at_utc", "comparison_sha256", "protocol_sha256", "approved_run_ids")},
                                                       "facts": facts, "final_test_used": False, "no_cards_or_upload": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "conditions": len(facts), "errors": errors, "restoration_steps": restoration_steps,
                      "wall_seconds": review["review_wall_seconds"]}, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
