import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
from uuid import uuid4

from .maze import Maze, make_splits
from .plots import plot_run
from .storage import ROOT, runtime_metadata, snapshot_source, write_json


def aggregate(summaries):
    complete = [s for s in summaries if s["status"] == "completed"]
    result = {"completed_seeds": len(complete), "requested_seeds": len(summaries),
              "incomplete_seeds": [s["seed"] for s in summaries if s["status"] != "completed"],
              "metrics": {}}
    for name in ("success_rate", "average_success_steps", "path_efficiency", "average_failure_steps",
                 "training_time_seconds", "sample_efficiency_steps"):
        values = [(s.get(name) if name in ("training_time_seconds", "sample_efficiency_steps")
                   else (s.get("last_validation") or {}).get(name)) for s in complete]
        values = [v for v in values if v is not None]
        result["metrics"][name] = {"mean": statistics.mean(values) if values else None,
                                   "std": statistics.stdev(values) if len(values) > 1 else None,
                                   "n": len(values)}
    result["metrics"]["sample_efficiency_steps"]["note"] = "Only seeds reaching the threshold; see n."
    return result


def validate_manifest(config, manifest):
    names = ("train", "validation", "test")
    counts = (config.evaluation.train_mazes, config.evaluation.validation_mazes, config.evaluation.test_mazes)
    seen = set()
    for name, count in zip(names, counts):
        maps = manifest["splits"][name]
        if len(maps) != count:
            raise ValueError(f"{name} map count differs from config")
        for raw in maps:
            if raw["hash"] in seen:
                raise ValueError("Maze geometry overlaps within/across splits")
            seen.add(raw["hash"])
            # Final-test geometry is copied unchanged; never evaluated for model selection.
            if name != "test":
                maze = Maze.from_dict(raw)
                if maze.grid.shape != (config.env.size, config.env.size):
                    raise ValueError("Maze dimensions differ from config")


def launch_worker(run_dir, config, seed):
    print(f"{run_dir.name}: seed={seed}, budget={config.train.steps} steps / {config.train.max_seconds}s", flush=True)
    log = run_dir / f"seed_{seed}_worker.log"
    command = [sys.executable, "-X", "utf8", "-m", "rl_lab", "_worker", "--run", str(run_dir), "--seed", str(seed)]
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=config.train.hard_timeout_seconds)
            error = None if process.returncode == 0 else f"worker exit code {process.returncode}"
        except subprocess.TimeoutExpired:
            error = "hard_timeout"
    summary_path = run_dir / f"seed_{seed}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
        "seed": seed, "status": "hard_timeout" if error == "hard_timeout" else "failed"}
    if error:
        summary.update(status="hard_timeout" if error == "hard_timeout" else "failed", worker_error=error)
        write_json(run_dir / f"seed_{seed}_failure.json", summary)
    print(f"{run_dir.name}: seed={seed} {summary['status']}", flush=True)
    return summary


def run_experiment(config, output, reuse_completed_from=None, maze_manifest=None, workers=1):
    config.validate()
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("workers must be 1 or 2")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_dir = Path(output).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if reuse_completed_from is not None:
        from .config import from_dict
        reuse_completed_from = Path(reuse_completed_from).resolve()
        original = json.loads((reuse_completed_from / "config.json").read_text(encoding="utf-8"))
        if from_dict(original).to_dict() != config.to_dict():
            raise ValueError("Recovery can reuse seeds only with exactly the same configuration")
        write_json(run_dir / "recovery_provenance.json", {
            "original_run": str(reuse_completed_from),
            "original_metadata": json.loads((reuse_completed_from / "metadata.json").read_text(encoding="utf-8")),
            "policy": "Copy only completed seed artifacts; restart failed seeds from initialization. Original run retained.",
            "change": "Bounded checkpoint/JSON replacement retry; learning algorithm unchanged."})
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "metadata.json", {"experiment_id": run_id, **runtime_metadata(config.train.device),
                                          "source_hashes": snapshot_source(run_dir),
                                          "worker_concurrency": workers,
                                          "provided_manifest": maze_manifest is not None})
    write_json(run_dir / "status.json", {"status": "running"})
    summaries = []
    try:
        manifest = maze_manifest if maze_manifest is not None else make_splits(config)
        validate_manifest(config, manifest)
        if reuse_completed_from and json.loads((reuse_completed_from / "mazes.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Recovery cannot reuse a seed from a different maze manifest")
        write_json(run_dir / "mazes.json", manifest)
        pending = []
        for seed in config.train.seeds:
            prior_summary = reuse_completed_from / f"seed_{seed}" / "summary.json" if reuse_completed_from else None
            if prior_summary and prior_summary.exists():
                prior = json.loads(prior_summary.read_text(encoding="utf-8"))
                if prior["status"] == "completed":
                    shutil.copytree(prior_summary.parent, run_dir / f"seed_{seed}")
                    old_log = reuse_completed_from / f"seed_{seed}_worker.log"
                    if old_log.exists():
                        shutil.copyfile(old_log, run_dir / old_log.name)
                    summaries.append(prior)
                    write_json(run_dir / "summary.json", {"seeds": summaries, "aggregate": aggregate(summaries)})
                    print(f"{run_id}: reused completed seed={seed} from {reuse_completed_from.name}", flush=True)
                    continue
            pending.append(seed)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(launch_worker, run_dir, config, seed) for seed in pending]
            for future in as_completed(futures):
                summaries.append(future.result())
                summaries.sort(key=lambda row: config.train.seeds.index(row["seed"]))
                # Only this parent thread writes aggregate files.
                write_json(run_dir / "summary.json", {"seeds": summaries, "aggregate": aggregate(summaries)})
        fields = ["seed", "status", "environment_steps", "training_time_seconds", "sample_efficiency_steps",
                  "success_rate", "average_success_steps", "path_efficiency", "failed_episodes", "average_failure_steps"]
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in summaries:
                writer.writerow({**(row.get("last_validation") or {}), **row})
        plot_run(run_dir, summaries, manifest)
        completed = all(s["status"] == "completed" for s in summaries)
        write_json(run_dir / "status.json", {"status": "completed" if completed else "incomplete"})
        write_json(run_dir / "research_notes.json", {
            "hypothesis": config.hypothesis, "changes": config.change_description,
            "comparison_to_previous": None,
            "conclusion": "See recorded metrics; no cross-algorithm or convergence claim is generated automatically.",
            "next_experiment": "Follow the predefined study protocol; use validation only for development.",
            "final_test_used": False})
        print(f"Results: {run_dir}", flush=True)
        return run_dir, completed
    except Exception as error:
        import traceback
        write_json(run_dir / "status.json", {"status": "failed", "error": str(error),
                                            "traceback": traceback.format_exc()})
        raise
