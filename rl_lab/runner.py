import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from uuid import uuid4

from .maze import make_splits
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


def run_experiment(config, output):
    config.validate()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_dir = Path(output).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "metadata.json", {"experiment_id": run_id, **runtime_metadata(),
                                          "source_hashes": snapshot_source(run_dir)})
    write_json(run_dir / "status.json", {"status": "running"})
    summaries = []
    try:
        manifest = make_splits(config)
        write_json(run_dir / "mazes.json", manifest)
        for seed in config.train.seeds:
            print(f"{run_id}: seed={seed}, budget={config.train.steps} steps / {config.train.max_seconds}s", flush=True)
            log = run_dir / f"seed_{seed}_worker.log"
            command = [sys.executable, "-m", "rl_lab", "_worker", "--run", str(run_dir), "--seed", str(seed)]
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
            summaries.append(summary)
            # Persist completed/failed seed results even if the next worker crashes.
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
            "conclusion": "Pipeline check only. No convergence or algorithm-ranking claim.",
            "next_experiment": "After Phase 1 review, compare DQN/Double/Dueling with >=3 training seeds.",
            "final_test_used": False})
        print(f"Results: {run_dir}", flush=True)
        return run_dir, completed
    except Exception as error:
        import traceback
        write_json(run_dir / "status.json", {"status": "failed", "error": str(error),
                                            "traceback": traceback.format_exc()})
        raise
