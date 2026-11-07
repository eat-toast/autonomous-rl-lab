"""Run the approved MLP14 action mask comparison with one CUDA worker."""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from rl_lab.paths import resolve_legacy_path, study_runs, run_path
import os
import shutil
import subprocess
import sys
import time
import traceback

from scripts.study_modules import load as _load_study
_run_mlp01 = _load_study("MLP-01", "run_mlp01")
ROOT = _run_mlp01.ROOT
np = _run_mlp01.np
read = _run_mlp01.read
runtime_metadata = _run_mlp01.runtime_metadata
sha = _run_mlp01.sha
snapshot_source = _run_mlp01.snapshot_source
torch = _run_mlp01.torch
train_worker = _run_mlp01.train_worker
utc_now = _run_mlp01.utc_now
write_json = _run_mlp01.write_json


BASE = ROOT / "studies/MLP-01/results"
RESULTS = ROOT / "studies/MLP-02/results"
LOCK = ROOT / "studies/MLP-01/.worker.lock"
SEEDS = (0, 1, 2)


def legal_actions(observation):
    return np.flatnonzero(observation[:8:2] == 0)


def explore(rng, observation):
    return int(rng.choice(legal_actions(observation)))


def choose(model, observation):
    legal = legal_actions(observation)
    with torch.inference_mode():
        device = next(model.parameters()).device
        values = model(torch.as_tensor(observation, device=device)[None])[0]
        return int(legal[values[legal].argmax().item()])


def bootstrap(q_values, next_observations, terminated):
    blocked = next_observations[:, :8:2].bool()
    best = q_values.masked_fill(blocked, -torch.inf).max(dim=1).values
    return best.masked_fill(terminated, 0)


def worker(seed):
    train_worker("MLP14", seed, study=RESULTS, raw=read(RESULTS / "config_MLP14.json"),
                 choose=choose, explore=explore, bootstrap=bootstrap)


def main():
    started = time.monotonic()
    recovery = subprocess.check_output(
        ["git", "rev-parse", "codex/mlp02-pre-execution"], cwd=ROOT, text=True
    ).strip()
    with LOCK.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    try:
        RESULTS.mkdir(parents=True, exist_ok=False)
        run_study(started, recovery)
    finally:
        LOCK.unlink()


def run_study(started, recovery):
    runs = study_runs(RESULTS)
    runs.mkdir()
    protocol = {
        "study_id": "MLP-02", "status": "running", "started_at_utc": utc_now(),
        "approval": "User approved MLP-02 with one RTX 4060 worker",
        "recovery_commit": recovery, "seeds": SEEDS, "steps_per_seed": 20000,
        "max_total_steps": 60000, "max_retries": 0, "max_elapsed_seconds": 1800,
        "soft_seconds_per_seed": 300, "hard_seconds_per_seed": 330,
        "primary_checkpoint_step": 20000, "validation_mazes": 8,
        "final_test_used": False, "adoption_min_successes": 21,
        "adoption_requires_no_seed_decline": True,
    }
    attempts = [{"seed": seed, "status": "not_started"} for seed in SEEDS]
    try:
        protocol["runtime"] = runtime_metadata("cuda")
        assert "RTX 4060" in torch.cuda.get_device_name(0)
        config = read(BASE / "config_MLP14.json")
        config["train"]["method"] = "mask"
        config["hypothesis"] = "Can excluding blocked directions increase MLP14 validation success?"
        config["change_description"] = "Mask cardinal wall bits during exploration, greedy selection and target maximization."
        write_json(RESULTS / "config_MLP14.json", config)
        shutil.copyfile(BASE / "maze_manifest.json", RESULTS / "maze_manifest.json")
        protocol["maze_manifest_sha256"] = sha(RESULTS / "maze_manifest.json")
        protocol["config_sha256"] = sha(RESULTS / "config_MLP14.json")
        protocol["baseline_config_sha256"] = sha(BASE / "config_MLP14.json")
        protocol["baseline_comparison_sha256"] = sha(BASE / "comparison.json")
        protocol["code_hashes"] = snapshot_source(RESULTS)
        write_json(RESULTS / "protocol.json", protocol)
        for entry in attempts:
            if time.monotonic() - started + 330 > 1800:
                break
            seed = entry["seed"]
            run_id = f"MLP14_seed_{seed}"
            entry.update(run_id=run_id, status="running")
            write_json(RESULTS / "ledger.json", attempts)
            seed_started = time.monotonic()
            print(f"Starting seed {seed}", flush=True)
            with (runs / f"{run_id}_worker.log").open("x", encoding="utf-8") as log:
                try:
                    child = subprocess.run(
                        [sys.executable, __file__, "--worker", str(seed)], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, timeout=330,
                    )
                    entry["return_code"] = child.returncode
                    entry["status"] = "completed" if child.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    entry["status"] = "hard_timeout"
            folder = runs / run_id
            entry["elapsed_seconds"] = time.monotonic() - seed_started
            for name in ("summary.json", "failure.json"):
                if (folder / name).is_file():
                    entry["environment_steps"] = read(folder / name)["environment_steps"]
                    break
            entry["charged_steps"] = entry.get("environment_steps", 20000)
            write_json(RESULTS / "ledger.json", attempts)
            print(f"Seed {seed}: {entry['status']}", flush=True)
            if entry["status"] != "completed":
                break
        protocol["status"] = "completed" if all(e["status"] == "completed" for e in attempts) else "incomplete"
    except Exception:
        protocol.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        protocol["finished_at_utc"] = utc_now()
        protocol["execution_seconds"] = time.monotonic() - started
        write_json(RESULTS / "ledger.json", attempts)
        write_json(RESULTS / "protocol.json", protocol)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
