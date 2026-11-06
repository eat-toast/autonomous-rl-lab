"""Bounded MLP-01 study runner; preserves every seed attempt under the study."""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rl_lab.config import from_dict
from rl_lab.dqn import ReplayBuffer, seed_everything
from rl_lab.maze import Maze, MazeEnv
from rl_lab.mlp01 import FeelerQNetwork, greedy_action, observation, parameter_count
from rl_lab.storage import runtime_metadata, snapshot_source, utc_now, write_json

STUDY = ROOT / "studies/MLP-01/results"
BASE_RUN = ROOT / "runs/20261104T075746Z_862a1f55"
LOCK = ROOT / "runs/.mlp01_worker.lock"
ACTION_NAMES = ["north", "east", "south", "west"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(stage):
    raw = read(BASE_RUN / "config.json")
    raw["algorithm"] = "mlp_dqn"
    raw["train"].update(device="cuda", method="baseline", seeds=[0, 1, 2], steps=20000,
                        max_seconds=300, hard_timeout_seconds=330, max_total_steps=60000,
                        replay_capacity=2000, max_replay_mb=256.0, batch_size=32,
                        learning_starts=64, train_every=4, target_update=100,
                        learning_rate=0.001, gamma=0.99, epsilon_start=1.0,
                        epsilon_end=0.1, epsilon_decay_steps=16000, gradient_clip=10.0,
                        replay_event_step=0, replay_clear=False, replay_pause_steps=0)
    raw["evaluation"].update(interval=500, train_mazes=16, validation_mazes=8,
                              test_mazes=16, diagnose_training_fit=False)
    raw["network"] = {"hidden_units": 64, "input_size": 10 if stage == "MLP10" else 14,
                      "architecture": "Linear(64)-ReLU-Linear(64)-ReLU-Linear(4)"}
    raw["hypothesis"] = ("Can eight neighboring wall bits and normalized goal displacement support maze solving?"
                         if stage == "MLP10" else
                         "Does adding this-episode cardinal neighbor visitation help vs MLP10?")
    raw["change_description"] = ("New MLP DQN; no legal action mask; eight wall feelers + row/column goal delta."
                                  if stage == "MLP10" else
                                  "MLP10 input plus four cardinal neighbor visited-this-episode bits; fresh initialization.")
    # Runtime config dataclass does not know MLP input metadata; only common env/train/eval fields are validated.
    validation_raw = copy.deepcopy(raw)
    validation_raw["algorithm"] = "dqn"
    validation_raw["network"] = {"hidden_units": 64}
    from_dict(validation_raw)
    return raw


def make_agent(input_size, seed, lr, device):
    online = FeelerQNetwork(input_size).to(device)
    target = FeelerQNetwork(input_size).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    return online, target, optimizer


class FeatureReplay:
    def __init__(self, capacity, inputs, seed):
        self.capacity, self.position, self.count = capacity, 0, 0
        self.states = np.empty((capacity, inputs), dtype=np.float32)
        self.next_states = np.empty((capacity, inputs), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.rng = np.random.default_rng(seed)
        self.samples = 0

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminated[i] = next_state, terminated
        self.position = (i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch):
        ids = self.rng.choice(self.count, batch, replace=False)
        self.samples += len(ids)
        return (torch.from_numpy(self.states[ids].copy()), torch.from_numpy(self.actions[ids].copy()),
                torch.from_numpy(self.rewards[ids].copy()), torch.from_numpy(self.next_states[ids].copy()),
                torch.from_numpy(self.terminated[ids].copy()))


def classify(episode):
    path = episode["path"]
    if episode["success"]:
        return "success"
    if len(path) >= 9 and len(set(map(tuple, path[-9:]))) == 1:
        return "stationary_collision"
    tail = list(map(tuple, path[-9:]))
    if len(tail) == 9 and tail[0] != tail[1] and all(tail[i] == tail[i % 2] for i in range(9)):
        return "two_cell_oscillation"
    if episode["collisions"]:
        return "other_with_collision"
    return "other"


def evaluate_policy(model, config, mazes, include_visited, capture=False, choose=greedy_action):
    episodes = []
    model.eval()
    with torch.inference_mode():
        for raw in mazes:
            maze = Maze.from_dict(raw)
            env = MazeEnv(config.env, maze)
            visited = {env.position}
            path, actions, observations, rewards, collisions = [env.position], [], [], [], []
            state = env.reset()
            for _ in range(config.env.max_steps):
                obs = observation(maze.grid, env.position, (9, 9), visited, include_visited)
                action = choose(model, obs)
                observations.append(obs.tolist())
                actions.append(action)
                state, reward, terminated, truncated, info = env.step(action)
                if env.position != path[-1]:
                    visited.add(env.position)
                path.append(env.position)
                rewards.append(float(reward))
                collisions.append(bool(info["collision"]))
                if terminated or truncated:
                    break
            row = {"maze_seed": raw["seed"], "maze_hash": raw["hash"], "success": bool(terminated),
                   "steps": env.steps, "terminated": bool(terminated), "truncated": bool(truncated),
                   "return": float(sum(rewards)), "collisions": int(sum(collisions)),
                   "shortest_length": raw["shortest_length"],
                   "path_efficiency": raw["shortest_length"] / env.steps if terminated else None,
                   "failure_type": classify({"success": bool(terminated), "path": path,
                                              "collisions": int(sum(collisions))})}
            if capture:
                row.update(path=[list(p) for p in path], actions=actions, action_names=[ACTION_NAMES[a] for a in actions],
                           observations=observations, rewards=rewards, collision_flags=collisions)
            episodes.append(row)
    model.train()
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]
    metrics = {"episodes": len(episodes), "successes": len(successes), "success_rate": len(successes)/len(episodes),
               "failed_episodes": len(failures), "path_efficiency":
               statistics.mean(e["path_efficiency"] for e in successes) if successes else None,
               "mean_success_steps": statistics.mean(e["steps"] for e in successes) if successes else None,
               "collisions": sum(e["collisions"] for e in episodes),
               "failure_types": {kind: sum(e["failure_type"] == kind for e in episodes)
                                 for kind in ("two_cell_oscillation", "stationary_collision", "other_with_collision", "other")}}
    return {"metrics": metrics, "episodes": episodes}


def train_worker(stage, seed, *, study=STUDY, raw=None, choose=greedy_action,
                 explore=None, bootstrap=None):
    folder = study / "runs" / f"{stage}_seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    raw = make_config(stage) if raw is None else raw
    config_raw = copy.deepcopy(raw)
    config_raw["algorithm"] = "dqn"
    config_raw["network"] = {"hidden_units": 64}
    config = from_dict(config_raw)
    manifest = read(study / "maze_manifest.json")
    include_visited = stage == "MLP14"
    inputs = 14 if include_visited else 10
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
    seed_everything(seed)
    online, target, optimizer = make_agent(inputs, seed, config.train.learning_rate, device)
    replay = FeatureReplay(config.train.replay_capacity, inputs, seed)
    rng = np.random.default_rng(seed)
    train_pool = [Maze.from_dict(m) for m in manifest["splits"]["train"]]
    eval_pool = manifest["splits"]["validation"]
    env = MazeEnv(config.env, train_pool[int(rng.integers(len(train_pool)))])
    state = env.reset(); visited = {env.position}; episode_return = 0.0; episode_collisions = 0
    episode_rows, history = [], []
    optimizer_updates, completed_episodes, started = 0, 0, time.monotonic()
    best_score, best_step = (-1.0, -1.0), None
    device_name = torch.cuda.get_device_name(0)

    def save_checkpoint(step):
        torch.save({"schema_version": 1, "stage": stage, "seed": seed, "step": step,
                    "input_size": inputs, "model_state": online.state_dict(), "device": device_name},
                   folder / "latest.pt")

    def eval_at(step):
        nonlocal best_score, best_step
        result = evaluate_policy(online, config, eval_pool, include_visited, capture=True, choose=choose)
        row = {"step": step, "elapsed_seconds": time.monotonic()-started,
               "epsilon": max(.1, 1.0 - .9 * step / 16000), **result["metrics"]}
        history.append(row)
        write_json(folder / f"validation_{step:05d}.json", result)
        save_checkpoint(step)
        score = (row["success_rate"], row["path_efficiency"] if row["path_efficiency"] is not None else -1.0)
        if score > best_score:
            best_score, best_step = score, step
            torch.save({"schema_version": 1, "stage": stage, "seed": seed, "step": step,
                        "input_size": inputs, "model_state": online.state_dict(), "device": device_name},
                       folder / "best.pt")
            write_json(folder / "best_validation.json", result)
        write_json(folder / "progress.json", {"status": "running", "step": step,
                    "elapsed_seconds": time.monotonic()-started, "optimizer_updates": optimizer_updates,
                    "evaluations": len(history)})

    write_json(folder / "metadata.json", {"stage": stage, "seed": seed, "started_at_utc": utc_now(),
                "input_size": inputs, "parameters": parameter_count(online), "device": device_name,
                "source_hashes": {"rl_lab/mlp01.py": sha(ROOT/"rl_lab/mlp01.py"),
                                  "scripts/run_mlp01.py": sha(ROOT/"scripts/run_mlp01.py")},
                "maze_manifest_sha256": sha(study/"maze_manifest.json")})
    step = 0
    executed_steps = 0
    try:
        eval_at(0)
        with (folder / "training_episodes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", "end_step", "maze_seed", "steps", "return", "success", "collisions"])
            writer.writeheader()
            for step in range(1, 20001):
                elapsed = time.monotonic() - started
                if elapsed >= 300:
                    raise TimeoutError("seed soft wall-time limit of 300 seconds reached")
                eps = max(.1, 1.0 - .9 * (step-1) / 16000)
                obs = observation(env.maze.grid, env.position, (9, 9), visited, include_visited)
                if rng.random() < eps:
                    action = int(rng.integers(4)) if explore is None else explore(rng, obs)
                else:
                    action = choose(online, obs)
                old_position = env.position
                next_state, reward, terminated, truncated, info = env.step(action)
                executed_steps = step
                if env.position != old_position:
                    visited.add(env.position)
                next_obs = observation(env.maze.grid, env.position, (9, 9), visited, include_visited)
                replay.add(obs, action, reward, next_obs, terminated)
                state = next_state; episode_return += reward; episode_collisions += int(info["collision"])
                if step >= 64 and step % 4 == 0 and replay.count >= 32:
                    s, a, r, ns, term = (x.to(device) for x in replay.sample(32))
                    q = online(s).gather(1, a[:, None]).squeeze(1)
                    with torch.no_grad():
                        next_q = target(ns)
                        continuation = next_q.max(1).values if bootstrap is None else bootstrap(next_q, ns, term)
                        targets = r + .99 * (~term).float() * continuation
                    loss = torch.nn.functional.smooth_l1_loss(q, targets)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0, error_if_nonfinite=True)
                    optimizer.step(); optimizer_updates += 1
                if step % 100 == 0:
                    target.load_state_dict(online.state_dict())
                if terminated or truncated:
                    completed_episodes += 1
                    row = {"episode": completed_episodes, "end_step": step, "maze_seed": env.maze.seed,
                           "steps": env.steps, "return": episode_return, "success": bool(terminated),
                           "collisions": episode_collisions}
                    episode_rows.append(row); writer.writerow(row); stream.flush()
                    env = MazeEnv(config.env, train_pool[int(rng.integers(len(train_pool)))])
                    state = env.reset(); visited = {env.position}; episode_return = 0.0; episode_collisions = 0
                if step % 500 == 0:
                    eval_at(step)
        final_validation = read(folder / "validation_20000.json")
        train_eval = evaluate_policy(online, config, manifest["splits"]["train"], include_visited,
                                     capture=True, choose=choose)
        write_json(folder / "final_training_evaluation.json", train_eval)
        write_json(folder / "summary.json", {"status": "completed", "stage": stage, "seed": seed,
                    "environment_steps": 20000, "optimizer_updates": optimizer_updates,
                    "processed_samples": replay.samples, "training_time_seconds": time.monotonic()-started,
                    "completed_training_episodes": completed_episodes,
                    "exploration_training_return_mean_last20": statistics.mean(r["return"] for r in episode_rows[-20:]) if episode_rows else None,
                    "exploration_training_return_episodes": episode_rows,
                    "training_greedy": train_eval["metrics"], "last_validation": final_validation["metrics"],
                    "validation_history": history, "intermediate_best_step": best_step,
                    "intermediate_best_success_rate": best_score[0], "final_checkpoint": "latest.pt",
                    "best_checkpoint": "best.pt", "parameter_count": parameter_count(online),
                    "final_test_used": False})
        write_json(folder / "progress.json", {"status": "completed", "step": 20000,
                    "elapsed_seconds": time.monotonic()-started, "optimizer_updates": optimizer_updates})
    except Exception as exc:
        write_json(folder / "failure.json", {"status": "failed", "stage": stage, "seed": seed,
                    "error": str(exc), "traceback": traceback.format_exc(),
                    "environment_steps": executed_steps,
                    "last_progress": read(folder/"progress.json") if (folder/"progress.json").exists() else None,
                    "elapsed_seconds": time.monotonic()-started})
        raise


def run_stage(stage):
    config = make_config(stage)
    (STUDY / f"config_{stage}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results = []
    for seed in (0, 1, 2):
        remaining = 3600 - (time.monotonic() - GLOBAL_STARTED)
        if remaining < 330:
            break
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage, str(seed)]
        log_path = STUDY / "runs" / f"{stage}_seed_{seed}_worker.log"
        before = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=330, check=False)
            status = "completed" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "hard_timeout"
        elapsed = time.monotonic()-before
        results.append({"seed": seed, "status": status, "elapsed_seconds": elapsed,
                        "run_path": str(STUDY / "runs" / f"{stage}_seed_{seed}"),
                        "return_code": completed.returncode if "completed" in locals() else None})
        write_json(STUDY / "ledger.json", build_ledger(results))
        if status != "completed":
            break
    return results


def build_ledger(completed):
    attempts = []
    for stage in ("MLP10", "MLP14"):
        for seed in (0, 1, 2):
            folder = STUDY / "runs" / f"{stage}_seed_{seed}"
            progress = read(folder/"progress.json") if (folder/"progress.json").exists() else None
            summary = read(folder/"summary.json") if (folder/"summary.json").exists() else None
            status = summary["status"] if summary else (progress or {}).get("status", "not_started")
            attempts.append({"stage": stage, "seed": seed, "status": status,
                             "steps": summary.get("environment_steps", (progress or {}).get("step", 0)) if summary else (progress or {}).get("step", 0),
                             "elapsed_seconds": summary.get("training_time_seconds", (progress or {}).get("elapsed_seconds")) if summary else (progress or {}).get("elapsed_seconds"),
                             "optimizer_updates": summary.get("optimizer_updates", (progress or {}).get("optimizer_updates")) if summary else (progress or {}).get("optimizer_updates"),
                             "retry_count": 0})
    actual = sum(a["steps"] for a in attempts)
    return {"max_training_steps": 120000, "actual_training_steps": actual,
            "max_wall_seconds": 3600, "elapsed_seconds": time.monotonic()-GLOBAL_STARTED,
            "retry_count": 0, "cuda_worker_count": 1, "attempts": attempts}


def aggregate_results(decision, results_by_stage):
    rows = []
    for stage in ("MLP10", "MLP14"):
        stage_seeds = []
        for seed in (0, 1, 2):
            folder = STUDY/"runs"/f"{stage}_seed_{seed}"
            summary_file = folder/"summary.json"
            if summary_file.exists():
                summary = read(summary_file)
                stage_seeds.append({"seed": seed, "status": "completed", **summary})
            else:
                failure = read(folder/"failure.json") if (folder/"failure.json").exists() else {}
                progress = read(folder/"progress.json") if (folder/"progress.json").exists() else {}
                stage_seeds.append({"seed": seed, "status": failure.get("status", progress.get("status", "not_started")),
                                    "environment_steps": progress.get("step", 0),
                                    "elapsed_seconds": progress.get("elapsed_seconds"), "failure": failure})
        completed = [s for s in stage_seeds if s["status"] == "completed"]
        values = [s["last_validation"]["success_rate"] for s in completed]
        efficiencies = [s["last_validation"]["path_efficiency"] for s in completed
                        if s["last_validation"]["path_efficiency"] is not None]
        best = [s["intermediate_best_success_rate"] for s in completed]
        rows.append({"stage": stage, "status": "completed" if len(completed) == 3 else "incomplete",
                     "requested_seeds": 3, "completed_seeds": len(completed), "validation_denominator_per_seed": 8,
                     "validation_success_by_seed": [s.get("last_validation", {}).get("success_rate") for s in stage_seeds],
                     "validation_success_mean": statistics.mean(values) if values else None,
                     "validation_success_sample_sd": statistics.stdev(values) if len(values)>1 else None,
                     "validation_success_range": [min(values), max(values)] if values else None,
                     "successful_path_efficiency_mean": statistics.mean(efficiencies) if efficiencies else None,
                     "successful_path_efficiency_n": len(efficiencies),
                     "intermediate_best_success_by_seed": best,
                     "last_checkpoint_success_by_seed": [s.get("last_validation", {}).get("success_rate") for s in completed],
                     "two_cell_oscillation_failures": decision.get("two_cell_oscillation_failures_by_seed") if stage == "MLP10" else None,
                     "seeds": stage_seeds, "run_statuses": results_by_stage.get(stage, [])})
    comparison = {"study_id": "MLP-01", "conditions": rows, "mlp14_entry_decision": decision,
                  "primary_checkpoint": "latest.pt at 20,000 steps", "intermediate_best_is_secondary": True,
                  "final_test_used": False, "reference_run_id": BASE_RUN.name,
                  "cnn_comparison_note": "CNN-00..04 use full-map CNN observation and a different model; reference only."}
    write_json(STUDY/"comparison.json", comparison)
    with (STUDY/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["stage", "status", "completed_seeds", "validation_denominator_per_seed",
                  "validation_success_by_seed", "validation_success_mean", "validation_success_sample_sd",
                  "validation_success_range", "successful_path_efficiency_mean", "successful_path_efficiency_n"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})
    lines = ["# MLP-01 결과", "", f"실행 상태: MLP10 {rows[0]['status']}; MLP14 진입 판단: {decision['decision']}.",
             "마지막 20,000스텝 체크포인트를 주 결과로 사용했다. 중간 최고값은 보조 기록이다.",
             "탐색 훈련 return, 훈련 pool greedy, 개발 검증 greedy를 별개로 기록했다. 최종 test는 사용하지 않았다.",
             "기존 CNN 결과는 전체 지도 관측과 CNN 구조가 함께 다른 참고 자료이며 직접 재현/인과 비교가 아니다.", "",
             "| 조건 | 완료 seed | 마지막 개발 검증 성공률 (seed별) | 평균 ± 표본 SD | 범위 |", "|---|---:|---|---:|---:|"]
    for row in rows:
        if row["validation_success_mean"] is None:
            stats_text = "미완료"
        else:
            sd = row["validation_success_sample_sd"]
            stats_text = f"{row['validation_success_mean']:.1%} ± {sd:.1%}" if sd is not None else f"{row['validation_success_mean']:.1%} ± n/a"
        lines.append(f"| {row['stage']} | {row['completed_seeds']}/3 | {row['validation_success_by_seed']} | {stats_text} | {row['validation_success_range']} |")
    lines += ["", "## MLP14 진입 기준", "", json.dumps(decision, ensure_ascii=False, indent=2),
              "", "## 원본", "", "- 지도와 split: `maze_manifest.json` (CUDA B 원본 manifest에서 추출)",
              "- seed별 checkpoint, 전체 validation 경로/관측/행동, 탐색 훈련 에피소드: `runs/`", "- 실행 예산: `ledger.json`; 설정과 해시: `protocol.json`"]
    (STUDY/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["stage"] for r in rows if r["validation_success_mean"] is not None]
        means = [r["validation_success_mean"]*100 for r in rows if r["validation_success_mean"] is not None]
        errors = [(r["validation_success_sample_sd"] or 0)*100 for r in rows if r["validation_success_mean"] is not None]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, means, yerr=errors, capsize=4, color=["#477d8a", "#dc8b58"][:len(names)])
        for i, row in enumerate(r for r in rows if r["validation_success_mean"] is not None):
            for value in row["validation_success_by_seed"]:
                if value is not None: ax.scatter([i], [value*100], color="black", s=20, zorder=4)
        ax.set(ylabel="Last checkpoint validation greedy success (%)", ylim=(0, 105))
        ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(STUDY/"comparison.png", dpi=160); plt.close(fig)
    except Exception as error:
        write_json(STUDY/"plot_error.json", {"error": str(error)})


def oscillation_count(episodes):
    found = []
    for episode in episodes:
        p = episode["path"]
        tail = [tuple(x) for x in p[-9:]]
        if (not episode["success"] and episode["truncated"] and episode["steps"] == 100 and len(tail) == 9
                and tail[0] != tail[1] and all(tail[i] == tail[i % 2] for i in range(9))):
            found.append(episode)
    return found


def stage_decision():
    seeds = [STUDY/"runs"/f"MLP10_seed_{seed}" for seed in (0, 1, 2)]
    if any(not (path/"summary.json").exists() for path in seeds):
        return {"decision": "deferred_incomplete_mlp10", "mlp14_started": False,
                "reason": "MLP10 did not complete all three seeds"}
    found = []; per_seed = {}
    for seed in (0, 1, 2):
        episodes = read(seeds[seed]/"validation_20000.json")["episodes"]
        oscillations = oscillation_count(episodes)
        per_seed[str(seed)] = len(oscillations)
        found.extend((seed, episode) for episode in oscillations)
    eligible = sum(per_seed[str(seed)] > 0 for seed in (0, 1, 2)) >= 2
    proceed = len(found) >= 6 and eligible
    return {"decision": "run_mlp14" if proceed else "criteria_not_met_mlp14_not_run",
            "mlp14_started": proceed, "mlp10_last_checkpoint_validation_episodes": 24,
            "two_cell_oscillation_failures_total": len(found),
            "two_cell_oscillation_failures_by_seed": per_seed,
            "threshold_total": 6, "threshold_seed_count": 2,
            "reason": "both predefined thresholds met" if proceed else "one or both predefined thresholds not met"}


def main():
    global GLOBAL_STARTED
    GLOBAL_STARTED = time.monotonic()
    STUDY.mkdir(parents=True, exist_ok=False)
    (STUDY/"runs").mkdir()
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, str(os.getpid()).encode()); os.close(lock_fd)
    try:
        raw = read(BASE_RUN/"mazes.json")
        manifest = {"source_run_id": BASE_RUN.name,
                    "source_manifest_sha256": sha(BASE_RUN/"mazes.json"),
                    "splits": {key: raw["splits"][key] for key in ("train", "validation", "test")}}
        write_json(STUDY/"maze_manifest.json", manifest)
        protocol = {"study_id": "MLP-01", "status": "running", "started_at_utc": utc_now(),
                    "approval": "user-approved MLP10 and conditional MLP14; RTX 4060 CUDA worker=1",
                    "base_run_id": BASE_RUN.name, "base_config_sha256": sha(BASE_RUN/"config.json"),
                    "base_manifest_sha256": sha(BASE_RUN/"mazes.json"), "train_maze_count": 16,
                    "validation_maze_count": 8, "test_maze_count": 16, "final_test_used": False,
                    "seeds": [0,1,2], "steps_per_seed": 20000, "stage_step_cap": 60000,
                    "total_step_cap": 120000, "max_retries": 0, "soft_seconds_per_seed": 300,
                    "hard_seconds_per_seed": 330, "max_elapsed_seconds": 3600,
                    "runtime": runtime_metadata("cuda"), "conditions": {"MLP10": make_config("MLP10")}}
        protocol["code_hashes"] = {"rl_lab/mlp01.py": sha(ROOT/"rl_lab/mlp01.py"),
                                   "scripts/run_mlp01.py": sha(Path(__file__))}
        protocol["manifest_hash"] = sha(STUDY/"maze_manifest.json")
        write_json(STUDY/"protocol.json", protocol)
        write_json(STUDY/"implementation_log.json", {
            "implementation_completed_at_utc": utc_now(), "training_started_after_this_record": True,
            "files": ["rl_lab/mlp01.py", "scripts/run_mlp01.py", "tests/test_mlp01.py"],
            "unit_validation": {"command": "python -m unittest tests.test_mlp01 -v",
                                "tests": 5, "passed": 5, "failed": 0,
                                "checks": ["8-direction wall order/boundary", "goal delta normalization",
                                           "visited-neighbor cardinal order", "10/14 inputs and 4 outputs",
                                           "5124/5380 parameter counts", "replay feature copy",
                                           "two-cell oscillation classifier"],
                                "initial_expectation_mismatch_corrected": True},
            "syntax_validation": "python -m py_compile rl_lab/mlp01.py scripts/run_mlp01.py tests/test_mlp01.py passed",
            "implementation_steps": 0, "implementation_time_seconds_recorded_separately": True,
            "runtime": runtime_metadata("cuda"),
            "recovery_tag": "codex/mlp01-pre-execution", "reference_run_id": BASE_RUN.name,
            "reference_manifest_sha256": sha(BASE_RUN/"mazes.json"),
            "unrelated_preexisting_untracked": ["docs/next_session_experiments_prompt.md"]})
        snapshot_source(STUDY)
        mlp10 = run_stage("MLP10")
        decision = stage_decision()
        write_json(STUDY/"mlp14_entry_decision.json", decision)
        if decision["mlp14_started"] and time.monotonic()-GLOBAL_STARTED < 2610:
            protocol["conditions"]["MLP14"] = make_config("MLP14")
            write_json(STUDY/"protocol.json", protocol)
            mlp14 = run_stage("MLP14")
        else:
            mlp14 = []
        write_json(STUDY/"ledger.json", build_ledger(mlp10 + mlp14))
        protocol["status"] = "completed" if all(x["status"] == "completed" for x in mlp10) and (not decision["mlp14_started"] or all(x["status"] == "completed" for x in mlp14)) else "incomplete"
        protocol["mlp10_results"] = mlp10; protocol["mlp14_entry_decision"] = decision; protocol["mlp14_results"] = mlp14
        protocol["elapsed_seconds"] = time.monotonic()-GLOBAL_STARTED; protocol["finished_at_utc"] = utc_now()
        aggregate_results(decision, {"MLP10": mlp10, "MLP14": mlp14})
        write_json(STUDY/"protocol.json", protocol)
    except Exception as error:
        write_json(STUDY/"protocol.json", {**(read(STUDY/"protocol.json") if (STUDY/"protocol.json").exists() else {}),
                    "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic()-GLOBAL_STARTED, "finished_at_utc": utc_now()})
        raise
    finally:
        if LOCK.exists(): LOCK.unlink()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        train_worker(sys.argv[2], int(sys.argv[3]))
    else:
        main()
