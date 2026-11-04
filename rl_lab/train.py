import csv
import json
import hashlib
import struct
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .config import load_config
from .dqn import DQNAgent, seed_everything
from .evaluation import evaluate
from .maze import Maze, MazeEnv
from .storage import replace_with_retry, state_digest, utc_now, write_json


class BudgetExpired(Exception):
    pass


def save_checkpoint(path, agent, config, seed, step):
    temporary = path.with_suffix(".tmp")
    torch.save({"schema_version": 1, "algorithm": "dqn", "config": config.to_dict(),
                "seed": seed, "step": step, "model_state": agent.online.state_dict()}, temporary)
    replace_with_retry(temporary, path)


def train_seed(run_dir, seed):
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.json")
    if seed not in config.train.seeds:
        raise ValueError("Worker seed not in experiment config")
    folder = run_dir / f"seed_{seed}"
    folder.mkdir(exist_ok=False)
    started, step, status = time.monotonic(), 0, "running"
    evaluations, first_target, best_score, last_loss = [], None, (-1.0, -1.0), None
    agent = None
    training_fit = None
    optimizer_updates, paused_updates, paused_env_steps = 0, 0, 0
    trajectory = hashlib.sha256()
    resume_recorded = False
    evaluation_environment_steps = 0

    def check_budget():
        if time.monotonic() - started >= config.train.max_seconds:
            raise BudgetExpired("Per-seed wall time budget reached")

    def record_validation(at_step, stream):
        nonlocal first_target, best_score, evaluation_environment_steps
        result = evaluate(agent.online, config.env, manifest["splits"]["validation"], check_budget)
        evaluation_environment_steps += sum(ep["steps"] for ep in result["episodes"])
        metrics = result["metrics"]
        row = {"step": at_step, "elapsed_seconds": time.monotonic() - started,
               "epsilon": agent.epsilon(at_step), "loss": last_loss, **metrics}
        evaluations.append(row)
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        write_json(folder / "latest_validation.json", result)
        with (folder / "replay.jsonl").open("a", encoding="utf-8") as replay_stream:
            replay_stream.write(json.dumps({"step": at_step, "optimizer_updates": optimizer_updates,
                                            **agent.replay.snapshot(at_step)})+"\n")
        if first_target is None and metrics["success_rate"] >= config.evaluation.target_success_rate:
            first_target = at_step
        efficiency = metrics["path_efficiency"]
        score = (metrics["success_rate"], efficiency if efficiency is not None else -1.0)
        if score > best_score:
            best_score = score
            save_checkpoint(folder / "best.pt", agent, config, seed, at_step)
            write_json(folder / "best_validation.json", {"step": at_step, **result})

    try:
        seed_everything(seed)
        agent = DQNAgent(config, seed)
        manifest = json.loads((run_dir / "mazes.json").read_text(encoding="utf-8"))
        train_pool = [Maze.from_dict(m) for m in manifest["splits"]["train"]]
        rng = np.random.default_rng(seed)

        def new_environment():
            return MazeEnv(config.env, train_pool[int(rng.integers(len(train_pool)))])

        env = new_environment()
        state, episode_reward, collisions = env.reset(), 0.0, 0
        with (folder / "validation.jsonl").open("w", encoding="utf-8") as stream, \
                (folder / "training_episodes.csv").open("w", encoding="utf-8", newline="") as episode_file:
            writer = csv.DictWriter(episode_file, fieldnames=["end_step", "maze_seed", "steps", "return",
                                                            "success", "truncated", "collisions"])
            writer.writeheader()
            record_validation(0, stream)
            for next_step in range(1, config.train.steps + 1):
                check_budget()
                action = agent.act(state, next_step - 1)
                next_state, reward, terminated, truncated, info = env.step(action)
                step = next_step
                trajectory.update(struct.pack("<qqqqqd??", step, env.maze.seed, action,
                                              env.position[0], env.position[1], reward, terminated, truncated))
                agent.replay.add(state, action, reward, next_state, terminated, step, env.maze.seed)
                state = next_state
                episode_reward += reward
                collisions += int(info["collision"])
                due_update = step >= config.train.learning_starts and step % config.train.train_every == 0
                paused = config.train.replay_event_step < step <= \
                    config.train.replay_event_step + config.train.replay_pause_steps
                paused_env_steps += int(paused)
                if due_update and paused:
                    paused_updates += 1
                elif due_update and agent.replay.count >= config.train.batch_size:
                    last_loss = agent.update()
                    optimizer_updates += 1
                    if config.train.replay_event_step and step > config.train.replay_event_step+config.train.replay_pause_steps \
                            and not resume_recorded:
                        write_json(folder / "replay_first_resume_update.json", {
                            "step": step, "optimizer_updates": optimizer_updates,
                            **agent.replay.snapshot(step)})
                        resume_recorded = True
                if step % config.train.target_update == 0:
                    agent.sync_target()
                if config.train.replay_event_step and step == config.train.replay_event_step:
                    def fingerprints():
                        return {"online": state_digest(agent.online.state_dict()),
                                "target": state_digest(agent.target.state_dict()),
                                "optimizer": state_digest(agent.optimizer.state_dict()),
                                "sampler_rng": state_digest(agent.replay.rng.bit_generator.state),
                                "action_rng": state_digest(agent.rng.bit_generator.state),
                                "maze_rng": state_digest(rng.bit_generator.state)}
                    before = fingerprints()
                    replay_before = agent.replay.snapshot(step)
                    if config.train.replay_clear:
                        agent.replay.clear()
                    after = fingerprints()
                    if before != after:
                        raise AssertionError("Replay intervention changed model/optimizer/RNG state")
                    write_json(folder / "replay_event.json", {
                        "step": step, "clear": config.train.replay_clear,
                        "pause_first_step": step+1, "pause_last_step": step+config.train.replay_pause_steps,
                        "optimizer_updates": optimizer_updates,
                        "trajectory_sha256": trajectory.hexdigest(),
                        "before": before, "after": after, "replay_before": replay_before,
                        "replay_after": agent.replay.snapshot(step)})
                if terminated or truncated:
                    writer.writerow({"end_step": step, "maze_seed": env.maze.seed, "steps": env.steps,
                                     "return": episode_reward, "success": terminated,
                                     "truncated": truncated, "collisions": collisions})
                    episode_file.flush()
                    env = new_environment()
                    state, episode_reward, collisions = env.reset(), 0.0, 0
                if step % config.evaluation.interval == 0 or step == config.train.steps:
                    save_checkpoint(folder / "latest.pt", agent, config, seed, step)
                    record_validation(step, stream)
                if config.train.replay_event_step and step == config.train.replay_event_step+config.train.replay_pause_steps:
                    write_json(folder / "replay_after_refill.json", {
                        "step": step, "optimizer_updates": optimizer_updates,
                        "trajectory_sha256": trajectory.hexdigest(), "state_hashes": fingerprints(),
                        **agent.replay.snapshot(step)})
        status = "completed"
        if config.evaluation.diagnose_training_fit:
            training_fit = evaluate(agent.online, config.env, manifest["splits"]["train"], check_budget)
            evaluation_environment_steps += sum(ep["steps"] for ep in training_fit["episodes"])
            write_json(folder / "final_training_evaluation.json", training_fit)
    except BudgetExpired as error:
        status = "time_limit"
        write_json(folder / "error.json", {"type": type(error).__name__, "message": str(error)})
    except Exception as error:
        status = "failed"
        write_json(folder / "error.json", {"type": type(error).__name__, "message": str(error),
                                            "traceback": traceback.format_exc()})
    finally:
        if agent is not None:
            save_checkpoint(folder / "latest.pt", agent, config, seed, step)
        summary = {"seed": seed, "status": status, "environment_steps": step,
                   "training_time_seconds": time.monotonic() - started,
                   "sample_efficiency_steps": first_target,
                   "parameter_count": sum(p.numel() for p in agent.online.parameters()) if agent else None,
                   "device": str(agent.device) if agent else config.train.device,
                   "last_loss": last_loss, "validation_history": evaluations,
                   "optimizer_updates": optimizer_updates, "paused_updates": paused_updates,
                   "evaluation_environment_steps": evaluation_environment_steps,
                   "paused_env_steps": paused_env_steps,
                   "trajectory_sha256": trajectory.hexdigest(),
                   "final_replay": agent.replay.snapshot(step) if agent else None,
                   "training_greedy": training_fit["metrics"] if training_fit else None,
                   "last_validation": evaluations[-1] if evaluations else None,
                   "finished_at_utc": utc_now(),
                   "note": "Validation performance only; held-out test has not been evaluated."}
        write_json(folder / "summary.json", summary)
    return summary
