import csv
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .config import load_config
from .dqn import DQNAgent, seed_everything
from .evaluation import evaluate
from .maze import Maze, MazeEnv
from .storage import replace_with_retry, utc_now, write_json


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

    def check_budget():
        if time.monotonic() - started >= config.train.max_seconds:
            raise BudgetExpired("Per-seed wall time budget reached")

    def record_validation(at_step, stream):
        nonlocal first_target, best_score
        result = evaluate(agent.online, config.env, manifest["splits"]["validation"], check_budget)
        metrics = result["metrics"]
        row = {"step": at_step, "elapsed_seconds": time.monotonic() - started,
               "epsilon": agent.epsilon(at_step), "loss": last_loss, **metrics}
        evaluations.append(row)
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        write_json(folder / "latest_validation.json", result)
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
                agent.replay.add(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward
                collisions += int(info["collision"])
                if step >= config.train.learning_starts and step % config.train.train_every == 0 \
                        and agent.replay.count >= config.train.batch_size:
                    last_loss = agent.update()
                if step % config.train.target_update == 0:
                    agent.sync_target()
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
        status = "completed"
        if config.evaluation.diagnose_training_fit:
            training_fit = evaluate(agent.online, config.env, manifest["splits"]["train"], check_budget)
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
                   "last_loss": last_loss, "validation_history": evaluations,
                   "training_greedy": training_fit["metrics"] if training_fit else None,
                   "last_validation": evaluations[-1] if evaluations else None,
                   "finished_at_utc": utc_now(),
                   "note": "Validation performance only; held-out test has not been evaluated."}
        write_json(folder / "summary.json", summary)
    return summary
