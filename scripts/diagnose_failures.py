"""Read saved DQN policies; diagnose failures without retraining or test-set use."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_lab.config import from_dict
from rl_lab.dqn import QNetwork, seed_everything
from rl_lab.evaluation import evaluate, summarize
from rl_lab.maze import ACTIONS, Maze, MazeEnv, astar
from rl_lab.storage import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cycle_info(episode):
    seen = {}
    for t, raw in enumerate(episode["path"]):
        position = tuple(raw)
        if position in seen:
            return {"first_repeat_step": t, "cycle_start_step": seen[position],
                    "period": t-seen[position], "position": list(position),
                    "unique_positions": len(set(map(tuple, episode["path"])))}
        seen[position] = t
    return None


def valid_actions(grid, position):
    n = len(grid)
    return [a for a, (dr, dc) in enumerate(ACTIONS)
            if 0 <= position[0]+dr < n and 0 <= position[1]+dc < n
            and not grid[position[0]+dr, position[1]+dc]]


def mask_diagnostic(model, env_config, mazes):
    """Different evaluation policy: forbid wall/boundary actions, no learning.

    This is a counterfactual diagnostic, never substituted into benchmark metrics.
    """
    episodes = []
    for raw in mazes:
        env = MazeEnv(env_config, Maze.from_dict(raw))
        obs = env.reset()
        path, total_reward, collisions = [env.position], 0., 0
        while not env.finished:
            valid = valid_actions(env.maze.grid, env.position)
            with torch.inference_mode():
                q = model(torch.from_numpy(obs).unsqueeze(0).to(next(model.parameters()).device))[0].clone()
                q[[a for a in range(4) if a not in valid]] = -torch.inf
                action = int(q.argmax())
            obs, reward, terminated, truncated, info = env.step(action)
            path.append(env.position)
            total_reward += reward
            collisions += info["collision"]
        episodes.append({"success": terminated, "steps": env.steps, "return": total_reward,
                         "collisions": collisions, "path": path,
                         "path_efficiency": raw["shortest_length"]/env.steps if terminated else None})
    return {"metrics": summarize(episodes), "episodes": episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "runs/20261104T042433Z_2393292b")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/DIAG-01/results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = from_dict(read(args.run / "config.json"))
    manifest = read(args.run / "mazes.json")
    result = {"run_id": args.run.name, "retraining": False, "final_test_used": False,
              "seeds": [], "maze_difficulty": {},
              "counterfactual_scope": "Invalid-action mask changes evaluation policy only; diagnostic, not a new training result."}
    # Only train/validation maps are examined. Test geometry is not used for diagnosis.
    for split in ("train", "validation"):
        maps = manifest["splits"][split]
        result["maze_difficulty"][split] = {
            "count": len(maps), "shortest_lengths": [m["shortest_length"] for m in maps],
            "mean_density": statistics.mean(m["actual_density"] for m in maps)}
    all_failures, all_successes, visual_examples = [], [], []
    for seed in config.train.seeds:
        seed_everything(seed)
        folder = args.run / f"seed_{seed}"
        payload = torch.load(folder / "latest.pt", map_location="cpu", weights_only=True)
        model = QNetwork(config.env.size, config.network.hidden_units).to(config.train.device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        recorded = read(folder / "latest_validation.json")
        restored = evaluate(model, config.env, manifest["splits"]["validation"])
        assert json.dumps(recorded, sort_keys=True) == json.dumps(restored, sort_keys=True)
        failures, successes = [], []
        for episode in recorded["episodes"]:
            cycle = cycle_info(episode)
            raw = next(m for m in manifest["splits"]["validation"] if m["hash"] == episode["maze_hash"])
            entry = {"maze_seed": episode["maze_seed"], "success": episode["success"],
                     "steps": episode["steps"], "collisions": episode["collisions"],
                     "shortest_length": episode["shortest_length"], "cycle": cycle}
            if cycle:
                env = MazeEnv(config.env, Maze.from_dict(raw))
                env.position = tuple(cycle["position"])
                with torch.inference_mode():
                    q = model(torch.from_numpy(env.observe()).unsqueeze(0).to(config.train.device))[0].tolist()
                chosen = int(np.argmax(q))
                current_distance = len(astar(env.maze.grid, env.position))-1
                shortest_actions = []
                for a in valid_actions(env.maze.grid, env.position):
                    dr, dc = ACTIONS[a]
                    nxt = env.position[0]+dr, env.position[1]+dc
                    if len(astar(env.maze.grid, nxt))-1 == current_distance-1:
                        shortest_actions.append(a)
                entry["trap_q"] = {"q_values_up_right_down_left": q, "greedy_action": chosen,
                                   "valid_actions": valid_actions(env.maze.grid, env.position),
                                   "shortest_path_actions": shortest_actions,
                                   "chosen_action_valid": chosen in valid_actions(env.maze.grid, env.position),
                                   "q_advantage_over_best_shortest_action": q[chosen]-max(q[a] for a in shortest_actions)}
            (successes if episode["success"] else failures).append(entry)
        masked = mask_diagnostic(model, config.env, manifest["splits"]["validation"])
        write_json(args.output / f"seed_{seed}_masked_diagnostic.json", masked)
        with (folder / "training_episodes.csv").open(encoding="utf-8", newline="") as stream:
            training = list(csv.DictReader(stream))
        goals = [r for r in training if r["success"] == "True"]
        seed_result = {"seed": seed, "restored_evaluation_exact_match": True,
                       "original_validation": recorded["metrics"],
                       "training_greedy": read(folder / "summary.json")["training_greedy"],
                       "failure_details": failures, "success_details": successes,
                       "training_goal_transitions": len(goals),
                       "goal_transition_fraction": len(goals)/config.train.steps,
                       "first_training_goal_step": int(goals[0]["end_step"]) if goals else None,
                       "goals_in_last_2000_steps": sum(int(r["end_step"]) > config.train.steps-2000 for r in goals),
                       "invalid_action_mask_diagnostic": masked["metrics"]}
        result["seeds"].append(seed_result)
        all_failures.extend(failures)
        all_successes.extend(successes)
        visual_examples.append((seed, manifest["splits"]["validation"][0], recorded["episodes"][0]))
    result["failure_summary"] = {
        "failed_episodes": len(all_failures), "successful_episodes": len(all_successes),
        "cycles_by_period": dict(Counter(f["cycle"]["period"] if f["cycle"] else "none" for f in all_failures)),
        "mean_first_repeat_step": statistics.mean(f["cycle"]["first_repeat_step"] for f in all_failures if f["cycle"]),
        "mean_unique_positions": statistics.mean(f["cycle"]["unique_positions"] for f in all_failures if f["cycle"]),
        "failures_with_any_collision": sum(f["collisions"] > 0 for f in all_failures),
        "mean_collision_count": statistics.mean(f["collisions"] for f in all_failures),
        "total_training_goal_transitions": sum(s["training_goal_transitions"] for s in result["seeds"]),
        "goal_transition_fraction": sum(s["training_goal_transitions"] for s in result["seeds"])/(len(config.train.seeds)*config.train.steps),
        "invalid_action_mask_mean_success": statistics.mean(s["invalid_action_mask_diagnostic"]["success_rate"] for s in result["seeds"])}
    write_json(args.output / "diagnosis.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (seed, raw, ep) in zip(axes, visual_examples):
        ax.imshow(raw["grid"], cmap="Greys", vmin=0, vmax=1)
        shortest, path = raw["shortest_path"], ep["path"]
        ax.plot([p[1] for p in shortest], [p[0] for p in shortest], "--", color="#078c80", label="A* shortest")
        ax.plot([p[1] for p in path], [p[0] for p in path], color="#e8724d", linewidth=3, label="Recorded DQN")
        cycle = cycle_info(ep)
        if cycle:
            r, c = cycle["position"]
            ax.scatter([c], [r], marker="o", s=180, facecolors="none", edgecolors="#be3d45", linewidths=2)
        ax.scatter([0,9], [0,9], c=["#397cb4", "#d6a633"], s=60)
        cycle_text = f"cycle={cycle['period']} from action {cycle['first_repeat_step']}" if cycle else "no repeated state"
        ax.set_title(f"seed {seed} | success={ep['success']}\n{cycle_text}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Same first validation maze | latest B checkpoints | recorded greedy paths")
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(args.output / "failure_paths.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result["failure_summary"], indent=2))
    print(json.dumps(result["maze_difficulty"], indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
