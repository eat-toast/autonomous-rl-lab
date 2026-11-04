import statistics
from .dqn import greedy_action
from .maze import Maze, MazeEnv


def summarize(episodes):
    successes = [row for row in episodes if row["success"]]
    failures = [row for row in episodes if not row["success"]]
    return {"episodes": len(episodes), "success_rate": len(successes) / len(episodes),
            "average_success_steps": statistics.mean(r["steps"] for r in successes) if successes else None,
            "path_efficiency": statistics.mean(r["path_efficiency"] for r in successes) if successes else None,
            "failed_episodes": len(failures),
            "average_failure_steps": statistics.mean(r["steps"] for r in failures) if failures else None}


def evaluate(model, env_config, mazes, deadline_check=None):
    episodes = []
    was_training = model.training
    model.eval()
    try:
        for raw in mazes:
            env = MazeEnv(env_config, Maze.from_dict(raw))
            state = env.reset()
            path, total_reward, collisions = [env.position], 0.0, 0
            for _ in range(env_config.max_steps):
                if deadline_check:
                    deadline_check()
                state, reward, terminated, truncated, info = env.step(greedy_action(model, state))
                path.append(env.position)
                total_reward += reward
                collisions += int(info["collision"])
                if terminated or truncated:
                    break
            episodes.append({"maze_seed": raw["seed"], "maze_hash": raw["hash"],
                             "success": terminated, "steps": env.steps,
                             "terminated": terminated, "truncated": truncated,
                             "return": total_reward, "collisions": collisions,
                             "shortest_length": raw["shortest_length"],
                             "path_efficiency": raw["shortest_length"] / env.steps if terminated else None,
                             "path": path})
    finally:
        model.train(was_training)
    return {"metrics": summarize(episodes), "episodes": episodes}
