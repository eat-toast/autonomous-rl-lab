import json
from collections import deque
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from rl_lab.config import Config, EnvConfig, from_dict
from rl_lab.dqn import DQNAgent, QNetwork, ReplayBuffer, seed_everything, td_targets
from rl_lab.evaluation import evaluate, summarize
from rl_lab.maze import ACTIONS, Maze, MazeEnv, astar, generate_maze, make_splits
from rl_lab.runner import aggregate
from rl_lab.train import save_checkpoint, train_seed


def bfs_length(grid):
    queue, seen = deque([((0, 0), 0)]), {(0, 0)}
    while queue:
        (r, c), distance = queue.popleft()
        if (r, c) == (len(grid) - 1, len(grid) - 1):
            return distance
        for dr, dc in ACTIONS:
            nxt = r + dr, c + dc
            if 0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid) and not grid[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


class MazeTests(unittest.TestCase):
    def test_astar_matches_independent_bfs_and_reproducibility(self):
        for size in (5, 10, 15, 20, 30, 40):
            config = EnvConfig(size=size)
            for seed in range(5):
                maze = generate_maze(config, seed)
                self.assertEqual(maze.fingerprint, generate_maze(config, seed).fingerprint)
                path = astar(maze.grid)
                self.assertEqual(len(path) - 1, bfs_length(maze.grid))
                self.assertTrue(all(not maze.grid[cell] for cell in path))
                self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(path, path[1:])))

    def test_astar_unreachable(self):
        grid = np.zeros((3, 3), dtype=np.uint8)
        grid[1] = 1
        self.assertEqual(astar(grid), [])

    def test_splits_no_seed_or_geometry_leakage(self):
        manifest = make_splits(Config())
        pools = list(manifest["splits"].values())
        self.assertEqual(manifest, make_splits(Config()))
        self.assertEqual(sum(map(len, pools)), len({m["hash"] for p in pools for m in p}))
        self.assertEqual(sum(map(len, pools)), len({m["seed"] for p in pools for m in p}))

    def test_collision_goal_truncation_and_observation(self):
        grid = np.zeros((3, 3), dtype=np.uint8)
        grid[0, 1] = 1
        env = MazeEnv(EnvConfig(size=3, max_steps=5), Maze(grid, 0, 1, .2))
        state = env.reset()
        self.assertEqual(state.shape, (3, 3, 3))
        self.assertEqual(state.dtype, np.float32)
        obs, reward, term, trunc, info = env.step(1)
        self.assertEqual(reward, -.05)
        self.assertEqual(env.position, (0, 0))
        self.assertTrue(info["collision"])
        self.assertTrue(np.array_equal(obs[0], state[0]))
        for action in (2, 2, 1, 1):
            obs, reward, term, trunc, _ = env.step(action)
        self.assertTrue(term)
        self.assertFalse(trunc)  # Goal at time limit is a successful termination.
        self.assertEqual(reward, 1)
        self.assertEqual(obs[1, -1, -1], 1)
        with self.assertRaises(RuntimeError):
            env.step(1)
        short = MazeEnv(EnvConfig(size=3, max_steps=1), Maze(grid, 0, 1, .2))
        _, reward, term, trunc, _ = short.step(2)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(reward, -.01)


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        seed_everything(7)

    def test_bellman_target_uses_target_max_and_only_terminal_mask(self):
        q = torch.tensor([[2., 5., 1., 0.], [100., 4., 8., 0.], [2., 4., 1., 0.]])
        target = td_targets(torch.tensor([1., 1., -1.]), torch.tensor([False, True, False]), q, .5)
        torch.testing.assert_close(target, torch.tensor([3.5, 1., 1.]))

    def test_replay_wrap_and_sampling(self):
        replay = ReplayBuffer(3, 3, 0)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        for i in range(5):
            replay.add(obs, i % 4, float(i), obs, False)
        self.assertEqual(replay.count, 3)
        self.assertEqual(set(replay.sample(3)[2].tolist()), {2., 3., 4.})

    def test_optimizer_updates_online_target_is_frozen_and_syncs(self):
        config = from_dict({"env": {"size": 3}, "train": {"batch_size": 2}})
        agent = DQNAgent(config, 7)
        maze = generate_maze(config.env, 2)
        obs = MazeEnv(config.env, maze).reset()
        agent.replay.add(obs, 0, 1., obs, True)
        agent.replay.add(obs, 1, -1., obs, False)
        before = [p.detach().clone() for p in agent.online.parameters()]
        target_before = [p.detach().clone() for p in agent.target.parameters()]
        self.assertTrue(np.isfinite(agent.update()))
        self.assertTrue(any(not torch.equal(p, b) for p, b in zip(agent.online.parameters(), before)))
        self.assertTrue(all(torch.equal(p, b) for p, b in zip(agent.target.parameters(), target_before)))
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))
        agent.sync_target()
        self.assertTrue(all(torch.equal(p, b) for p, b in zip(agent.online.parameters(), agent.target.parameters())))

    def test_network_shapes_all_stages(self):
        for size in (10, 15, 20, 30, 40):
            self.assertEqual(QNetwork(size)(torch.zeros(2, 3, size, size)).shape, (2, 4))

    def test_evaluation_does_not_mutate_training_model_or_rng(self):
        config = Config()
        model = QNetwork(10)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng_before = torch.random.get_rng_state().clone()
        result = evaluate(model, config.env, make_splits(config)["splits"]["validation"])
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(result["metrics"]["episodes"], 4)

    def test_checkpoint_roundtrip(self):
        config = Config()
        agent = DQNAgent(config, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, agent, config, 7, 3)
            payload = torch.load(path, weights_only=True)
            restored = QNetwork(10)
            restored.load_state_dict(payload["model_state"])
            obs = torch.randn(2, 3, 10, 10)
            torch.testing.assert_close(agent.online(obs), restored(obs))


class PipelineTests(unittest.TestCase):
    def test_invalid_configs(self):
        for raw in ({"train": {"seeds": [0, 0]}}, {"train": {"max_replay_mb": .0001}},
                    {"train": {"steps": 10000}}, {"env": {"goal_reward": 2}},
                    {"train": {"steps": 3.5}}, {"env": {"wall_probability": float("nan")}}):
            with self.assertRaises(ValueError):
                from_dict(raw)

    def test_success_and_failure_metrics_and_undefined_values(self):
        result = summarize([{"success": True, "steps": 10, "path_efficiency": .8},
                            {"success": False, "steps": 100, "path_efficiency": None}])
        self.assertEqual(result["success_rate"], .5)
        self.assertEqual(result["average_success_steps"], 10)
        self.assertEqual(result["average_failure_steps"], 100)
        self.assertIsNone(summarize([{"success": False, "steps": 10}])["path_efficiency"])

    def test_failed_seeds_not_silently_included(self):
        result = aggregate([{"seed": 0, "status": "completed", "last_validation": {"success_rate": 1.}},
                            {"seed": 1, "status": "failed"}])
        self.assertEqual(result["incomplete_seeds"], [1])
        self.assertEqual(result["metrics"]["success_rate"]["mean"], 1.)
        self.assertIsNone(result["metrics"]["success_rate"]["std"])

    def test_small_training_reproduces_and_never_evaluates_test(self):
        config = from_dict({"env": {"size": 5, "max_steps": 10},
                            "train": {"steps": 40, "learning_starts": 8, "batch_size": 4,
                                      "replay_capacity": 50, "target_update": 10},
                            "evaluation": {"interval": 20, "train_mazes": 4,
                                           "validation_mazes": 2, "test_mazes": 2}})
        runs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
                (root / "mazes.json").write_text(json.dumps(make_splits(config)), encoding="utf-8")
                manifest = make_splits(config)
                allowed = {m["hash"] for m in manifest["splits"]["validation"]}
                def validation_only(model, env_config, mazes, deadline_check=None):
                    self.assertEqual({m["hash"] for m in mazes}, allowed)
                    return evaluate(model, env_config, mazes, deadline_check)
                with patch("rl_lab.train.evaluate", side_effect=validation_only):
                    result = train_seed(root, 0)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["environment_steps"], 40)
                self.assertTrue((root / "seed_0" / "best.pt").exists())
                self.assertFalse(list((root / "seed_0").glob("evaluation_test_*.json")))
                history = [{k: v for k, v in row.items() if k != "elapsed_seconds"}
                           for row in result["validation_history"]]
                payload = torch.load(root / "seed_0" / "latest.pt", weights_only=True)
                runs.append((history, payload["model_state"]))
        self.assertEqual(runs[0][0], runs[1][0])
        for key in runs[0][1]:
            torch.testing.assert_close(runs[0][1][key], runs[1][1][key], rtol=0, atol=0)

    def test_time_limit_saves_status_error_and_checkpoint(self):
        config = from_dict({"train": {"max_seconds": 0.000001}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            (root / "mazes.json").write_text(json.dumps(make_splits(config)), encoding="utf-8")
            result = train_seed(root, 0)
            self.assertEqual(result["status"], "time_limit")
            self.assertEqual(result["environment_steps"], 0)
            self.assertTrue((root / "seed_0" / "error.json").exists())
            self.assertTrue((root / "seed_0" / "latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
