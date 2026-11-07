import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rl_lab.config import from_dict
from rl_lab.dqn import ReplayBuffer
from rl_lab.maze import make_splits
from rl_lab.runner import run_experiment, validate_manifest
from rl_lab.storage import state_digest
from rl_lab.train import train_seed


class ReplayStudyTests(unittest.TestCase):
    def test_clear_preserves_rng_and_never_samples_old_experience(self):
        buffer = ReplayBuffer(4, 3, 9)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        for step in range(1, 9):
            buffer.add(obs, 0, float(step), obs, step == 8, step, 100+step)
        before = copy.deepcopy(buffer.rng.bit_generator.state)
        buffer.clear()
        self.assertEqual(buffer.rng.bit_generator.state, before)
        self.assertEqual((buffer.count, buffer.position), (0, 0))
        self.assertIsNone(buffer.snapshot(8)["oldest_step"])
        with self.assertRaises(ValueError):
            buffer.sample(1)
        for step in (9, 10):
            buffer.add(obs, 0, float(step), obs, False, step, 500)
        self.assertEqual(set(buffer.sample(2)[2].tolist()), {9., 10.})
        self.assertEqual(buffer.snapshot(10)["maze_counts"], {"500": 2})

    def test_clear_is_local_and_paired_pause_has_equal_updates(self):
        self.assert_paired_pause("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_clear_pair_is_reproducible_and_keeps_device(self):
        self.assert_paired_pause("cuda")

    def assert_paired_pause(self, device):
        base = {"env": {"size": 5, "max_steps": 10},
                "train": {"device": device, "steps": 40, "learning_starts": 8, "batch_size": 4,
                          "replay_capacity": 50, "target_update": 10,
                          "replay_event_step": 20, "replay_pause_steps": 8},
                "evaluation": {"interval": 20, "train_mazes": 4, "validation_mazes": 2, "test_mazes": 2}}
        results = []
        for clear in (False, True):
            raw = copy.deepcopy(base)
            raw["train"]["replay_clear"] = clear
            config = from_dict(raw)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
                (root / "mazes.json").write_text(json.dumps(make_splits(config)), encoding="utf-8")
                summary = train_seed(root, 0)
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["device"], device)
                self.assertEqual(summary["optimizer_updates"], 7)
                self.assertEqual(summary["paused_updates"], 2)
                self.assertEqual(summary["paused_env_steps"], 8)
                read = lambda name: json.loads((root / "seed_0" / name).read_text(encoding="utf-8"))
                event, refill, resume = [read(name) for name in
                                         ("replay_event.json", "replay_after_refill.json", "replay_first_resume_update.json")]
                self.assertEqual(event["before"], event["after"])
                self.assertEqual(event["optimizer_updates"], 4)
                self.assertEqual(refill["optimizer_updates"], 4)
                self.assertEqual(resume["step"], 32)
                self.assertEqual(resume["optimizer_updates"], 5)
                if clear:
                    self.assertEqual(event["replay_after"]["count"], 0)
                    self.assertEqual(refill["count"], 8)
                    self.assertEqual(resume["count"], 12)
                    self.assertEqual(refill["oldest_step"], 21)
                results.append((event, refill))
        self.assertEqual(results[0][0]["before"], results[1][0]["before"])
        self.assertEqual(results[0][1]["state_hashes"], results[1][1]["state_hashes"])
        self.assertEqual(results[0][1]["trajectory_sha256"], results[1][1]["trajectory_sha256"])

    def test_fixed_manifest_parallel_workers_and_no_test_evaluation(self):
        config = from_dict({"env": {"size": 5, "max_steps": 10},
                            "train": {"seeds": [0, 1], "steps": 16, "learning_starts": 4, "batch_size": 2},
                            "evaluation": {"interval": 8, "train_mazes": 4, "validation_mazes": 2, "test_mazes": 2}})
        manifest = make_splits(config)
        bad = copy.deepcopy(manifest)
        bad["splits"]["validation"][0] = bad["splits"]["train"][0]
        with self.assertRaises(ValueError):
            validate_manifest(config, bad)
        with tempfile.TemporaryDirectory() as directory:
            root, completed = run_experiment(config, directory, maze_manifest=manifest, workers=2)
            self.assertTrue(completed)
            self.assertEqual(json.loads((root / "mazes.json").read_text(encoding="utf-8")), json.loads(json.dumps(manifest)))
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([s["seed"] for s in summary["seeds"]], [0, 1])
            self.assertTrue(all(s["optimizer_updates"] == 4 for s in summary["seeds"]))
            self.assertFalse(list(root.glob("seed_*/evaluation_test_*.json")))

    def test_invalid_replay_interventions(self):
        for patch in ({"replay_clear": True}, {"replay_pause_steps": 8},
                      {"replay_event_step": 400, "replay_pause_steps": 120},
                      {"replay_event_step": 100, "replay_clear": True, "replay_pause_steps": 1}):
            with self.assertRaises(ValueError):
                from_dict({"train": patch})


if __name__ == "__main__":
    unittest.main()
