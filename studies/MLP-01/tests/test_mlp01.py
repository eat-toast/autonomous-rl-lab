import unittest

import numpy as np
import torch

from rl_lab.mlp01 import FeelerQNetwork, observation, parameter_count
from scripts.study_modules import load as _load_study
_mlp = _load_study("MLP-01", "run_mlp01")
FeatureReplay = _mlp.FeatureReplay
classify = _mlp.classify
new_visit_history = _mlp.new_visit_history
record_arrival = _mlp.record_arrival


class FeelerObservationTests(unittest.TestCase):
    def test_direction_order_and_boundary_are_nw_ordered(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[0, 1] = 1
        values = observation(grid, (0, 0), (9, 9), {(0, 0)})
        self.assertEqual(values[:8].tolist(), [1., 1., 1., 0., 0., 1., 1., 1.])
        self.assertEqual(values[8:].tolist(), [1., 1.])

    def test_visit_bits_are_cardinal_and_ordered(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        visited = {(0, 0), (0, 1), (1, 0)}
        values = observation(grid, (0, 0), (9, 9), visited, True)
        self.assertEqual(values.shape, (14,))
        self.assertEqual(values[10:].tolist(), [0., 1., 1., 0.])

    def test_network_has_four_outputs_and_expected_parameter_counts(self):
        ten, fourteen = FeelerQNetwork(10), FeelerQNetwork(14)
        self.assertEqual(ten(torch.zeros(2, 10)).shape, (2, 4))
        self.assertEqual(parameter_count(ten), 5124)
        self.assertEqual(parameter_count(fourteen), 5380)

    def test_replay_keeps_feature_values(self):
        replay = FeatureReplay(4, 10, 1)
        state = np.arange(10, dtype=np.float32)
        replay.add(state, 2, 0.1, state + 1, False)
        state[:] = -1
        batch = replay.sample(1)
        self.assertEqual(batch[0].shape, (1, 10))
        self.assertEqual(batch[0][0].tolist(), list(range(10)))

    def test_visit_counts_track_arrivals_and_replay_keeps_temporal_values(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[1, 1] = 1
        left, right = (0, 0), (0, 1)
        training = new_visit_history(left, True)
        evaluation = new_visit_history(left, True)
        replay = FeatureReplay(4, 14, 1)

        def observe(position, visits):
            return observation(grid, position, (9, 9), visits, visit_counts=True)

        self.assertEqual(training[left], 1)
        before = observe(left, training)
        record_arrival(training, left, left, True)  # boundary collision
        np.testing.assert_array_equal(observe(left, training), before)
        record_arrival(training, right, left, True)
        after = observe(right, training)
        replay.add(before, 1, -.01, after, False)
        np.testing.assert_allclose(after[-4:], [0, 0, 0, 1 / 3])
        record_arrival(training, right, right, True)  # wall collision
        self.assertEqual(training[right], 1)
        record_arrival(training, left, right, True)
        record_arrival(training, right, left, True)
        np.testing.assert_allclose(observe(right, training)[-4:], [0, 0, 0, 2 / 3])
        for _ in range(4):
            record_arrival(training, left, right, True)
            record_arrival(training, right, left, True)
        training[(1, 1)] = 2  # invalid history must still emit zero on a wall
        np.testing.assert_array_equal(observe(right, training)[-4:], [0, 0, 0, 1])
        np.testing.assert_array_equal(observe(left, evaluation), before)
        self.assertEqual(new_visit_history(left, True), {left: 1})
        sampled_before, _, _, sampled_after, _ = replay.sample(1)
        np.testing.assert_array_equal(sampled_before[0], before)
        np.testing.assert_array_equal(sampled_after[0], after)

    def test_two_cell_oscillation_classifier(self):
        self.assertEqual(classify({"success": False, "path": [[4, 4], [4, 5]] * 4 + [[4, 4]],
                                   "collisions": 0}), "two_cell_oscillation")


if __name__ == "__main__":
    unittest.main()
