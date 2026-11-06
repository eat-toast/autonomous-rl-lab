import unittest

import numpy as np
import torch

from rl_lab.mlp01 import FeelerQNetwork, observation, parameter_count
from scripts.run_mlp01 import FeatureReplay, classify


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

    def test_two_cell_oscillation_classifier(self):
        self.assertEqual(classify({"success": False, "path": [[4, 4], [4, 5]] * 4 + [[4, 4]],
                                   "collisions": 0}), "two_cell_oscillation")


if __name__ == "__main__":
    unittest.main()
