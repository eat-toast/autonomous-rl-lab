"""Local wall-feeler observations and MLP policies for study MLP-01."""
import numpy as np
import torch
from torch import nn

from .maze import ACTIONS


FEELER_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
                     (1, 0), (1, -1), (0, -1), (-1, -1))
CARDINAL_DIRECTIONS = ACTIONS  # north, east, south, west; matches action indices


def observation(grid, position, goal, visited, include_visited=False, *, visit_counts=False):
    """Return float32 wall bits, normalized goal delta, and optional neighbor visits."""
    n = int(grid.shape[0])
    row, col = position
    features = [float(not (0 <= row + dr < n and 0 <= col + dc < n) or
                       bool(grid[row + dr, col + dc])) for dr, dc in FEELER_DIRECTIONS]
    features.extend(((goal[0] - row) / 9.0, (goal[1] - col) / 9.0))
    if visit_counts:
        features.extend(min(visited.get((row + dr, col + dc), 0), 3) / 3
                        if features[direction * 2] == 0 else 0.0
                        for direction, (dr, dc) in enumerate(CARDINAL_DIRECTIONS))
    elif include_visited:
        features.extend(float((row + dr, col + dc) in visited)
                        if 0 <= row + dr < n and 0 <= col + dc < n else 0.0
                        for dr, dc in CARDINAL_DIRECTIONS)
    return np.asarray(features, dtype=np.float32)


class FeelerQNetwork(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.layers(x)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def greedy_action(model, features):
    with torch.inference_mode():
        device = next(model.parameters()).device
        return int(model(torch.as_tensor(features, dtype=torch.float32, device=device)[None]).argmax(1).item())

