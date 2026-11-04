from dataclasses import dataclass
import hashlib
import heapq
import numpy as np

from .config import EnvConfig


ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def astar(grid, start=(0, 0), goal=None):
    """Four-connected unit-cost A*; returns endpoints too, or [] if unreachable."""
    n = len(grid)
    goal = goal or (n - 1, n - 1)
    if grid[start] or grid[goal]:
        return []
    frontier = [(0, 0, start)]
    distance, parents = {start: 0}, {}
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cost != distance[cell]:
            continue
        if cell == goal:
            path = [cell]
            while cell != start:
                cell = parents[cell]
                path.append(cell)
            return path[::-1]
        for dr, dc in ACTIONS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or grid[nxt]:
                continue
            if cost + 1 < distance.get(nxt, float("inf")):
                distance[nxt], parents[nxt] = cost + 1, cell
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (cost + 1 + heuristic, cost + 1, nxt))
    return []


@dataclass
class Maze:
    grid: np.ndarray
    seed: int
    attempts: int
    requested_density: float

    @property
    def fingerprint(self):
        return hashlib.sha256(self.grid.tobytes()).hexdigest()

    def to_dict(self):
        path = astar(self.grid)
        return {"grid": self.grid.tolist(), "seed": self.seed, "attempts": self.attempts,
                "requested_density": self.requested_density,
                "actual_density": float(self.grid.mean()), "hash": self.fingerprint,
                "shortest_path": path, "shortest_length": len(path) - 1}

    @classmethod
    def from_dict(cls, raw):
        maze = cls(np.array(raw["grid"], dtype=np.uint8), raw["seed"],
                   raw["attempts"], raw["requested_density"])
        if maze.fingerprint != raw["hash"] or not astar(maze.grid):
            raise ValueError("Maze manifest corrupted")
        return maze


def generate_maze(config, seed):
    rng = np.random.default_rng(seed)
    for attempt in range(1, 10001):
        grid = (rng.random((config.size, config.size)) < config.wall_probability).astype(np.uint8)
        grid[0, 0] = grid[-1, -1] = 0
        if astar(grid):
            return Maze(grid, seed, attempt, config.wall_probability)
    raise RuntimeError("Unable to generate a reachable maze within 10000 attempts")


def make_splits(config):
    """Disjoint seed namespaces AND geometry hashes, independent of policy seed."""
    splits, seen = {}, set()
    counts = (config.evaluation.train_mazes, config.evaluation.validation_mazes,
              config.evaluation.test_mazes)
    for split_index, (name, count) in enumerate(zip(("train", "validation", "test"), counts)):
        pool = []
        for candidate in range(10000):
            seed = config.evaluation.maze_seed + split_index * 10000 + candidate
            maze = generate_maze(config.env, seed)
            if maze.fingerprint not in seen:
                pool.append(maze.to_dict())
                seen.add(maze.fingerprint)
                if len(pool) == count:
                    break
        if len(pool) != count:
            raise RuntimeError(f"Not enough unique geometries for {name}; reduce pool sizes")
        splits[name] = pool
    return {"schema_version": 1, "generator": "reachable-bernoulli-v1", "splits": splits}


class MazeEnv:
    def __init__(self, config: EnvConfig, maze: Maze):
        self.config, self.maze = config, maze
        if maze.grid.shape != (config.size, config.size) or not astar(maze.grid):
            raise ValueError("Invalid maze for this environment")
        self.reset()

    def reset(self):
        self.position, self.steps, self.finished = (0, 0), 0, False
        return self.observe()

    def observe(self):
        n = self.config.size
        obs = np.zeros((3, n, n), dtype=np.float32)
        obs[0] = self.maze.grid
        obs[1, self.position[0], self.position[1]] = 1
        obs[2, -1, -1] = 1
        return obs

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset() is required after episode end")
        if action not in range(4):
            raise ValueError("action must be 0..3")
        dr, dc = ACTIONS[action]
        nxt = (self.position[0] + dr, self.position[1] + dc)
        n = self.config.size
        collision = not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or bool(self.maze.grid[nxt])
        if not collision:
            self.position = nxt
        self.steps += 1
        terminated = self.position == (n - 1, n - 1)
        truncated = self.steps >= self.config.max_steps and not terminated
        self.finished = terminated or truncated
        reward = (self.config.goal_reward if terminated else
                  self.config.collision_reward if collision else self.config.step_reward)
        return self.observe(), reward, terminated, truncated, {"collision": collision}
