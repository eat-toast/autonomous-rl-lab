import random
import os
from collections import Counter
import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class QNetwork(nn.Module):
    def __init__(self, size, hidden_units=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * size * size, hidden_units), nn.ReLU(), nn.Linear(hidden_units, 4))

    def forward(self, observations):
        return self.layers(observations)


def td_targets(rewards, terminated, next_q, gamma):
    """Standard DQN: selection and evaluation both use target-network max."""
    return rewards + gamma * (~terminated).float() * next_q.max(dim=1).values


class ReplayBuffer:
    def __init__(self, capacity, size, seed):
        self.capacity, self.position, self.count = capacity, 0, 0
        shape = (capacity, 3, size, size)
        self.states = np.empty(shape, dtype=np.uint8)
        self.next_states = np.empty(shape, dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.insertion_steps = np.empty(capacity, dtype=np.int64)
        self.maze_seeds = np.empty(capacity, dtype=np.int64)
        self.last_sample_stats = None
        self.sampled_transitions = 0
        self.sampled_goals = 0
        self.sampled_age_sum = 0
        self.sampled_maze_counts = Counter()
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminated, step=0, maze_seed=-1):
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminated[i] = next_state, terminated
        self.insertion_steps[i], self.maze_seeds[i] = step, maze_seed
        self.position = (i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size):
        ids = self.rng.choice(self.count, batch_size, replace=False)
        newest = int(self.insertion_steps[:self.count].max())
        ages = newest-self.insertion_steps[ids]
        sampled_goals = int(self.terminated[ids].sum())
        self.sampled_transitions += len(ids)
        self.sampled_goals += sampled_goals
        self.sampled_age_sum += int(ages.sum())
        self.sampled_maze_counts.update(map(int, self.maze_seeds[ids]))
        self.last_sample_stats = {
            "mean_age_steps": float(ages.mean()),
            "goal_transitions": sampled_goals,
            "unique_maze_count": int(len(np.unique(self.maze_seeds[ids])))}
        return (torch.tensor(self.states[ids], dtype=torch.float32),
                torch.tensor(self.actions[ids]), torch.tensor(self.rewards[ids]),
                torch.tensor(self.next_states[ids], dtype=torch.float32),
                torch.tensor(self.terminated[ids]))

    def clear(self):
        """Logically remove stored experience; preserve sampler RNG and allocated arrays."""
        self.position, self.count = 0, 0
        self.last_sample_stats = None

    def snapshot(self, current_step):
        sampling = {"scope": "since run start; survives clear", "transitions": self.sampled_transitions,
                    "goal_transitions": self.sampled_goals,
                    "mean_age_steps": self.sampled_age_sum/self.sampled_transitions if self.sampled_transitions else None,
                    "maze_counts": {str(k): v for k, v in sorted(self.sampled_maze_counts.items())}}
        if not self.count:
            return {"count": 0, "position": self.position, "capacity": self.capacity, "goal_transitions": 0,
                    "oldest_step": None, "newest_step": None, "mean_age_steps": None,
                    "maze_counts": {}, "last_sample": self.last_sample_stats, "sampling_totals": sampling}
        ids, counts = np.unique(self.maze_seeds[:self.count], return_counts=True)
        steps = self.insertion_steps[:self.count]
        return {"count": self.count, "position": self.position, "capacity": self.capacity,
                "goal_transitions": int(self.terminated[:self.count].sum()),
                "oldest_step": int(steps.min()), "newest_step": int(steps.max()),
                "mean_age_steps": float((current_step-steps).mean()),
                "age_quantiles_0_25_50_75_100": np.quantile(current_step-steps, [0, .25, .5, .75, 1]).tolist(),
                "maze_counts": {str(int(k)): int(v) for k, v in zip(ids, counts)},
                "last_sample": self.last_sample_stats, "sampling_totals": sampling}


class DQNAgent:
    def __init__(self, config, seed):
        self.config = config
        self.device = torch.device(config.train.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; no silent CPU fallback")
        self.online = QNetwork(config.env.size, config.network.hidden_units).to(self.device)
        self.target = QNetwork(config.env.size, config.network.hidden_units).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.train.learning_rate)
        self.replay = ReplayBuffer(config.train.replay_capacity, config.env.size, seed)
        self.rng = np.random.default_rng(seed)

    def epsilon(self, step):
        t = self.config.train
        fraction = min(1.0, step / t.epsilon_decay_steps)
        return t.epsilon_start + fraction * (t.epsilon_end - t.epsilon_start)

    def act(self, state, step):
        if self.rng.random() < self.epsilon(step):
            return int(self.rng.integers(4))
        return greedy_action(self.online, state)

    def update(self):
        states, actions, rewards, next_states, terminated = self.replay.sample(self.config.train.batch_size)
        states, actions, rewards, next_states, terminated = (
            tensor.to(self.device) for tensor in (states, actions, rewards, next_states, terminated))
        values = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = td_targets(rewards, terminated, self.target(next_states), self.config.train.gamma)
        loss = nn.functional.smooth_l1_loss(values, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.train.gradient_clip,
                                error_if_nonfinite=True)
        self.optimizer.step()
        return float(loss.detach())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())


def greedy_action(model, state):
    with torch.inference_mode():
        device = next(model.parameters()).device
        states = torch.from_numpy(state).unsqueeze(0).to(device)
        values = model(states)
        if getattr(model, "action_mask", False):
            from .methods import legal_masks
            values = values.masked_fill(~legal_masks(states), -torch.inf)
        return int(values.argmax(dim=1).item())
