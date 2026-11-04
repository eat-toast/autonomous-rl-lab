import random
import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class QNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * size * size, 64), nn.ReLU(), nn.Linear(64, 4))

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
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminated[i] = next_state, terminated
        self.position = (i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size):
        ids = self.rng.choice(self.count, batch_size, replace=False)
        return (torch.tensor(self.states[ids], dtype=torch.float32),
                torch.tensor(self.actions[ids]), torch.tensor(self.rewards[ids]),
                torch.tensor(self.next_states[ids], dtype=torch.float32),
                torch.tensor(self.terminated[ids]))


class DQNAgent:
    def __init__(self, config, seed):
        self.config = config
        self.online = QNetwork(config.env.size)
        self.target = QNetwork(config.env.size)
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
        return int(model(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
