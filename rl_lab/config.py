from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class EnvConfig:
    size: int = 10
    wall_probability: float = 0.20
    max_steps: int = 100
    step_reward: float = -0.01
    collision_reward: float = -0.05
    goal_reward: float = 1.0


@dataclass(frozen=True)
class NetworkConfig:
    hidden_units: int = 64


@dataclass(frozen=True)
class TrainConfig:
    method: str = "baseline"
    device: str = "cpu"
    seeds: list[int] = field(default_factory=lambda: [0])
    steps: int = 512
    max_seconds: float = 60.0
    hard_timeout_seconds: float = 120.0
    max_total_steps: int = 6000
    replay_capacity: int = 2000
    max_replay_mb: float = 256.0
    batch_size: int = 32
    learning_starts: int = 64
    train_every: int = 4
    target_update: int = 100
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 400
    gradient_clip: float = 10.0
    replay_event_step: int = 0
    replay_clear: bool = False
    replay_pause_steps: int = 0


@dataclass(frozen=True)
class EvalConfig:
    interval: int = 128
    target_success_rate: float = 0.80
    train_mazes: int = 16
    validation_mazes: int = 4
    test_mazes: int = 8
    maze_seed: int = 20261104
    diagnose_training_fit: bool = False


@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    algorithm: str = "dqn"
    hypothesis: str = "Phase 1 smoke: verify the pipeline; no performance claim."
    change_description: str = "New whole-map CNN DQN baseline; legacy code not used."

    def validate(self):
        e, t, v = self.env, self.train, self.evaluation
        if t.method not in ("baseline", "success", "per", "nstep", "double", "mask"):
            raise ValueError("Unknown method")
        if t.method == "success" and t.replay_capacity != 2000:
            raise ValueError("success method requires fixed 1800+200 slots")
        # Reject invalid values before directories, processes, or tensors are created.
        for section in (asdict(e), asdict(t), asdict(v)):
            for value in section.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Configuration must contain finite numbers")
        if self.algorithm != "dqn":
            raise ValueError("Phase 1 supports only dqn")
        if t.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if type(self.network.hidden_units) is not int or not 1 <= self.network.hidden_units <= 512:
            raise ValueError("hidden_units must be an integer in [1, 512]")
        if type(e.size) is not int or not 3 <= e.size <= 40:
            raise ValueError("size must be an integer in [3, 40]")
        if not 0 < e.wall_probability < 0.7 or e.max_steps < 1:
            raise ValueError("wall_probability must be in (0, .7), max_steps positive")
        if (e.step_reward, e.collision_reward, e.goal_reward) != (-0.01, -0.05, 1.0):
            raise ValueError("reward-v1 is frozen for this study")
        if not t.seeds or len(set(t.seeds)) != len(t.seeds):
            raise ValueError("Training seeds must be nonempty and unique")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in t.seeds):
            raise ValueError("Seeds must be unsigned 32-bit integers")
        integer_fields = [e.max_steps, t.steps, t.max_total_steps, t.replay_capacity,
                          t.batch_size, t.learning_starts, t.train_every, t.target_update,
                          t.epsilon_decay_steps, v.interval, v.train_mazes,
                          v.validation_mazes, v.test_mazes]
        if any(type(n) is not int or n < 1 for n in integer_fields):
            raise ValueError("Step counts, pool sizes, and capacities must be positive integers")
        if t.steps * len(t.seeds) > t.max_total_steps:
            raise ValueError("Total step budget exceeded")
        if t.max_seconds <= 0 or t.hard_timeout_seconds <= t.max_seconds:
            raise ValueError("hard timeout must exceed positive soft timeout")
        if t.batch_size > t.replay_capacity:
            raise ValueError("batch_size exceeds replay capacity")
        if t.learning_starts >= t.steps:
            raise ValueError("learning_starts must be less than steps")
        if not 0 <= t.gamma <= 1 or not 0 <= t.epsilon_end <= t.epsilon_start <= 1:
            raise ValueError("Invalid discount or exploration schedule")
        if t.learning_rate <= 0 or t.gradient_clip <= 0 or t.max_replay_mb <= 0:
            raise ValueError("Optimizer and memory limits must be positive")
        if type(t.replay_clear) is not bool:
            raise ValueError("replay_clear must be boolean")
        if any(type(v) is not int or v < 0 for v in (t.replay_event_step, t.replay_pause_steps)):
            raise ValueError("Replay event/pause steps must be nonnegative integers")
        if t.replay_event_step:
            if t.replay_event_step + t.replay_pause_steps >= t.steps:
                raise ValueError("Replay event/pause must finish before training ends")
            if t.replay_event_step < t.learning_starts:
                raise ValueError("Replay event must occur after learning starts")
        elif t.replay_clear or t.replay_pause_steps:
            raise ValueError("Replay clearing/pause requires an event step")
        if t.replay_clear and t.replay_pause_steps < t.batch_size:
            raise ValueError("After clear, refill pause must supply at least a batch")
        if not 0 < v.target_success_rate <= 1 or type(v.maze_seed) is not int or v.maze_seed < 0:
            raise ValueError("Invalid evaluation settings")
        # Preallocated uint8 states: two states per transition + scalar arrays.
        replay_bytes = t.replay_capacity * (2 * 3 * e.size * e.size + 29)
        if replay_bytes > t.max_replay_mb * 1024**2:
            raise ValueError("Replay allocation exceeds max_replay_mb")
        return self

    def to_dict(self):
        return asdict(self)


def from_dict(raw):
    raw = dict(raw)
    raw["env"] = EnvConfig(**raw.get("env", {}))
    raw["train"] = TrainConfig(**raw.get("train", {}))
    raw["evaluation"] = EvalConfig(**raw.get("evaluation", {}))
    raw["network"] = NetworkConfig(**raw.get("network", {}))
    return Config(**raw).validate()


def load_config(path):
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
