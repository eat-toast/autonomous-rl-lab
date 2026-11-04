"""Runtime-only microbenchmark with synthetic replay; not maze performance evidence."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rl_lab.config import from_dict
from rl_lab.dqn import DQNAgent, greedy_action, seed_everything


def main():
    raw = json.loads((ROOT / "runs/20261104T042433Z_2393292b/config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(123)
    obs = (rng.random((3, 10, 10)) > .8).astype(np.float32)
    results = []
    for device in ("cpu", "cuda"):
        settings = copy.deepcopy(raw)
        settings["train"]["device"] = device
        seed_everything(123)
        agent = DQNAgent(from_dict(settings), 123)
        for i in range(2000):
            agent.replay.add(obs, i % 4, -.01, obs, False, i+1, 0)
        def workload(n):
            for i in range(n):
                # Approximate collection + validation inference, and 1 optimizer update/4 steps.
                for _ in range(3):
                    greedy_action(agent.online, obs)
                if i % 4 == 0:
                    agent.update()
        workload(40)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(3):
            started = time.perf_counter()
            workload(500)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        update_times = []
        for _ in range(3):
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(250):
                agent.update()
            if device == "cuda":
                torch.cuda.synchronize()
            update_times.append(time.perf_counter()-started)
        results.append({"device": device, "seconds_3_trials": times, "median_seconds": float(np.median(times)),
                        "update_only_250_seconds": update_times,
                        "update_only_median_seconds": float(np.median(update_times)),
                        "scope": "synthetic replay; 1500 single-state inferences + 125 updates per trial; no RL result"})
        del agent
        if device == "cuda":
            torch.cuda.empty_cache()
    out = ROOT / "diagnostics/device_benchmark_split_20261104.json"
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
