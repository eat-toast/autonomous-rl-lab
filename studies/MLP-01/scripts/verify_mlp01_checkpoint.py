"""Independent checkpoint replay audit using a separately implemented maze loop."""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from rl_lab.paths import resolve_legacy_path, study_runs, run_path
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "studies/MLP-01/results"
RUNS = study_runs(RESULTS)
DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


class RestoredPolicy(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, values):
        return self.layers(values)


def read(path):
    return json.loads(resolve_legacy_path(path).read_text(encoding="utf-8"))


def make_observation(grid, pos, visited, include_visited):
    row, col = pos
    n = len(grid)
    values = [float(not (0 <= row+dr < n and 0 <= col+dc < n) or grid[row+dr][col+dc] == 1)
              for dr, dc in DIRS]
    values += [(9-row)/9, (9-col)/9]
    if include_visited:
        values += [float((row+dr, col+dc) in visited) if 0 <= row+dr < n and 0 <= col+dc < n else 0.0
                   for dr, dc in MOVES]
    return values


def oscillation(path):
    tail = [tuple(x) for x in path[-9:]]
    return len(tail) == 9 and tail[0] != tail[1] and all(tail[i] == tail[i % 2] for i in range(9))


def audit_one(stage, seed, manifest):
    include_visited = stage == "MLP14"
    inputs = 14 if include_visited else 10
    model = RestoredPolicy(inputs)
    checkpoint_path = RUNS / f"{stage}_seed_{seed}" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    source = read(RUNS / f"{stage}_seed_{seed}" / "validation_20000.json")["episodes"]
    rebuilt = []
    with torch.inference_mode():
        for maze_raw, saved in zip(manifest["splits"]["validation"], source):
            grid = maze_raw["grid"]; pos = (0, 0); visited = {pos}
            path, actions, observations, rewards, collisions = [pos], [], [], [], []
            terminated = False
            for step in range(100):
                obs = make_observation(grid, pos, visited, include_visited)
                action = int(model(torch.tensor([obs], dtype=torch.float32)).argmax(1).item())
                dr, dc = MOVES[action]; nxt = (pos[0]+dr, pos[1]+dc)
                hit = not (0 <= nxt[0] < 10 and 0 <= nxt[1] < 10) or grid[nxt[0]][nxt[1]] == 1
                if not hit:
                    pos = nxt; visited.add(pos)
                terminated = pos == (9, 9)
                reward = 1.0 if terminated else (-0.05 if hit else -0.01)
                actions.append(action); observations.append(obs); rewards.append(reward)
                collisions.append(bool(hit)); path.append(pos)
                if terminated: break
            episode = {"maze_seed": maze_raw["seed"], "maze_hash": maze_raw["hash"],
                       "success": terminated, "steps": len(actions), "terminated": terminated,
                       "truncated": not terminated and len(actions) == 100,
                       "path": [list(p) for p in path], "actions": actions,
                       "observations": observations, "rewards": rewards,
                       "collision_flags": collisions, "collisions": sum(collisions),
                       "shortest_length": maze_raw["shortest_length"],
                       "path_efficiency": maze_raw["shortest_length"]/len(actions) if terminated else None}
            fields = ("maze_hash", "success", "steps", "terminated", "truncated", "path", "actions",
                      "observations", "rewards", "collision_flags", "collisions", "path_efficiency")
            mismatches = []
            for field in fields:
                if field == "observations":
                    # The policy stored float32 observations; this independent implementation uses Python floats.
                    equal = (len(episode[field]) == len(saved[field]) and all(
                        len(left) == len(right) and all(abs(a-b) <= 1e-7 for a, b in zip(left, right))
                        for left, right in zip(episode[field], saved[field])))
                else:
                    equal = episode[field] == saved[field]
                if not equal:
                    mismatches.append(field)
            rebuilt.append({"maze_seed": maze_raw["seed"], "matched": not mismatches, "mismatches": mismatches,
                            "success": terminated, "failure_100_step_oscillation":
                            bool(not terminated and len(actions) == 100 and oscillation(path)),
                            "collisions": sum(collisions), "path_efficiency": episode["path_efficiency"]})
    return {"stage": stage, "seed": seed, "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "checkpoint_step": checkpoint["step"], "input_size": checkpoint["input_size"],
            "episodes": rebuilt, "episodes_matched": sum(e["matched"] for e in rebuilt),
            "episodes_total": len(rebuilt), "all_match": all(e["matched"] for e in rebuilt)}


def main():
    manifest = read(RESULTS / "maze_manifest.json")
    audits = [audit_one(stage, seed, manifest) for stage in ("MLP10", "MLP14") for seed in range(3)]
    m10 = [e for a in audits if a["stage"] == "MLP10" for e in a["episodes"]]
    per_seed = {str(a["seed"]): sum(e["failure_100_step_oscillation"] for e in a["episodes"])
                for a in audits if a["stage"] == "MLP10"}
    result = {"review_method": "independently reconstructed Sequential network, checkpoint load and maze/action loop",
              "device": "cpu", "manifest_sha256": hashlib.sha256((RESULTS/"maze_manifest.json").read_bytes()).hexdigest(),
              "all_6_checkpoints_replayed": len(audits) == 6 and all(a["checkpoint_step"] == 20000 for a in audits),
              "all_48_validation_episodes_match": sum(a["episodes_matched"] for a in audits) == 48,
              "entry_criterion_8_move_oscillations_total": sum(per_seed.values()),
              "entry_criterion_by_seed": per_seed, "audits": audits}
    (RESULTS / "checkpoint_restore_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "audits"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
