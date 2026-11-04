import argparse
import json
from pathlib import Path
from uuid import uuid4
import torch

from .config import from_dict, load_config
from .dqn import QNetwork, seed_everything
from .evaluation import evaluate
from .storage import write_json


def main():
    parser = argparse.ArgumentParser(description="Autonomous RL Lab — Phase 1")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run a bounded DQN experiment")
    train.add_argument("--config", default="configs/smoke.json")
    train.add_argument("--output", default="runs")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run", required=True)
    worker.add_argument("--seed", type=int, required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate a checkpoint independently")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--final-test", action="store_true", help="Consume held-out test evaluation once for this run/seed")
    args = parser.parse_args()
    if args.command == "train":
        from .runner import run_experiment
        _, completed = run_experiment(load_config(args.config), args.output)
        return 0 if completed else 1
    if args.command == "_worker":
        from .train import train_seed
        summary = train_seed(args.run, args.seed)
        return 0 if summary["status"] == "completed" else 1
    checkpoint = Path(args.checkpoint).resolve()
    # Load only this project's trusted local checkpoints; never arbitrary Internet files.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = from_dict(payload["config"])
    seed_everything(payload["seed"])
    model = QNetwork(config.env.size)
    model.load_state_dict(payload["model_state"])
    run_dir = checkpoint.parent.parent
    manifest = json.loads((run_dir / "mazes.json").read_text(encoding="utf-8"))
    split = "test" if args.final_test else "validation"
    if args.final_test:
        # A second checkpoint from the same run/seed cannot repeatedly inspect the test set.
        marker = checkpoint.parent / "final_test_consumed.json"
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"checkpoint": checkpoint.name, "step": payload["step"]}, stream)
    import time
    deadline = time.monotonic() + config.train.max_seconds

    def check_budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("Evaluation wall time budget reached")

    result = evaluate(model, config.env, manifest["splits"][split], check_budget)
    result.update(split=split, checkpoint=checkpoint.name, step=payload["step"], seed=payload["seed"])
    destination = checkpoint.parent / f"evaluation_{split}_{uuid4().hex[:8]}.json"
    write_json(destination, result)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
