import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import numpy as np


def trailing_mean(values, window=20):
    """Causal average; early episodes use all available observations, no future data."""
    values = np.asarray(values, dtype=float)
    totals = np.concatenate(([0.0], np.cumsum(values)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (totals[end] - totals[start]) / (end - start)


def load_episodes(run_dir, seed):
    path = run_dir / f"seed_{seed}" / "training_episodes.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_rewards(run_dir, summaries):
    """Training exploration returns, distinct from greedy validation performance."""
    for axis_name in ("episode", "environment_steps"):
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        for summary in summaries:
            rows = load_episodes(run_dir, summary["seed"])
            if not rows:
                continue
            returns = [float(row["return"]) for row in rows]
            x = (np.arange(1, len(rows)+1) if axis_name == "episode"
                 else [int(row["end_step"]) for row in rows])
            color = f"C{summary['seed'] % 10}"
            axes[0].plot(x, returns, color=color, alpha=.16, linewidth=.8)
            axes[0].plot(x, trailing_mean(returns), color=color,
                         label=f"seed {summary['seed']} / trailing 20 episodes")
            success = [row["success"] == "True" for row in rows]
            axes[1].plot(x, trailing_mean(success)*100, color=color)
        axes[0].set(ylabel="Episode return (sum of rewards)",
                    title="Training returns with epsilon-greedy exploration")
        axes[0].axhline(0, color="grey", linewidth=.7, linestyle="--")
        axes[1].set(ylabel="Training success (%)", ylim=(-2, 102),
                    xlabel="Completed episode" if axis_name == "episode" else "Training environment steps")
        for ax in axes:
            ax.grid(alpha=.2)
        if axes[0].lines:
            axes[0].legend(fontsize=8)
        fig.text(.5, .01, "Faint lines: raw returns. Solid: causal moving averages. Incomplete final episode excluded.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
        name = "reward_curve.png" if axis_name == "episode" else "reward_curve_steps.png"
        fig.savefig(run_dir / name, dpi=160)
        plt.close(fig)


def plot_run(run_dir, summaries, manifest):
    plot_rewards(run_dir, summaries)
    fig, ax = plt.subplots(figsize=(7, 4))
    for summary in summaries:
        rows = summary.get("validation_history", [])
        if rows:
            ax.plot([r["step"] for r in rows], [r["success_rate"] for r in rows],
                    marker="o", label=f"seed {summary['seed']}")
    ax.set(xlabel="Training environment steps", ylabel="Validation success rate", ylim=(-0.02, 1.02),
           title="DQN greedy validation performance")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curve.png", dpi=140)
    plt.close(fig)

    import json
    first = next((s for s in summaries if (run_dir / f"seed_{s['seed']}" / "latest_validation.json").exists()), None)
    if first is None:
        return
    data = json.loads((run_dir / f"seed_{first['seed']}" / "latest_validation.json").read_text(encoding="utf-8"))
    episode = data["episodes"][0]
    raw = next(m for m in manifest["splits"]["validation"] if m["hash"] == episode["maze_hash"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(raw["grid"], cmap="Greys", vmin=0, vmax=1)
    shortest, actual = raw["shortest_path"], episode["path"]
    ax.plot([p[1] for p in shortest], [p[0] for p in shortest], "--", color="#139b87", label="A* shortest")
    ax.plot([p[1] for p in actual], [p[0] for p in actual], color="#ee7955", label="Greedy DQN")
    ax.scatter([0, len(raw["grid"]) - 1], [0, len(raw["grid"]) - 1], c=["#3677d2", "#f1c747"], s=80)
    ax.set_title(f"Validation maze | success={episode['success']}\n"
                 f"Actions {episode['steps']} | A* {episode['shortest_length']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "path_comparison.png", dpi=140)
    plt.close(fig)
