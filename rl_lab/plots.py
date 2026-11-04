import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run(run_dir, summaries, manifest):
    fig, ax = plt.subplots(figsize=(7, 4))
    for summary in summaries:
        rows = summary.get("validation_history", [])
        if rows:
            ax.plot([r["step"] for r in rows], [r["success_rate"] for r in rows],
                    marker="o", label=f"seed {summary['seed']}")
    ax.set(xlabel="Training environment steps", ylabel="Validation success rate", ylim=(-0.02, 1.02),
           title="DQN pipeline check (not a convergence study)")
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
