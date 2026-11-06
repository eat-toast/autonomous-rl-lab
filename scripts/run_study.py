"""Four sequential, bounded DQN experiments authorized on 2026-11-04."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_lab.config import from_dict
from rl_lab.plots import load_episodes, plot_rewards, trailing_mean
from rl_lab.runner import run_experiment
from rl_lab.storage import utc_now, write_json

PILOT = ROOT / "runs/20261104T031358Z_d9a39f92"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selection_score(run):
    seeds = read(run / "summary.json")["seeds"]
    if len(seeds) != 3 or any(s["status"] != "completed" for s in seeds):
        raise ValueError("Selection requires all three completed seeds")
    rates = [s["last_validation"]["success_rate"] for s in seeds]
    efficiencies = [s["last_validation"]["path_efficiency"] for s in seeds]
    efficiencies = [e for e in efficiencies if e is not None]
    # Mean success first, lower seed dispersion second, successful-path efficiency third.
    return (statistics.mean(rates), -statistics.stdev(rates),
            statistics.mean(efficiencies) if efficiencies else -1.)


def choose_reference(reference, candidate):
    # Exact ties keep the earlier condition, avoiding arbitrary switches.
    return candidate if selection_score(candidate) > selection_score(reference) else reference


def summarize_condition(label, run):
    summary = read(run / "summary.json")
    config = read(run / "config.json")
    train_rates = [s["training_greedy"]["success_rate"] for s in summary["seeds"]
                   if s.get("training_greedy")]
    if not train_rates and (run / "fit_diagnostic.json").exists():
        train_rates = [s["training_greedy"]["success_rate"]
                       for s in read(run / "fit_diagnostic.json")["seeds"]]
    episodes = [row for seed in config["train"]["seeds"] for row in load_episodes(run, seed)]
    return {"label": label, "run_id": run.name,
            "steps": config["train"]["steps"],
            "epsilon_decay_steps": config["train"]["epsilon_decay_steps"],
            "learning_rate": config["train"]["learning_rate"],
            "hidden_units": config.get("network", {}).get("hidden_units", 64),
            "parameters": summary["seeds"][0]["parameter_count"],
            "validation_success_mean": summary["aggregate"]["metrics"]["success_rate"]["mean"],
            "validation_success_std": summary["aggregate"]["metrics"]["success_rate"]["std"],
            "training_greedy_success_mean": statistics.mean(train_rates) if train_rates else None,
            "completed_training_episodes": len(episodes),
            "successful_training_episodes": sum(r["success"] == "True" for r in episodes),
            "time_mean_seconds": summary["aggregate"]["metrics"]["training_time_seconds"]["mean"],
            "seed_validation_success": {s["seed"]: s["last_validation"]["success_rate"] for s in summary["seeds"]}}


def compare_plot(folder, conditions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (label, run) in enumerate(conditions):
        color = f"C{i}"
        seeds = read(run / "summary.json")["seeds"]
        histories = [s["validation_history"] for s in seeds]
        x = [r["step"] for r in histories[0]]
        values = np.array([[r["success_rate"] for r in h] for h in histories]) * 100
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[0, 0].plot(x, mean, color=color, label=label)
        axes[0, 0].fill_between(x, mean-std, mean+std, color=color, alpha=.10)
        curves = []
        for s in seeds:
            rows = load_episodes(run, s["seed"])
            end_steps = np.array([int(r["end_step"]) for r in rows])
            smoothed = trailing_mean([float(r["return"]) for r in rows])
            curves.append((end_steps, smoothed))
            axes[0, 1].plot(end_steps, smoothed, color=color, alpha=.16, linewidth=.8)
        grid = np.linspace(max(c[0][0] for c in curves), min(c[0][-1] for c in curves), 120)
        mean_return = np.array([np.interp(grid, a, b) for a, b in curves]).mean(axis=0)
        axes[0, 1].plot(grid, mean_return, color=color, label=label)
    axes[0, 0].set(title="Greedy validation success: mean +/- sample SD", ylabel="Success (%)",
                   xlabel="Training environment steps", ylim=(-5, 105))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Training returns: trailing 20 episodes", ylabel="Episode return",
                   xlabel="Training environment steps")
    axes[0, 1].legend(fontsize=8)
    rows = [summarize_condition(label, run) for label, run in conditions]
    x = np.arange(len(rows))
    axes[1, 0].bar(x-.18, [r["training_greedy_success_mean"]*100 for r in rows], .36,
                   label="Training mazes (greedy)", color="#80b9b0")
    axes[1, 0].bar(x+.18, [r["validation_success_mean"]*100 for r in rows], .36,
                   yerr=[r["validation_success_std"]*100 for r in rows], capsize=3,
                   label="Validation (greedy)", color="#e88862")
    axes[1, 0].set(title="Final checkpoint: familiar vs unseen mazes", ylabel="Success (%)", ylim=(0, 110))
    axes[1, 0].set_xticks(x, [r["label"] for r in rows], rotation=12)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(x, [r["time_mean_seconds"] for r in rows], color=[f"C{i}" for i in range(len(rows))])
    axes[1, 1].set(title="Mean wall time per seed (includes validation)", ylabel="Seconds")
    axes[1, 1].set_xticks(x, [r["label"] for r in rows], rotation=12)
    for ax in axes.flat:
        ax.grid(alpha=.18, axis="y")
    fig.suptitle("DQN sequential study | 3 policy seeds | same 8 validation mazes | final test unused", fontsize=14)
    fig.text(.5, .012, "Pilot ends at 2k. Return curves use exploration; greedy evaluation does not. Small development set; provisional findings.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    return rows


def write_report(folder, conditions, decisions):
    rows = compare_plot(folder, conditions)
    write_json(folder / "comparison.json", {"conditions": rows, "decisions": decisions,
                                            "final_test_used": False})
    report = ["# 4단계 DQN 실험 결과", "", "모든 조건은 3개 학습 seed와 동일한 미로 세트를 사용했다.",
              "State/보상/알고리즘은 고정했고 최종 테스트는 평가하지 않았다.", "",
              "| 조건 | 스텝 | 탐색 감소 | 학습률 | FC | 파라미터 | 훈련 greedy | 검증 평균 ± 표준편차 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['label']} | {r['steps']} | {r['epsilon_decay_steps']} | {r['learning_rate']} | "
                      f"{r['hidden_units']} | {r['parameters']:,} | {r['training_greedy_success_mean']:.1%} | "
                      f"{r['validation_success_mean']:.1%} ± {r['validation_success_std']*100:.1f}%p |")
    report += ["", "## 선택 기록", ""] + [f"- {d}" for d in decisions]
    report += ["", "## 그래프 해석", "",
               "- reward_curve.png: x축 완료 에피소드, y축 에피소드 누적 보상. 얇은 선은 원본, 굵은 선은 최근 20개 이동평균.",
               "- reward_curve_steps.png: 같은 보상 데이터를 환경 스텝에 표시. 성공 시 일찍 종료하므로 에피소드 번호만으로 조건을 비교하면 안 된다.",
               "- 초기 20개 미만은 사용 가능한 과거 에피소드만 평균한다. 마지막 미완료 에피소드는 포함하지 않는다.",
               "- comparison.png 보상 패널의 굵은 선은 각 seed의 이동평균을 공통 환경 스텝 구간에 보간해 평균한 값이다. 얇은 선은 각 seed.",
               "- 훈련 return은 탐색을 포함한다. 충돌 감소만으로도 오를 수 있어 성공률과 함께 해석한다.",
               "- 선택 기준은 마지막 검증 성공률 평균 → seed 표준편차가 작은 쪽 → 성공 경로 효율. 완전 동률이면 기존 조건 유지.",
               "- 8개 검증 미로를 반복 사용한 개발 실험이다. 조건 선택은 잠정적이며 최종 성능 확정이 아니다.",
               "- 단계별로 하나의 학습 변수를 바꾼다. 시간/총스텝 제한과 진단 로깅 추가는 실행 관리 변경이다.",
               "- 마지막 모델 기준으로 비교한다. best.pt의 중간 최고 성공률과 혼동하지 않는다.",
               "", "## 원본 실행", ""]
    report += [f"- {label}: {run}" for label, run in conditions]
    (folder / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    import csv
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "studies/CNN-02/results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-run", type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume)
    previous_elapsed = 0.0
    previous_conditions = {}
    if args.resume:
        previous = read(folder / "protocol.json")
        if previous["status"] != "failed":
            raise ValueError("Resume is allowed only after a recorded study failure")
        write_json(folder / f"protocol_failed_{int(time.time())}.json", previous)
        previous_elapsed = previous.get("elapsed_seconds", 0.)
        previous_conditions = {r["label"]: Path(r["run"])
                               for r in read(folder / "progress.json")["completed"]}
    start = time.monotonic()
    protocol = {"created_at_utc": utc_now(), "status": "running", "seeds": [0, 1, 2],
                "max_training_steps": 240000, "max_wall_seconds": 2700,
                "max_recovery_training_steps": 20000 if args.retry_run else 0,
                "selection": "final validation mean success, lower sample SD, path efficiency; exact tie keeps reference",
                "steps": ["A: 2k -> 20k", "B: epsilon decay 1600 -> 16000 vs A",
                          "C: learning rate .001 -> .0003 vs selected A/B",
                          "D: FC 64 -> 128 vs selected A/B/C"], "final_test_used": False}
    write_json(folder / "protocol.json", protocol)
    conditions, decisions = [("Pilot 2k", PILOT)], []
    expected_manifest = read(PILOT / "mazes.json")
    # Add requested graphs to earlier runs without modifying historical metrics/checkpoints.
    for old in (PILOT, ROOT / "runs/20261104T030730Z_0be51831"):
        plot_rewards(old, read(old / "summary.json")["seeds"])
    base = from_dict(read(PILOT / "config.json")).to_dict()
    base["train"].update(steps=20000, max_total_steps=60000, max_seconds=180, hard_timeout_seconds=210)
    base["evaluation"]["diagnose_training_fit"] = True

    def execute(label, raw, reference):
        if label in previous_conditions:
            run = previous_conditions[label]
            if read(run / "config.json") != raw:
                raise ValueError("Previously completed condition config mismatch")
            conditions.append((label, run))
            print(f"REUSE CONDITION {label}: {run.name}", flush=True)
            return run
        if previous_elapsed + time.monotonic() - start + 3*raw["train"]["hard_timeout_seconds"] > protocol["max_wall_seconds"]:
            raise TimeoutError("Not enough suite wall budget to safely start another condition")
        write_json(folder / f"config_{label}.json", raw)
        print(f"START {label}: {raw['hypothesis']}", flush=True)
        retry = None
        if args.retry_run and from_dict(read(args.retry_run / "config.json")).to_dict() == from_dict(raw).to_dict():
            retry = args.retry_run
        run, complete = run_experiment(from_dict(raw), ROOT / "runs", reuse_completed_from=retry)
        if not complete:
            raise RuntimeError(f"{label} incomplete; retained {run}")
        if read(run / "mazes.json") != expected_manifest:
            raise AssertionError("Maze sets changed between conditions")
        conditions.append((label, run))
        write_json(folder / "progress.json", {"completed": [{"label": a, "run": str(b)} for a, b in conditions],
                                               "decisions": decisions, "elapsed_seconds": previous_elapsed+time.monotonic()-start})
        stats = summarize_condition(label, run)
        print(f"DONE {label}: train={stats['training_greedy_success_mean']:.1%}, "
              f"validation={stats['validation_success_mean']:.1%} +/- {stats['validation_success_std']:.1%}", flush=True)
        return run

    try:
        a = copy.deepcopy(base)
        a.update(hypothesis="Does 10x more interaction improve DQN with all learning settings fixed?",
                 change_description="A vs pilot: environment interaction steps 2000 -> 20000 only; diagnostics/budgets expanded.")
        arun = execute("A_budget", a, PILOT)
        b = copy.deepcopy(a)
        b["train"]["epsilon_decay_steps"] = 16000
        b.update(hypothesis="Does longer exploration improve DQN at the same 20k budget?",
                 change_description="B vs A: epsilon_decay_steps 1600 -> 16000 only.")
        brun = execute("B_explore", b, arun)
        selected = choose_reference(arun, brun)
        decisions.append(f"C reference selected from A/B: {selected.name}; scores A={selection_score(arun)}, B={selection_score(brun)}")
        c = read(selected / "config.json")
        c["train"]["learning_rate"] = .0003
        c.update(hypothesis="Does a lower learning rate improve the selected exploration condition?",
                 change_description=f"C vs {selected.name}: learning_rate .001 -> .0003 only.")
        crun = execute("C_lr", c, selected)
        selected = choose_reference(selected, crun)
        decisions.append(f"D reference selected after C: {selected.name}; C score={selection_score(crun)}")
        d = read(selected / "config.json")
        d["network"]["hidden_units"] = 128
        d.update(hypothesis="Does doubling the FC width improve the selected DQN condition?",
                 change_description=f"D vs {selected.name}: hidden_units 64 -> 128 only.")
        drun = execute("D_width", d, selected)
        winner = choose_reference(selected, drun)
        decisions.append(f"Provisional selected condition: {winner.name}; D score={selection_score(drun)}")
        write_report(folder, conditions, decisions)
        protocol.update(status="completed", elapsed_seconds=previous_elapsed+time.monotonic()-start,
                        completed_at_utc=utc_now(), provisional_selected_run=winner.name)
        write_json(folder / "protocol.json", protocol)
        write_json(folder / "progress.json", {"completed": [{"label": a, "run": str(b)} for a, b in conditions],
                                               "decisions": decisions, "status": "completed"})
        print(f"STUDY COMPLETE: {folder}", flush=True)
    except Exception as error:
        import traceback
        protocol.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                        elapsed_seconds=previous_elapsed+time.monotonic()-start)
        write_json(folder / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    main()
