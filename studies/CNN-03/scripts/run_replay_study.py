"""First bounded batch from experiment_candidates_20261104.md; no final-test use."""
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[3]))
from rl_lab.paths import resolve_legacy_path, study_runs, run_path
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import math

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rl_lab.config import from_dict
from rl_lab.runner import run_experiment
from rl_lab.storage import utc_now, write_json
from scripts.study_modules import load as _load_study
_study = _load_study("CNN-02", "run_study")
compare_plot = _study.compare_plot
selection_score = _study.selection_score
summarize_condition = _study.summarize_condition

BASE = ROOT / "studies/CNN-02/runs/20261104T042433Z_2393292b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(run, label, expected_manifest):
    assert read(run / "mazes.json") == expected_manifest
    summary = read(run / "summary.json")
    assert [s["seed"] for s in summary["seeds"]] == [0, 1, 2]
    paused = label in ("P_pause", "F_clear")
    expected_updates = 4921 if paused else 4985
    for s in summary["seeds"]:
        assert s["status"] == "completed" and s["environment_steps"] == 20000
        assert s["parameter_count"] == 210212
        assert s["optimizer_updates"] == expected_updates
        assert s["paused_updates"] == (64 if paused else 0)
        assert s["paused_env_steps"] == (256 if paused else 0)
        assert s["final_replay"]["sampling_totals"]["transitions"] == expected_updates*32
        assert s["final_replay"]["oldest_step"] == (10001 if label == "R_buffer10k" else 18001)
        assert s["final_replay"]["newest_step"] == 20000
        seed_dir = run / f"seed_{s['seed']}"
        if paused:
            event, refill, resumed = [read(seed_dir / n) for n in (
                "replay_event.json", "replay_after_refill.json", "replay_first_resume_update.json")]
            assert event["step"] == 10000 and event["optimizer_updates"] == 2485
            assert event["before"] == event["after"]
            assert refill["step"] == 10256 and refill["optimizer_updates"] == 2485
            for key in ("online", "target", "optimizer", "sampler_rng"):
                assert event["after"][key] == refill["state_hashes"][key]
            assert resumed["step"] == 10260 and resumed["optimizer_updates"] == 2486
            if label == "F_clear":
                assert event["replay_after"]["count"] == 0
                assert refill["count"] == 256 and refill["oldest_step"] == 10001
                assert resumed["count"] == 260
    assert not list(run.glob("seed_*/evaluation_test_*.json"))
    return summary


def verify_pair(pause_run, clear_run):
    result = []
    for seed in (0, 1, 2):
        p = pause_run / f"seed_{seed}"
        f = clear_run / f"seed_{seed}"
        pe, fe = [read(x / "replay_event.json") for x in (p, f)]
        pr, fr = [read(x / "replay_after_refill.json") for x in (p, f)]
        assert pe["before"] == fe["before"]
        assert pe["trajectory_sha256"] == fe["trajectory_sha256"]
        assert pr["state_hashes"] == fr["state_hashes"]
        assert pr["trajectory_sha256"] == fr["trajectory_sha256"]
        # The operational-only additions must preserve the B trajectory before the intervention.
        with (BASE / f"seed_{seed}" / "training_episodes.csv").open(encoding="utf-8", newline="") as stream:
            old = [r for r in csv.DictReader(stream) if int(r["end_step"]) <= 10000]
        with (p / "training_episodes.csv").open(encoding="utf-8", newline="") as stream:
            new = [r for r in csv.DictReader(stream) if int(r["end_step"]) <= 10000]
        same_device = from_dict(read(pause_run / "config.json")).train.device == from_dict(read(BASE / "config.json")).train.device
        if same_device:
            assert old == new
        result.append({"seed": seed, "same_model_and_control_at_10000": True,
                       "identical_trajectory_through_10256": True,
                       "baseline_training_episode_prefix_matches": True if same_device else None,
                       "baseline_prefix_scope": "verified" if same_device else "not required: CPU reference vs CUDA new condition",
                       "prefix_complete_episodes": len(old)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "studies/CNN-03/results")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = read(BASE / "mazes.json")
    base = from_dict(read(BASE / "config.json")).to_dict()
    base["train"].update(max_seconds=240, hard_timeout_seconds=270, max_total_steps=60000)
    base["train"]["device"] = args.device
    workers = 1 if args.device == "cuda" else 2
    cross_device_reference = args.device != from_dict(read(BASE / "config.json")).train.device
    single = copy.deepcopy(base)
    single["evaluation"]["train_mazes"] = 1
    single.update(hypothesis="Can the unchanged DQN reliably fit one fixed training maze across seeds?",
                  change_description="S: replace training pool with original B train[0] only; validation/test unchanged.")
    larger = copy.deepcopy(base)
    larger["train"]["replay_capacity"] = 10000
    larger.update(hypothesis="Does retaining 10k rather than 2k uniformly sampled transitions improve learning?",
                  change_description="R vs B: FIFO replay capacity 2000 -> 10000 only.")
    pause = copy.deepcopy(base)
    pause["train"].update(replay_event_step=10000, replay_pause_steps=256)
    pause.update(hypothesis="Matched optimizer-pause control for the replay clearing experiment.",
                 change_description="P vs B: skip optimizer updates for environment steps 10001..10256; retain replay.")
    clear = copy.deepcopy(pause)
    clear["train"]["replay_clear"] = True
    clear.update(hypothesis="Does clearing experience at t=10k improve learning compared with the matched pause control?",
                 change_description="F vs P: clear replay once after step10000 update; no other state reset.")
    plan = [("S_single", single), ("R_buffer10k", larger), ("P_pause", pause), ("F_clear", clear)]
    protocol = {"status": "running", "created_at_utc": utc_now(), "reference_run": BASE.name,
                "reference_manifest_sha256": digest(BASE / "mazes.json"),
                "source_protocol": "studies/CNN-03/replay_study_protocol_20261104.md",
                "conditions": [label for label, _ in plan], "seeds": [0, 1, 2],
                "planned_training_steps": 240000, "maximum_including_recovery": 260000,
                "max_wall_seconds": 3600, "workers": workers, "threads_per_worker": 1,
                "device": args.device, "cross_device_reference": cross_device_reference,
                "reference_scope": "observed CPU reference, not a matched-device causal control" if cross_device_reference else "matched CPU reference",
                "comparison_pairs": {"capacity": ["R_buffer10k", "B_reference"],
                                     "pause": ["P_pause", "B_reference"], "clear": ["F_clear", "P_pause"]},
                "single_map_excluded_from_generalization_selection": True, "final_test_used": False}
    write_json(folder / "protocol.json", protocol)
    conditions, ledger, charged_steps, recovery_used = [("B_reference", BASE)], [], 0, False
    try:
        for label, raw in plan:
            config = from_dict(raw)
            manifest = copy.deepcopy(original)
            if label == "S_single":
                manifest["splits"]["train"] = [copy.deepcopy(original["splits"]["train"][0])]
            assert manifest["splits"]["validation"] == original["splits"]["validation"]
            assert manifest["splits"]["test"] == original["splits"]["test"]
            initial_reserved = sum(a["reserved_training_steps"] for a in ledger if a["kind"] == "initial")
            condition_wall_reserve = math.ceil(3/workers)*270+30
            if initial_reserved+60000 > 240000 or charged_steps+60000 > 260000 or time.monotonic()-started+condition_wall_reserve > 3600:
                raise TimeoutError("Insufficient remaining budget for another full condition")
            write_json(folder / f"config_{label}.json", raw)
            charged_steps += 60000
            attempt = {"condition": label, "kind": "initial", "reserved_training_steps": 60000,
                       "started_at_utc": utc_now(), "status": "running"}
            ledger.append(attempt)
            write_json(folder / "ledger.json", {"reserved_steps": charged_steps, "attempts": ledger})
            print(f"START {label}", flush=True)
            run, complete = run_experiment(config, ROOT / "studies/CNN-03/runs", maze_manifest=manifest, workers=workers)
            attempt.update(run_id=run.name, status="completed" if complete else "incomplete")
            if not complete:
                failed = [s for s in read(run / "summary.json")["seeds"] if s["status"] != "completed"]
                if recovery_used or len(failed) != 1 or charged_steps+20000 > 260000 or time.monotonic()-started+270 > 3600:
                    raise RuntimeError(f"Incomplete condition retained without further recovery: {run}")
                recovery_used = True
                charged_steps += 20000
                repair = {"condition": label, "kind": "one-seed recovery", "source_run_id": run.name,
                          "reserved_training_steps": 20000, "seed": failed[0]["seed"], "status": "running"}
                ledger.append(repair)
                write_json(folder / "ledger.json", {"reserved_steps": charged_steps, "attempts": ledger})
                run, complete = run_experiment(config, ROOT / "studies/CNN-03/runs", reuse_completed_from=run,
                                                maze_manifest=manifest, workers=workers)
                repair.update(run_id=run.name, status="completed" if complete else "incomplete")
                if not complete:
                    raise RuntimeError(f"Recovery incomplete; retained {run}")
            summaries = verify_run(run, label, manifest)
            conditions.append((label, run))
            write_json(folder / "progress.json", {"completed": [{"label": a, "run": str(b)} for a, b in conditions],
                                                   "elapsed_seconds": time.monotonic()-started})
            write_json(folder / "ledger.json", {"reserved_steps": charged_steps, "attempts": ledger})
            stats = summarize_condition(label, run)
            print(f"DONE {label}: train={stats['training_greedy_success_mean']:.1%}, validation="
                  f"{stats['validation_success_mean']:.1%} +/- {stats['validation_success_std']:.1%}", flush=True)
        mapping = dict(conditions)
        paired_checks = verify_pair(mapping["P_pause"], mapping["F_clear"])
        eligible = [(label, run) for label, run in conditions if label != "S_single"
                    and (label != "B_reference" or not cross_device_reference)]
        winner = eligible[0]
        for candidate in eligible[1:]:
            if selection_score(candidate[1]) > selection_score(winner[1]):
                winner = candidate
        rows = compare_plot(folder, conditions)
        for row, (label, run) in zip(rows, conditions):
            raw = read(run / "config.json")
            summaries = read(run / "summary.json")["seeds"]
            row.update(replay_capacity=raw["train"]["replay_capacity"],
                       device=from_dict(raw).train.device,
                       train_maze_count=raw["evaluation"]["train_mazes"],
                       diagnostic_only=label == "S_single",
                       optimizer_updates_by_seed=[s.get("optimizer_updates") for s in summaries],
                       evaluation_steps_by_seed=[s.get("evaluation_environment_steps") for s in summaries])
        write_json(folder / "comparison.json", {"conditions": rows, "paired_checks": paired_checks,
                                                "selected_condition": winner[0], "final_test_used": False})
        with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = ["# Replay 첫 묶음 결과", "", "S는 단일 지도 진단이며 일반화 순위에서 제외한다.",
                  "F/P는 같은 장치의 clear 직접 대조다.",
                  "기존 B가 CPU이고 새 조건이 CUDA이면 R/B·P/B 차이에는 장치도 포함된다. 단일 기법의 효과로 단정하지 않는다.", "",
                  "| 조건 | 훈련 지도 수 | replay | 훈련 greedy 평균 | 검증 평균 ± 표본 SD |",
                  "|---|---:|---:|---:|---:|"]
        for row in rows:
            report.append(f"| {row['label']} | {row['train_maze_count']} | {row['replay_capacity']} | "
                          f"{row['training_greedy_success_mean']:.1%} | {row['validation_success_mean']:.1%} ± "
                          f"{row['validation_success_std']*100:.1f}%p |")
        report += ["", f"잠정 일반화 선택: {winner[0]} ({winner[1].name}).",
                   "같은 8개 개발 검증 미로·3개 학습 seed의 결과이며 최종 테스트는 사용하지 않았다.",
                   "P/F는 모델·Adam·RNG를 유지했고 10256까지 궤적이 동일했다. 이후 차이는 학습에 쓰는 버퍼 내용의 차이다.",
                   "B의 optimizer 갱신4985회는 구 코드에서 유도한 값이며 새 조건은 실제 카운터로 확인한다.",
                   "벽시계 시간은 병렬 실행과 데스크톱 부하의 영향을 받으므로 연산속도 순위로 해석하지 않는다.",
                   "훈련 return은 탐색을 포함하며 훈련/검증 greedy 성공률과 구분한다.", "", "## 원본 실행"]
        report += [f"- {label}: {run}" for label, run in conditions]
        (folder / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        actual, unknown_attempts = 0, []
        for attempt in ledger:
            for seed_row in read(run_path(attempt["run_id"], "CNN-03") / "summary.json")["seeds"]:
                if attempt["kind"] != "initial" and seed_row["seed"] != attempt["seed"]:
                    continue
                if "environment_steps" not in seed_row:
                    unknown_attempts.append({"run_id": attempt["run_id"], "seed": seed_row["seed"]})
                actual += seed_row.get("environment_steps", 0)
        protocol.update(status="completed", finished_at_utc=utc_now(), elapsed_seconds=time.monotonic()-started,
                        reserved_training_steps=charged_steps,
                        actual_training_steps=actual if not unknown_attempts else None,
                        actual_training_steps_lower_bound=actual, unknown_step_attempts=unknown_attempts,
                        selected_condition=winner[0], selected_run_id=winner[1].name)
        write_json(folder / "protocol.json", protocol)
        print(f"STUDY COMPLETE: {folder}", flush=True)
    except Exception as error:
        import traceback
        protocol.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                        elapsed_seconds=time.monotonic()-started, reserved_training_steps=charged_steps)
        write_json(folder / "ledger.json", {"reserved_steps": charged_steps, "attempts": ledger})
        write_json(folder / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    main()
