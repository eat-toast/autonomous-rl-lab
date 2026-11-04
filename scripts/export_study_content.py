"""Create data-driven study cards after all four conditions have completed."""
import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import ImageDraw
from scripts.export_content import card, text, TEAL, INK, MUTED, PAPER, ORANGE, LINE
from rl_lab.plots import trailing_mean
from scripts.commit_review import commit_review, validate_review

NAMES = {"Pilot 2k": "기존 2천 스텝", "A_budget": "1. 학습량", "B_explore": "2. 탐색 기간",
         "C_lr": "3. 학습률", "D_width": "4. 모델 크기"}
COLORS = ("#397CB4", "#E68B46", "#4B987B")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "studies/20261104_four_steps")
    parser.add_argument("--output", type=Path, default=ROOT / "exports/20261104_dqn_four_steps")
    args = parser.parse_args()
    protocol, comparison = validate_review(args.study)
    rows = comparison["conditions"]
    actual = rows[1:]
    best = next(r for r in actual if r["run_id"] == protocol["provisional_selected_run"])
    width = actual[-1]
    reference = next(r for r in actual[:-1] if r["steps"] == width["steps"] and
                     r["learning_rate"] == width["learning_rate"] and
                     r["epsilon_decay_steps"] == width["epsilon_decay_steps"])
    if args.output.exists():
        raise FileExistsError(args.output)
    review_commit = commit_review(args.study)
    # A commit hook must not silently change the reviewed publication inputs.
    current_protocol, current_comparison = validate_review(args.study)
    if current_protocol != protocol or current_comparison != comparison:
        raise ValueError("Study changed while committing its review")
    args.output.mkdir(parents=True, exist_ok=False)

    im, d = card(1, "연구 노트 02 / 학습량 · 탐색 · 학습률 · 모델 크기")
    text(d, (64, 204), "모델을 키우기 전에,\n보상부터 그려봤다.", 66, INK, True)
    text(d, (64, 432), "10×10 미로 · 4개 조건 · 조건당 3개 seed", 33, TEAL, True)
    text(d, (64, 560), "이번에 선택된 조건의 검증 성공률", 31, MUTED)
    text(d, (64, 634), f"{best['validation_success_mean']*100:.1f}%", 158, TEAL, True)
    text(d, (72, 855), f"{NAMES[best['label']]} 조건 / seed 평균", 39, INK, True)
    d.rounded_rectangle((64, 1000, 1016, 1194), radius=20, fill=PAPER)
    text(d, (96, 1032), "같은 입력·보상·미로를 유지했습니다.\n최종 테스트 대신 개발 검증 결과입니다.", 34)
    im.save(args.output / "01_cover.png")

    im, d = card(2, f"보상 곡선 / {NAMES[best['label']]} 조건")
    text(d, (64, 204), "한 에피소드의 보상을\n모두 더하면 return.", 60, INK, True)
    text(d, (64, 405), "얇은 선: 실제 값 / 굵은 선: 최근 20개 평균", 30, MUTED)
    x, y, w, h = 125, 530, 865, 430
    series = []
    for seed in (0, 1, 2):
        with (ROOT / "runs" / best["run_id"] / f"seed_{seed}/training_episodes.csv").open(encoding="utf-8") as stream:
            series.append([float(r["return"]) for r in csv.DictReader(stream)])
    max_episode = max(map(len, series))
    for tick in (-5, -3, -1, 1):
        yy = y + (1-tick)/6*h
        d.line((x, yy, x+w, yy), fill=LINE, width=2)
        text(d, (67, yy-18), str(tick), 25, MUTED)
    for i, values in enumerate(series):
        color = COLORS[i]
        rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
        faint = tuple(int(c*.22+244*.78) for c in rgb)
        def points(vs):
            return [(x+(j+1)/max_episode*w, y+(1-v)/6*h) for j, v in enumerate(vs)]
        d.line(points(values), fill=faint, width=2)
        d.line(points(trailing_mean(values)), fill=color, width=5)
        text(d, (135+i*295, 1008), f"seed {i}", 30, color, True)
    for tick in (0, max_episode//2, max_episode):
        xx = x+tick/max_episode*w
        text(d, (xx-20, y+h+15), str(tick), 24, MUTED)
    text(d, (64, 1090), "가로: 완료 에피소드 / 세로: 누적 보상\n탐색 행동이 포함된 훈련 기록입니다.", 31, MUTED)
    im.save(args.output / "02_reward.png")

    im, d = card(3, "조건별 결과 / 마지막 모델의 동일 검증 미로 8개 평가")
    text(d, (64, 204), "보상과 함께,\n성공률도 확인했다.", 66, INK, True)
    for i, r in enumerate(actual):
        yy = 461+i*160
        text(d, (64, yy), NAMES[r["label"]], 34, TEAL, True)
        text(d, (535, yy-9), f"{r['validation_success_mean']*100:.1f}%", 47, INK, True)
        rates = list(r["seed_validation_success"].values())
        text(d, (65, yy+63), "seed별 " + " / ".join(f"{v*100:g}%" for v in rates)
             + f"    SD {r['validation_success_std']*100:.1f}%p", 27, MUTED)
        d.line((64, yy+130, 1016, yy+130), fill=LINE, width=2)
    text(d, (64, 1150), "각 평균은 3개 seed 기준 / 같은 검증 미로를 사용", 27, MUTED)
    im.save(args.output / "03_comparison.png")

    im, d = card(4, "모델 크기 실험 / 동일 학습 설정에서 FC 너비만 변경")
    text(d, (64, 204), "파라미터는 약 2배.\n성공률은 어땠을까?", 64, INK, True)
    for xx, r, title in ((64, reference, "FC 64"), (560, width, "FC 128")):
        d.rounded_rectangle((xx, 485, xx+455, 914), radius=20, fill=PAPER)
        text(d, (xx+28, 515), title, 34, TEAL, True)
        text(d, (xx+28, 597), f"{r['parameters']:,}", 54, INK, True)
        text(d, (xx+28, 679), "parameters", 27, MUTED)
        text(d, (xx+28, 759), f"{r['validation_success_mean']*100:.1f}%", 66, ORANGE, True)
        text(d, (xx+28, 850), "평균 검증 성공률", 27, MUTED)
    text(d, (64, 1006), f"각 20,000 환경 스텝 / 3개 학습 seed\n탐색 감소 {width['epsilon_decay_steps']:,}스텝 · 학습률 {width['learning_rate']}", 31, MUTED)
    text(d, (64, 1145), "모델 크기 증가의 효과는 이번 조건 안에서 해석합니다.", 28, INK, True)
    im.save(args.output / "04_size.png")

    im, d = card(5, "해석 / 익숙한 미로와 새로운 미로를 구분하기")
    text(d, (64, 204), "학습 곡선 하나로는\n설명되지 않았다.", 65, INK, True)
    text(d, (64, 458), f"선택 조건: {NAMES[best['label']]}", 34, TEAL, True)
    text(d, (64, 542), f"훈련 미로  {best['training_greedy_success_mean']*100:.1f}%\n검증 미로  {best['validation_success_mean']*100:.1f}%", 51, INK, True, 28)
    d.line((64, 755, 1016, 755), fill=LINE, width=2)
    text(d, (64, 802), "두 값 모두 무작위 행동 없이 평가한 평균입니다.\n훈련 중 보상과는 다른 지표입니다.", 32, MUTED)
    text(d, (64, 956), "다음 가설", 30, TEAL, True)
    next_hypothesis = ("훈련 미로를 더 다양하게 만들면\n새로운 미로에서도 나아질까?"
                       if best["training_greedy_success_mean"] > best["validation_success_mean"]+.20
                       else "더 많은 seed와 검증 미로에서도\n이번 차이가 유지될까?")
    text(d, (64, 1016), next_hypothesis, 38, INK, True)
    text(d, (64, 1167), "개발 검증 결과 / 최종 테스트 미사용 / 다음 가설은 미실행", 25, MUTED)
    im.save(args.output / "05_interpretation.png")

    lines = ["모델을 키우기 전에, 보상부터 그려봤습니다.", "",
             "10×10 장애물 미로에서 같은 전체 지도 입력과 보상을 유지하고 DQN을 비교했습니다.",
             "학습 seed는 0·1·2, 검증은 학습에 쓰지 않은 동일 미로 8개입니다.", "",
             "1) 학습량 2,000 → 20,000스텝", "2) 탐색 감소 기간 1,600 → 16,000스텝",
             "3) 앞선 선택 조건에서 학습률 0.001 → 0.0003", "4) 다시 선택한 조건에서 FC 너비 64 → 128", "",
             "조건별 마지막 모델의 검증 성공률 평균 ± seed 표준편차:"]
    lines += [f"{NAMES[r['label']]}: {r['validation_success_mean']*100:.1f}% ± {r['validation_success_std']*100:.1f}%p"
              for r in actual]
    lines += ["", "보상 그래프는 에피소드의 실제 보상을 더한 return과 최근 20개 이동평균입니다.",
              "탐색을 포함한 훈련 보상은 충돌이 줄어도 좋아질 수 있으므로, 목표 도달 성공률과 함께 확인했습니다.",
              "에피소드 길이가 다르기 때문에 실험 간 비교는 같은 환경 스텝에서도 확인했습니다.", "",
              "3·4단계는 바로 직전 조건이 아니라, 사전에 정한 검증 기준으로 선택한 조건에서 한 변수만 바꿨습니다.",
              "같은 8개 미로를 반복 평가한 개발 실험입니다. 더 넓은 미로 분포의 성능으로 일반화하지 않습니다.",
              "최종 테스트 세트는 사용하지 않았으며, 다음 가설은 아직 실행하지 않았습니다.", "",
              "#강화학습 #DQN #ReinforcementLearning #학습곡선 #인공지능 #AI실험 #연구기록"]
    (args.output / "caption.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for name in ("comparison.png", "comparison.csv", "comparison.json", "report.md", "protocol.json",
                 "review.json", "verified_facts.json"):
        (args.output / name).write_bytes((args.study / name).read_bytes())
    report = (args.output / "report.md").read_text(encoding="utf-8")
    for row in rows:
        report = report.replace(f"{row['validation_success_mean']:.1%} ± {row['validation_success_std']:.1%} |",
                                f"{row['validation_success_mean']:.1%} ± {row['validation_success_std']*100:.1f}%p |")
    (args.output / "report.md").write_text(report, encoding="utf-8")
    graphs = args.output / "graphs"
    graphs.mkdir()
    for row in rows:
        for name in ("reward_curve.png", "reward_curve_steps.png", "learning_curve.png"):
            source = ROOT / "runs" / row["run_id"] / name
            (graphs / f"{row['label'].replace(' ', '_')}_{name}").write_bytes(source.read_bytes())
    (args.output / "provenance.json").write_text(json.dumps({"study": str(args.study),
        "selected_run": best["run_id"], "width_reference_run": reference["run_id"],
        "original_run_ids": [r["run_id"] for r in rows], "review_commit": review_commit,
        "no_instagram_upload": True}, indent=2), encoding="utf-8")
    archive = args.output.parent / "Instagram_DQN_four_steps_20261104.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in args.output.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(args.output).as_posix())
    print(f"Content: {args.output.resolve()}\nArchive: {archive.resolve()}")


if __name__ == "__main__":
    main()
