"""Data-driven Instagram assets. No simulated/fabricated agent trajectories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".tools"))

BG = "#F4F1E9"
INK = "#182C2A"
TEAL = "#078C80"
ORANGE = "#E8724D"
MUTED = "#647570"
PAPER = "#FFFFFF"
LINE = "#D9DFD7"


def font(size, bold=False):
    windows = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    if not windows.exists():
        raise FileNotFoundError("Korean Malgun font not found; provide a Korean font before exporting")
    return ImageFont.truetype(str(windows), size)


def text(draw, xy, value, size=32, fill=INK, bold=False, spacing=15):
    draw.multiline_text(xy, str(value), font=font(size, bold), fill=fill, spacing=spacing)


def card(number, kicker):
    image = Image.new("RGB", (1080, 1350), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((64, 64, 74, 96), radius=5, fill=TEAL)
    text(draw, (92, 61), "RL LAB / RESEARCH NOTES", 24, TEAL, True)
    text(draw, (64, 121), kicker, 27, MUTED)
    draw.line((64, 1264, 1016, 1264), fill=LINE, width=2)
    text(draw, (64, 1283), "DQN · 10×10 · 예비 실험 / 2026.11.04", 22, MUTED)
    text(draw, (958, 1281), f"{number:02}", 25, TEAL, True)
    return image, draw


def maze_panel(image, raw, episode, xy, side, action_index=None, show_astar=True):
    draw = ImageDraw.Draw(image)
    grid, n = raw["grid"], len(raw["grid"])
    x, y = xy
    cell = side / n
    draw.rounded_rectangle((x-10, y-10, x+side+10, y+side+10), radius=16, fill=PAPER)
    for r, row in enumerate(grid):
        for c, wall in enumerate(row):
            box = (x+c*cell+1, y+r*cell+1, x+(c+1)*cell-1, y+(r+1)*cell-1)
            draw.rounded_rectangle(box, radius=max(2, int(cell*.07)), fill=INK if wall else "#EDF0E9")

    def center(position):
        return x+(position[1]+.5)*cell, y+(position[0]+.5)*cell

    if show_astar:
        points = [center(p) for p in raw["shortest_path"]]
        draw.line(points, fill=TEAL, width=max(4, int(cell*.08)), joint="curve")
    path = episode["path"]
    index = len(path)-1 if action_index is None else action_index
    visible_path = path[:index+1]
    if len(visible_path) > 1:
        draw.line([center(p) for p in visible_path], fill=ORANGE,
                  width=max(5, int(cell*.12)), joint="curve")
    for position, color in ((path[0], "#3B80C7"), ((n-1, n-1), "#E5B94C")):
        cx, cy = center(position)
        radius = cell*.18
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color)
    cx, cy = center(path[index])
    radius = cell*.26
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=ORANGE, outline=PAPER, width=3)


def load_and_audit(run):
    from rl_lab.config import from_dict
    from rl_lab.dqn import QNetwork, seed_everything
    from rl_lab.evaluation import evaluate
    import torch

    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Only completed experiments may be exported")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "mazes.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    if len(summary["seeds"]) != len(config["train"]["seeds"]):
        raise ValueError("Missing training seed results")
    records, checks, training_successes, training_episodes = [], [], 0, 0
    for seed in config["train"]["seeds"]:
        folder = run / f"seed_{seed}"
        checkpoint = folder / "latest.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seed_everything(seed)
        model = QNetwork(config["env"]["size"])
        model.load_state_dict(payload["model_state"])
        restored = evaluate(model, from_dict(payload["config"]).env, manifest["splits"]["validation"])
        recorded = json.loads((folder / "latest_validation.json").read_text(encoding="utf-8"))
        if json.dumps(restored, sort_keys=True) != json.dumps(recorded, sort_keys=True):
            raise AssertionError(f"Checkpoint/trajectory mismatch for seed {seed}")
        if payload["step"] != config["train"]["steps"]:
            raise AssertionError("Checkpoint is not at the declared training budget")
        for episode in recorded["episodes"]:
            raw = next(m for m in manifest["splits"]["validation"] if m["hash"] == episode["maze_hash"])
            assert len(episode["path"]) == episode["steps"] + 1
            assert all(not raw["grid"][r][c] for r, c in episode["path"])
            distances = [abs(a[0]-b[0])+abs(a[1]-b[1])
                         for a, b in zip(episode["path"], episode["path"][1:])]
            assert all(d in (0, 1) for d in distances)
            assert distances.count(0) == episode["collisions"]
        with (folder / "training_episodes.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                training_episodes += 1
                training_successes += row["success"] == "True"
        records.append(recorded)
        checks.append({"seed": seed, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "checkpoint_step": payload["step"], "restored_evaluation_matches": True,
                       "trajectory_geometry_valid": True, "success_rate": recorded["metrics"]["success_rate"]})
    audit = {"run_id": run.name, "training_git_commit": metadata["git_commit"],
             "seeds": checks, "training_completed_episodes": training_episodes,
             "training_successful_episodes": training_successes,
             "validation_mazes_shared_across_seeds": len(manifest["splits"]["validation"]),
             "final_test_evaluated": False,
             "video_selection": "first configured seed, latest checkpoint, first validation maze; not cherry-picked",
             "video_is": "Replay of recorded greedy evaluation actions, not live training or interpolated learning"}
    return config, summary, manifest, records, audit


def export_cards(output, config, summary, manifest, records, audit):
    # This editorial package describes this exact failed pilot, never guesses about new runs.
    if any(r["metrics"]["success_rate"] != 0 for r in records):
        raise ValueError("This editorial template is for the all-zero pilot; revise copy for different results")
    if config["env"]["size"] != 10 or config["train"]["steps"] != 2000 or len(records) != 3:
        raise ValueError("Revise the editorial template for other experiment conditions")
    raw, episode = manifest["splits"]["validation"][0], records[0]["episodes"][0]
    assert raw["hash"] == episode["maze_hash"]
    total_evaluations = sum(r["metrics"]["episodes"] for r in records)
    nval = len(manifest["splits"]["validation"])

    im, d = card(1, "01 / 전체 지도를 보는 AI의 첫 미로 실험")
    text(d, (64, 204), "지도는 보인다.\n길 찾기는 아직이다.", 68, INK, True, 18)
    text(d, (68, 406), "10×10 장애물 미로 × CNN DQN", 34, TEAL, True)
    maze_panel(im, raw, episode, (290, 535), 500)
    text(d, (157, 1090), "초록: A* 최단 경로    주황: 실제 평가 이동", 29, MUTED)
    text(d, (206, 1154), "3개 학습 seed로 확인한 예비 결과", 32, INK, True)
    im.save(output / "01_cover.png")

    im, d = card(2, "02 / 이번에는 무엇을 학습시켰나")
    text(d, (64, 204), "문제는 단순하게,\n조건은 같게.", 67, INK, True)
    rows = [("관측", "벽 · 현재 위치 · 목표의 전체 지도"),
            ("행동", "상 · 하 · 좌 · 우, 총 4가지"),
            ("보상", "도착 +1 / 이동 -0.01 / 충돌 -0.05"),
            ("학습", "독립 seed 3개 / 각 2,000 환경 스텝"),
            ("검증", f"학습에 쓰지 않은 동일 미로 {nval}개")]
    for i, (label, value) in enumerate(rows):
        y = 459 + i*120
        text(d, (65, y), label, 30, TEAL, True)
        text(d, (191, y), value, 30, INK)
        d.line((64, y+77, 1016, y+77), fill=LINE, width=2)
    text(d, (64, 1120), "훈련 16개 / 검증 8개 / 최종 테스트 16개 분리\n최종 테스트는 이번에 평가하지 않았습니다.", 29, MUTED, spacing=16)
    im.save(output / "02_conditions.png")

    im, d = card(3, "03 / 최종 체크포인트의 개발 검증 결과")
    text(d, (64, 204), "세 모델 모두\n아직 목표에 닿지 못했다.", 62, INK, True)
    text(d, (64, 418), "0%", 185, TEAL, True)
    text(d, (73, 668), f"학습 seed별 검증 성공률: 0/{nval}", 38, INK, True)
    for i, seed in enumerate(config["train"]["seeds"]):
        x = 64 + i*325
        d.rounded_rectangle((x, 775, x+300, 995), radius=20, fill=PAPER)
        text(d, (x+28, 800), f"SEED {seed}", 26, MUTED, True)
        text(d, (x+28, 855), f"0/{nval}", 67, INK, True)
    text(d, (64, 1043), f"성공률 평균 0% / 표본 표준편차 0%p", 33, INK, True)
    text(d, (64, 1120), f"같은 {nval}개 미로 × 3개 모델 = {total_evaluations}회 평가\n{total_evaluations}개의 서로 다른 미로를 평가한 것은 아닙니다.", 27, MUTED)
    im.save(output / "03_result.png")

    im, d = card(4, "04 / 실제 평가 기록을 들여다보면")
    text(d, (64, 204), "경로는 존재한다.\n정책이 찾지 못했을 뿐.", 62, INK, True)
    maze_panel(im, raw, episode, (274, 450), 530)
    text(d, (93, 1030), f"A* 최단 {raw['shortest_length']}칸", 40, TEAL, True)
    text(d, (570, 1030), f"DQN {episode['steps']}행동 · 실패", 36, ORANGE, True)
    text(d, (64, 1116), f"seed 0 · 검증 미로 1번 · 충돌 {episode['collisions']}회\n실패 에피소드의 경로 효율은 계산하지 않습니다.", 29, MUTED)
    im.save(output / "04_path.png")

    im, d = card(5, "05 / 이번 결과에서 말할 수 있는 것")
    text(d, (64, 204), "실패의 원인은\n아직 모른다.", 73, INK, True)
    text(d, (64, 452), "확인한 것", 29, TEAL, True)
    text(d, (64, 511), "이 설정 · 이 학습량 · 이 검증 세트에서\n세 모델 모두 목표에 도달하지 못했다.", 38)
    d.line((64, 672, 1016, 672), fill=LINE, width=2)
    text(d, (64, 721), "다음에 확인할 것 — 아직 실행 전", 29, TEAL, True)
    text(d, (64, 783), "학습 예산만\n2,000 → 20,000 스텝", 49, INK, True)
    text(d, (64, 967), "같은 입력 · 보상 · 탐색 스케줄 · 검증 미로\n학습량 부족이라는 가설부터 검증한다.", 32, MUTED)
    d.rounded_rectangle((64, 1103, 1016, 1208), radius=18, fill=INK)
    text(d, (91, 1124), "예비 실험입니다. DQN의 한계로 일반화하지 않습니다.", 29, PAPER, True)
    im.save(output / "05_next.png")
    return raw, episode


def export_video(output, config, raw, episode, audit):
    import imageio_ffmpeg
    fps, duration = 20, 16
    writer = imageio_ffmpeg.write_frames(str(output / "reel.mp4"), (1080, 1920), fps=fps,
                                         codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                         macro_block_size=2, ffmpeg_log_level="error",
                                         output_params=["-movflags", "+faststart", "-crf", "20"])
    writer.send(None)
    # Render exactly the recorded action sequence, including stationary collisions.
    # First 2 seconds hold the start, 10 seconds replay, last 4 seconds hold the end.
    try:
        for frame_index in range(fps*duration):
            elapsed = frame_index / fps
            progress = min(1., max(0., (elapsed-2)/10))
            index = min(episode["steps"], int(progress*episode["steps"]))
            im = Image.new("RGB", (1080, 1920), BG)
            d = ImageDraw.Draw(im)
            text(d, (84, 166), "RL LAB  /  DQN PILOT", 28, TEAL, True)
            text(d, (84, 235), "지도를 다 보여주면\n길을 찾을 수 있을까?", 62, INK, True)
            text(d, (84, 425), "10×10 · CNN DQN · seed 0 · 학습 2,000스텝", 29, MUTED)
            maze_panel(im, raw, episode, (108, 548), 840, index, show_astar=elapsed >= 12)
            collision_count = sum(a == b for a, b in zip(episode["path"][:index], episode["path"][1:index+1]))
            if 2 <= elapsed < 12 and index > 0 and episode["path"][index] == episode["path"][index-1]:
                row, col = episode["path"][index]
                cx, cy = 108+(col+.5)*84, 548+(row+.5)*84
                radius = 25 + (frame_index % 10)*1.4
                d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=ORANGE, width=3)
            d.rounded_rectangle((84, 1445, 948, 1543), radius=16, fill=INK)
            text(d, (112, 1464), f"평가 행동 {index:03d}/{episode['steps']}  |  A* 최단 {raw['shortest_length']}칸", 35, PAPER, True)
            if elapsed < 2:
                headline = "학습 후 평가 이동을 재생합니다."
            elif elapsed < 12:
                headline = f"충돌 {collision_count}회 · 제자리" if collision_count else "주황 점: AI의 실제 위치"
            else:
                headline = "100행동 후 목표 미도달."
            text(d, (84, 1583), headline, 39, ORANGE, True)
            text(d, (84, 1663), "검증 성공률: 세 seed 모두 0/8 (0%)\n예비 결과 · 실패 원인은 아직 검증 전", 29, MUTED)
            writer.send(im.tobytes())
            if frame_index in (0, fps*8, fps*14):
                im.save(output / f"reel_frame_{frame_index//fps:02d}s.png")
    finally:
        writer.close()
    reader = imageio_ffmpeg.read_frames(str(output / "reel.mp4"))
    metadata = next(reader)
    frame_count = sum(1 for _ in reader)
    if metadata["size"] != (1080, 1920) or frame_count != fps*duration:
        raise AssertionError("Encoded video dimensions or duration mismatch")
    audit["video"] = {"width": 1080, "height": 1920, "fps": fps,
                      "frames_decoded": frame_count, "duration_seconds": duration,
                      "audio": False, "format": "H.264 MP4 / yuv420p"}


def write_copy(output, run, config, audit, episode):
    caption = f"""지도를 다 보여주면, AI는 길을 찾을 수 있을까요?

이번에는 코인 없이, 장애물 미로의 시작점에서 목표까지 이동하는 CNN DQN을 새로 만들었습니다.
벽·현재 위치·목표의 전체 지도를 입력하고, 상하좌우 네 방향으로 움직입니다.

10×10 미로에서 학습 seed 0·1·2를 각각 2,000 환경 스텝씩 학습했습니다.
세 모델을 학습에 사용하지 않은 동일한 8개 검증 미로에서 greedy 정책으로 평가했습니다.

결과는 세 모델 모두 0/8. 검증 성공률 평균 0%, 표본 표준편차 0%p였습니다.
같은 미로 8개를 세 모델로 평가한 총 24회이며, 서로 다른 미로 24개는 아닙니다.
각 평가는 최대 100행동으로 제한했고, 도달하지 못하면 실패로 기록했습니다.

영상은 seed 0 모델의 첫 번째 검증 미로에서 저장한 실제 평가 이동입니다.
A*로 경로가 존재하는지 먼저 확인했습니다. 이 미로의 최단 경로는 {episode['shortest_length']}칸입니다.
학습 중에 길을 배우는 타임랩스가 아니라, 학습 후 평가 경로의 재생입니다.

아직은 짧은 예비 실험입니다. 이 결과만으로 DQN의 한계를 말할 수는 없습니다.
학습량이 부족했는지, 탐색이나 다른 설정이 원인인지는 아직 구분하지 못했습니다.
다음에는 입력·보상·탐색 스케줄·검증 미로를 유지하고, 학습 예산만 20,000스텝으로 늘려 비교할 예정입니다. 아직 실행하지 않았습니다.

최종 테스트 세트는 이번 실험에서 평가하지 않았습니다.

#강화학습 #DQN #ReinforcementLearning #인공지능 #미로찾기 #AI실험 #연구기록
"""
    (output / "caption.txt").write_text(caption, encoding="utf-8")
    (output / "posting_notes.md").write_text(f"""# 게시물 사용 안내

- 카드뉴스: `01_cover.png`부터 `05_next.png`까지 순서대로. 각 1080×1350.
- 릴스: `reel.mp4`, 1080×1920, 16초, 무음. 실제 기록된 행동을 압축 재생.
- 릴스 프리뷰 PNG는 확인용이며 카드뉴스 묶음에는 포함하지 않는다.
- 캡션: `caption.txt` 사용. 인스타 업로드는 실행하지 않았다.
- 원본 실험: `{run.name}` / 학습 코드 커밋 `{audit['training_git_commit']}`.
- 근거와 재검증: `audit.json`, `source_summary.json`.
- 영상 선택: seed 0의 latest checkpoint, 검증 세트 1번. 결과를 보고 고른 성공 장면이 아니다.
- 초록 A*는 평가가 끝난 마지막 4초에 표시된다. 실제 에이전트에는 A* 행동을 주지 않았다.
- AI 정책 이동 중 벽/경계에 부딪히면 제자리에 머문다. 같은 화면이 유지되는 것은 기록상 실제 정지다.
- 이번 결과는 짧은 DQN 예비 실험. 알고리즘 간 비교, 수렴 결과, 일반적 성능 결론으로 소개하지 않는다.
- '자율 연구 AI가 스스로 개선했다', 'DQN은 미로를 못 푼다', '학습 성공 영상'이라는 표현은 근거가 없다.
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, summary, manifest, records, audit = load_and_audit(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw, episode = export_cards(args.output, config, summary, manifest, records, audit)
    export_video(args.output, config, raw, episode, audit)
    write_copy(args.output, args.run, config, audit, episode)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created: {args.output.resolve()}")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
