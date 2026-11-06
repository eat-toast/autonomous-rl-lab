"""Episode 1: verified map/rules/model introduction, without performance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import ImageDraw
from scripts.export_content import card, text, TEAL, INK, MUTED, PAPER, ORANGE, LINE, BG

BLUE = "#3C80B4"
GOLD = "#DCAE45"


def intro_card(number, kicker):
    im, d = card(number, kicker)
    d.rectangle((0, 1270, 1080, 1350), fill=BG)
    text(d, (64, 1283), "MAZE LAB / SERIES 01 · INTRODUCTION", 22, MUTED)
    text(d, (958, 1281), f"{number:02}", 25, TEAL, True)
    return im, d


def draw_map(image, grid, x, y, side, channel=None):
    d = ImageDraw.Draw(image)
    n, cell = len(grid), side/len(grid)
    d.rounded_rectangle((x-8,y-8,x+side+8,y+side+8), radius=16, fill=PAPER)
    for r in range(n):
        for c in range(n):
            if channel is None or channel == "walls":
                fill = INK if grid[r][c] else "#E8ECE4"
            else:
                fill = "#E8ECE4"
                if channel == "agent" and (r,c) == (0,0):
                    fill = ORANGE
                if channel == "goal" and (r,c) == (n-1,n-1):
                    fill = GOLD
            d.rounded_rectangle((x+c*cell+1,y+r*cell+1,x+(c+1)*cell-1,y+(r+1)*cell-1),
                                radius=max(2,int(cell*.06)), fill=fill)
    if channel is None:
        cx, cy = x+cell*.5, y+cell*.5
        radius = cell*.30
        d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), fill=BLUE)
        radius = cell*.16
        d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), fill=ORANGE)
        cx, cy = x+(n-.5)*cell, y+(n-.5)*cell
        radius = cell*.30
        d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius), fill=GOLD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, default=ROOT / "exports/instagram_01_maze_intro_facts.json")
    parser.add_argument("--output", type=Path, default=ROOT / "exports/instagram_01_maze_intro")
    args = parser.parse_args()
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    if facts.get("status") != "approved_for_intro":
        raise ValueError("Introduction facts have not passed independent review")
    raw = facts["example_maze"]
    if raw["split"] != "train" or len(raw["grid"]) != 10:
        raise ValueError("Use a verified 10x10 training-map illustration")
    grid = raw["grid"]
    import numpy as np
    if hashlib.sha256(np.asarray(grid,dtype=np.uint8).tobytes()).hexdigest() != raw["hash"]:
        raise ValueError("Illustration map hash mismatch")
    args.output.mkdir(parents=True,exist_ok=True)
    if list(args.output.glob("0[1-6]_*.png")):
        raise FileExistsError("Existing introduction images are preserved; choose a new output directory")

    im, d = intro_card(1, "미로 실험 노트 01 / 우리의 실험장을 소개합니다")
    text(d,(64,204),"지도를 다 보여주면,\nAI는 길을 찾을까?",68,INK,True)
    text(d,(64,411),"10×10 장애물 미로에서 시작하는 강화학습 실험",31,TEAL,True)
    draw_map(im,grid,290,525,500)
    text(d,(164,1092),"출발점에서 목표까지. 규칙은 단순합니다.",32,INK,True)
    text(d,(164,1157),"길을 배우는 과정은 어떨까요?",34,MUTED)
    im.save(args.output/"01_cover.png")

    im,d = intro_card(2,"지도 읽기 / 길이 존재하는 미로에서 시작합니다")
    text(d,(64,204),"출발점에서\n목표까지.",72,INK,True)
    draw_map(im,grid,64,465,650)
    legend = [(BLUE,"출발점"),(ORANGE,"현재 위치"),(GOLD,"목표"),(INK,"벽"),("#E8ECE4","통로")]
    for i,(color,label) in enumerate(legend):
        yy=500+i*115
        d.rounded_rectangle((765,yy,801,yy+36),radius=8,fill=color)
        text(d,(822,yy-4),label,30,INK,True)
    text(d,(64,1160),"처음에는 출발점과 현재 위치가 겹칩니다.",29,MUTED)
    im.save(args.output/"02_map.png")

    im,d = intro_card(3,"행동과 보상 / 목표에 도착하면 한 시도가 끝납니다")
    text(d,(64,204),"움직임은 네 방향.\n피드백은 보상으로.",64,INK,True)
    for xx,label in ((64,"상"),(310,"하"),(556,"좌"),(802,"우")):
        d.rounded_rectangle((xx,445,xx+214,550),radius=18,fill=PAPER)
        text(d,(xx+82,466),label,43,TEAL,True)
    rows = [("목표 도착","+1",TEAL),("일반 이동","-0.01",INK),("벽·경계 충돌","-0.05",ORANGE)]
    for i,(label,value,color) in enumerate(rows):
        yy=621+i*135
        text(d,(64,yy),label,38,INK,True)
        text(d,(762,yy-5),value,48,color,True)
        d.line((64,yy+87,1016,yy+87),fill=LINE,width=2)
    text(d,(64,1073),"충돌하면 제자리에 머뭅니다.\n한 시도는 최대 100행동까지 진행합니다.",31,MUTED)
    im.save(args.output/"03_rules.png")

    im,d = intro_card(4,"AI의 입력 / 현재 모델은 CNN DQN입니다")
    text(d,(64,204),"AI가 보는 지도는\n세 장의 정보입니다.",64,INK,True)
    for xx,label,channel in ((64,"벽 지도","walls"),(400,"현재 위치","agent"),(736,"목표 위치","goal")):
        text(d,(xx,460),label,33,TEAL,True)
        draw_map(im,grid,xx,538,280,channel)
    d.rounded_rectangle((64,910,1016,1053),radius=20,fill=INK)
    text(d,(108,940),"3채널 전체 지도 → CNN DQN",43,PAPER,True)
    text(d,(64,1111),"각 방향으로 움직였을 때\n앞으로 받을 보상을 추정하도록 학습합니다.",30,MUTED)
    im.save(args.output/"04_model.png")

    im,d = intro_card(5,"모델 비교 / 현재 DQN에서 순서대로 확장할 계획입니다")
    text(d,(64,204),"같은 미로에서,\n다른 DQN을 비교합니다.",60,INK,True)
    rows=[("DQN · 현재","행동별 가치를 학습"),
          ("Double · 예정","Q값 과대평가를 완화"),
          ("Dueling · 예정","상태 가치와 행동 이점을 분리"),
          ("PER-DQN · 예정","중요한 경험을 더 자주 복습"),
          ("Rainbow · 예정","여러 개선 기법을 결합")]
    for i,(label,desc) in enumerate(rows):
        yy=449+i*135
        text(d,(64,yy),label,31,TEAL,True)
        text(d,(438,yy),desc,29,INK)
        d.line((64,yy+91,1016,yy+91),fill=LINE,width=2)
    text(d,(64,1156),"Double+Dueling 결합형도 비교할 계획입니다.",28,MUTED)
    im.save(args.output/"05_roadmap.png")

    im,d = intro_card(6,"함께 볼 질문 / 결과와 과정을 차근차근 기록합니다")
    text(d,(64,204),"무엇을 바꾸면\n목표 도달이 늘어날까요?",61,INK,True)
    questions=["얼마나 빨리 배울까?","처음 보는 미로에서도 통할까?","어떤 경험을 기억해야 할까?"]
    for i,question in enumerate(questions):
        yy=491+i*155
        d.rounded_rectangle((64,yy,1016,yy+115),radius=18,fill=PAPER)
        text(d,(93,yy+25),question,38,INK,True)
    # Keep the final copy well above the bottom footer so platform previews
    # and carousel crops cannot clip the closing line.
    text(d,(64,1018),"성공과 실패, 보상 곡선과 이동 경로까지.",32,MUTED)
    text(d,(64,1080),"다음 실험, 함께 지켜봐주세요.",44,TEAL,True)
    im.save(args.output/"06_follow.png")

    (args.output/"verified_intro_facts.json").write_bytes(args.facts.read_bytes())
    provenance={"facts_sha256":hashlib.sha256(args.facts.read_bytes()).hexdigest(),
                "example_maze_hash":raw["hash"],"example_maze_split":"train",
                "performance_claims":False,"future_models_status":"planned", "uploaded":False}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(f"Created six introduction cards: {args.output.resolve()}")


if __name__=="__main__":
    main()
