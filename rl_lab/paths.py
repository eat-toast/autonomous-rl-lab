"""Study layout and read-only resolution of paths embedded in historical records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_legacy_path(value, root=ROOT):
    root = Path(root).resolve()
    original = Path(value)
    candidate = original if original.is_absolute() else root / original
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate
    mapping = root / "docs/path_migrations.json"
    if mapping.is_file():
        moves = json.loads(mapping.read_text(encoding="utf-8"))["moves"]
        for old in sorted(moves, key=len, reverse=True):
            if relative == old or relative.startswith(old + "/"):
                return root / (moves[old] + relative[len(old):])
    return candidate


def study_runs(results):
    results = Path(results)
    return (results.parent if results.name == "results" else results) / "runs"


def run_path(run_id, study_id):
    historical = resolve_legacy_path(Path("runs") / run_id)
    return historical if historical.is_dir() else ROOT / "studies" / study_id / "runs" / run_id
