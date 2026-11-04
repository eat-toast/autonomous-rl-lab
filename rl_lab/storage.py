from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parent.parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=10)
    return result.stdout.strip() if result.returncode == 0 else None


def snapshot_source(run_dir):
    """Archive only project source/config, never credentials or other local folders."""
    paths = [ROOT / name for name in ("README.md", "requirements.txt", "requirements-tested.txt", ".gitignore")]
    for directory in ("rl_lab", "tests", "configs", "docs", "scripts"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".json", ".md", ".ps1"))
    manifest = {}
    with zipfile.ZipFile(run_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if path.is_file():
                relative = path.relative_to(ROOT).as_posix()
                manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.write(path, relative)
    return manifest


def runtime_metadata():
    versions = {}
    for name in ("torch", "numpy", "matplotlib", "Pillow"):
        versions[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "device": "cpu", "torch_threads": 1,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "git_diff": git("diff", "HEAD"), "created_at_utc": utc_now()}
