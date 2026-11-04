from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parent.parent


def state_digest(value):
    """Stable hash of tensor/scalar state, used for event invariants (not checkpoint IDs)."""
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: str(k)):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(source, destination, attempts=10):
    """Bounded retry for transient Windows/OneDrive sharing violations."""
    for attempt in range(attempts):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(.05 * 2**attempt, 1.0))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


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


def runtime_metadata(device="cpu"):
    versions = {}
    for name in ("torch", "numpy", "matplotlib", "Pillow"):
        versions[name] = importlib.metadata.version(name)
    accelerator = {}
    if device == "cuda":
        import torch
        accelerator = {"gpu_name": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version(),
                       "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                       "precision": "float32; TF32 and AMP disabled"}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "device": device, "accelerator": accelerator, "torch_threads": 1,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "git_diff": git("diff", "HEAD"), "created_at_utc": utc_now()}
