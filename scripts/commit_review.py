"""Workflow hook: preserve an independent review in a local Git commit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
FILES = ("protocol.json", "comparison.json", "review.json", "verified_facts.json")


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8").strip()


def validate_review(study):
    """Publication requires both reviewer records to approve these exact inputs."""
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    comparison = json.loads((study / "comparison.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "completed":
        raise ValueError("Study must be completed before creating publication assets")
    for record in ("verified_facts.json", "review.json"):
        review = json.loads((study / record).read_text(encoding="utf-8"))
        if review.get("status") != "approved_for_dev_report":
            raise ValueError(f"Independent results review has not approved {record}")
        for name in ("comparison.json", "protocol.json"):
            digest = hashlib.sha256((study / name).read_bytes()).hexdigest()
            if digest != review.get(name.replace(".json", "_sha256"), "").lower():
                raise ValueError(f"Verified facts no longer match {name} ({record})")
        if set(review.get("approved_run_ids", [])) != {r["run_id"] for r in comparison["conditions"]}:
            raise ValueError(f"Verified run IDs do not match the study ({record})")
    return protocol, comparison


def commit_review(study, root=ROOT):
    root = Path(root).resolve()
    study = Path(study).resolve()
    relative = study.relative_to(root).as_posix()
    if git(root, "rev-parse", "--show-toplevel").replace("\\", "/") != root.as_posix():
        raise ValueError("Hook root must be the repository root")
    # Facts may be absent for an incomplete/rejected review. Preserve its status as-is.
    payload = {name: (study / name).read_bytes() for name in FILES if (study / name).is_file()}
    if "review.json" not in payload:
        raise ValueError("Independent review.json is required before committing")
    status = json.loads(payload["review.json"].decode("utf-8"))["status"]
    manifest = {"study": relative, "review_status": status,
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    identity = hashlib.sha256(manifest_bytes).hexdigest()
    snapshot = root / "review_snapshots" / identity
    payload["manifest.json"] = manifest_bytes
    paths = [(snapshot / name).relative_to(root).as_posix() for name in payload]
    # Never sweep another task's staged work into this commit.
    staged = git(root, "diff", "--cached", "--name-only", "-z").split("\0")
    if any(path and path not in paths for path in staged):
        raise RuntimeError("Unrelated staged files exist; preserve/commit them before running the review hook")
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        target = snapshot / name
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"Existing review snapshot was modified: {target}")
        else:
            with target.open("xb") as stream:
                stream.write(data)
    git(root, "add", "--", *paths)
    if git(root, "diff", "--cached", "--name-only", "--", *paths):
        git(root, "commit", "--only", "-m", f"research: preserve independent review {identity[:12]}", "--", *paths)
    commit = git(root, "log", "-1", "--format=%H", "--", *paths)
    if not commit:
        raise RuntimeError("Review snapshot has no Git commit")
    # Confirm committed bytes, including if a local Git hook changed a file.
    for name, data in payload.items():
        path = (snapshot / name).relative_to(root).as_posix()
        saved = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True, check=True).stdout
        if saved != data or (snapshot / name).read_bytes() != data:
            raise ValueError(f"Committed review differs from input: {name}")
        if name in FILES and (study / name).read_bytes() != data:
            raise ValueError(f"Study changed while committing its review: {name}")
    print(f"Review commit: {commit}")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    commit_review(args.study)


if __name__ == "__main__":
    main()
