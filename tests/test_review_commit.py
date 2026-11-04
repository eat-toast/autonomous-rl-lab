import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.commit_review import commit_review, git, validate_review


class ReviewCommitTests(unittest.TestCase):
    def prepare(self, root):
        git(root, "init")
        git(root, "config", "user.name", "Hook Test")
        git(root, "config", "user.email", "hook-test@example.invalid")
        git(root, "config", "core.autocrlf", "true")
        (root / ".gitattributes").write_text("review_snapshots/** -text\n", encoding="utf-8")
        (root / ".gitignore").write_text("studies/\n", encoding="utf-8")
        git(root, "add", ".gitignore", ".gitattributes")
        git(root, "commit", "-m", "initial")
        study = root / "studies" / "example"
        study.mkdir(parents=True)
        (study / "protocol.json").write_text('{"status":"completed"}', encoding="utf-8")
        (study / "comparison.json").write_text('{"conditions":[{"run_id":"run-1"}]}', encoding="utf-8")
        record = {"status": "approved_for_dev_report", "approved_run_ids": ["run-1"]}
        for name in ("protocol", "comparison"):
            record[name + "_sha256"] = hashlib.sha256((study / (name + ".json")).read_bytes()).hexdigest()
        for name in ("review.json", "verified_facts.json"):
            (study / name).write_text(json.dumps(record), encoding="utf-8")
        return study

    def test_snapshot_commit_is_exact_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            study = self.prepare(root)
            (root / "unrelated.txt").write_text("leave this alone", encoding="utf-8")
            commit = commit_review(study, root)
            self.assertEqual(commit_review(study, root), commit)
            self.assertEqual(git(root, "rev-list", "--count", "HEAD"), "2")
            self.assertEqual(git(root, "status", "--porcelain"), "?? unrelated.txt")
            snapshot = next((root / "review_snapshots").iterdir())
            for name in ("review.json", "verified_facts.json", "protocol.json", "comparison.json"):
                path = (snapshot / name).relative_to(root).as_posix()
                saved = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True, check=True).stdout
                self.assertEqual(saved, (study / name).read_bytes())

    def test_unrelated_staging_blocks_hook_without_changing_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            study = self.prepare(root)
            (root / "other.txt").write_text("existing work", encoding="utf-8")
            git(root, "add", "other.txt")
            before = git(root, "diff", "--cached")
            with self.assertRaisesRegex(RuntimeError, "Unrelated staged"):
                commit_review(study, root)
            self.assertEqual(git(root, "diff", "--cached"), before)
            self.assertFalse((root / "review_snapshots").exists())

    def test_rejected_review_is_recorded_but_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            study = self.prepare(root)
            (study / "review.json").write_text('{"status":"rejected"}', encoding="utf-8")
            commit_review(study, root)
            with self.assertRaisesRegex(ValueError, "Independent results review"):
                validate_review(study)

    def test_changed_result_and_changed_review_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            study = self.prepare(root)
            validate_review(study)
            record = json.loads((study / "review.json").read_text(encoding="utf-8"))
            record["comparison_sha256"] = "0" * 64
            (study / "review.json").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer match comparison.json"):
                validate_review(study)


if __name__ == "__main__":
    unittest.main()
