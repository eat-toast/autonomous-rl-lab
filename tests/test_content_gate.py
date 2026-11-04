import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_study_content import main


class ContentGateTests(unittest.TestCase):
    def prepare(self, folder, status):
        protocol = folder / "protocol.json"
        comparison = folder / "comparison.json"
        protocol.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        comparison.write_text(json.dumps({"conditions": []}), encoding="utf-8")
        verified = {"status": status,
                    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                    "comparison_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
                    "approved_run_ids": []}
        (folder / "verified_facts.json").write_text(json.dumps(verified), encoding="utf-8")
        (folder / "review.json").write_text(json.dumps(verified), encoding="utf-8")

    def test_completed_experiment_without_independent_review_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.prepare(folder, "pending")
            output = folder / "output"
            with patch.object(sys, "argv", ["export", "--study", str(folder), "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "Independent results review"):
                    main()
            self.assertFalse(output.exists())

    def test_changed_results_after_review_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.prepare(folder, "approved_for_dev_report")
            (folder / "comparison.json").write_text(json.dumps({"conditions": [], "changed": True}), encoding="utf-8")
            output = folder / "output"
            with patch.object(sys, "argv", ["export", "--study", str(folder), "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "no longer match comparison.json"):
                    main()
            self.assertFalse(output.exists())

    def test_commit_failure_blocks_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.prepare(folder, "approved_for_dev_report")
            rows = [{"run_id": "pilot"},
                    {"run_id": "base", "steps": 20, "learning_rate": .001, "epsilon_decay_steps": 16},
                    {"run_id": "wide", "steps": 20, "learning_rate": .001, "epsilon_decay_steps": 16}]
            protocol = {"provisional_selected_run": "base"}
            output = folder / "output"
            with patch.object(sys, "argv", ["export", "--study", str(folder), "--output", str(output)]), \
                 patch("scripts.export_study_content.validate_review", return_value=(protocol, {"conditions": rows})), \
                 patch("scripts.export_study_content.commit_review", side_effect=RuntimeError("Git commit failed")):
                with self.assertRaisesRegex(RuntimeError, "Git commit failed"):
                    main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
