import json
from pathlib import Path
import tempfile
import unittest

from rl_lab.paths import resolve_legacy_path, study_runs


class LayoutTests(unittest.TestCase):
    def test_historical_paths_resolve_without_rewriting_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "docs").mkdir()
            mapping = root / "docs/path_migrations.json"
            mapping.write_text(json.dumps({"moves": {
                "runs/old": "studies/A/runs/old",
                "studies/A/results/runs": "studies/A/runs",
            }}), encoding="utf-8")
            before = mapping.read_bytes()
            expected = root / "studies/A/runs/old/seed_0/latest.pt"
            self.assertEqual(resolve_legacy_path("runs/old/seed_0/latest.pt", root), expected)
            self.assertEqual(resolve_legacy_path(root / "runs/old/seed_0/latest.pt", root), expected)
            self.assertEqual(resolve_legacy_path("runs/older/file", root), root / "runs/older/file")
            self.assertEqual(resolve_legacy_path("studies/A/results/runs/x/file", root),
                             root / "studies/A/runs/x/file")
            self.assertEqual(mapping.read_bytes(), before)

    def test_study_runs_are_beside_results(self):
        self.assertEqual(study_runs(Path("studies/A/results")), Path("studies/A/runs"))
