"""Keep the common test command discovering the study-owned test suites."""
import importlib.util
from pathlib import Path


def load_tests(loader, tests, pattern):
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "studies").glob("*/tests/test_*.py")):
        spec = importlib.util.spec_from_file_location(f"{path.parents[1].name}_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tests.addTests(loader.loadTestsFromModule(module))
    return tests
