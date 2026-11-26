"""The word-level full pilot must never train or download on import."""

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "word_pilot_full.py"


def test_pilot_import_has_no_optional_dependencies():
    spec = importlib.util.spec_from_file_location("optional_word_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_pilot_help_runs_without_data():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_pilot_refuses_existing_output_before_loading_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(tmp_path / "absent"),
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Output exists" in result.stderr
    assert not list(tmp_path.iterdir())
