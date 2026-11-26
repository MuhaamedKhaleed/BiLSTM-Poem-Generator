"""The optional recipe must not import ML or touch data on import/help."""

import runpy
import subprocess
import sys
from pathlib import Path


def test_improvement_import_is_inert(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/improvement_cpu.py"
    runpy.run_path(str(script), run_name="not_main")
    assert list(tmp_path.iterdir()) == []


def test_improvement_help():
    script = Path(__file__).resolve().parents[1] / "scripts/improvement_cpu.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
