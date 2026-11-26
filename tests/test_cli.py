import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from poem_generator.cli import app

runner = CliRunner()


def test_help_no_import_time_compute() -> None:
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;import poem_generator.cli;"
                "assert 'tensorflow' not in sys.modules"
            ),
        ],
        capture_output=True,
    )
    assert p.returncode == 0
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["train", "--help"]).exit_code == 0


def test_data_config_and_precedence(tmp_path: Path) -> None:
    path = tmp_path / "poem.txt"
    path.write_text("a b c\nd e f\ng h i\nj k l\nm n o\n")
    config = tmp_path / ".env"
    config.write_text(f"POEM_CORPUS_PATH={path}\n")
    result = runner.invoke(app, ["check-data", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "unique_lines": 5,
        "train_lines": 4,
        "validation_lines": 1,
        "training_words": 12,
        "context": 2,
        "train_prefixes": 8,
        "validation_prefixes": 2,
        "validation_oov_targets": 2,
    }
    config.write_text("POEM_CORPUS_PATH=/missing\n")
    assert (
        runner.invoke(
            app, ["check-data", "--data", str(path), "--config", str(config)]
        ).exit_code
        == 0
    )


def test_bad_cli_paths(tmp_path: Path) -> None:
    for args in [
        [],
        ["--data", str(tmp_path / "missing")],
        ["--config", str(tmp_path / "missing")],
    ]:
        result = runner.invoke(app, ["check-data", *args])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
    config = tmp_path / ".env"
    config.write_text("UNRELATED=none\n")
    assert runner.invoke(app, ["check-data", "--config", str(config)]).exit_code == 2
    assert runner.invoke(app, ["generate", str(tmp_path), "hello"]).exit_code == 2
