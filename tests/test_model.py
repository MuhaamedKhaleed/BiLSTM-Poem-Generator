import json
from pathlib import Path

import pytest

from poem_generator.core import Vocabulary, prefixes
from poem_generator.model import backend, build_model, generate, load_generate, train

np = pytest.importorskip("numpy")
pytest.importorskip("tensorflow")


class Fake:
    def __init__(self, output):
        self.output = output

    def __call__(self, *args, **kwargs):
        return self

    def numpy(self):
        return self.output


def test_greedy_excludes_special_tokens_keeps_length() -> None:
    vocab = Vocabulary({"moon": 2, "sky": 3}, 2)
    # Highest raw scores belong to padding/OOV, both must be excluded.
    fake = Fake(np.array([[0.6, 0.3, 0.05, 0.05]]))
    assert generate(fake, vocab, "Unseen seed", 3) == "Unseen seed moon moon moon"
    for prompt, n in [("", 2), ("x", 0), ("x", 1001)]:
        with pytest.raises(ValueError):
            generate(fake, vocab, prompt, n)
    with pytest.raises(ValueError):
        generate(Fake(np.array([[float("nan")] * 4])), vocab, "x")
    with pytest.raises(ValueError):
        generate(Fake(np.ones((1, 3))), vocab, "x")


def test_actual_forward_sparse_loss() -> None:
    tf = backend()
    tf.keras.utils.set_random_seed(42)
    vocab = Vocabulary.fit(["moon shines bright", "stars glow softly"])
    x, y = prefixes(["moon shines bright"], vocab)
    model = build_model(vocab, 4, 3, 2)
    output = model(np.array(x), training=False).numpy()
    assert output.shape == (2, 8)
    assert np.isfinite(output).all()
    assert all(np.isfinite(model.train_on_batch(np.array(x), np.array(y))))


def test_synthetic_training_round_trip(tmp_path: Path) -> None:
    lines = ["moon shines bright", "stars glow softly"]
    validation = ["moon shines softly"]
    output = tmp_path / "tiny"
    scores = train(
        lines,
        validation,
        output,
        epochs=1,
        batch_size=2,
        embedding=4,
        units=3,
        second_units=2,
    )
    assert scores["validation_prefixes"] == 2 and scores["validation_oov_targets"] == 0
    assert np.isfinite(scores["validation_loss_with_regularization"])
    assert len(load_generate(output, "unknown moon", 2).split()) == 4
    assert json.loads((output / "validation.json").read_text()) == scores
    with pytest.raises(ValueError):
        train(lines, validation, output)
    (output / "vocabulary.json").write_text("[]")
    with pytest.raises(ValueError):
        load_generate(output, "moon")
    (output / "vocabulary.json").write_text(
        json.dumps({"words": {"moon": 2}, "context": 1})
    )
    with pytest.raises(ValueError):
        load_generate(output, "moon")


def test_bad_training(tmp_path: Path) -> None:
    lines = ["a b", "c d"]
    with pytest.raises(ValueError):
        train(lines, ["a b"], tmp_path / "overlap")
    with pytest.raises(ValueError):
        train(lines, ["e f"], tmp_path / "invalid", epochs=0)
    with pytest.raises(ValueError):
        build_model(Vocabulary.fit(lines), embedding=0)


def test_explicit_cli_training(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from poem_generator.cli import app

    path = tmp_path / "poem.txt"
    path.write_text(
        "moon shines bright\nstars glow softly\nnight falls gently\nsun shines softly\n"
    )
    runner = CliRunner()
    out = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "train",
            str(path),
            str(out),
            "--epochs",
            "1",
            "--embedding",
            "4",
            "--units",
            "3",
            "--second-units",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["validation_prefixes"] == 2
    result = runner.invoke(
        app, ["generate", str(out), "moon shines", "--next-words", "2"]
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["text"].split()) == 4
