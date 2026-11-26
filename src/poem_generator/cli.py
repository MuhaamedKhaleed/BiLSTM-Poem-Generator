"""Thin explicit local corpus/train/generate commands."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from dotenv import dotenv_values

from .core import Vocabulary, load_lines, prefixes, split_lines

app = typer.Typer(help="Local BiLSTM next-word generator. No implicit training.")
logger = logging.getLogger(__name__)


def run(action: Callable[[], Any]) -> None:
    """JSON stdout and expected local failures on stderr with exit 2."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        typer.echo(json.dumps(action()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def resolve(data: Path | None, config: Path | None) -> Path:
    """Explicit corpus wins over opt-in dotenv POEM_CORPUS_PATH."""
    if data:
        return data
    if config:
        if not config.is_file():
            raise ValueError("Config not found. Supply an existing dotenv file.")
        value = dotenv_values(config).get("POEM_CORPUS_PATH")
        if value:
            return Path(value)
    raise ValueError("Supply --data or --config containing POEM_CORPUS_PATH.")


@app.command()
def check_data(
    data: Path | None = None,
    config: Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Validate local corpus and whole-line split without importing TensorFlow."""

    def action() -> dict[str, int]:
        lines = load_lines(resolve(data, config))
        training, validation = split_lines(lines, validation_fraction, seed)
        vocab = Vocabulary.fit(training)
        x, _ = prefixes(training, vocab)
        vx, vy = prefixes(validation, vocab)
        return {
            "unique_lines": len(lines),
            "train_lines": len(training),
            "validation_lines": len(validation),
            "training_words": len(vocab.words),
            "context": vocab.context,
            "train_prefixes": len(x),
            "validation_prefixes": len(vx),
            "validation_oov_targets": sum(i == 1 for i in vy),
        }

    run(action)


@app.command()
def train(
    data: Path,
    output: Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 150,
    batch_size: int = 32,
    embedding: int = 100,
    units: int = 150,
    second_units: int = 100,
) -> None:
    """Start local CPU training deliberately; never downloads data or weights."""

    def action() -> dict[str, int | float]:
        from .model import train as fit

        training, validation = split_lines(load_lines(data), validation_fraction, seed)
        return fit(
            training,
            validation,
            output,
            epochs,
            batch_size,
            seed,
            embedding,
            units,
            second_units,
        )

    run(action)


@app.command()
def generate(artifact: Path, prompt: str, next_words: int = 8) -> None:
    """Greedy continuation from your own trusted local Keras model."""

    def action() -> dict[str, str]:
        from .model import load_generate

        return {"text": load_generate(artifact, prompt, next_words)}

    run(action)
