"""Optional local TensorFlow adapter, never loaded on help or corpus checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .core import Vocabulary, prefixes


def backend() -> Any:
    """Load optional ML dependencies only for an explicit model operation."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("ML backend missing. Install uv sync --extra ml.") from exc
    return tf


def build_model(
    vocab: Vocabulary, embedding: int = 100, units: int = 150, second_units: int = 100
) -> Any:
    """Preserve historical BiLSTM/dropout/LSTM/dense architecture, sparse labels."""
    if min(embedding, units, second_units) < 1:
        raise ValueError("All architecture dimensions must be positive.")
    tf = backend()
    size = len(vocab.words) + 2
    inputs = tf.keras.Input(shape=(vocab.context,), dtype="int32")
    x = tf.keras.layers.Embedding(size, embedding)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(second_units)(x)
    x = tf.keras.layers.Dense(
        size // 2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    output = tf.keras.layers.Dense(size, activation="softmax")(x)
    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def train(
    train_lines: list[str],
    validation: list[str],
    output: Path,
    epochs: int = 150,
    batch_size: int = 32,
    seed: int = 42,
    embedding: int = 100,
    units: int = 150,
    second_units: int = 100,
) -> dict[str, int | float]:
    """Train explicitly on local data, save new artifacts and validation metrics."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if set(train_lines) & set(validation):
        raise ValueError("Train/validation lines overlap. Split before prefixes.")
    if output.exists():
        raise ValueError("Output exists. Choose a new directory; never overwrite.")
    tf = backend()
    import numpy as np

    vocab = Vocabulary.fit(train_lines)
    x, y = prefixes(train_lines, vocab)
    vx, vy = prefixes(validation, vocab)
    tf.keras.utils.set_random_seed(seed)
    model = build_model(vocab, embedding, units, second_units)
    model.fit(
        np.array(x, dtype=np.int32),
        np.array(y, dtype=np.int32),
        validation_data=(np.array(vx, dtype=np.int32), np.array(vy, dtype=np.int32)),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    loss, accuracy = model.evaluate(
        np.array(vx, dtype=np.int32), np.array(vy, dtype=np.int32), verbose=0
    )
    if not math.isfinite(float(loss)) or not math.isfinite(float(accuracy)):
        raise ValueError("Training produced nonfinite metrics, no artifact saved.")
    scores = {
        "validation_loss_with_regularization": float(loss),
        "validation_next_token_accuracy": float(accuracy),
        "validation_prefixes": len(vy),
        "validation_oov_targets": sum(i == 1 for i in vy),
    }
    output.mkdir(parents=True)
    model.save(output / "model.keras")
    (output / "vocabulary.json").write_text(
        json.dumps({"words": vocab.words, "context": vocab.context}), encoding="utf-8"
    )
    (output / "validation.json").write_text(json.dumps(scores), encoding="utf-8")
    return scores


def generate(model: Any, vocab: Vocabulary, prompt: str, next_words: int = 8) -> str:
    """Greedy decoding, excluding padding and OOV output classes.

    Prompt unknowns remain OOV context positions, they are never silently removed.
    """
    if not prompt.strip() or not 1 <= next_words <= 1000:
        raise ValueError("Supply nonempty prompt and next words in [1,1000].")
    import numpy as np

    text = prompt.strip()
    reverse = {v: k for k, v in vocab.words.items()}
    ids = vocab.encode(text)
    for _ in range(next_words):
        output = model(
            np.array([vocab.context_ids(ids)], dtype=np.int32), training=False
        ).numpy()
        if output.shape != (1, len(vocab.words) + 2) or not np.isfinite(output).all():
            raise ValueError(
                "Model output is nonfinite or incompatible with vocabulary."
            )
        index = int(output[0, 2:].argmax()) + 2
        ids.append(index)
        text += " " + reverse[index]
    return text


def load_generate(directory: Path, prompt: str, next_words: int = 8) -> str:
    """Load only a trusted local Keras artifact. Safe mode is not a sandbox."""
    tf = backend()
    payload = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"words", "context"}:
        raise ValueError("Vocabulary JSON needs words and context.")
    vocab = Vocabulary(payload["words"], payload["context"])
    model = tf.keras.models.load_model(
        directory / "model.keras", compile=False, safe_mode=True
    )
    if (
        model.input_shape[1] != vocab.context
        or model.output_shape[-1] != len(vocab.words) + 2
    ):
        raise ValueError("Model shapes do not match vocabulary.")
    return generate(model, vocab, prompt, next_words)
