from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import random
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("poem.txt")
DEFAULT_EPOCHS = 150
DEFAULT_SEED_TEXT = "his drumming"
DEFAULT_NEXT_WORDS = 8
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds across libraries for reproducible execution."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_corpus(file_path: Path | str) -> list[str]:
    """Read and normalize the text corpus from a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as file:
        data = file.read()

    if not data.strip():
        raise ValueError(f"Corpus file is empty: {path.resolve()}")

    # Generate corpus lines, convert to lowercase, and deduplicate deterministically
    corpus_lines = [line.strip().lower() for line in data.split("\n") if line.strip()]
    unique_corpus = list(dict.fromkeys(corpus_lines))

    if not unique_corpus:
        raise ValueError("Corpus contains no non-empty lines.")

    return unique_corpus


def create_sequences(
    corpus: list[str],
) -> Tuple[Tokenizer, np.ndarray, np.ndarray, int, int]:
    """Tokenize corpus and create n-gram training sequences and one-hot labels."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences: list[list[int]] = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    if not input_sequences:
        raise ValueError(
            "No n-gram sequences could be generated. Ensure lines contain at least two words."
        )

    max_sequence_len = max(len(seq) for seq in input_sequences)
    padded_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )

    predictors = padded_sequences[:, :-1]
    labels = padded_sequences[:, -1]
    one_hot_labels = ku.to_categorical(labels, num_classes=total_words)

    return tokenizer, predictors, one_hot_labels, max_sequence_len, total_words


def build_model(max_sequence_len: int, total_words: int) -> Model:
    """Build and compile the Bi-Directional LSTM text generation model."""
    input_layer = Input(shape=(max_sequence_len - 1,))
    embedding_layer = Embedding(input_dim=total_words, output_dim=100)(input_layer)
    bi_lstm = Bidirectional(LSTM(150, return_sequences=True))(embedding_layer)
    dropout_layer = Dropout(0.2)(bi_lstm)
    lstm_layer = LSTM(100)(dropout_layer)
    dense_layer = Dense(
        total_words // 2,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.01),
    )(lstm_layer)
    output_layer = Dense(total_words, activation="softmax")(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def generate_text(
    model: Model,
    tokenizer: Tokenizer,
    max_sequence_len: int,
    seed_text: str,
    next_words: int,
) -> str:
    """Generate poetic continuation from seed text using iterative next-word prediction."""
    if not seed_text.strip():
        raise ValueError("Seed text cannot be empty.")
    if next_words <= 0:
        raise ValueError("next_words must be greater than zero.")

    current_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        predicted_index = int(predicted[0]) if isinstance(predicted, np.ndarray) else int(predicted)

        output_word = tokenizer.index_word.get(predicted_index, "")
        if not output_word:
            break

        current_text += " " + output_word

    return current_text


def main() -> None:
    """Parse CLI arguments, train model, evaluate, and generate verse extension."""
    parser = argparse.ArgumentParser(description="BiLSTM Poem Generator")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the training poem corpus file (default: poem.txt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: 150)",
    )
    parser.add_argument(
        "--seed-text",
        type=str,
        default=DEFAULT_SEED_TEXT,
        help='Seed text for poem generation (default: "his drumming")',
    )
    parser.add_argument(
        "--next-words",
        type=int,
        default=DEFAULT_NEXT_WORDS,
        help="Number of words to generate (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    set_seed(args.seed)

    logger.info("Loading corpus from %s", args.data_path)
    corpus = load_corpus(args.data_path)
    logger.info("Loaded %d unique lines from corpus", len(corpus))

    tokenizer, predictors, label, max_sequence_len, total_words = create_sequences(corpus)
    print("Total Words:", total_words)

    model = build_model(max_sequence_len=max_sequence_len, total_words=total_words)
    print(model.summary())

    logger.info("Training BiLSTM model for %d epochs...", args.epochs)
    model.fit(predictors, label, epochs=args.epochs, verbose=1)

    loss, accuracy = model.evaluate(predictors, label, verbose=0)
    print("Accuracy: %f" % (accuracy * 100))

    logger.info("Generating text with seed: '%s'", args.seed_text)
    generated_poem = generate_text(
        model=model,
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
        seed_text=args.seed_text,
        next_words=args.next_words,
    )
    print(generated_poem)


if __name__ == "__main__":
    main()
