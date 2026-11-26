from pathlib import Path

import pytest

from poem_generator.core import Vocabulary, load_lines, prefixes, split_lines


def test_normalized_deduplication(tmp_path: Path) -> None:
    path = tmp_path / "poem.txt"
    path.write_text(" Hello  WORLD!\nhello world!\n\nNext line\n")
    assert load_lines(path) == ["hello world!", "next line"]


@pytest.mark.parametrize("text", ["", "  \n", "word\na b"])
def test_bad_corpus(tmp_path: Path, text: str) -> None:
    path = tmp_path / "poem.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_lines(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lines(tmp_path / "missing")


def test_line_split_before_prefixes() -> None:
    lines = ["a b c", "d e f", "g h i", "j k l", "m n o"]
    train, val = split_lines(lines, 0.2, 42)
    assert len(train) == 4 and len(val) == 1
    assert set(train).isdisjoint(val)
    assert (train, val) == split_lines(lines, 0.2, 42)
    assert lines[0] == "a b c"
    vocab = Vocabulary.fit(train)
    assert all(w not in vocab.words for w in val[0].split())
    assert prefixes(val, vocab)[1] == [1, 1]


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan"), float("inf")])
def test_bad_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        split_lines(["a b", "c d"], fraction)


def test_invalid_split() -> None:
    for lines in [[], ["a b"], ["a b", "a b"]]:
        with pytest.raises(ValueError):
            split_lines(lines)


def test_sparse_prefix_alignment_and_context() -> None:
    vocab = Vocabulary.fit(["a b c", "d e"])
    assert vocab.context == 2
    assert vocab.encode("A unknown B") == [2, 1, 3]
    assert vocab.context_ids([2, 3, 4]) == [3, 4]
    assert prefixes(["a b c"], vocab) == ([[0, 2], [2, 3]], [3, 4])
    assert vocab.context_ids([]) == [0, 0]


@pytest.mark.parametrize(
    "words,context",
    [
        ({}, 1),
        ({"a": 1}, 1),
        ({"a": 2}, 0),
        ({"a": 2}, True),
        ({"a": 2.0}, 1),
        ({"": 2}, 1),
        ([], 1),
    ],
)
def test_invalid_vocab(words, context) -> None:
    with pytest.raises(ValueError):
        Vocabulary(words, context)


def test_no_prefixes_or_training() -> None:
    for lines in [[], ["single"]]:
        with pytest.raises(ValueError):
            Vocabulary.fit(lines)
        with pytest.raises(ValueError):
            prefixes(lines, Vocabulary({"a": 2}, 1))
