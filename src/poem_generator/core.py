"""Corpus validation, line split and train-only prefix construction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path


def load_lines(path: Path) -> list[str]:
    """Read UTF-8, lowercase and deduplicate normalized whitespace-separated lines.

    Punctuation is preserved. This differs from the old Keras punctuation filter.
    Lines with fewer than two tokens are rejected, not silently discarded.
    """
    lines = list(
        dict.fromkeys(
            " ".join(s.lower().split())
            for s in path.read_text(encoding="utf-8").splitlines()
            if s.strip()
        )
    )
    if not lines:
        raise ValueError("Corpus is empty. Supply nonempty UTF-8 text lines.")
    if any(len(s.split()) < 2 for s in lines):
        raise ValueError("Every nonempty corpus line needs at least two tokens.")
    return lines


def split_lines(
    lines: list[str], fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split complete unique lines before generating overlapping prefixes."""
    if len(lines) < 2 or len(set(lines)) != len(lines):
        raise ValueError("Need at least two unique lines for held-out validation.")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Validation fraction must be finite and between 0 and 1.")
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    n = max(1, min(len(lines) - 1, math.ceil(len(lines) * fraction)))
    return shuffled[n:], shuffled[:n]


@dataclass(frozen=True)
class Vocabulary:
    """Training-only words, 0 padding and 1 unknown. No evaluation token drops."""

    words: dict[str, int]
    context: int

    def __post_init__(self) -> None:
        if type(self.context) is not int or self.context < 1:
            raise ValueError("Context length must be a positive integer.")
        if not isinstance(self.words, dict) or not self.words:
            raise ValueError("Words must be a nonempty JSON object.")
        if any(
            not isinstance(k, str) or not k.strip() or type(v) is not int
            for k, v in self.words.items()
        ):
            raise ValueError("Words require nonempty string keys and integer IDs.")
        if sorted(self.words.values()) != list(range(2, len(self.words) + 2)):
            raise ValueError("Word IDs must be contiguous from 2.")

    @classmethod
    def fit(cls, lines: list[str]) -> Vocabulary:
        """Fit vocabulary and maximum context on training lines only."""
        if not lines or any(len(s.split()) < 2 for s in lines):
            raise ValueError("Training lines need at least two tokens each.")
        words = sorted({word for s in lines for word in s.split()})
        return cls(
            {w: i + 2 for i, w in enumerate(words)},
            max(len(s.split()) - 1 for s in lines),
        )

    def encode(self, text: str) -> list[int]:
        """Lowercase whitespace tokenization, unknown tokens retain their slot."""
        return [self.words.get(word, 1) for word in text.lower().split()]

    def context_ids(self, tokens: list[int]) -> list[int]:
        """Pre-pad and retain last training-context tokens, matching generation."""
        kept = tokens[-self.context :]
        return [0] * (self.context - len(kept)) + kept


def prefixes(lines: list[str], vocab: Vocabulary) -> tuple[list[list[int]], list[int]]:
    """Generate one sparse next-token target per noninitial token."""
    x: list[list[int]] = []
    y: list[int] = []
    for line in lines:
        ids = vocab.encode(line)
        if len(ids) < 2:
            raise ValueError("Every line needs at least two tokens.")
        for i in range(1, len(ids)):
            x.append(vocab.context_ids(ids[:i]))
            y.append(ids[i])
    if not x:
        raise ValueError("No prefixes. Supply at least one two-token line.")
    return x, y
