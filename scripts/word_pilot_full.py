"""Full-training original-path word-level BiLSTM follow-up.

The recorded word-level baseline (docs/quality-cpu.md) used a reduced
64/64/32 architecture, about ten epochs, and its greedy sample collapsed into
repeated words. This script reruns the original word-level BiLSTM path with
the package's full default architecture (100/150/100), longer training with
development-checkpoint selection, and a fixed generation characterization:
collapse and train-line memorization checks mirroring the character-level
pilot's methodology. Same chronological unique-line recipe: first 1200 train,
next 200 development, next 200 test. The test slice was already read once by
the earlier baseline; this second read is documented and must not be treated
as an untouched benchmark. No download, no import-time work.
"""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    if args.epochs < 1 or args.patience < 1:
        parser.error("Epochs and patience must be positive.")
    import os

    for key in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"]:
        os.environ[key] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import hashlib
    import json
    import math
    import resource
    import time

    import numpy as np

    from poem_generator.core import Vocabulary, prefixes
    from poem_generator.model import backend, build_model, generate

    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    raw = (args.data / "1041.txt").read_bytes()
    text = raw.decode("utf-8-sig")
    body = text.split("THE SONNETS", 1)[1].split("*** END OF THIS PROJECT", 1)[0]
    lines = list(
        dict.fromkeys(
            line.strip().lower()
            for line in body.splitlines()
            if len(line.split()) >= 4 and not line.strip().isupper()
        )
    )
    training, dev, test = lines[:1200], lines[1200:1400], lines[1400:1600]
    vocab = Vocabulary.fit(training)
    x, y = map(np.array, prefixes(training, vocab))
    vx, vy = map(np.array, prefixes(dev, vocab))
    tx, ty = map(np.array, prefixes(test, vocab))
    tf = backend()
    tf.keras.utils.set_random_seed(42)
    model = build_model(vocab, 100, 150, 100)

    def unregularized_nll(data_x, data_y):
        prob = model(data_x, training=False).numpy()
        chosen = prob[np.arange(len(data_y)), data_y]
        return float(-np.log(np.maximum(chosen, 1e-10)).mean())

    start = time.time()
    best = float("inf")
    best_weights = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.fit(x, y, batch_size=64, epochs=1, verbose=0)
        loss, acc = model.evaluate(vx, vy, verbose=0)
        nll = unregularized_nll(vx, vy)
        history.append(
            {
                "epoch": epoch,
                "development_regularized_loss": float(loss),
                "development_accuracy": float(acc),
                "development_crossentropy": nll,
                "development_perplexity": math.exp(nll),
            }
        )
        print(history[-1], flush=True)
        if nll < best - 1e-4:
            best, best_epoch, stale = nll, epoch, 0
            best_weights = [w.copy() for w in model.get_weights()]
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.set_weights(best_weights)
    test_loss, test_acc = model.evaluate(tx, ty, verbose=0)
    test_nll = unregularized_nll(tx, ty)

    def sample(prompt, tokens, temperature, top_k, seed):
        rng = np.random.default_rng(seed)
        reverse = {v: k for k, v in vocab.words.items()}
        ids = vocab.encode(prompt)
        for _ in range(tokens):
            prob = model(
                np.array([vocab.context_ids(ids)], dtype=np.int32), training=False
            ).numpy()[0][2:]
            if temperature is None:
                index = int(prob.argmax())
            else:
                logits = np.log(np.maximum(prob, 1e-10)) / temperature
                keep = logits.argsort()[-top_k:]
                weights_ = np.exp(logits[keep] - logits[keep].max())
                index = int(keep[rng.choice(len(keep), p=weights_ / weights_.sum())])
            ids.append(index + 2)
        return " ".join(reverse[i] for i in ids[len(vocab.encode(prompt)) :])

    prompts = ["shall i compare", "when i have seen", "my mistress", "love is not"]
    greedy = {p: generate(model, vocab, p, 20) for p in prompts}
    sampled = {p: sample(p, 40, 0.8, 12, 42) for p in prompts}

    def distinct_ratio(text):
        words = text.split()
        return len(set(words)) / max(1, len(words))

    train_text = " ".join(training)

    padded_train = " " + train_text + " "

    def longest_shared(text):
        words = text.split()
        for n in range(len(words), 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " " + " ".join(words[i : i + n]) + " "
                if phrase in padded_train:
                    return n
        return 0

    generation = {
        "greedy_20_tokens": greedy,
        "sampled_temperature0.8_top12_seed42": sampled,
        "greedy_distinct_word_ratio": {p: distinct_ratio(g) for p, g in greedy.items()},
        "sampled_distinct_word_ratio": {
            p: distinct_ratio(s) for p, s in sampled.items()
        },
        "longest_shared_train_word_run": {
            p: longest_shared(s) for p, s in {**greedy, **sampled}.items()
        },
    }
    report = {
        "source": "https://www.gutenberg.org/ebooks/1041",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model": (
            "Original-path word-level BiLSTM, full default 100/150/100 "
            "architecture, sparse targets, train-only vocabulary."
        ),
        "train_lines": len(training),
        "dev_lines": len(dev),
        "test_lines": len(test),
        "selection": (
            "Best development unregularized cross-entropy checkpoint with "
            f"patience {args.patience}; epoch {best_epoch} selected before "
            "the single test read below."
        ),
        "history": history,
        "selected_epoch": best_epoch,
        "development_perplexity": math.exp(best),
        "test_next_token_accuracy": float(test_acc),
        "test_crossentropy_excluding_regularization": test_nll,
        "test_perplexity": math.exp(test_nll),
        "test_oov_targets": int((ty == 1).sum()),
        "test_prefixes": len(ty),
        "generation": generation,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Second read of this exposed test slice, not an untouched "
            "benchmark. Chronological line slices can retain near-repeated "
            "phrases across splits and are not sonnet-disjoint. Word "
            "perplexity is not comparable to character cross-entropy and "
            "neither measures poetic quality, meter or rhyme. Memorization "
            "check is limited to exact shared word runs against training "
            "lines. Not production poetry."
        ),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k not in {"history", "generation"}}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
