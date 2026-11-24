"Explicit measured CPU quality experiment. No import-time work or downloads."


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, help="Translation only: trusted local Marian"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    import os

    for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"]:
        os.environ[k] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import json
    import math
    import resource
    import time
    from pathlib import Path

    import numpy as np

    from poem_generator.core import Vocabulary, prefixes
    from poem_generator.model import backend, build_model, generate

    root = Path(str(args.output))
    root.mkdir(exist_ok=False)
    text = (args.data / "1041.txt").read_text(encoding="utf-8-sig")
    body = text.split("THE SONNETS", 1)[1].split("*** END OF THIS PROJECT", 1)[0]
    lines = list(
        dict.fromkeys(
            line.strip().lower()
            for line in body.splitlines()
            if len(line.split()) >= 4 and not line.strip().isupper()
        )
    )
    training = lines[:1200]
    dev = lines[1200:1400]
    test = lines[1400:1600]
    vocab = Vocabulary.fit(training)
    x, y = map(np.array, prefixes(training, vocab))
    vx, vy = map(np.array, prefixes(dev, vocab))
    tf = backend()
    tf.keras.utils.set_random_seed(42)
    model = build_model(vocab, 64, 64, 32)
    start = time.time()
    best = 1e9
    history = []
    for epoch in range(10):
        model.fit(x, y, batch_size=64, epochs=1, verbose=0)
        loss, acc = model.evaluate(vx, vy, verbose=0)
        row = {
            "epoch": epoch + 1,
            "development_regularized_loss": float(loss),
            "development_accuracy": float(acc),
        }
        history.append(row)
        if loss < best:
            best = loss
            model.save(root / "best.keras")
        if epoch in [0, 4, 9]:
            print(row, flush=True)
    model = tf.keras.models.load_model(root / "best.keras")
    tx, ty = map(np.array, prefixes(test, vocab))
    loss, acc = model.evaluate(tx, ty, verbose=0)
    prob = model(tx, training=False).numpy()
    nll = float(-np.log(np.maximum(prob[np.arange(len(ty)), ty], 1e-10)).mean())
    r = {
        "source": "https://www.gutenberg.org/ebooks/1041",
        "train_lines": len(training),
        "dev_lines": len(dev),
        "test_lines": len(test),
        "history": history,
        "test_next_token_accuracy": float(acc),
        "test_crossentropy_excluding_regularization": nll,
        "test_perplexity": math.exp(nll),
        "test_oov_targets": int((ty == 1).sum()),
        "test_prefixes": len(ty),
        "generated": generate(model, vocab, "shall i compare", 12),
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Line-disjoint chronological source slices, near-"
            "repeatedphrases can remain, not sonnet/author-disjoint. "
            "Heldout test read once after devcheckpointselection; poem "
            "aesthetic quality not measured."
        ),
    }
    (root / "report.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "history"}))


if __name__ == "__main__":
    main()
