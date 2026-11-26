"""Explicit local development experiment. No downloads or import-time work."""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory.")
    """Measured train-only n-gram alternative. Dev only, no test access."""
    import collections
    import json
    import math
    import resource
    import time
    from pathlib import Path

    out = args.output
    out.mkdir(exist_ok=False)
    text = (args.data / "1041.txt").read_text(encoding="utf-8-sig")
    body = text.split("THE SONNETS", 1)[1].split("*** END OF THIS PROJECT", 1)[0]
    lines = list(
        dict.fromkeys(
            x.strip().lower()
            for x in body.splitlines()
            if len(x.split()) >= 4 and not x.strip().isupper()
        )
    )
    train = lines[:1200]
    dev = lines[1200:1400]
    vocab = sorted(set(" ".join(train).split()))
    ids = {w: i + 1 for i, w in enumerate(vocab)}
    V = len(ids) + 1

    def encode(line):
        return [ids.get(w, 0) for w in line.split()]

    counts = [collections.defaultdict(collections.Counter) for _ in range(3)]
    for line in train:
        a = encode(line)
        for i in range(1, len(a)):
            for n in range(3):
                if i >= n:
                    counts[n][tuple(a[i - n : i])][a[i]] += 1
    start = time.time()
    trials = []
    for alpha in [0.01, 0.1, 1.0]:
        for mix in [(1.0, 0.0, 0.0), (0.2, 0.8, 0.0), (0.2, 0.4, 0.4), (0.1, 0.2, 0.7)]:
            loss = correct = total = oov = 0
            for line in dev:
                a = encode(line)
                for i in range(1, len(a)):
                    p = [0.0] * V
                    for n, w in enumerate(mix):
                        if not w:
                            continue
                        c = (
                            counts[n].get(tuple(a[max(0, i - n) : i]), {})
                            if i >= n
                            else {}
                        )
                        denom = sum(c.values()) + alpha * V
                        for j in range(V):
                            p[j] += w * (c.get(j, 0) + alpha) / denom
                    loss -= math.log(p[a[i]])
                    correct += int(max(range(V), key=p.__getitem__) == a[i])
                    total += 1
                    oov += a[i] == 0
            r = dict(
                alpha=alpha,
                mix=mix,
                dev_perplexity=math.exp(loss / total),
                dev_next_token_accuracy=correct / total,
                dev_oov_targets=oov,
                dev_targets=total,
            )
            trials.append(r)
            print(r, flush=True)
    best = min(trials, key=lambda r: r["dev_perplexity"])
    r = dict(
        model="explicit interpolated word ngram alternative, not BiLSTM",
        train_lines=len(train),
        dev_lines=len(dev),
        vocabulary=V,
        selection="12 configurations selected on dev only; no test read",
        trials=trials,
        selected=best,
        elapsed_seconds=time.time() - start,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limits=(
            "Development comparison only, same tokenization/train vocabulary "
            "and noninitial targets as old recipe; word perplexity not "
            "aesthetic quality; old test is exposed "
        ),
    )
    (out / "report.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
