"""Small local character GRU experiment. No import-time training or downloads."""

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path


def sonnets(path):
    text = path.read_text(encoding="utf-8-sig")
    body = text.split("THE SONNETS", 1)[1].split("*** END OF THIS PROJECT", 1)[0]
    parts = re.split(r"(?m)^\s*([IVXLCDM]+)\s*$", body)
    rows = []
    for i in range(1, len(parts), 2):
        lines = [s.strip().lower() for s in parts[i + 1].splitlines() if s.strip()]
        if not lines:
            raise ValueError("Empty sonnet")
        rows.append({"id": parts[i], "text": "\n".join(lines) + "\n\n"})
    if len(rows) != 154:
        raise ValueError(f"Expected154sonnets, got{len(rows)}")
    return rows


def run(source, out, epochs=20):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    if out.exists():
        raise FileExistsError("Output exists. Choose a new directory.")
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    rows = sonnets(source)
    splits = {"train": rows[:120], "dev": rows[120:137], "test": rows[137:]}
    texts = {k: "".join(r["text"] for r in v) for k, v in splits.items()}
    chars = ["<unk>"] + sorted(set(texts["train"]))
    ids = {c: i for i, c in enumerate(chars)}
    size = len(chars)

    def encode(t):
        return [ids.get(c, 0) for c in t]

    def examples(key, stride):
        x = []
        y = []
        for r in splits[key]:
            seq = encode(r["text"])
            for i in range(0, len(seq) - 81, stride):
                x.append(seq[i : i + 80])
                y.append(seq[i + 1 : i + 81])
        return torch.tensor(x), torch.tensor(y)

    x, y = examples("train", 40)
    dx, dy = examples("dev", 80)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.Embedding(size, 48)
            self.r = nn.GRU(48, 128, batch_first=True)
            self.f = nn.Linear(128, size)

        def forward(self, x, h=None):
            z, h = self.r(self.e(x), h)
            return self.f(z), h

    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    loss = nn.CrossEntropyLoss()
    history = []
    best = math.inf

    def evaluate(a, b):
        m.eval()
        tot = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for j in range(0, len(a), 64):
                z, _ = m(a[j : j + 64])
                v = loss(z.reshape(-1, size), b[j : j + 64].reshape(-1))
                n = b[j : j + 64].numel()
                tot += float(v) * n
                count += n
                correct += int((z.argmax(-1) == b[j : j + 64]).sum())
        return {
            "cross_entropy": tot / count,
            "accuracy": correct / count,
            "targets": count,
        }

    for ep in range(1, epochs + 1):
        m.train()
        for ix in torch.randperm(len(x)).split(64):
            opt.zero_grad()
            z, _ = m(x[ix])
            step_loss = loss(z.reshape(-1, size), y[ix].reshape(-1))
            step_loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1)
            opt.step()
        dev = evaluate(dx, dy)
        history.append({"epoch": ep, **dev})
        if dev["cross_entropy"] < best:
            best = dev["cross_entropy"]
            torch.save(m.state_dict(), out / "best.pt")
        print(history[-1], flush=True)
    m.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    tx, ty = examples("test", 80)
    test = evaluate(tx, ty)
    counts = np.bincount(encode(texts["train"]), minlength=size) + 1
    probs = counts / counts.sum()
    test["unigram_cross_entropy"] = float(-np.log(probs[ty.numpy()]).mean())
    test["bits_per_char"] = test["cross_entropy"] / math.log(2)

    def generate(prompt, temperature, seed):
        g = torch.Generator().manual_seed(seed)
        seq = encode(prompt)
        out_text = prompt
        m.eval()
        with torch.no_grad():
            z, h = m(torch.tensor([seq]))
            logits = z[0, -1]
            for _ in range(400):
                logits = logits / temperature
                logits[0] = -1e9
                values, indices = torch.topk(logits, 12)
                choice = indices[
                    torch.multinomial(torch.softmax(values, 0), 1, generator=g)
                ].item()
                out_text += chars[choice]
                z, h = m(torch.tensor([[choice]]), h)
                logits = z[0, -1]
        return out_text

    samples = [
        {
            "prompt": p,
            "temperature": 0.8,
            "seed": 100 + i,
            "text": generate(p, 0.8, 100 + i),
        }
        for i, p in enumerate(
            ["shall i compare", "when winter comes", "my love is", "the silent night"]
        )
    ]
    report = {
        "source": "https://www.gutenberg.org/ebooks/1041",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split_sonnets": {k: len(v) for k, v in splits.items()},
        "split_ids": {k: [r["id"] for r in v] for k, v in splits.items()},
        "vocabulary": chars,
        "train_windows": len(x),
        "context": 80,
        "history": history,
        "test": test,
        "samples": samples,
        "elapsed_seconds": time.time() - start,
        "limits": [
            "Single-author small corpus, exposed in earlier word-model experiments.",
            "Sonnet-disjoint split, not a new untouched external benchmark.",
            "Teacher-forced character accuracy is not poetry quality.",
            "No meter/rhyme/meaning or memorization guarantees.",
            "Lowercase and whitespace cleanup; source header excluded.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["history", "split_ids", "vocabulary", "samples"]
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args()
    run(a.source, a.output, a.epochs)
