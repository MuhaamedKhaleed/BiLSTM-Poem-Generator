"""DistilGPT2 sonnet fine-tune pilot. No import-time training or downloads.

Replacement route, distinct from the original from-scratch BiLSTM: a small
public pretrained language model (distilbert/distilgpt2, Apache-2.0) is
fine-tuned on the same public-domain sonnet source and same 120/17/17
sonnet-level split as scripts/char_poetry.py. Runs on one CPU. Default
--freeze-bottom 3 trains the top three transformer blocks in
about 1 GB on one CPU; --freeze-bottom 0 trains all blocks and needs
roughly 2 GB.
"""

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

PROMPTS = [
    "the quiet river knows my name",
    "when winter steals the golden leaves",
    "thy voice is softer than the rain",
    "i dreamed of thee beneath the moon",
]


def sonnets(path):
    """Same source parsing and 120/17/17 sonnet-level split as char_poetry.py."""
    text = path.read_text(encoding="utf-8-sig")
    body = text.split("THE SONNETS", 1)[1].split("*** END OF THIS PROJECT", 1)[0]
    parts = re.split(r"(?m)^\s*([IVXLCDM]+)\s*$", body)
    rows = []
    for i in range(1, len(parts), 2):
        lines = [s.strip().lower() for s in parts[i + 1].splitlines() if s.strip()]
        if not lines:
            raise ValueError("Empty sonnet")
        rows.append("\n".join(lines) + "\n\n")
    if len(rows) != 154:
        raise ValueError(f"Expected 154 sonnets, got {len(rows)}")
    return {"train": rows[:120], "dev": rows[120:137], "test": rows[137:]}


def longest_shared(text, corpus):
    """Longest common substring length between text and corpus (characters)."""
    for length in range(len(text), 0, -1):
        for start in range(0, len(text) - length + 1):
            if text[start : start + length] in corpus:
                return length
    return 0


def run(source, model_dir, out, epochs, freeze_bottom, block, batch, lr, seed):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    if out.exists():
        raise FileExistsError("Output exists. Choose a new directory.")
    import torch
    import transformers
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()

    splits = sonnets(source)
    tokenizer = GPT2TokenizerFast.from_pretrained(str(model_dir))
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.eval()
    train_text = "".join(splits["train"])

    # Base-model samples first, while the loaded model is still the public
    # checkpoint: avoids holding two 330 MB models in memory at once.
    def sample(net, prompt, gen_seed):
        ids = tokenizer.encode(prompt, return_tensors="pt")
        torch.manual_seed(gen_seed)
        generated = net.generate(
            ids,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_k=40,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(generated[0][ids.shape[1] :])

    base_samples = [
        sample(model, prompt, seed * 100 + index)
        for index, prompt in enumerate(PROMPTS)
    ]
    torch.manual_seed(seed)
    random.seed(seed)

    def blocks(key):
        ids = []
        for sonnet in splits[key]:
            ids.extend(tokenizer.encode(sonnet))
        cut = len(ids) - (len(ids) % block)
        return [ids[i : i + block] for i in range(0, cut, block)]

    data = {key: blocks(key) for key in splits}

    def evaluate(key):
        model.eval()
        total = 0.0
        with torch.no_grad():
            rows = data[key]
            for i in range(0, len(rows), 8):
                chunk = torch.tensor(rows[i : i + 8])
                total += float(model(chunk, labels=chunk).loss) * len(chunk)
        model.train()
        return total / len(rows)

    base_test_loss = evaluate("test")

    trainable_names = [f"transformer.h.{i}." for i in range(freeze_bottom, 6)]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(p) for p in trainable_names) or (
            name.startswith("transformer.ln_f.")
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    model.train()
    history = []
    best_loss = None
    for epoch in range(epochs):
        order = list(range(len(data["train"])))
        random.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = torch.tensor([data["train"][j] for j in order[i : i + batch]])
            loss = model(chunk, labels=chunk).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
        dev_loss = evaluate("dev")
        history.append({"epoch": epoch + 1, "dev_loss": dev_loss})
        if best_loss is None or dev_loss < best_loss:
            best_loss = dev_loss
            model.save_pretrained(out / "best-model")
            tokenizer.save_pretrained(out / "best-model")

    del model
    model = GPT2LMHeadModel.from_pretrained(str(out / "best-model"))
    model.eval()
    test_loss = evaluate("test")
    model.eval()

    samples = []
    for index, prompt in enumerate(PROMPTS):
        text = sample(model, prompt, seed * 100 + index)
        samples.append(
            {
                "prompt": prompt,
                "base": base_samples[index],
                "base_shared_with_train_chars": longest_shared(
                    base_samples[index], train_text
                ),
                "finetuned": text,
                "finetuned_shared_with_train_chars": longest_shared(text, train_text),
            }
        )

    report = {
        "route": (
            "pretrained distilgpt2 fine-tune (replacement for from-scratch BiLSTM)"
        ),
        "model": "distilbert/distilgpt2",
        "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        "model_license": "apache-2.0",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": "154 sonnets: 120 train / 17 dev / 17 test, sonnet-level, no crossing",
        "arguments": {
            "epochs": epochs,
            "freeze_bottom": freeze_bottom,
            "block": block,
            "batch": batch,
            "lr": lr,
            "seed": seed,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_blocks": len(data["train"]),
        "dev_blocks": len(data["dev"]),
        "test_blocks": len(data["test"]),
        "epoch_dev_losses": history,
        "best_dev_loss": best_loss,
        "base_test_loss": base_test_loss,
        "test_loss": test_loss,
        "test_perplexity": float(__import__("math").exp(min(test_loss, 20))),
        "generation": {
            "max_new_tokens": 80,
            "temperature": 0.9,
            "top_k": 40,
            "no_repeat_ngram_size": 3,
            "seed_rule": "seed*100+prompt_index",
        },
        "samples": samples,
        "wall_seconds": round(time.time() - start, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "test_loss": test_loss,
                "best_dev_loss": best_loss,
                "wall_seconds": report["wall_seconds"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local 1041.txt")
    parser.add_argument(
        "--model", required=True, type=Path, help="Local model directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New output directory"
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--freeze-bottom", type=int, default=3, choices=range(0, 6))
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        args.source,
        args.model,
        args.output,
        args.epochs,
        args.freeze_bottom,
        args.block,
        args.batch,
        args.lr,
        args.seed,
    )


if __name__ == "__main__":
    main()
