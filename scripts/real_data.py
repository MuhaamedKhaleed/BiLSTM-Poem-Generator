"""Explicit real-data CPU development recipe. Never runs on import."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared local source directory")
    parser.add_argument(
        "--base-model", help="Trusted compatible local DistilBERT directory"
    )
    parser.add_argument(
        "--manifest", help="Reviewed selected Open Images source manifest"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs (default: the small recorded run)")
    args = parser.parse_args()
    import hashlib
    import json
    import os
    import pathlib
    import time

    for k, v in {
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }.items():
        os.environ[k] = v
    from poem_generator.model import load_generate, train

    root = pathlib.Path(args.data)
    text = (root / "1041.txt").read_text(encoding="utf-8-sig")
    body = text.split("THE SONNETS", 1)[1].split("*** END OF THIS PROJECT", 1)[0]
    lines = list(
        dict.fromkeys(
            line.strip().lower()
            for line in body.splitlines()
            if len(line.split()) >= 4 and not line.strip().isupper()
        )
    )
    # Contiguous held-out tail; complete poetic lines before prefix expansion.
    training = lines[:512]
    validation = lines[512:640]
    assert len(training) == 512 and len(validation) == 128
    start = time.time()
    scores = train(
        training,
        validation,
        root / "model-real",
        epochs=args.epochs,
        batch_size=16,
        embedding=16,
        units=16,
        second_units=8,
    )
    generated = load_generate(root / "model-real", "shall i compare", 8)
    report = {
        "real_data": True,
        "source": "https://www.gutenberg.org/ebooks/1041",
        "mirror": "https://github.com/GITenberg/Shakespeare-s-Sonnets_1041",
        "corpus": "Shakespeare Sonnets, first512 train lines / next128 dev lines",
        "source_sha256": hashlib.sha256((root / "1041.txt").read_bytes()).hexdigest(),
        "rights": (
            "Source catalog: public domain USA. Gutenberg "
            "trademark/license and local-law cautions remain; no "
            "corpus/model redistributed."
        ),
        "training": scores,
        "elapsed_seconds": round(time.time() - start, 2),
        "sample": generated,
        "limits": (
            "2 CPU epochs small model; no poetry quality, memorization "
            "or final-test claim"
        ),
    }
    (root / "real-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
