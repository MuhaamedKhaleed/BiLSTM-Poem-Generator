"""Explicit pretrained download. No import-time network or weights redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

# distilbert/distilgpt2 at the pinned revision. License: Apache-2.0 per the model
# card front matter and the Hugging Face API license field at fetch time
# (25 November 2026). Served bytes verified against Hugging Face metadata:
# model.safetensors sha256 equals the x-linked-etag at the pinned revision;
# every small file's git blob SHA-1 equals the API blobId at that revision.
REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
BASE = f"https://huggingface.co/distilbert/distilgpt2/resolve/{REVISION}"

SOURCES = [
    (
        "config.json",
        f"{BASE}/config.json",
        "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a",
    ),
    (
        "generation_config.json",
        f"{BASE}/generation_config.json",
        "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
    ),
    (
        "merges.txt",
        f"{BASE}/merges.txt",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    (
        "tokenizer.json",
        f"{BASE}/tokenizer.json",
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    ),
    (
        "tokenizer_config.json",
        f"{BASE}/tokenizer_config.json",
        "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    ),
    (
        "vocab.json",
        f"{BASE}/vocab.json",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
    (
        "model.safetensors",
        f"{BASE}/model.safetensors",
        "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New model directory"
    )
    parser.add_argument(
        "--cached", type=Path, help="Use already downloaded files, without network"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=600) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "pretrained-manifest.json").write_text(
        json.dumps(
            {
                "model": "distilbert/distilgpt2",
                "revision": REVISION,
                "license": "apache-2.0",
                "license_source": (
                    "model card front matter and HF API license field, 2026-11-25"
                ),
                "files": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
