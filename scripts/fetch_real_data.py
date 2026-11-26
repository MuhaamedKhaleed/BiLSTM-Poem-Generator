"""Explicit source download. No import-time network or data redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New source directory"
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
                urllib.request.urlopen(url, timeout=120) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))


SOURCES = [
    (
        "1041.txt",
        "https://raw.githubusercontent.com/GITenberg/Shakespeare-s-Sonnets_1041/b6da8a3e73e7081cf71c57e930001d0b92a8f644/1041.txt",
        "e3580b25135e2d89467dd4cef0ccdcdd5f50f8779247fc24612ef653d27dc9d5",
    )
]

if __name__ == "__main__":
    main()
