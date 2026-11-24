# Real-data CPU development run

Source: https://www.gutenberg.org/ebooks/1041 . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Prepare `1041.txt` from the source catalog. Direct Gutenberg downloads returned503 here; the GITenberg mirror https://github.com/GITenberg/Shakespeare-s-Sonnets_1041 supplied the same work. Source says public domain USA; check local copyright and Gutenberg trademark/license conditions, do not assume worldwide clearance. Full source license is retained locally, not republished here. Script extracts poetic lines, first512 train/next128 development before prefix expansion. Two epochs with tiny layers yields repetitive output, not learned poetry quality. The exact source hash is recorded in the report.

## Actual limits

2 CPU epochs small model; no poetry quality, memorization or final-test claim

## Explicit download and preparation

The optional helper downloads only the documented primary-source files, checks SHA-256 and refuses an existing destination. It never runs during installation, import or tests. Source attribution/license cautions above still apply. No model/data is bundled. For an offline copy, add `--cached /path/to/source-files` with the same filenames; checksums are still required.

```sh
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/real_data.py --data ./real-source
```

The hashes pin the exact small development sources used here, not a permission warranty or a quality benchmark. If a source changes, the helper fails rather than silently training on different data.
