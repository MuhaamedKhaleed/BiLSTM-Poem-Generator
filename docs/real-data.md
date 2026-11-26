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

## Docker without a registry account

This repository supplies a Dockerfile, not a published registry image. Build locally once. The default remains the small invented-input wiring demo. No account, paid service or automatic source download is required by the image.

```sh
docker build -f Dockerfile.ml -t project-ml .
docker run --rm --network none project-ml
```

The image also includes `/app/scripts`. After explicit source preparation on the host, mount a new writable working directory and invoke the real script with `--entrypoint python`. Keep model/source mounts read-only when the script does not write beside them. The image uses uid10001, so grant that uid write access to the output folder, do not run the container as root to bypass this. Data/preprocessing scripts currently place artifacts beside source files; mount the working copy read-write and keep your original untouched. The environment disables Hugging Face downloads. Network-none is suitable only after sources/models are local; an explicit fetch helper needs network by design. CPU/dependency builds were checked on GitHub's Linux runner, not every laptop OS.
