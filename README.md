# Local BiLSTM Next-Word Generator

Train a small BiLSTM on your own text lines and generate a greedy continuation. Complete unique lines are split before prefix construction, so validation does not re-score training prefixes. Original architecture preserved: embedding(100), bidirectional LSTM(150), dropout(0.2), LSTM(100), L2 dense, softmax with sparse labels.

## Results (honest)

- Fresh-clone CPU run on public Shakespeare sonnets, 20 epochs (~2 minutes on one core): "shall i compare" -> "shall i compare and the love of love of love love". Repetitive; the small model is weak.
- A separate pinned DistilGPT2 fine-tune route (`docs/pretrained-pilot.md`, Apache-2.0 model, same sonnet source/split) gives usable output from a fresh clone. It is a replacement route, not a fix for the BiLSTM.
- The original path's generation collapse persists under the full default architecture with checkpoint selection, measured on the already-exposed test slice (`docs/word-pilot-full.md`).
- 29 offline tests, 97% statement coverage (forward shape, sparse loss, tiny synthetic epoch, Keras save/load, CLI generation). No poetry-quality evaluation has been done.

## Quickstart (fresh clone, about 1 minute)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source          # Shakespeare sonnets text
uv run --no-sync python scripts/real_data.py --data ./real-source --epochs 2
uv run --no-sync poem-generator generate ./real-source/model-real 'shall i compare' --next-words 8
```

Corpus format: UTF-8, one text line per example. `poem-generator check-data --data YOUR.txt` validates and reports line/word counts. Training is explicit: `poem-generator train CORPUS OUTPUT --epochs 150`. Use a new `--data`/output folder per run.

## Limits

- No trained quality, external benchmark, untouched test corpus or human review. Random line split is not poem/source independence; similar lines can cross splits.
- Greedy decoding only: no sampling/temperature/top-k, rhyme or repetition controls.
- Tokenization changed deliberately (punctuation stays attached to tokens), so old weights/tokenizers are incompatible.
- Verified on Linux/Python 3.11; the container runs the data-check CLI only (no TensorFlow). Free hosted CI covers tests, lint, CLI and Docker build.
- Corpus rights and memorization risk need review before real training or publication. No corpus or model is redistributed.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). Historical code, third-party models and corpora keep their own terms. Details and evidence: `docs/verification.md`, `docs/real-data.md`, `docs/quality-cpu.md`, `CHANGELOG.md`.
