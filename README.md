# Local BiLSTM next-word generator

Train a small BiLSTM on your own local text lines and generate a greedy continuation. Complete normalized unique lines are split before making overlapping prefixes, so validation is not a re-evaluation of the training prefixes.

## Evidence, not poetry-quality claims

29 offline tests, 97% software statement coverage locally. Real TensorFlow CPU checks exercise forward shape, sparse loss, a tiny synthetic epoch, Keras save/load and explicit CLI generation. No real corpus, full training, pretrained weights or poetry-quality evaluation was used. Random synthetic output is not a demo of learned poetry.

The historical model architecture remains embedding(100), bidirectional LSTM(150), dropout(0.2), LSTM(100), L2 dense and softmax, with sparse labels instead of vocabulary-sized one-hot matrices. Padding embedding is deliberately unmasked to preserve the original architecture; no claim of padding invariance. Greedy decoding now excludes padding/OOV outputs, and unknown prompt words retain an OOV slot instead of disappearing.

## Install and verify

Python 3.11/3.12, tested on Linux 3.11. Installation needs internet, processing your files does not. No keys/accounts or model/data downloads. ML is optional; omit `--extra ml` for data checking only.

```sh
uv sync --locked --extra dev --extra ml
. .venv/bin/activate
poem-generator --help
python -m poem_generator --help
python main.py --help
python -m pytest --cov=poem_generator --cov-report=term-missing
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pre_commit run --all-files
```

Runtime: Typer/Click/python-dotenv. Optional ML: TensorFlow CPU 2.18.0, NumPy 1.26.4. Dev: pytest/cov, Ruff/mypy/pre-commit. uv lock included; other platforms are not certified.

## Corpus and CLI

UTF-8, one text line per example. Lowercase, normalize whitespace, remove blank lines, deduplicate exact normalized lines. Every nonblank line needs at least two tokens. Punctuation remains attached to whitespace tokens, unlike the old Keras punctuation filtering. This explicit tokenization change requires retraining; old weights/tokenizer are not interchangeable.

```sh
poem-generator check-data --data /tmp/poem-demo.txt
```

For five invented three-token lines with distinct vocabularies, the tested command returns:

```json
{"unique_lines": 5, "train_lines": 4, "validation_lines": 1, "training_words": 12, "context": 2, "train_prefixes": 8, "validation_prefixes": 2, "validation_oov_targets": 2}
```

`check-data --config .env` optionally reads `POEM_CORPUS_PATH`; explicit `--data` wins. No implicit .env loading. Errors go to stderr and exit 2, results are JSON stdout.

Explicit training interface: `poem-generator train CORPUS OUTPUT --epochs 150 --batch-size 32`. This can take substantial local CPU time; help/check-data/import never starts it. Output must be a new directory. The artifact contains model.keras, vocabulary.json and validation.json. Metrics are validation next-token accuracy and loss including L2 regularization, not test accuracy or pure negative log likelihood. OOV validation targets are counted, not removed. Do not infer perplexity from regularized loss.

Generation interface: `poem-generator generate ARTIFACT 'his drumming' --next-words 8`. Input prompt case is preserved in output; model context uses lowercase whitespace tokens. Context is the training maximum and older prompt tokens are truncated consistently. Greedy continuation emits exactly 1 to 1000 requested known-vocabulary words, it does not measure style, meaning or originality. Only load your own trusted local Keras artifacts. Safe mode is not a sandbox for unknown models.

Synthetic one-epoch train and generate CLI paths have been run with tiny dimensions solely as plumbing checks. Full default training has not been run. No synthetic model is published.

## Architecture

```text
UTF-8 lines -> normalize/deduplicate -> whole-line split
  -> training-only vocabulary/context -> sparse prefix/target pairs
  -> BiLSTM next-token network -> held-out-line metrics
  -> trusted local artifact -> bounded greedy continuation
```

`legacy/original.py` preserves the original auto-training code with training-set accuracy, excluded from package and CI. `main.py` is now an explicit CLI, not an automatic 150-epoch run. Bug fixes change output behavior; they are not claimed equivalent.

## Container and CI

Multi-stage slim/non-root Docker includes only the data-check CLI, no TensorFlow, dataset, weights or dev tools. ML runs locally via the optional extra. Free public-only standard hosted CI checks real synthetic ML tests, quality/hooks, CLI, Docker build and data-check smoke. No image/artifact publishing or cache charges configured. Local Docker unavailable; only a passing hosted result verifies these Docker commands:

```sh
docker build --tag poem-generator:local .
docker run --rm poem-generator:local --help
```

## Limits and next work

- No trained quality, external benchmark, untouched test corpus, human poetry review or deployment.
- Random line split is not poem/author/source independence. Similar lines can cross splits; group by poem/source and audit near-duplicates before using metrics.
- All prefixes are materialized in memory. No streaming, early stopping, resume or production-size memory/timing measurements.
- A BiLSTM over a known prefix is not future-target leakage, but neither is it a modern causal transformer. Only training-only vocabulary and split checks are established here.
- No sampling/temperature/top-k, rhyme/meter controls, repetition guard, EOS stopping or calibrated quality estimate.
- Corpus rights and model memorization risks need review before real training or publication. No corpus/model redistributed, license change, release or benchmark badge.

Demo screenshot/GIF: TODO after real-data work. See docs/verification.md and unapplied docs/presentation.md.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Real-data CPU example

See [actual results, source terms and explicit download/preparation commands](docs/real-data.md). These are small development runs, not accuracy or production guarantees. No model/data is committed.

## Measured CPU quality follow-up

See [quality experiments, results and limits](docs/quality-cpu.md). These go beyond wiring smokes; some improve measured results, while others expose overfit or a stronger simple baseline. No quality guarantee.

See [full-architecture word-level follow-up](docs/word-pilot-full.md): the original path's generation collapse persists under the full default architecture with checkpoint selection, measured on the already-exposed test slice.

## Pretrained replacement pilot

See [pretrained fine-tune route, measured results and limits](docs/pretrained-pilot.md): a pinned public distilgpt2 (Apache-2.0) fine-tuned on the same sonnet source and split, runnable on one CPU. This is a separate replacement route for someone who wants usable output from a fresh clone; it does not change or rehabilitate the BiLSTM model above.
