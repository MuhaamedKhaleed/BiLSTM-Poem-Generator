# Verification and phase record

Python 3.11.18/Linux/local CPU. No external corpus or pretrained model download. Original isolated under legacy in a separate commit. Correction: old script trained on every prefix then reported accuracy on the same data; new implementation splits normalized unique lines before prefixes and labels metrics as validation, not poetry quality. Unknown prompts keep OOV positions; reserved output IDs are excluded.

1. Package: editable src layout and explicit module/root/console CLI help.
2. Cleanup: corpus/split/vocab/prefix core, optional model adapter and thin CLI. Sparse target labels, no one-hot allocation.
3. Validation: missing/empty/single-token corpus, invalid fraction/maps/context, overlap, existing output, invalid prompt/count/model output. JSON separate from stderr, exit 2.
4. Config: explicit dotenv POEM_CORPUS_PATH with data precedence, placeholders only.
5. Runtime/dev/optional ML separate, locked uv install. TensorFlow CPU 2.18.0/NumPy 1.26.4 actually exercised.
6. 29 tests, 97% software statement coverage locally. Actual TensorFlow forward/loss, tiny synthetic epoch/artifact round-trip and CLI generation, not mocks alone.
7. Ruff lint/format and strict mypy, tracked-file pre-commit pass.
8. Public-only standard free CI, no cache/artifacts/model publication. Tests/quality/hooks/CLI and container smoke.
9. Nonroot multi-stage data-only Docker, no ML runtime. Local Docker unavailable, hosted outcome separately verified.
10. check-data/train/generate, training is explicit, local files only, no overwrite.
11. README has real data-check JSON, architecture, behavior changes and limits.
12. Feature branch, baseline and corrective commits. No merge, PR, tag, release or license change.
13. Suggested GitHub copy prepared but not applied.

Verification commands: uv sync --locked --extra dev --extra ml; poem-generator --help; python -m poem_generator --help; python main.py --help; python -m pytest --cov=poem_generator --cov-report=term-missing; python -m ruff check src tests; python -m ruff format --check src tests; python -m mypy src; python -m pre_commit run --all-files. Tests constrain CPU thread counts for this environment. README data-check path uses an invented corpus, not a user's text. No model/poem quality score is retained as evidence.

Behavior changes: whitespace tokenizer preserves punctuation instead of Keras filtering, OOV reserved, split before prefixes, sparse loss, reserved generation IDs disallowed, no implicit training. Core architecture defaults retained, embedding remains unmasked like original. Old tokenizer/model files incompatible. TF seed does not guarantee identical devices or later-library runs; setting PYTHONHASHSEED after startup was removed rather than claimed effective.

Weaknesses: no poem/source-level independence, untouched test set, real corpus, pretrained weights, human quality evaluation, copyright review, memory/runtime benchmark, continuation plagiarism audit or deployment. Validation loss includes regularization and OOV targets, it is not pure cross entropy/perplexity. Safe mode does not justify loading unknown Keras files. Expected local errors are caught, arbitrary TensorFlow hardware faults are not hidden.
