# Changelog

## Unreleased

### Fixed
- Split unique normalized lines before prefix generation, replacing training-set evaluation with labeled validation metrics.
- Preserve OOV prompt positions and exclude reserved output IDs during greedy generation.
- Require explicit training, validate local artifacts and refuse overwrite.

### Changed
- Sparse next-token labels replace one-hot matrices. Whitespace tokenization preserves punctuation; retraining is required, old artifacts incompatible.
- Historical architecture defaults retained; original script kept under legacy/.

### Added
- Locked package, explicit CLI, real synthetic backend tests, quality checks, data-only Docker, free CI and honest scope docs.


### Engineering follow-up
- Use the committed uv lock in CI and key dependency-installer caching by that lock.
- Complete copy-only profile README drafts and clarify historical versus current verification.

### Synthetic demonstration follow-up

Added standalone invented-input backend demo, checked JSON output and documented limits. Opt-in ML container added where a model backend exists; default lightweight image unchanged. Hosted container verification is required separately.
