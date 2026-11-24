# Measured no-cost CPU quality experiment

1200train/200dev/200heldout lines beforeprefixes; larger64/64/32model. Moreepochs worseneddevloss7.04to8.46, earliestbestselected. Testaccuracy.02373/perplexity1115,343OOVof1433targets; output repeats of. Original20epoch attempt executiontimeout after10+, savedbestrecovered. Repro script caps10epochs, keepsbestdevcheckpoint. No sonnet/author-disjoint oraestheticquality claim.

Run only explicitly, with previously prepared local permitted data and a new output. No automatic downloads or import-time execution. Exact measured result is quality-output.json. This is a reproducible development experiment, not a productionquality guarantee.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/quality_cpu.py --data LOCAL_PREPARED_SOURCE --output NEW_OUTPUT
```

Test sets described as finalized have now been observed. Future choices must not be justified with their scores as if they remained unseen. Retain source terms/attribution from real-data.md. No model or raw text/photos committed.
