# Measured development alternative, 24 November 2026

Explicit interpolated word n-gram alternative, not BiLSTM. Same1200train/200dev lines, tokenization/vocabulary/noninitial targets. Dev perplexity962.45 vs saved neural1102.84, tokenaccuracy .05091 vs .02510.12configurations selected on dev. No new test read. Lower word perplexity does not establish good poetry. No corpus redistribution.

Only explicit local execution. No download/import side effects. Use a new output directory.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/improvement_cpu.py --data LOCAL_SOURCE --output NEW_OUTPUT
```

The portable script was rerun and reproduced the recorded metrics. Environment Python3.11 Linux x86_64 CPU. Raw data and trained weights stay outside GitHub. Original baseline and limitations remain in quality-cpu.md.

```json
{
  "model": "explicit interpolated word ngram alternative, not BiLSTM",
  "train_lines": 1200,
  "dev_lines": 200,
  "vocabulary": 2910,
  "selection": "12 configurations selected on dev only; no test read",
  "trials": [
    {
      "alpha": 0.01,
      "mix": [
        1.0,
        0.0,
        0.0
      ],
      "dev_perplexity": 2612.337506660349,
      "dev_next_token_accuracy": 0.023709902370990237,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.01,
      "mix": [
        0.2,
        0.8,
        0.0
      ],
      "dev_perplexity": 1077.906078811316,
      "dev_next_token_accuracy": 0.04184100418410042,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.01,
      "mix": [
        0.2,
        0.4,
        0.4
      ],
      "dev_perplexity": 962.44936073911,
      "dev_next_token_accuracy": 0.05090655509065551,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.01,
      "mix": [
        0.1,
        0.2,
        0.7
      ],
      "dev_perplexity": 1146.7338473130842,
      "dev_next_token_accuracy": 0.05090655509065551,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.1,
      "mix": [
        1.0,
        0.0,
        0.0
      ],
      "dev_perplexity": 1534.2489177307277,
      "dev_next_token_accuracy": 0.023709902370990237,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.1,
      "mix": [
        0.2,
        0.8,
        0.0
      ],
      "dev_perplexity": 1033.430956101194,
      "dev_next_token_accuracy": 0.044630404463040445,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.1,
      "mix": [
        0.2,
        0.4,
        0.4
      ],
      "dev_perplexity": 1090.5252694584665,
      "dev_next_token_accuracy": 0.043235704323570434,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 0.1,
      "mix": [
        0.1,
        0.2,
        0.7
      ],
      "dev_perplexity": 1359.9356821271508,
      "dev_next_token_accuracy": 0.05160390516039052,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 1.0,
      "mix": [
        1.0,
        0.0,
        0.0
      ],
      "dev_perplexity": 1036.9720332549375,
      "dev_next_token_accuracy": 0.023709902370990237,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 1.0,
      "mix": [
        0.2,
        0.8,
        0.0
      ],
      "dev_perplexity": 1320.611480158376,
      "dev_next_token_accuracy": 0.03486750348675035,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 1.0,
      "mix": [
        0.2,
        0.4,
        0.4
      ],
      "dev_perplexity": 1373.5380222096112,
      "dev_next_token_accuracy": 0.02580195258019526,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    },
    {
      "alpha": 1.0,
      "mix": [
        0.1,
        0.2,
        0.7
      ],
      "dev_perplexity": 1700.6442933172693,
      "dev_next_token_accuracy": 0.026499302649930265,
      "dev_oov_targets": 336,
      "dev_targets": 1434
    }
  ],
  "selected": {
    "alpha": 0.01,
    "mix": [
      0.2,
      0.4,
      0.4
    ],
    "dev_perplexity": 962.44936073911,
    "dev_next_token_accuracy": 0.05090655509065551,
    "dev_oov_targets": 336,
    "dev_targets": 1434
  },
  "elapsed_seconds": 15.772772789001465,
  "peak_rss_kb": 15092,
  "limits": "Development comparison only, same tokenization/train vocabulary and noninitial targets as old recipe; word perplexity not aesthetic quality; old test is exposed",
  "portable_rerun_verified": true
}
```
