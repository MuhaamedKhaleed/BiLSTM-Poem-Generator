# Pretrained distilgpt2 sonnet fine-tune pilot

25 November 2026. Research pilot, not production poetry. This is a
**replacement route**, clearly separate from the original from-scratch
BiLSTM: the BiLSTM is unchanged and remains the repository's model. This
pilot downloads a small public pretrained language model and fine-tunes it
on the same public-domain source as the character pilot.

## Route

- Model: `distilbert/distilgpt2`, revision
  `2290a62682d06624634c1f46a6ad5be0f47f38aa`, Apache-2.0 per the model card
  front matter and the HF API license field at fetch time (25 November
  2026). `scripts/fetch_pretrained.py` pins every needed file by SHA-256;
  the served bytes were verified against Hugging Face metadata (safetensors
  sha256 equals the `x-linked-etag`; small files' git blob SHA-1 equals the
  API blobId).
- Source: the same pinned GITenberg `1041.txt` as
  `scripts/fetch_real_data.py` (public domain USA per the catalog; local
  law and Gutenberg terms still apply, corpus not redistributed).
- Split: identical to the character pilot — 154 sonnets, first 120 train,
  next 17 development, last 17 test, no sonnet crossing splits. Earlier
  experiments already exposed parts of this source, so this is not an
  untouched benchmark.
- Fine-tune: causal LM blocks of 128 tokens, AdamW, grad clip 1.0,
  checkpoint selected by development loss, then the test loss is read once.
  Default `--freeze-bottom 4` trains only the top two transformer blocks
  plus the final norm so the run fits in about 1 GB RAM on one CPU;
  `--freeze-bottom 0` trains everything and needs about 2 GB.

## Reproduce

Python 3.12, torch 2.4.1+cpu, transformers 4.44.2, numpy 1.26.4. Install
dependencies locally, not remote code:

```sh
uv sync --locked --extra dev
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers==4.44.2 numpy==1.26.4
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/fetch_pretrained.py --output ./pretrained
uv run --no-sync python scripts/pretrained_pilot.py \
  --source ./real-source/1041.txt --model ./pretrained --output ./pretrained-out
```

The output directory contains `best-model/` (fine-tuned weights, not
committed) and `report.json` (arguments, versions, losses, fixed-seed
samples, memorization check).

## Measured result

Two configurations were run; the second was chosen after inspecting the
first run's test samples, so the test split is **touched**: config B's test
numbers are a touched-test readout, not a clean held-out estimate. No
further tuning was done after config B.

- Config A (first run): epochs 3, `--freeze-bottom 4` (top 2 blocks), no
  ngram guard. Test loss 4.676. Fixed-seed samples shifted to sonnet
  register but 2 of 4 degraded into repetition loops.
- Config B (shipped default): epochs 4, `--freeze-bottom 3` (top 3 blocks,
  21,265,152 of 81,912,576 parameters trainable), generation adds
  `no_repeat_ngram_size=3`. Development loss per epoch: 4.805, 4.758,
  4.719, 4.737; checkpoint from epoch 3. Test loss 4.619 (touched, see
  above) versus base distilgpt2 5.186 on the exact same 22 test blocks — a
  paired same-split comparison, not a benchmark. Wall time 280 seconds on
  one CPU; memory stayed under 1.5 GB.
- Samples (4 fixed prompts, fixed seeds, single draws, in
  `report.json`): no repetition loops; output keeps line breaks and
  sonnet register where the base model writes generic prose. Meaning is
  still weak, with occasional non-words ("fairtarted", "fairts") and
  mid-word truncation at 80 tokens. These are not good finished poems.
- Memorization screen: longest shared substring between each completion
  and the training text is 13-18 characters (character pilot: 13-17). Not
  a guarantee of originality.

The full numerical record is `report.json` from the reproduction commands
above; trained weights land in `pretrained-out/best-model/` and are not
committed.

## Limits

- The fine-tuned model continues in a sonnet-like register because the
  pretrained model is already fluent; this route does not show the
  repository's own architecture learned anything. That is why it is a
  documented replacement, not an improvement claim for the BiLSTM.
- Generation still uses sampling without rhyme, meter or repetition
  controls. Samples are fixed-seed single draws, not best-of selections.
- The memorization check is a longest-shared-substring screen against the
  training text, not a guarantee of originality.
- Model weights are not committed; every run re-derives them from the
  pinned public files.
