# Full-training word-level original-path follow-up

25 November 2026. Development comparison on an exposed test, not a benchmark.

The recorded word-level BiLSTM baseline (docs/quality-cpu.md) used a reduced
64/64/32 architecture and about ten epochs; its greedy sample collapsed into
repeated words. This follow-up reruns the original word-level path with the
package's full default architecture (embedding 100, BiLSTM 150, LSTM 100) and
development-checkpoint selection, then characterizes generation the same way
the character-level pilot did. The question: was the collapse an artifact of
the reduced model and short training?

## Actual measured results

Same chronological unique-line recipe as before: first 1200 train lines,
next 200 development, next 200 test; train-only vocabulary, sparse targets.

- Best development checkpoint was epoch 1 of 40 (patience 8 stopped the run
  at epoch 9): development perplexity 1023.15 versus the reduced run's
  recorded 1102.84. From epoch 2 onward the development cross-entropy rose
  monotonically (1464, 1864, 2709, ... 4780 by epoch 9): the larger model
  overfits the 1200-line corpus almost immediately, and longer training makes
  held-out prediction worse, not better.
- Single test read after selection (second exposure of this slice):
  perplexity 1039.48, next-token accuracy .0244, 343 OOV targets of 1433
  prefixes. Reduced run recorded 1114.61 / .0237 on the same slice. The gain
  is small and neither run produces usable language modeling.
- Generation still collapses. Greedy 20-token continuations of four fixed
  prompts are function-word loops ("the of the of the ..."), distinct-word
  ratios .18-.25. Temperature .8 top-12 sampling (fixed seed 42) removes the
  exact loop but only shuffles the same high-frequency words; samples remain
  meaningless. No sample shows meter, rhyme or coherent meaning.
- Memorization stays trivial: the longest exact word-run shared with the
  training lines is 2 words for every sample.

Conclusion: this full-architecture run with checkpoint selection did not fix
word-level collapse. Perplexity is marginally lower but generation is still
unusable. This bounded result does not rule out every other training choice. This is
direct evidence for keeping the documented original-path limitation, and it
contextualizes - but does not validate the quality of - the separate
character-level pilot, which uses a different modeling unit; scores are not
comparable across units.

Exact numbers, per-epoch history and all samples: `word-pilot-full.json`.

## Reproduce

Python 3.11, TensorFlow CPU 2.18.0, NumPy 1.26.4, locked environment. Local
source text only, no download at import or test time, new output directory
required. ~9 minutes on one CPU core, peak RSS ~840 MB.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/word_pilot_full.py --data LOCAL_SOURCE --output NEW_DIRECTORY
python -m pytest tests/test_word_pilot_full.py
```

Source catalog: https://www.gutenberg.org/ebooks/1041 (public domain in the
USA per the catalog; local-law and Gutenberg trademark conditions still
apply; corpus not redistributed). Source SHA-256 is recorded in the JSON.
No model weights are committed. Second read of the exposed test slice; do
not cite these test numbers as unseen. Not production poetry, no aesthetic
quality claim.
