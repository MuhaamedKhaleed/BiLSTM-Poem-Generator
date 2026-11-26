# Character-level English poetry pilot

25 November2026. Research pilot, not production poetry.

The existing word-level model collapsed into repeated words and had many unknown target words. This experiment changes the modeling unit, not the project's language. A small character GRU learns English Shakespeare sonnet text from scratch on CPU. No pretrained model, paid API, web upload or GPU service.

## Evidence

-154sonnets parsed with their Roman numeral IDs. First120train, next17development, last17test. Sonnet groups do not cross splits. Earlier experiments already exposed portions of this same source, so this is not an untouched external benchmark.
-48-dimensional embedding,128-unit single GRU, linear output; context80characters, training stride40, evaluation stride80. Only complete context windows are evaluated, up to80trailing characters per sonnet omitted. No windows cross sonnet boundaries.
-20fixedepochs, Adam.003, batch64, seed42. Checkpoint selected by development cross-entropy, best at epoch14. Test read afterward.
-Test teacher-forced character cross-entropy1.722692, accuracy.480987,2.48532bits per character,9,520targets. Train-only smoothed unigram cross-entropy on the same target positions3.000697.
-Two full local training runs produced exactly the same report, metrics and samples except wall-clock runtime.
-These character scores are not comparable to previous word-level perplexity. They measure prediction, not whether a poem makes sense or has meter and rhyme.

Read all four samples in results/report.json. They no longer collapse to a single repeated word but contain malformed words and weak meaning. They are NOT good finished poems. No score increase claimed; project stays55out of100. Four prompts, seeds and temperature fixed before sampling, no best-of output selection. Sampling uses temperature.8and top12characters.

Limited memorization check: none of the four400-character completions exactly matched a training line of at least4words; longest shared substring13–17characters. This is not a guarantee of originality. No human quality, meter or rhyme evaluation yet.

## Reproduce

Python3.11, numpy1.26.4, torch2.4.1+cpu. Install dependencies locally, not remote code. Put the original source text in a local path:

    python scripts/char_poetry.py --source YOUR_1041.txt --output NEW_DIRECTORY --epochs 20
    python -m pytest tests/test_char_poetry.py


Original catalog: https://www.gutenberg.org/ebooks/1041
Source mirror: https://github.com/GITenberg/Shakespeare-s-Sonnets_1041
Source SHA256:e3580b25135e2d89467dd4cef0ccdcdd5f50f8779247fc24612ef653d27dc9d5

Catalog says public domain in the USA. Local-law and Gutenberg trademark/license conditions still matter. Corpus is not redistributed. Trained weights are not included. Code is MIT; third-party text rights separate. Output samples and numerical reports supplied for review, not posted to GitHub or merged to main.


Numerical result JSON and any generated samples are supplied in the private experiment ZIP, not committed here. Run the commands from the repository root with authorized local inputs. These scripts are separate research recipes, not changes to the production model or alert policy.
