# Frugal Fongbe tonal ASR

An offline Fongbe speech-to-text baseline for the [Zindi BENIN Fongbe Tonal ASR Challenge](https://zindi.world/competitions/frugal-tonal-asr-challenge-for-fongbe). The model preserves `ɖ`, `ɛ`, `ɔ` and tone marks (acute: high; grave: low; caron: rising). No pretrained model, external training data, AutoML, API calls or GPU at inference.

## Pipeline

1. Read mono 16 kHz WAV and compute 64 log-mel bands, relative and absolute log-F0, and a voicing-strength feature in 40 ms windows every 10 ms. Absolute and utterance-relative pitch help distinguish tone across speaker registers.
2. Run a strided convolution and two bidirectional GRU layers at a 20 ms output step. CTC predicts Unicode graphemes, grouping a base letter and its combining marks into one unit; NFC/NFD spellings are equivalent and different tones remain different classes. `fongbe_asr.model.Encoder` is the isolated acoustic encoder.
3. Decode with bounded CTC prefix beam search and an optional smoothed grapheme bigram LM fitted only to the training transcripts. Tone-bearing graphemes have separate LM probabilities. Greedy decoding is available with `--beam-width 1` (omit `--lm`).
4. Emit raw tone-marked text or run the organizer's local `syllabify.py` to produce the syllable tokens required for the submission.

The decoder reads one recording at a time. This is utterance-level ASR, not streaming; bidirectional context and feature normalization use the whole clip. Memory and RTF are empirical and depend on the final model, input durations and CPU.

## Setup

From the repository root on Linux with Python 3.10+ and `libseccomp.so.2` (Debian/Ubuntu package `libseccomp2`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

Installing dependencies is a setup step and may use the network. Once installed, both training from local inputs and CPU inference use local files. Training can instead use an appropriate GPU-enabled PyTorch installation. The code uses open-source Python (PSF), PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), python-soundfile (BSD-3-Clause), libsndfile (LGPL-2.1-or-later), libseccomp (LGPL-2.1-only), and pytest (MIT, tests); no external acoustic weights or text corpora are needed.

## Train on the real data

1. Enrol on Zindi and download the official manifests, tools, and `audio_*.tar` archives referenced in the [Data tab](https://zindi.world/competitions/frugal-tonal-asr-challenge-for-fongbe/data). Verify the archive checksums and extract into a private workspace outside this repository. Set `--audio-root` to the directory containing `audio/`, so each relative `audio_path` resolves from there. Keep manifests, tools, audio, weights and submissions out of Git.
2. Read the downloaded `conventions.md` before training. The three train CSVs use `id,audio_path,transcription,speaker_id,duration` (duration in seconds). `Train_reliable.csv` is about 106 hours of validated transcriptions and includes `recommended_validation`. `Train_uncertain.csv` is about 198 hours of uncertain tone labels and includes `protocol`. `Train_religious.csv` is about 42 hours of Bible readings from one speaker and includes book/chapter metadata. `Test.csv` has `id,audio_path` and no reference transcripts; extra columns are ignored. The Data tab reports 9,407 test recordings, about 22 hours.
3. Start with reliable speech. The loader reserves all reliable records from any speaker flagged for validation and excludes those speakers from any supplied auxiliary training files. Flag value encodings aren't specified publicly: by default it accepts `true,1,yes,1.0` and `false,0,no,0.0` plus empty for false; override using `--validation-positive` and `--validation-negative` if needed. The best checkpoint is selected by dev grapheme error (tone-sensitive). The checkpoint and LM are a pair from the same training run.

```bash
python -m fongbe_asr.train \
  --reliable data/zindi/Train_reliable.csv --audio-root data/zindi \
  --output runs/reliable --epochs 30 --width 128 --batch-size 24 --workers 4 --device cuda
```

Replace `--device cuda` with `--device cpu` if necessary; training has no frugality cap. Use `--auxiliary data/zindi/Train_uncertain.csv` and/or `--auxiliary data/zindi/Train_religious.csv` to compare additional data against the reliable-only baseline, ideally also checking performance on whole held-out books before counting on religious-domain gains. Auxiliary labels have equal weight if enabled; uncertain tones can hurt. The training command creates `acoustic.pt` and `lm.json`.

## Run inference and prepare the submission

```bash
python -m fongbe_asr.infer \
  --checkpoint runs/reliable/acoustic.pt --lm runs/reliable/lm.json \
  --test-csv data/zindi/Test.csv --audio-root data/zindi \
  --sample-submission data/zindi/SampleSubmission.csv \
  --syllabify data/zindi/syllabify.py --output runs/reliable/submission.csv
```

`SampleSubmission.csv` defines the required `ID,Target` header and row order. IDs are matched as strings and must agree exactly with the test manifest. Keep the trusted organizer script and its local helper modules together; the CLI calls its `syllabify(text)` function. For raw orthographic output, omit both `--sample-submission` and `--syllabify`. Do not submit raw or mock-syllabified output.

Zindi scores edits on **syllable** tokens, not orthographic words. For example, the organizer gives `Un nɔ́ ɖu nǔ → un nɔ1 ɖu nu3`. The local estimator takes a reference CSV with `id,transcription` and a prediction CSV with `ID,Target` already syllabified. It weights each utterance's syllable edit rate by the square root of its reference syllable count and counts tone changes as syllable substitutions.

For the organizer's separate local diagnostics, save the speaker-held-out rows returned by `fongbe_asr.data.training_split` as a validation CSV. Decode that CSV with `fongbe_asr.infer`, omitting both `--sample-submission` and `--syllabify` to retain plain tone-marked text. Run the organizer's `test_score.py` and pass the validation CSV and these **plain** predictions to `python score.py --reference validation.csv --submission dev_predictions.csv`. The tool reports its local score, TSER, and conditional TER; neither this score nor the local weighted estimator is a leaderboard result.

## Benchmark

Repeat the same inference command with `python -m fongbe_asr.benchmark` in place of `python -m fongbe_asr.infer`. It checks total checkpoint and LM bytes (decimal units), pins the fresh process to one logical CPU, caps the math and PyTorch thread pools to one, disables CUDA, and blocks network socket creation with Linux seccomp. The benchmark requires the socket probe to fail before inference starts. Wall time includes heavyweight imports, weight loading, audio I/O, feature extraction, decoding, syllabification and CSV output; RTF is wall time divided by corpus audio time. Peak RSS is the OS process high-water mark, including the loaded runtime and model. It returns a nonzero status if weights exceed 1,000 MB, RTF exceeds 1, or peak RSS exceeds 2 GB.

Rebenchmark the actual competition model and test audio using the official `measure_frugality.py` and its instructions. Include both the checkpoint and LM as separate `--weights` arguments, and use a `--command` that passes `{csv}`, `{root}`, and `{output}` to the benchmark CLI. The organizer tool may be unable to create a network namespace on an unprivileged VM; the repo benchmark still enforces offline inference with seccomp. Passing here does not guarantee passing on Zindi's reference machine.

## Reproduce the SYNTHETIC validation

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=3
python -m fongbe_asr.synthetic data/synthetic
python -m fongbe_asr.train \
  --reliable data/synthetic/Train_reliable.csv --audio-root data/synthetic \
  --output runs/synthetic --width 80 --batch-size 24 --threads 3 --epochs 22
python -m fongbe_asr.benchmark \
  --checkpoint runs/synthetic/acoustic.pt --lm runs/synthetic/lm.json \
  --test-csv data/synthetic/Test.csv --audio-root data/synthetic \
  --sample-submission data/synthetic/SampleSubmission.csv --synthetic \
  --output runs/synthetic/predictions.csv
python -m fongbe_asr.evaluate \
  --references data/synthetic/TestReferences.csv --predictions runs/synthetic/predictions.csv \
  --synthetic
```

The generator creates 1,200 train, 100 dev and 120 test utterances in the official file schema, with disjoint synthetic speakers across splits. It uses open V/CV syllables, single- and two-syllable mock words, harmonically excited vowel formants, nasal/stop/fricative-like onsets, high/low/rising pitch contours, and varied vocal tract, pitch register, tempo, gain, noise and early reflections. Words are randomly assembled, not Fongbe vocabulary; the audio is controlled speech-like synthesis, not native speech. The toy syllabifier is only for the generated alphabet. Synthetic results test the pipeline and resource budget, not real-world or leaderboard accuracy. **Never train a competition model on synthetic data**; the rules restrict datasets to those provided.

**SYNTHETIC results** — Ubuntu x86-64, Intel Xeon VM, Python 3.10.12 and PyTorch 2.7.1+cpu. The benchmark used one pinned CPU and the beam decoder with the LM; the offline socket probe was denied.

| Measurement | SYNTHETIC result | Limit |
| --- | ---: | ---: |
| Weighted syllable WER, held-out test | 4.94% | Not a frugality metric |
| Grapheme error, best dev checkpoint (epoch 18) | 1.93% | Not a frugality metric |
| Total model and LM size | 916,442 bytes (0.916 MB) | ≤ 1,000 MB |
| Real-time factor | 0.01183 (3.897 s / 329.482 s audio) | ≤ 1 |
| Peak process RSS | 312,197,120 bytes (312.20 MB) | ≤ 2 GB |

All 11 unit tests passed. No real-data training or official scoring was performed.

## Participant checklist

- Enrol and download the official files yourself; then train and validate on your local copies.
- Verify the final submission and resource limits with the organizer tools. Use only the provided competition datasets for the competition. Keep data and derived artifacts private and follow Zindi's deletion and disclosure rules.
- Submit by the December 14, 2026 cutoff, but **do not wait until December 14**: the public page viewed during development displayed a submission-close tooltip of **December 13, 2026 at 21:30** without an explicit timezone and a December 14 reveal. Confirm the operative deadline in your Zindi account and submit before the earlier displayed cutoff. Limits: **5 submissions per day, 100 total**. The rules also ask you to select two submissions for the private leaderboard.
- If sharing this starter through the competition, follow Zindi's requirement to make shared code available to all participants through the platform.
