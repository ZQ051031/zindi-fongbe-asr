"""Generate a controlled speech-like corpus; not natural Fongbe speech."""
import argparse
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.signal import lfilter
import soundfile as sf

from .audio import RATE
from .data import write_csv
from .text import normalize

ONSETS = ("", "m", "n", "ɖ", "k", "s", "f", "l", "w", "b")
FORMANTS = {"a": (730, 1300), "ɛ": (570, 1860), "ɔ": (510, 900),
            "u": (300, 760), "i": (305, 2250)}
TONES = ("", "\u0301", "\u0300", "\u030c")
CODES = {"\u0301": "1", "\u0300": "2", "\u030c": "3"}
ONSETS_RE = "[m n ɖ k s f l w b]".replace(" ", "")
SYLLABLE_RE = re.compile(f"({ONSETS_RE}?[aɛɔui])([\u0301\u0300\u030c]?)")
FIELDS = ("id", "audio_path", "transcription", "speaker_id", "duration")


def synthetic_syllabify(text):
    return " ".join(base + CODES.get(tone, "") for base, tone in
                    SYLLABLE_RE.findall(unicodedata.normalize("NFD", normalize(text))))


def envelope(size, rise=0.13):
    t = np.linspace(0, 1, size)
    return np.minimum(1, np.minimum(t, 1 - t) / rise)


def voiced(tone, size, f0, formants, rng):
    start, end = ((0.98, 0.95), (1.28, 1.24), (0.76, 0.72), (0.73, 1.30))[tone]
    glide = np.linspace(start, end, size) + 0.012 * np.sin(np.linspace(0, 6, size))
    phase = np.cumsum(2 * np.pi * f0 * glide / RATE)
    out = np.zeros(size)
    harmonics = np.arange(1, min(45, int(7200 / (f0 * max(glide)))) + 1)
    frequencies = harmonics[:, None] * f0 * glide
    for center, bandwidth, gain in zip(formants, (95, 130, 195), (1., 0.60, 0.23)):
        weights = np.exp(-0.5 * ((frequencies - center) / bandwidth) ** 2)
        out += ((gain / np.sqrt(harmonics[:, None])) * weights *
                np.sin(harmonics[:, None] * phase + rng.uniform(0, 2*np.pi))).sum(axis=0)
    return out / max(np.std(out), 0.01) * 0.16 * envelope(size)


def onset_sound(onset, f0, speed, rng):
    size = int(RATE * rng.uniform(0.048, 0.085) * speed)
    if onset == "":
        return np.zeros(size // 4)
    if onset in "mnlw":
        pair = dict(m=(250, 700), n=(400, 1550), l=(480, 1950), w=(290, 950))[onset]
        return 0.55 * voiced(0, size, f0, (*pair, 2700), rng)
    noise = rng.normal(size=size)
    if onset in "sf":
        tap = 0.96 if onset == "s" else 0.35
        return envelope(size) * 0.105 * lfilter([1, -tap], [1], noise)
    center = dict(k=2800, b=800, ɖ=1600)[onset]
    phase = np.arange(size) * 2 * np.pi * center / RATE
    burst = (0.5 * noise + np.sin(phase)) * np.exp(-np.arange(size)/(0.014*RATE))
    if onset != "k":
        burst += 0.22 * np.sin(np.arange(size) * 2*np.pi*f0 / RATE)
    return np.concatenate((np.zeros(size // 3), burst * 0.135 * envelope(size, 0.04)))


def utterance(rng, f0, tract, speed):
    chunks = [np.zeros(int(RATE * rng.uniform(0.04, 0.13)))]
    words = []
    for _ in range(rng.integers(3, 9)):
        word = []
        for _ in range(1 + int(rng.random() < 0.33)):
            onset = str(rng.choice(ONSETS))
            vowel = str(rng.choice(list(FORMANTS)))
            tone = rng.integers(len(TONES))
            word.append(onset + vowel + TONES[tone])
            pitch = f0 * rng.uniform(0.96, 1.04)
            chunks.append(onset_sound(onset, pitch, speed, rng))
            size = int(RATE * rng.uniform(0.15, 0.25) * speed)
            formants = tuple(center * tract for center in (*FORMANTS[vowel], 2800))
            chunks.append(voiced(tone, size, pitch, formants, rng))
        words.append("".join(word))
        chunks.append(np.zeros(int(RATE * speed * rng.uniform(0.07, 0.14))))
    speech = np.concatenate(chunks)
    delay = rng.integers(100, 550)
    speech[delay:] += rng.uniform(0.015, 0.12) * speech[:-delay]
    speech += rng.normal(0, rng.uniform(0.002, 0.014), size=len(speech))
    return np.clip(speech * rng.uniform(0.65, 1.2), -0.99, 0.99), normalize(" ".join(words))


def generate(root, n_train, n_dev, n_test, seed):
    rng = np.random.default_rng(seed)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    train, test = [], []
    for split, count, n_speakers in (("train", n_train, 24), ("dev", n_dev, 6), ("test", n_test, 8)):
        voices = [(f0, rng.uniform(0.88, 1.12), rng.uniform(0.84, 1.18))
                  for f0 in np.linspace(112, 246, n_speakers) + rng.uniform(-7, 7, n_speakers)]
        for i in range(count):
            speaker = i % n_speakers
            speech, text = utterance(rng, *voices[speaker])
            uid = f"{split}-{i:06}"
            path = f"audio/{uid}.wav"
            sf.write(root / path, speech, RATE, subtype="PCM_16")
            row = dict(id=uid, audio_path=path, transcription=text, speaker_id=f"{split}-{speaker}",
                       duration=f"{len(speech)/RATE:.6f}")
            if split == "test":
                test.append(row)
            else:
                train.append(dict(row, recommended_validation=int(split == "dev")))
    write_csv(root / "Train_reliable.csv", (*FIELDS, "recommended_validation"), train)
    test_fields = tuple(field for field in FIELDS if field != "transcription")
    write_csv(root / "Test.csv", test_fields, ({key: row[key] for key in test_fields} for row in test))
    write_csv(root / "TestReferences.csv", FIELDS, test)
    write_csv(root / "SampleSubmission.csv", ("ID", "Target"),
              (dict(ID=row["id"], Target="") for row in reversed(test)))
    print(f"SYNTHETIC: {n_train} train, {n_dev} dev, {n_test} test utterances")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=410)
    args = parser.parse_args()
    if min(args.train, args.dev, args.test) < 1:
        parser.error("split sizes must be positive")
    generate(args.root, args.train, args.dev, args.test, args.seed)


if __name__ == "__main__":
    main()
