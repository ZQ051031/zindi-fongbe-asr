import math
import unicodedata

import pytest

from fongbe_asr.metric import weighted_syllable_wer
from fongbe_asr.synthetic import synthetic_syllabify
from fongbe_asr.text import Vocabulary, graphemes, normalize


def test_tones_and_canonical_equivalence():
    original = "  NƆ́,  ɖǔ! Ɛ̀ na  "
    target = "nɔ́ ɖǔ ɛ̀ na"
    assert normalize(original) == normalize(unicodedata.normalize("NFD", original)) == target
    assert graphemes(target) == ["n", "ɔ́", " ", "ɖ", "ǔ", " ", "ɛ̀", " ", "n", "a"]
    vocab = Vocabulary.from_texts([original, "nɔ̀"])
    assert vocab.render(vocab.encode(original)) == target
    assert vocab.encode("nɔ̀") != vocab.encode("nɔ́")


def test_synthetic_syllabifier_handles_words_with_multiple_syllables():
    assert synthetic_syllabify("Nɔ́ɖǔ ɛ̀") == "nɔ1 ɖu3 ɛ2"


def test_weighted_wer_counts_tone_substitution_and_insertions():
    score = weighted_syllable_wer(["nɔ1", "a wa ɛ ku"], ["nɔ2", "a wa ɛ ku ma"])
    assert score == pytest.approx((1 + math.sqrt(4) / 4) / (1 + math.sqrt(4)))
    assert weighted_syllable_wer(["a"], [""]) == 1
    assert weighted_syllable_wer(["a"], ["a"]) == 0
    with pytest.raises(ValueError):
        weighted_syllable_wer([""], ["a"])
    with pytest.raises(ValueError):
        weighted_syllable_wer(["a"], [])
