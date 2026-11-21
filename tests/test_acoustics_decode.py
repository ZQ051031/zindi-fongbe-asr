from collections import defaultdict
from itertools import product

import numpy as np
import pytest
import torch

from fongbe_asr.audio import FEATURES, RATE, features
from fongbe_asr.decode import BigramLM, beam_search, greedy
from fongbe_asr.model import AcousticModel
from fongbe_asr.train import ctc_targets


def test_pitch_feature_distinguishes_two_fundamentals():
    t = np.arange(RATE, dtype=np.float32) / RATE
    feature_matrices = [features(np.sin(2 * np.pi * pitch * t)) for pitch in (128, 192)]
    assert feature_matrices[0].shape == (101, FEATURES)
    assert np.isfinite(features(np.zeros(10, dtype=np.float32))).all()
    assert np.median(feature_matrices[1][:, -2]) - np.median(feature_matrices[0][:, -2]) > 0.5


def test_padded_batch_matches_individual_outputs():
    torch.manual_seed(10)
    model = AcousticModel(5, width=8).eval()
    frames = torch.randn(2, 23, FEATURES)
    frames[0, 12:] = 0
    lengths = torch.tensor([12, 23])
    with torch.inference_mode():
        together = model(frames, lengths)
        short = model(frames[0:1, :12])
    assert torch.equal(model.encoder.output_lengths(lengths), torch.tensor([6, 12]))
    torch.testing.assert_close(together[0, :6], short[0])


def test_repeated_graphemes_need_blank_capacity():
    assert ctc_targets([[1, 1]], [3], ["ok"]).tolist() == [1, 1]
    with pytest.raises(ValueError, match="bad"):
        ctc_targets([[1, 1]], [2], ["bad"])
    assert greedy([1, 1, 0, 1, 2, 2]) == [1, 1, 2]


def test_prefix_beam_matches_exact_ctc_marginal():
    probabilities = np.array([[0.15, 0.61, 0.24], [0.70, 0.22, 0.08], [0.12, 0.67, 0.21]])
    path_masses = defaultdict(float)
    for labels in product(range(3), repeat=3):
        path_masses[tuple(greedy(labels))] += np.prod(probabilities[np.arange(3), labels])
    result = beam_search(np.log(probabilities), width=30)
    assert result == max(path_masses, key=path_masses.get) == (1, 1)


def test_bigram_changes_tone_when_acoustic_scores_tie(tmp_path):
    path = tmp_path / "lm.json"
    BigramLM.fit([[2]] * 20, path)
    model = BigramLM.load(path, 3)
    scores = np.log(np.array([[0.02, 0.49, 0.49]]))
    assert beam_search(scores) == (1,)
    assert beam_search(scores, lm=model) == (2,)
