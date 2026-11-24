import os
from pathlib import Path
import subprocess
import sys

import pytest
import soundfile as sf
import torch

from fongbe_asr.data import read_csv, training_split, write_csv
from fongbe_asr.infer import ordered_test
from fongbe_asr.model import AcousticModel
from fongbe_asr.syllables import syllabifier
from fongbe_asr.synthetic import FIELDS, generate
from fongbe_asr.text import Vocabulary
from fongbe_asr.train import init_model


def test_corpus_uses_official_schema_and_speaker_holdout(tmp_path):
    generate(tmp_path, n_train=2, n_dev=2, n_test=2, seed=31)
    main = tmp_path / "Train_reliable.csv"
    aux = tmp_path / "Train_uncertain.csv"
    test = tmp_path / "Test.csv"
    sample = tmp_path / "SampleSubmission.csv"
    rows = read_csv(main, FIELDS)
    write_csv(aux, FIELDS, [dict({field: rows[2][field] for field in FIELDS}, id="aux-dev")])
    train, dev = training_split(main, [aux])
    assert [row["id"] for row in train] == ["train-000000", "train-000001"]
    assert not ({row["speaker_id"] for row in train} & {row["speaker_id"] for row in dev})
    info = sf.info(tmp_path / train[0]["audio_path"])
    assert (info.channels, info.samplerate, info.subtype) == (1, 16000, "PCM_16")
    assert [row["id"] for row in ordered_test(test, sample)] == ["test-000001", "test-000000"]
    write_csv(sample, ("ID", "Target"), [dict(ID="other", Target="")])
    with pytest.raises(ValueError, match="exactly match"):
        ordered_test(test, sample)


def test_official_hook_calls_the_local_module(tmp_path):
    script = tmp_path / "syllabify.py"
    script.write_text("def syllabify(text):\n    return text + ' hook'\n", encoding="utf-8")
    with syllabifier(script) as transform:
        assert transform("ɛ́") == "ɛ́ hook"


def test_training_can_initialize_from_matching_checkpoint(tmp_path):
    vocab = Vocabulary(["<blank>", "ɛ́", "ɔ́"])
    model = AcousticModel(len(vocab.units), width=8)
    checkpoint = tmp_path / "acoustic.pt"
    torch.save(dict(version=1, width=8, units=vocab.units, model=model.state_dict()), checkpoint)
    restored = init_model(vocab, 8, checkpoint)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight)
    with pytest.raises(ValueError, match="Checkpoint width"):
        init_model(vocab, 9, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint vocabulary"):
        init_model(Vocabulary(["<blank>", "ɔ́", "ɛ́"]), 8, checkpoint)


def test_training_can_expand_checkpoint_vocabulary(tmp_path):
    units = ["<blank>", "ɛ́", "ɔ́"]
    model = AcousticModel(len(units), width=8)
    checkpoint = tmp_path / "acoustic.pt"
    torch.save(dict(version=1, width=8, units=units, model=model.state_dict()), checkpoint)
    wider = Vocabulary(["<blank>", "a", "ɛ́", "ɔ́"])
    with pytest.raises(ValueError, match="Checkpoint vocabulary"):
        init_model(wider, 8, checkpoint)
    restored = init_model(wider, 8, checkpoint, expand_vocabulary=True)
    assert restored.classifier.out_features == 4
    for name, weight in model.encoder.state_dict().items():
        torch.testing.assert_close(restored.encoder.state_dict()[name], weight)
    torch.testing.assert_close(restored.classifier.weight[[0, 2, 3]], model.classifier.weight)
    torch.testing.assert_close(restored.classifier.bias[[0, 2, 3]], model.classifier.bias)
    with pytest.raises(ValueError, match="Checkpoint vocabulary"):
        init_model(Vocabulary(["<blank>", "ɛ́", "a", "b"]), 8, checkpoint, expand_vocabulary=True)


def test_inference_subprocess_writes_ordered_syllabified_csv(tmp_path):
    generate(tmp_path, n_train=1, n_dev=1, n_test=2, seed=12)
    model = AcousticModel(2, width=8)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-15., 15.]))
    checkpoint = tmp_path / "acoustic.pt"
    torch.save(dict(version=1, width=8, units=["<blank>", "ɔ́"], model=model.state_dict()), checkpoint)
    predictions = tmp_path / "predictions.csv"
    run = subprocess.run([
        sys.executable, "-m", "fongbe_asr.infer", "--checkpoint", str(checkpoint),
        "--test-csv", str(tmp_path / "Test.csv"), "--audio-root", str(tmp_path),
        "--sample-submission", str(tmp_path / "SampleSubmission.csv"),
        "--output", str(predictions), "--beam-width", "1", "--synthetic",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONUTF8": "1"})
    assert run.returncode == 0, run.stderr
    assert read_csv(predictions, ("ID", "Target")) == [
        dict(ID="test-000001", Target="ɔ1"), dict(ID="test-000000", Target="ɔ1"),
    ]
