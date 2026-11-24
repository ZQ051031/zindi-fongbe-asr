import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .audio import features, read_audio
from .data import training_split
from .decode import BigramLM, greedy
from .metric import edit_distance
from .model import AcousticModel, load_model
from .text import Vocabulary, graphemes, normalize


class SpeechDataset(Dataset):
    def __init__(self, rows, root):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        speech, _ = read_audio(self.root / row["audio_path"])
        return torch.from_numpy(features(speech)), normalize(row["transcription"]), row["id"]


def collate(batch):
    frames, texts, ids = zip(*batch)
    return pad_sequence(frames, batch_first=True), torch.tensor([len(x) for x in frames]), texts, ids


class DurationBatches:
    def __init__(self, rows, size, rng):
        self.durations = [float(row["duration"]) for row in rows]
        self.size = size
        self.rng = rng

    def __iter__(self):
        indices = list(range(len(self.durations)))
        self.rng.shuffle(indices)
        batches = []
        pool = 20 * self.size
        for start in range(0, len(indices), pool):
            block = sorted(indices[start:start + pool], key=self.durations.__getitem__)
            batches.extend(block[i:i + self.size] for i in range(0, len(block), self.size))
        self.rng.shuffle(batches)
        return iter(batches)


def ctc_targets(encoded, lengths, ids):
    for seq, length, row_id in zip(encoded, lengths, ids, strict=True):
        repeats = sum(left == right for left, right in zip(seq, seq[1:]))
        if not seq or len(seq) + repeats > length:
            raise ValueError(f"Empty target or impossible CTC alignment for {row_id}")
    return torch.tensor([token for seq in encoded for token in seq], dtype=torch.long)


@torch.inference_mode()
def validate(model, vocab, loader, device):
    model.eval()
    errors = total = 0
    for frames, lengths, refs, _ in loader:
        predictions = model(frames.to(device), lengths).argmax(-1).cpu()
        for pred, length, ref in zip(predictions, model.encoder.output_lengths(lengths), refs):
            reference = graphemes(ref)
            hypothesis = graphemes(vocab.render(greedy(pred[:length].tolist())))
            errors += edit_distance(reference, hypothesis)
            total += len(reference)
    if not total:
        raise ValueError("Validation transcripts are empty")
    return errors / total


def init_model(vocab, width, checkpoint, expand_vocabulary=False):
    if checkpoint is None:
        return AcousticModel(len(vocab.units), width)
    model, units = load_model(checkpoint)
    if model.encoder.proj.out_channels != width:
        raise ValueError("Checkpoint width does not match --width")
    if units == vocab.units:
        return model
    if not expand_vocabulary or not set(units) < set(vocab.units):
        raise ValueError("Checkpoint vocabulary does not match the training vocabulary")
    expanded = AcousticModel(len(vocab.units), width)
    expanded.encoder.load_state_dict(model.encoder.state_dict())
    positions = {unit: i for i, unit in enumerate(vocab.units)}
    indices = [positions[unit] for unit in units]
    with torch.no_grad():
        expanded.classifier.weight[indices] = model.classifier.weight
        expanded.classifier.bias[indices] = model.classifier.bias
    return expanded


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    pos = tuple(val.strip().casefold() for val in args.validation_positive.split(","))
    neg = tuple(val.strip().casefold() for val in args.validation_negative.split(","))
    rows, dev_rows = training_split(args.reliable, args.auxiliary, pos, neg)
    vocab = Vocabulary.from_texts(row["transcription"] for row in rows)
    if len(vocab.units) < 2:
        raise ValueError("Training transcriptions are empty")
    device = torch.device(args.device)
    model = init_model(vocab, args.width, args.init_checkpoint, args.expand_checkpoint_vocabulary).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    BigramLM.fit((vocab.encode(row["transcription"]) for row in rows), args.output / "lm.json")
    opts = dict(num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    loader = DataLoader(SpeechDataset(rows, args.audio_root),
                        batch_sampler=DurationBatches(rows, args.batch_size, rng), **opts)
    dev = DataLoader(SpeechDataset(dev_rows, args.audio_root), batch_size=args.batch_size, **opts)
    loss_fn = nn.CTCLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = batches = 0
        for frames, lengths, texts, ids in loader:
            encoded = [vocab.encode(text) for text in texts]
            out_lengths = model.encoder.output_lengths(lengths)
            targets = ctc_targets(encoded, out_lengths, ids)
            log_probs = model(frames.to(device), lengths)
            loss = loss_fn(log_probs.transpose(0, 1), targets.to(device), out_lengths,
                           torch.tensor([len(seq) for seq in encoded]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        error = validate(model, vocab, dev, device)
        print(f"epoch={epoch} ctc_loss={loss_sum / batches:.4f} dev_grapheme_error={error:.4f}",
              flush=True)
        if error < best:
            best = error
            torch.save(dict(version=1, width=args.width, units=vocab.units,
                            model=model.state_dict()), args.output / "acoustic.pt")


def main():
    parser = argparse.ArgumentParser(description="Train a compact tone-aware CTC model")
    parser.add_argument("--reliable", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, action="append", default=[])
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Start from matching acoustic weights with a fresh optimizer")
    parser.add_argument("--expand-checkpoint-vocabulary", action="store_true",
                        help="Allow new training graphemes when initializing from a checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-positive", default="true,1,yes,1.0")
    parser.add_argument("--validation-negative", default="false,0,no,0.0,")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.threads) < 1:
        parser.error("epochs, batch-size, width and threads must be positive")
    if args.expand_checkpoint_vocabulary and args.init_checkpoint is None:
        parser.error("--expand-checkpoint-vocabulary requires --init-checkpoint")
    train(args)


if __name__ == "__main__":
    main()
