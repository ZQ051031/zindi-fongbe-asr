import argparse
from pathlib import Path

from .data import read_csv, unique_index, write_csv
from .runtime import one_cpu
from .syllables import syllabifier


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    for name in ("checkpoint", "test-csv", "audio-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--lm", type=Path, help="Optional training-generated lm.json")
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--beam-width", type=int, default=8, help="1 uses greedy CTC")
    parser.add_argument("--token-topk", type=int, default=10)
    parser.add_argument("--lm-weight", type=float, default=0.15)
    script = parser.add_mutually_exclusive_group()
    script.add_argument("--syllabify", type=Path, help="Trusted organizer-provided syllabify.py")
    script.add_argument("--synthetic", action="store_true", help="Use toy syllabification only")
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if min(args.beam_width, args.token_topk) < 1:
        parser.error("beam-width and token-topk must be positive")
    if args.sample_submission and not (args.syllabify or args.synthetic):
        parser.error("submission output requires --syllabify (or --synthetic for a smoke test)")
    if args.beam_width == 1 and args.lm:
        parser.error("greedy decoding does not use an LM; omit --lm or increase beam-width")
    return args


def ordered_test(test_path, sample_path):
    rows = read_csv(test_path, ("id", "audio_path"))
    index = unique_index(rows, "id")
    if sample_path is None:
        return rows
    sample = read_csv(sample_path, ("ID", "Target"))
    ids = unique_index(sample, "ID")
    if ids.keys() != index.keys():
        raise ValueError("SampleSubmission IDs do not exactly match Test.csv IDs")
    return [index[row["ID"]] for row in sample]


def run(args):
    import torch

    from .audio import features, read_audio
    from .decode import BigramLM, beam_search, greedy
    from .model import load_model
    from .text import Vocabulary

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, tokens = load_model(args.checkpoint)
    vocab = Vocabulary(tokens)
    lm = BigramLM.load(args.lm, len(tokens)) if args.lm else None
    paths = ordered_test(args.test_csv, args.sample_submission)
    if not paths:
        raise ValueError("Test manifest is empty")
    duration = 0.0
    rows = []
    with torch.inference_mode(), syllabifier(args.syllabify, args.synthetic) as transform:
        for row in paths:
            audio, seconds = read_audio(args.audio_root / row["audio_path"])
            duration += seconds
            scores = model(torch.from_numpy(features(audio)).unsqueeze(0))[0]
            ids = (greedy(scores.argmax(-1).tolist()) if args.beam_width == 1 else
                   beam_search(scores.numpy(), args.beam_width, args.token_topk, lm, args.lm_weight))
            rows.append(dict(ID=row["id"], Target=transform(vocab.render(ids))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, ("ID", "Target"), rows)
    return len(rows), duration


def main():
    args = parse_args(make_parser("Decode local audio on a single CPU"))
    one_cpu()
    count, duration = run(args)
    print(f"Transcribed {count} recordings ({duration:.3f} audio seconds): {args.output}")


if __name__ == "__main__":
    main()
