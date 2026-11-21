import argparse
from pathlib import Path

from .data import read_csv, unique_index
from .metric import weighted_syllable_wer
from .syllables import syllabifier


def main():
    parser = argparse.ArgumentParser(description="Local weighted syllable WER, not official score.py")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path, help="Already syllabified ID,Target CSV")
    scripts = parser.add_mutually_exclusive_group(required=True)
    scripts.add_argument("--syllabify", type=Path)
    scripts.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    refs = unique_index(read_csv(args.references, ("id", "transcription")), "id")
    preds = unique_index(read_csv(args.predictions, ("ID", "Target")), "ID")
    if refs.keys() != preds.keys():
        parser.error("reference IDs and prediction IDs must match")
    with syllabifier(args.syllabify, args.synthetic) as transform:
        score = weighted_syllable_wer((transform(ref["transcription"]) for ref in refs.values()),
                                      (preds[key]["Target"] for key in refs))
    label = "SYNTHETIC" if args.synthetic else "LOCAL ESTIMATE"
    print(f"{label} weighted syllable WER: {score:.2%} ({len(refs)} recordings)")


if __name__ == "__main__":
    main()
