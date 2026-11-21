import csv
from pathlib import Path


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows, key):
    index = {row[key]: row for row in rows}
    if "" in index or None in index or len(index) != len(rows):
        raise ValueError(f"Empty or duplicate {key}")
    return index


def training_split(reliable, auxiliary=(), positive=("true", "1", "yes", "1.0"),
                   negative=("false", "0", "no", "0.0", "")):
    required = ("id", "audio_path", "transcription", "speaker_id", "duration")
    main = read_csv(reliable, (*required, "recommended_validation"))
    available = main + [row for file in auxiliary for row in read_csv(file, required)]
    unique_index(available, "id")
    heldout = set()
    for row in main:
        flag = row["recommended_validation"].strip().casefold()
        if flag in positive:
            heldout.add(row["speaker_id"])
        elif flag not in negative:
            raise ValueError(f"Unknown recommended_validation value {flag!r}; set flag options")
    if not heldout or "" in heldout or any(not row["speaker_id"] for row in available):
        raise ValueError("A nonempty, identified speaker holdout is required")
    train = [row for row in available if row["speaker_id"] not in heldout]
    dev = [row for row in main if row["speaker_id"] in heldout]
    if not train:
        raise ValueError("Speaker holdout leaves no training records")
    return train, dev
