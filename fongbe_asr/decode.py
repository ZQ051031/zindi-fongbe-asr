from collections import Counter, defaultdict
import json
import math
from pathlib import Path


NEG_INF = -math.inf


def logadd(a, b):
    a, b = max(a, b), min(a, b)
    return a if b == NEG_INF else a + math.log1p(math.exp(b - a))


class BigramLM:
    """Add-smoothed grapheme bigram; tone-bearing graphemes are distinct."""
    def __init__(self, counts, classes):
        self.scores = {}
        for context, outcomes in counts.items():
            total = sum(outcomes.values()) + 0.3 * classes
            self.scores[int(context)] = [
                math.log((outcomes.get(str(i), 0) + 0.3) / total)
                for i in range(classes)
            ]
        self.uniform = -math.log(classes)

    def score(self, context, token):
        table = self.scores.get(context)
        return self.uniform if table is None else table[token]

    @staticmethod
    def fit(sequences, path):
        counts = defaultdict(Counter)
        for seq in sequences:
            for left, right in zip([0, *seq], [*seq, 0]):
                counts[left][right] += 1
        Path(path).write_text(json.dumps(counts), encoding="utf-8")

    @classmethod
    def load(cls, path, classes):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), classes)


def greedy(frame_ids):
    result = []
    prev = 0
    for token in frame_ids:
        if token and token != prev:
            result.append(token)
        prev = token
    return result


def beam_search(log_probs, width=8, token_topk=10, lm=None, lm_weight=0.15):
    """CTC prefix beam search, keeping blank and nonblank mass separately."""
    beam = {(): (0.0, NEG_INF)}
    for frame in log_probs.tolist():
        candidates = sorted(range(1, len(frame)), key=frame.__getitem__,
                            reverse=True)[:token_topk]
        next_beam = {}

        def update(prefix, blank=NEG_INF, emit=NEG_INF):
            old_blank, old_emit = next_beam.get(prefix, (NEG_INF, NEG_INF))
            next_beam[prefix] = logadd(blank, old_blank), logadd(emit, old_emit)

        for prefix, (blank, emit) in beam.items():
            total = logadd(blank, emit)
            last = prefix[-1] if prefix else 0
            update(prefix, blank=total + frame[0])
            if last:
                update(prefix, emit=emit + frame[last])
            for token in candidates:
                score = blank if token == last else total
                if score == NEG_INF:
                    continue
                bonus = lm_weight * lm.score(last, token) if lm else 0
                update(prefix + (token,), emit=score + frame[token] + bonus)
        beam = dict(sorted(next_beam.items(), key=lambda item: logadd(*item[1]),
                           reverse=True)[:width])

    def final_score(prefix):
        bonus = lm_weight * lm.score(prefix[-1] if prefix else 0, 0) if lm else 0
        return logadd(*beam[prefix]) + bonus

    return max(beam, key=final_score)
