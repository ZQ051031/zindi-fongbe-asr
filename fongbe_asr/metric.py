import math


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, 1):
        current = [i]
        for j, right in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def weighted_syllable_wer(references, predictions):
    """Weighted mean of utterance WER, with sqrt(reference length) weights.

    Arguments contain already syllabified strings; empty references are invalid.
    Use the competition score.py to verify the official score.
    """
    weights = []
    terms = []
    for ref, hyp in zip(references, predictions, strict=True):
        left, right = ref.split(), hyp.split()
        if not left:
            raise ValueError("Syllabified references must be nonempty")
        weight = math.sqrt(len(left))
        weights.append(weight)
        terms.append(weight * edit_distance(left, right) / len(left))
    if not weights:
        raise ValueError("No scored utterances")
    return math.fsum(terms) / math.fsum(weights)
