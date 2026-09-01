"""Extractive segment scoring and budget-constrained selection.

Segments are scored by TF-IDF keyword density: a segment scores high when
it's dense with words that recur within it but are rare across the rest
of the document, the way a paragraph naming a specific failure or fix
stands out against surrounding boilerplate that repeats words already
common everywhere. select_by_score then picks the highest-scoring
segments that fit a token budget, greedily, while keeping the surviving
segments in their original document order.
"""

import math
import re

from ctx_squeeze.tokens import estimate_tokens

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def score_segments(segments):
    """Score each segment by TF-IDF keyword density.

    Each segment is treated as a document; a word's weight is its
    within-segment frequency times its inverse document frequency across
    all segments (smoothed so no word gets a zero or negative weight),
    and a segment's score is the sum of its word weights divided by its
    word count, so long and short segments compare on equal footing.
    Returns a list of floats aligned by index with `segments`; a segment
    with no words (pure punctuation, or an empty string) scores 0.0.
    """
    n = len(segments)
    if n == 0:
        return []

    word_lists = [_WORD_RE.findall(segment.text.lower()) for segment in segments]

    doc_freq = {}
    for words in word_lists:
        for word in set(words):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    scores = []
    for words in word_lists:
        if not words:
            scores.append(0.0)
            continue
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1
        total = 0.0
        for word, tf in term_freq.items():
            idf = math.log((n + 1) / (doc_freq[word] + 1)) + 1
            total += tf * idf
        scores.append(total / len(words))
    return scores


def select_by_score(segments, budget, scores=None):
    """Select the highest-scoring segments that fit within `budget` tokens.

    Segments are ranked best score first (ties broken by original
    position, for a stable result) and added greedily until the next one
    would push the running total over budget; a segment that alone
    exceeds the remaining budget is skipped rather than truncated, since
    truncation is `truncate_to_tokens`'s job, not this one's. The kept
    segments are returned in their original document order, not score
    order, so the result still reads like the source document with gaps.

    `scores` can be supplied to reuse a prior score_segments call (e.g.
    after filtering); otherwise it's computed from `segments`. Raises
    ValueError if a supplied `scores` isn't the same length as `segments`.
    """
    if scores is None:
        scores = score_segments(segments)
    elif len(scores) != len(segments):
        raise ValueError("scores must be the same length as segments")

    if budget <= 0:
        return []

    ranked = sorted(range(len(segments)), key=lambda i: (-scores[i], i))

    kept = set()
    used = 0
    for i in ranked:
        cost = estimate_tokens(segments[i].text)
        if used + cost <= budget:
            kept.add(i)
            used += cost

    return [segments[i] for i in sorted(kept)]
