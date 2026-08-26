"""Near-duplicate detection via word shingling and Jaccard similarity.

Agent transcripts repeat near-verbatim text with small local differences
(a timestamp, a retry count, a changed line number). Comparing whole
segments for equality misses those; comparing overlapping windows of
words catches them, since most windows in the repeated text are still
identical even where a few are not.
"""

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def shingles(text, size=5):
    """Return the set of overlapping `size`-word shingles in `text`.

    Words are lowercased before shingling so that case alone never
    prevents a match. If `text` has fewer than `size` words, the whole
    word sequence is returned as a single shingle; an empty or
    whitespace-only text yields an empty set.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a, b):
    """Jaccard similarity between two shingle sets: |a ∩ b| / |a ∪ b|.

    Two empty sets are defined as identical (similarity 1.0) rather than
    undefined, so comparing two blank segments doesn't need special
    casing at the call site.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
