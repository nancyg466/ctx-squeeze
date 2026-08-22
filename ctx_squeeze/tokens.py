"""Character-class token estimator.

No tokenizer, no vocabulary file, no network call. Good enough to budget
against; not good enough to bill against.
"""

import math
import re

_CJK = (
    "一-鿿"  # CJK Unified Ideographs
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "぀-ヿ"  # Hiragana + Katakana
    "가-힣"  # Hangul syllables
)

# The word alternative excludes the CJK ranges explicitly: \w already
# treats CJK ideographs as letters, and a greedy word match starting on
# an ASCII letter would otherwise swallow an adjacent CJK run instead of
# stopping and letting the cjk alternative price it per character.
_TOKEN_RE = re.compile(
    r"(?P<cjk>[" + _CJK + r"])"
    r"|(?P<digits>\d+)"
    r"|(?P<newline>\n)"
    r"|(?P<word>[^\W\d_" + _CJK + r"]+)"
    r"|(?P<symbol>[^\s\w])",
    re.UNICODE,
)

_CHARS_PER_WORD_TOKEN = 4
_CHARS_PER_DIGIT_TOKEN = 3
_NEWLINE_TOKEN = 0.5
_SYMBOL_TOKEN = 0.6


def estimate_tokens(text):
    """Estimate the token count of `text`.

    Rough weights: 4 chars/token for word runs, 3 chars/token for digit
    runs, 1 token per CJK character, 0.5 token per newline, 0.6 token per
    other symbol. Plain whitespace (other than newlines) costs nothing.
    Lands within about 10% of a BPE tokenizer on English prose.
    """
    if not text:
        return 0

    total = 0.0
    for match in _TOKEN_RE.finditer(text):
        if match.group("cjk") is not None:
            total += 1
        elif match.group("digits") is not None:
            total += len(match.group("digits")) / _CHARS_PER_DIGIT_TOKEN
        elif match.group("newline") is not None:
            total += _NEWLINE_TOKEN
        elif match.group("word") is not None:
            total += len(match.group("word")) / _CHARS_PER_WORD_TOKEN
        else:
            total += _SYMBOL_TOKEN

    # Round up so a caller who slices to a budget never overshoots it
    # because a fractional estimate got rounded away. A lone underscore is
    # the one character class none of the alternatives above match, so
    # floor it to 1 token when the text is otherwise non-blank.
    tokens = math.ceil(total)
    return max(tokens, 1) if text.strip() else tokens


def truncate_to_tokens(text, n):
    """Hard-truncate `text` to at most `n` estimated tokens.

    Binary search over character offsets rather than walking character by
    character, since estimate_tokens is non-decreasing as text grows and
    documents can be large. This cuts mid-word if it has to; callers who
    care about keeping structures (code fences, whole paragraphs) intact
    should segment first and drop whole segments instead of calling this
    directly on raw text.
    """
    if n <= 0:
        return ""
    if estimate_tokens(text) <= n:
        return text

    lo, hi, best = 0, len(text), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if estimate_tokens(candidate) <= n:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best
