"""Split a document into paragraph and code-block segments.

A fenced code block is kept as a single segment no matter how many blank
lines it contains internally, so later pipeline stages (scoring, dedupe,
budget selection) can drop or keep it whole instead of slicing through it.
"""

from dataclasses import dataclass

_FENCE_MARKERS = ("```", "~~~")


@dataclass(frozen=True)
class Segment:
    text: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    is_code: bool


def _fence_marker(stripped_line):
    for marker in _FENCE_MARKERS:
        if stripped_line.startswith(marker):
            return marker
    return None


def split_segments(text):
    lines = text.split("\n")
    segments = []
    buf = []
    buf_start = None
    fence = None

    def flush(end_line, is_code):
        if buf:
            segments.append(
                Segment(
                    text="\n".join(buf),
                    start_line=buf_start,
                    end_line=end_line,
                    is_code=is_code,
                )
            )
        buf.clear()

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        if fence is not None:
            buf.append(line)
            if stripped.startswith(fence):
                flush(lineno, is_code=True)
                fence = None
            continue

        opening = _fence_marker(stripped)
        if opening is not None:
            flush(lineno - 1, is_code=False)
            fence = opening
            buf.append(line)
            buf_start = lineno
            continue

        if stripped == "":
            flush(lineno - 1, is_code=False)
            continue

        if not buf:
            buf_start = lineno
        buf.append(line)

    flush(len(lines), is_code=fence is not None)
    return segments
