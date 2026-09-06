"""Top-level pipeline: segment, run the requested stages, assemble.

squeeze() is the one call the CLI and most library callers need. It
splits a document into segments, runs a comma-separated pipeline of
stages over them, and reassembles the survivors with elision markers
where whole segments were dropped. The budget is a promise, not a
suggestion: if the requested stages don't happen to land at or under it
(most commonly because "dedupe" ran without a following "score" or
"head-tail" to actually pick a budget-fitting subset, or because the
elision markers themselves pushed the assembled text back over), a
final hard truncation makes sure the caller never gets back more than
they asked for.
"""

from dataclasses import dataclass

from ctx_squeeze.dedupe import jaccard, shingles
from ctx_squeeze.scoring import select_by_score
from ctx_squeeze.segments import split_segments
from ctx_squeeze.tokens import estimate_tokens, truncate_to_tokens

_BUDGET_SELECTING_STAGES = ("head-tail", "score")
_VALID_STAGES = _BUDGET_SELECTING_STAGES + ("dedupe",)


@dataclass(frozen=True)
class SqueezeResult:
    text: str
    original_tokens: int
    final_tokens: int
    segments_in: int
    segments_out: int
    notes: list


def _dedupe_stage(segments, jaccard_threshold, shingle_size):
    """Drop segments that are near-duplicates of an earlier kept segment.

    Each survivor is compared against every previously kept segment's
    shingle set, so a duplicate that appears three times only keeps the
    first occurrence rather than pairing up duplicates two at a time.
    """
    kept = []
    kept_shingles = []
    dropped = 0
    for segment in segments:
        current = shingles(segment.text, size=shingle_size)
        if any(jaccard(current, other) >= jaccard_threshold for other in kept_shingles):
            dropped += 1
            continue
        kept.append(segment)
        kept_shingles.append(current)

    notes = [f"dedupe dropped {dropped} near-duplicate segment(s)"] if dropped else []
    return kept, notes


def _head_tail_stage(segments, budget, head_ratio):
    """Keep a run of whole segments from the start and from the end.

    The budget is split head/tail by `head_ratio` and each side is
    filled independently, so a document that's short enough to keep
    everything on one side still doesn't dip into the other side's
    share. The two runs never overlap: the tail scan stops at wherever
    the head scan stopped.
    """
    if budget <= 0 or not segments:
        return [], []

    head_budget = int(budget * head_ratio)
    tail_budget = budget - head_budget

    head = []
    used = 0
    i = 0
    while i < len(segments):
        cost = estimate_tokens(segments[i].text)
        if used + cost > head_budget:
            break
        head.append(segments[i])
        used += cost
        i += 1

    tail = []
    used = 0
    j = len(segments) - 1
    while j >= i:
        cost = estimate_tokens(segments[j].text)
        if used + cost > tail_budget:
            break
        tail.append(segments[j])
        used += cost
        j -= 1
    tail.reverse()

    kept = head + tail
    dropped = len(segments) - len(kept)
    notes = [f"head-tail dropped {dropped} segment(s)"] if dropped else []
    return kept, notes


def _score_stage(segments, budget):
    kept = select_by_score(segments, budget)
    dropped = len(segments) - len(kept)
    notes = [f"score dropped {dropped} segment(s)"] if dropped else []
    return kept, notes


def _assemble(segments, kept, marker):
    """Rejoin surviving segments, marking runs of dropped ones.

    Segments compare by value including their line numbers, so two
    segments with identical text at different positions never collide
    in the membership check below.
    """
    kept_set = set(kept)
    pieces = []
    gap = 0
    for segment in segments:
        if segment in kept_set:
            if gap:
                if marker:
                    plural = "s" if gap != 1 else ""
                    pieces.append(f"[{gap} segment{plural} elided]")
                gap = 0
            pieces.append(segment.text)
        else:
            gap += 1
    if gap and marker:
        plural = "s" if gap != 1 else ""
        pieces.append(f"[{gap} segment{plural} elided]")
    return "\n\n".join(pieces)


def squeeze(
    text,
    budget,
    strategy="score",
    head_ratio=0.5,
    jaccard_threshold=0.8,
    shingle_size=5,
    marker=True,
):
    """Compact `text` to at most `budget` estimated tokens.

    `strategy` is a comma-separated pipeline run left to right, chosen
    from "head-tail", "score", and "dedupe". "dedupe" removes near-
    duplicate segments without regard to the budget, so it's meant to
    run before one of the other two; if the pipeline never runs a
    budget-selecting stage, a "head-tail" pass is appended automatically
    so the result still honors the budget.

    Returns a SqueezeResult. `final_tokens` is guaranteed to be at most
    `budget` regardless of which stages ran or how much the elision
    markers themselves cost, since a hard truncation is applied to the
    assembled text as a last resort if it's still over.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")

    stages = [s.strip() for s in strategy.split(",") if s.strip()]
    for stage in stages:
        if stage not in _VALID_STAGES:
            raise ValueError(f"unknown strategy stage: {stage!r}")

    original_tokens = estimate_tokens(text)
    segments = split_segments(text)

    kept = segments
    notes = []
    budget_selected = False

    for stage in stages:
        if stage == "dedupe":
            kept, stage_notes = _dedupe_stage(kept, jaccard_threshold, shingle_size)
        elif stage == "head-tail":
            kept, stage_notes = _head_tail_stage(kept, budget, head_ratio)
            budget_selected = True
        else:  # "score"
            kept, stage_notes = _score_stage(kept, budget)
            budget_selected = True
        notes.extend(stage_notes)

    if not budget_selected:
        kept, stage_notes = _head_tail_stage(kept, budget, head_ratio)
        notes.extend(stage_notes)

    result_text = _assemble(segments, kept, marker)
    final_tokens = estimate_tokens(result_text)
    if final_tokens > budget:
        result_text = truncate_to_tokens(result_text, budget)
        final_tokens = estimate_tokens(result_text)
        notes.append("hard-truncated the assembled text to stay within budget")

    return SqueezeResult(
        text=result_text,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        segments_in=len(segments),
        segments_out=len(kept),
        notes=notes,
    )
