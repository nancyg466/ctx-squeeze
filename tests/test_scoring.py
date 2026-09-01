from ctx_squeeze.scoring import score_segments, select_by_score
from ctx_squeeze.segments import Segment
from ctx_squeeze.tokens import estimate_tokens


def _seg(text):
    return Segment(text=text, start_line=1, end_line=1, is_code=False)


def test_no_segments_scores_empty():
    assert score_segments([]) == []


def test_wordless_segment_scores_zero():
    scores = score_segments([_seg("---"), _seg("hello world")])
    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_distinctive_segment_outscores_repeated_boilerplate():
    boilerplate = "boilerplate filler boilerplate filler standard message"
    segments = [_seg(boilerplate), _seg(boilerplate), _seg(
        "critical failure alert threshold exceeded urgent action required"
    )]
    scores = score_segments(segments)
    assert scores[0] == scores[1]
    assert scores[2] > scores[0]


def test_select_by_score_zero_budget_selects_nothing():
    segments = [_seg("alpha beta"), _seg("gamma delta")]
    assert select_by_score(segments, 0) == []


def test_select_by_score_stays_within_budget():
    segments = [
        _seg("the nightly build failed after the runner image was bumped"),
        _seg("every run now spends eleven minutes reinstalling dependencies"),
        _seg("add an alert that fires when the job runs too long"),
    ]
    budget = 12
    kept = select_by_score(segments, budget)
    assert sum(estimate_tokens(s.text) for s in kept) <= budget


def test_select_by_score_prefers_higher_scoring_segment():
    boilerplate = "boilerplate filler boilerplate filler standard message"
    distinct = "critical failure alert threshold exceeded urgent action required"
    segments = [_seg(boilerplate), _seg(distinct)]
    budget = estimate_tokens(distinct)
    kept = select_by_score(segments, budget)
    assert kept == [segments[1]]


def test_select_by_score_preserves_original_order():
    boilerplate = "boilerplate filler boilerplate filler standard message"
    distinct = "critical failure alert threshold exceeded urgent action required"
    segments = [_seg(boilerplate), _seg(boilerplate), _seg(distinct)]
    budget = estimate_tokens(boilerplate) + estimate_tokens(distinct)
    kept = select_by_score(segments, budget)
    # highest score (segments[2]) is picked first by the greedy search,
    # then segments[0] (lower index than the tied segments[1]), but the
    # result should read in document order, not selection order.
    assert kept == [segments[0], segments[2]]


def test_select_by_score_rejects_mismatched_scores_length():
    segments = [_seg("one"), _seg("two")]
    try:
        select_by_score(segments, 10, scores=[1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
