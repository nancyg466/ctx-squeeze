from ctx_squeeze.squeeze import squeeze
from ctx_squeeze.tokens import estimate_tokens

_DOC = (
    "# Nightly build postmortem\n"
    "\n"
    "The nightly job started failing on Tuesday after the runner image "
    "was bumped.\n"
    "\n"
    "## Fix\n"
    "\n"
    "Restore the cache step and pin the runner image to the previous "
    "minor version.\n"
    "\n"
    "## Follow up\n"
    "\n"
    "Add an alert that fires when the nightly job runs longer than "
    "eight minutes.\n"
)


def test_empty_text_squeezes_to_empty():
    result = squeeze("", budget=100)
    assert result.text == ""
    assert result.original_tokens == 0
    assert result.final_tokens == 0
    assert result.segments_in == 0
    assert result.segments_out == 0


def test_generous_budget_keeps_everything():
    budget = estimate_tokens(_DOC) + 50
    result = squeeze(_DOC, budget=budget)
    assert result.segments_out == result.segments_in
    assert "elided" not in result.text
    assert result.final_tokens <= budget


def test_result_never_exceeds_budget():
    for budget in (0, 1, 5, 20, 1000):
        result = squeeze(_DOC, budget=budget, strategy="score")
        assert result.final_tokens <= budget
        assert estimate_tokens(result.text) <= budget


def test_tight_budget_inserts_elision_marker():
    # head-tail's selection is deterministic (cost-driven, not scored), so
    # the outcome here doesn't depend on how score_segments ranks anything.
    result = squeeze(_DOC, budget=30, strategy="head-tail")
    assert result.segments_out < result.segments_in
    assert "elided" in result.text


def test_no_marker_omits_elision_text():
    result = squeeze(_DOC, budget=20, strategy="score", marker=False)
    assert "elided" not in result.text


def test_dedupe_then_score_removes_repeats_before_selecting():
    paragraph = "the nightly job failed after the image bump and retries"
    doc = "\n\n".join([paragraph, paragraph, "a distinct closing note"])
    budget = estimate_tokens(doc)
    result = squeeze(doc, budget=budget, strategy="dedupe,score")
    assert any("dedupe dropped" in note for note in result.notes)
    assert result.text.count(paragraph) == 1


def test_dedupe_alone_still_honors_budget():
    paragraph = "the nightly job failed after the image bump and retries"
    doc = "\n\n".join([paragraph, paragraph, "a distinct closing note"])
    result = squeeze(doc, budget=5, strategy="dedupe")
    assert result.final_tokens <= 5


def test_head_tail_keeps_start_and_end():
    budget = 40
    result = squeeze(_DOC, budget=budget, strategy="head-tail", head_ratio=0.5)
    assert result.text.startswith("# Nightly build postmortem")
    assert "eight minutes" in result.text
    assert result.final_tokens <= budget


def test_zero_budget_yields_empty_text():
    result = squeeze(_DOC, budget=0)
    assert result.text == ""
    assert result.final_tokens == 0


def test_unknown_strategy_stage_raises():
    try:
        squeeze(_DOC, budget=10, strategy="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_negative_budget_raises():
    try:
        squeeze(_DOC, budget=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
