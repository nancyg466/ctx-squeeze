import json
from pathlib import Path

from ctx_squeeze import (
    jaccard,
    parse_messages,
    prune_messages,
    shingles,
    split_segments,
    squeeze,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _sample_text():
    return (_FIXTURES / "sample.md").read_text(encoding="utf-8")


def _chat_history():
    return json.loads((_FIXTURES / "chat.json").read_text(encoding="utf-8"))


def test_sample_md_keeps_its_code_fence_as_one_segment():
    segments = split_segments(_sample_text())
    code_segments = [s for s in segments if s.is_code]
    assert len(code_segments) == 1
    assert code_segments[0].text.startswith("```python")
    assert code_segments[0].text.rstrip().endswith("```")


def test_sample_md_retry_paragraphs_are_near_duplicates():
    segments = split_segments(_sample_text())
    retry_1 = next(s for s in segments if s.text.startswith("Retry 1"))
    retry_2 = next(s for s in segments if s.text.startswith("Retry 2"))
    assert jaccard(shingles(retry_1.text), shingles(retry_2.text)) >= 0.8


def test_squeeze_dedupe_drops_the_repeated_retry_note():
    result = squeeze(_sample_text(), budget=1000, strategy="dedupe")
    assert "Retry 1 hit the same failure" in result.text
    assert "Retry 2 hit the same failure" not in result.text
    assert "[1 segment elided]" in result.text
    assert result.segments_out == result.segments_in - 1
    assert "dedupe dropped 1 near-duplicate segment(s)" in result.notes


def test_squeeze_score_strategy_never_exceeds_the_budget():
    budget = 50
    result = squeeze(_sample_text(), budget=budget, strategy="score")
    assert result.final_tokens <= budget
    assert result.segments_out < result.segments_in


def test_squeeze_head_tail_keeps_the_title_and_the_last_paragraph():
    result = squeeze(_sample_text(), budget=50, strategy="head-tail")
    assert result.text.startswith("# Nightly build postmortem")
    assert result.text.rstrip().endswith(
        "Add an alert that fires when the nightly job runs longer than eight minutes."
    )
    assert result.final_tokens <= 50


def test_chat_json_pruning_keeps_system_message_and_last_turn():
    history = _chat_history()
    messages = parse_messages(history)
    result = prune_messages(messages, budget=80, recent_turns=1)

    assert result.final_tokens <= 80
    assert result.messages[0].role == "system"
    assert result.messages[0].content == history[0]["content"]
    assert "elided" in result.messages[1].content
    assert result.messages[-1].content == history[-1]["content"]
    assert result.messages[-2].content == history[-2]["content"]


def test_chat_json_tool_calls_and_results_are_never_split():
    history = _chat_history()
    messages = parse_messages(history)
    result = prune_messages(messages, budget=10_000, recent_turns=2)

    issued = set()
    fulfilled = set()
    for message in result.messages:
        if message.tool_calls:
            issued.update(call["id"] for call in message.tool_calls)
        if message.tool_call_id is not None:
            fulfilled.add(message.tool_call_id)
    assert issued == fulfilled
