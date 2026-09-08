from ctx_squeeze.messages import Message, parse_messages, prune_messages, to_dicts


def _msg(role, content, tool_calls=None, tool_call_id=None):
    d = {"role": role, "content": content}
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    if tool_call_id is not None:
        d["tool_call_id"] = tool_call_id
    return d


def test_parse_messages_round_trips_through_to_dicts():
    raw = [
        _msg("system", "be careful"),
        _msg("user", "patch the workflow file"),
        _msg("assistant", None, tool_calls=[{"id": "call_1", "type": "function"}]),
        _msg("tool", "patched", tool_call_id="call_1"),
    ]
    parsed = parse_messages(raw)
    assert all(isinstance(m, Message) for m in parsed)
    assert to_dicts(parsed) == raw


def test_parse_messages_rejects_entry_without_role():
    try:
        parse_messages([{"content": "no role here"}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_empty_history_prunes_to_empty():
    result = prune_messages([], budget=100)
    assert result.messages == []
    assert result.original_tokens == 0
    assert result.final_tokens == 0
    assert result.messages_in == 0
    assert result.messages_out == 0


def test_negative_budget_raises():
    try:
        prune_messages(parse_messages([_msg("user", "hi")]), budget=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_system_messages_always_survive_outside_the_recent_window():
    raw = [
        _msg("system", "you are a careful build engineer"),
        _msg("user", "old question about the first failure"),
        _msg("assistant", "old answer about the first failure"),
        _msg("user", "second, unrelated question"),
        _msg("assistant", "second answer"),
    ]
    result = prune_messages(parse_messages(raw), budget=1000, recent_turns=1)
    roles = [m.role for m in result.messages]
    assert roles[0] == "system"
    assert result.messages[0].content == raw[0]["content"]


def test_recent_turns_drops_older_turns_as_one_gap():
    raw = [
        _msg("system", "sys"),
        _msg("user", "turn one question"),
        _msg("assistant", "turn one answer"),
        _msg("user", "turn two question"),
        _msg("assistant", "turn two answer"),
    ]
    result = prune_messages(parse_messages(raw), budget=1000, recent_turns=1)
    contents = [m.content for m in result.messages]
    assert contents == [
        "sys",
        "[2 earlier messages elided]",
        "turn two question",
        "turn two answer",
    ]
    assert result.messages_out == 3
    assert result.messages_in == 5


def test_tool_call_and_result_kept_together_when_call_is_outside_window():
    raw = [
        _msg("system", "sys"),
        _msg("user", "turn one, please read the file"),
        _msg("assistant", None, tool_calls=[{"id": "call_9", "type": "function"}]),
        _msg("user", "turn two"),
        _msg("tool", "file contents from turn one's call", tool_call_id="call_9"),
        _msg("assistant", "turn two answer"),
    ]
    result = prune_messages(parse_messages(raw), budget=1000, recent_turns=1)
    tool_call_ids = [m.tool_calls[0]["id"] for m in result.messages if m.tool_calls]
    assert tool_call_ids == ["call_9"]
    assert any(m.tool_call_id == "call_9" for m in result.messages)
    assert "call_9" in result.pinned_tool_results


def test_unpaired_tool_call_outside_window_is_dropped():
    raw = [
        _msg("system", "sys"),
        _msg("user", "turn one, please read the file"),
        _msg("assistant", None, tool_calls=[{"id": "call_9", "type": "function"}]),
        _msg("tool", "file contents", tool_call_id="call_9"),
        _msg("assistant", "turn one answer"),
        _msg("user", "turn two"),
        _msg("assistant", "turn two answer"),
    ]
    result = prune_messages(parse_messages(raw), budget=1000, recent_turns=1)
    assert not any(m.tool_calls for m in result.messages)
    assert not any(m.tool_call_id for m in result.messages)
    assert result.pinned_tool_results == frozenset()


def test_anthropic_style_tool_use_and_result_blocks_are_paired():
    # Anthropic-shaped tool results are user-role messages (there's no
    # separate "tool" role), so the tool_use call at index 2 falls just
    # outside a recent_turns=2 window that starts at its tool_result (a
    # "user" message) at index 3 - exercising the same pull-in-from-
    # outside-the-window path as the OpenAI-style test above.
    raw = [
        _msg("system", "sys"),
        _msg("user", "please read the file"),
        _msg(
            "assistant",
            [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}],
        ),
        _msg("user", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]),
        _msg("assistant", "turn one answer using the file contents"),
        _msg("user", "a second, real question"),
        _msg("assistant", "turn two answer"),
    ]
    result = prune_messages(parse_messages(raw), budget=1000, recent_turns=2)
    assert any(
        isinstance(m.content, list)
        and any(b.get("type") == "tool_use" for b in m.content)
        for m in result.messages
    )
    assert "toolu_1" in result.pinned_tool_results


def test_result_never_exceeds_budget():
    raw = [
        _msg("system", "you are a careful build engineer with detailed instructions"),
        _msg("user", "first question about a failing build"),
        _msg("assistant", "first detailed answer about the failing build"),
        _msg("user", "second question that is completely different"),
        _msg("assistant", "second detailed answer with more words than needed"),
    ]
    parsed = parse_messages(raw)
    for budget in (0, 1, 5, 20, 1000):
        result = prune_messages(parsed, budget=budget, recent_turns=1)
        assert result.final_tokens <= budget


def test_no_marker_omits_elision_text():
    raw = [
        _msg("system", "sys"),
        _msg("user", "turn one"),
        _msg("assistant", "turn one answer"),
        _msg("user", "turn two"),
    ]
    result = prune_messages(parse_messages(raw), budget=1000, recent_turns=1, marker=False)
    assert not any("elided" in str(m.content) for m in result.messages)
