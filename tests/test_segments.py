from ctx_squeeze.segments import split_segments


def test_empty_text_has_no_segments():
    assert split_segments("") == []


def test_single_paragraph():
    segments = split_segments("just one line")
    assert len(segments) == 1
    assert segments[0].text == "just one line"
    assert segments[0].start_line == 1
    assert segments[0].end_line == 1
    assert segments[0].is_code is False


def test_paragraphs_split_on_blank_lines():
    text = "first paragraph\nstill first\n\nsecond paragraph"
    segments = split_segments(text)
    assert [s.text for s in segments] == [
        "first paragraph\nstill first",
        "second paragraph",
    ]
    assert segments[0].start_line == 1
    assert segments[0].end_line == 2
    assert segments[1].start_line == 4
    assert segments[1].end_line == 4


def test_multiple_blank_lines_collapse():
    text = "a\n\n\n\nb"
    segments = split_segments(text)
    assert [s.text for s in segments] == ["a", "b"]


def test_code_fence_is_one_segment_even_with_blank_lines_inside():
    text = "intro\n\n```python\ndef f():\n\n    return 1\n```\n\noutro"
    segments = split_segments(text)
    assert [s.text for s in segments] == [
        "intro",
        "```python\ndef f():\n\n    return 1\n```",
        "outro",
    ]
    assert [s.is_code for s in segments] == [False, True, False]


def test_code_fence_line_numbers():
    text = "```\ncode\n```"
    segments = split_segments(text)
    assert segments[0].start_line == 1
    assert segments[0].end_line == 3


def test_unterminated_code_fence_is_still_a_segment():
    text = "```\ndangling code"
    segments = split_segments(text)
    assert len(segments) == 1
    assert segments[0].is_code is True
    assert segments[0].text == "```\ndangling code"


def test_text_immediately_before_and_after_fence_split_correctly():
    text = "before\n```\ncode\n```\nafter"
    segments = split_segments(text)
    assert [s.text for s in segments] == ["before", "```\ncode\n```", "after"]


def test_tilde_fence_is_also_recognized():
    text = "~~~\ncode\n~~~"
    segments = split_segments(text)
    assert len(segments) == 1
    assert segments[0].is_code is True
