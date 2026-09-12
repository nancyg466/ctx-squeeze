import json

import pytest

from ctx_squeeze.cli import main

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
)

_CHAT = [
    {"role": "system", "content": "You are a careful build engineer."},
    {"role": "user", "content": "Can you patch the workflow file?"},
    {"role": "assistant", "content": "Sure, done."},
]


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_text_mode_writes_squeezed_text_to_stdout(tmp_path, capsys):
    src = _write(tmp_path, "doc.md", _DOC)
    rc = main([src, "--budget", "1000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() == _DOC.strip()


def test_stats_go_to_stderr_not_stdout(tmp_path, capsys):
    src = _write(tmp_path, "doc.md", _DOC)
    main([src, "--budget", "10", "--strategy", "head-tail", "--stats"])
    captured = capsys.readouterr()
    assert "kept" in captured.err
    assert "tokens (budget 10)" in captured.err
    assert "kept" not in captured.out


def test_json_report_has_expected_keys(tmp_path, capsys):
    src = _write(tmp_path, "doc.md", _DOC)
    main([src, "--budget", "1000", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "text",
        "original_tokens",
        "final_tokens",
        "segments_in",
        "segments_out",
        "notes",
    }


def test_output_flag_writes_to_file_instead_of_stdout(tmp_path, capsys):
    src = _write(tmp_path, "doc.md", _DOC)
    dest = tmp_path / "out.txt"
    main([src, "--budget", "1000", "-o", str(dest)])
    assert capsys.readouterr().out == ""
    assert dest.read_text(encoding="utf-8").strip() == _DOC.strip()


def test_dash_reads_from_stdin(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(_DOC))
    main(["-", "--budget", "1000"])
    assert capsys.readouterr().out.strip() == _DOC.strip()


def test_messages_mode_default_output_is_json_array(tmp_path, capsys):
    src = _write(tmp_path, "chat.json", json.dumps(_CHAT))
    main([src, "--messages", "--budget", "1000"])
    messages = json.loads(capsys.readouterr().out)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_messages_mode_json_report_has_expected_keys(tmp_path, capsys):
    src = _write(tmp_path, "chat.json", json.dumps(_CHAT))
    main([src, "--messages", "--budget", "1000", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "messages",
        "original_tokens",
        "final_tokens",
        "messages_in",
        "messages_out",
        "notes",
        "pinned_tool_results",
    }


def test_messages_mode_rejects_invalid_json(tmp_path):
    src = _write(tmp_path, "chat.json", "not json")
    with pytest.raises(SystemExit):
        main([src, "--messages", "--budget", "100"])


def test_messages_mode_rejects_non_array_json(tmp_path):
    src = _write(tmp_path, "chat.json", json.dumps({"role": "user"}))
    with pytest.raises(SystemExit):
        main([src, "--messages", "--budget", "100"])


def test_unknown_strategy_exits_with_error(tmp_path):
    src = _write(tmp_path, "doc.md", _DOC)
    with pytest.raises(SystemExit):
        main([src, "--budget", "100", "--strategy", "bogus"])


def test_no_marker_flag_omits_elision_text(tmp_path, capsys):
    src = _write(tmp_path, "doc.md", _DOC)
    main([src, "--budget", "10", "--strategy", "score", "--no-marker"])
    assert "elided" not in capsys.readouterr().out
