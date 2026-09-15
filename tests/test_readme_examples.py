"""Check that the ```console examples in README.md still match real CLI output.

Each block is parsed straight out of the README rather than duplicated here,
so a code change that shifts a kept-segment count or a token total fails this
test instead of quietly rotting in the docs.
"""

import io
import re
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ctx_squeeze.cli import main

_README = Path(__file__).resolve().parent.parent / "README.md"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"

_CONSOLE_BLOCK_RE = re.compile(r"```console\n(.*?)```", re.DOTALL)
_PROMPT_PREFIX = "$ ctx-squeeze "


def _parse_console_blocks():
    blocks = []
    for match in _CONSOLE_BLOCK_RE.finditer(_README.read_text(encoding="utf-8")):
        lines = match.group(1).splitlines()
        assert lines and lines[0].startswith(_PROMPT_PREFIX), (
            f"unrecognized console block in README.md:\n{match.group(1)}"
        )
        blocks.append((lines[0][len(_PROMPT_PREFIX):], lines[1:]))
    return blocks


_BLOCKS = _parse_console_blocks()


def _resolve_fixture_args(args):
    """Point bare filenames like sample.md at their checked-in fixture."""
    resolved = []
    for arg in args:
        candidate = _FIXTURES / arg
        resolved.append(str(candidate) if candidate.is_file() else arg)
    return resolved


def _run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        main(args)

    lines = []
    stderr_text = err.getvalue().rstrip("\n")
    if stderr_text:
        lines.extend(stderr_text.split("\n"))
    lines.extend(out.getvalue().rstrip("\n").split("\n"))
    return lines


def _assert_matches_readme(expected_lines, actual_lines):
    """Compare line by line, tolerating a README's own "..." elisions.

    Most examples are reproduced verbatim and must match exactly. A few are
    deliberately trimmed with a "..." line for readability; for those, every
    line the README does show must still appear in the CLI's output, in the
    same order, and the stats line must match exactly either way.
    """
    if not any(line.strip() == "..." for line in expected_lines):
        assert actual_lines == expected_lines
        return

    assert actual_lines[0] == expected_lines[0], "the stats line must match exactly"
    pos = 0
    for line in expected_lines:
        if line.strip() == "...":
            continue
        while pos < len(actual_lines) and actual_lines[pos] != line:
            pos += 1
        assert pos < len(actual_lines), f"README shows a line the CLI never produced: {line!r}"
        pos += 1


def test_readme_has_console_examples_to_check():
    assert len(_BLOCKS) >= 3


@pytest.mark.parametrize("command,expected_lines", _BLOCKS, ids=[c for c, _ in _BLOCKS])
def test_readme_console_example_matches_cli_output(command, expected_lines):
    args = _resolve_fixture_args(shlex.split(command))
    actual_lines = _run_cli(args)
    _assert_matches_readme(expected_lines, actual_lines)
