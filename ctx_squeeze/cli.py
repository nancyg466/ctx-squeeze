"""Command-line interface: wires argparse flags to the library.

Two modes share this entry point: plain-text/segment squeezing (the
default) and structural message pruning (--messages). They share
--budget, --stats, --json, and -o; flags specific to one mode are
simply unused by the other rather than rejected, so a stored alias
doesn't break when the input type changes.
"""

import argparse
import json
import sys

from ctx_squeeze.messages import parse_messages, prune_messages, to_dicts
from ctx_squeeze.squeeze import squeeze


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_output(path, text):
    if not text.endswith("\n"):
        text += "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit long documents and chat transcripts into an LLM token budget.",
    )
    parser.add_argument("input", help="path to the input file, or - for stdin")
    parser.add_argument("--budget", type=int, required=True, help="target size in estimated tokens")
    parser.add_argument(
        "--strategy", default="score", help="comma-separated pipeline: head-tail, score, dedupe"
    )
    parser.add_argument(
        "--head-ratio", type=float, default=0.5, help="share of the budget spent on the head in head-tail"
    )
    parser.add_argument(
        "--jaccard", type=float, default=0.8, help="similarity at which two segments count as duplicates"
    )
    parser.add_argument(
        "--shingle-size", type=int, default=5, help="words per shingle in the dedupe stage"
    )
    parser.add_argument(
        "--messages", action="store_true", help="treat the input as a JSON chat transcript"
    )
    parser.add_argument(
        "--recent-turns", type=int, default=2, help="user turns kept whole in --messages mode"
    )
    parser.add_argument(
        "--no-marker", action="store_true", help="omit the [N segments elided] markers"
    )
    parser.add_argument("--stats", action="store_true", help="print a token summary to stderr")
    parser.add_argument("--json", action="store_true", help="emit a JSON report instead of plain text")
    parser.add_argument("-o", "--output", metavar="PATH", help="write the result to a file (default: stdout)")
    return parser


def _run_messages(args, raw):
    try:
        history = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ctx-squeeze: invalid JSON input: {exc}")
    if not isinstance(history, list):
        raise SystemExit("ctx-squeeze: --messages input must be a JSON array")

    messages = parse_messages(history)
    result = prune_messages(
        messages,
        budget=args.budget,
        recent_turns=args.recent_turns,
        marker=not args.no_marker,
    )

    if args.stats:
        print(
            f"kept {result.messages_out} of {result.messages_in} messages | "
            f"{result.original_tokens} -> {result.final_tokens} tokens (budget {args.budget})",
            file=sys.stderr,
        )

    if args.json:
        report = {
            "messages": to_dicts(result.messages),
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "messages_in": result.messages_in,
            "messages_out": result.messages_out,
            "notes": result.notes,
            "pinned_tool_results": sorted(result.pinned_tool_results),
        }
        return json.dumps(report, indent=2)
    return json.dumps(to_dicts(result.messages), indent=2)


def _run_text(args, raw):
    result = squeeze(
        raw,
        budget=args.budget,
        strategy=args.strategy,
        head_ratio=args.head_ratio,
        jaccard_threshold=args.jaccard,
        shingle_size=args.shingle_size,
        marker=not args.no_marker,
    )

    if args.stats:
        print(
            f"kept {result.segments_out} of {result.segments_in} segments | "
            f"{result.original_tokens} -> {result.final_tokens} tokens (budget {args.budget})",
            file=sys.stderr,
        )

    if args.json:
        report = {
            "text": result.text,
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "segments_in": result.segments_in,
            "segments_out": result.segments_out,
            "notes": result.notes,
        }
        return json.dumps(report, indent=2)
    return result.text


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw = _read_input(args.input)

    try:
        if args.messages:
            output = _run_messages(args, raw)
        else:
            output = _run_text(args, raw)
    except ValueError as exc:
        raise SystemExit(f"ctx-squeeze: {exc}")

    _write_output(args.output, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
