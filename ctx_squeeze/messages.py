"""Structural pruning of chat transcripts.

Unlike squeeze(), which drops arbitrary text segments, a chat transcript has
shape that matters: a tool result read out of order looks like nonsense to
a model, and dropping a system prompt changes its behavior rather than just
its context. So pruning here is structural first and budget-driven second.
System messages are always kept. The most recent `recent_turns` user turns
are kept whole. A tool call and the result that answers it are kept or
dropped together, in either direction, by grouping messages that reference
the same tool-call id into one unit before anything is dropped. Only once
that structure is settled does the budget get enforced, oldest units first,
with a hard drop-everything fallback so the result never exceeds it.

Both OpenAI-style messages (top-level `tool_calls` / `tool_call_id`) and
Anthropic-style messages (`tool_use` / `tool_result` blocks inside
`content`) are understood, since both shapes show up in transcripts pulled
from real agent runs.
"""

from dataclasses import dataclass

from ctx_squeeze.tokens import estimate_tokens


@dataclass(frozen=True)
class Message:
    role: str
    content: object
    tool_calls: object
    tool_call_id: object

    def to_dict(self):
        d = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass(frozen=True)
class PruneResult:
    messages: list
    pinned_tool_results: frozenset
    original_tokens: int
    final_tokens: int
    messages_in: int
    messages_out: int
    notes: list


def parse_messages(history):
    """Parse a JSON-decoded list of message dicts into `Message` objects.

    Each entry must be an object with at least a `role`. `content`,
    `tool_calls`, and `tool_call_id` are carried over verbatim (including
    Anthropic-style `content` block lists) so `to_dicts()` can round-trip
    the shape the caller passed in.
    """
    messages = []
    for entry in history:
        if not isinstance(entry, dict) or "role" not in entry:
            raise ValueError("each message must be an object with a 'role' field")
        messages.append(
            Message(
                role=entry["role"],
                content=entry.get("content"),
                tool_calls=entry.get("tool_calls"),
                tool_call_id=entry.get("tool_call_id"),
            )
        )
    return messages


def to_dicts(messages):
    return [m.to_dict() for m in messages]


def _message_text(message):
    """Best-effort plain-text rendering of a message, for token estimation."""
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "content", "input"):
                    if key in block:
                        parts.append(str(block[key]))
                        break
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _issued_tool_call_ids(message):
    """Ids of tool calls this message asks the caller to run."""
    ids = set()
    if message.tool_calls:
        for call in message.tool_calls:
            if isinstance(call, dict) and "id" in call:
                ids.add(call["id"])
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and "id" in block:
                ids.add(block["id"])
    return ids


def _fulfilled_tool_call_ids(message):
    """Ids of tool calls this message reports the result of."""
    ids = set()
    if message.tool_call_id is not None:
        ids.add(message.tool_call_id)
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and "tool_use_id" in block:
                ids.add(block["tool_use_id"])
    return ids


def _link_components(messages):
    """Group message indices that share a tool-call id into one unit.

    Union-find over "who issued this id" / "who fulfilled this id" so that
    a call and its result end up in the same component regardless of which
    side, or how many messages separate them. Messages with no tool-call
    involvement end up as singleton components.
    """
    parent = list(range(len(messages)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    issuer_of_id = {}
    for i, message in enumerate(messages):
        for call_id in _issued_tool_call_ids(message):
            issuer_of_id[call_id] = i
    for i, message in enumerate(messages):
        for call_id in _fulfilled_tool_call_ids(message):
            if call_id in issuer_of_id:
                union(i, issuer_of_id[call_id])

    groups = {}
    for i in range(len(messages)):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def _assemble(messages, kept, marker):
    result = []
    gap = 0
    for i, message in enumerate(messages):
        if i in kept:
            if gap:
                if marker:
                    plural = "s" if gap != 1 else ""
                    result.append(
                        Message(
                            role="system",
                            content=f"[{gap} earlier message{plural} elided]",
                            tool_calls=None,
                            tool_call_id=None,
                        )
                    )
                gap = 0
            result.append(message)
        else:
            gap += 1
    if gap and marker:
        plural = "s" if gap != 1 else ""
        result.append(
            Message(
                role="system",
                content=f"[{gap} earlier message{plural} elided]",
                tool_calls=None,
                tool_call_id=None,
            )
        )
    return result


def _tokens(messages):
    return sum(estimate_tokens(_message_text(m)) for m in messages)


def prune_messages(messages, budget, recent_turns=2, marker=True):
    """Prune a parsed transcript to at most `budget` estimated tokens.

    Keeps every system message and the last `recent_turns` user turns
    (a turn is a user message plus everything up to the next user
    message) whole, then pulls in whatever tool calls or results those
    kept messages are paired with even if they fall outside that window.
    `pinned_tool_results` on the result reports which tool-call ids got
    pulled in that way.

    If the structurally-kept set is still over budget, whole linked
    units are dropped oldest-first: units pulled in purely for pairing
    go first, then non-system units in the kept window, and only as a
    last resort a unit containing a system message. If even an empty
    keep set with just the elision marker doesn't fit, the marker is
    dropped too, so the result never exceeds `budget`.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")

    n = len(messages)
    original_tokens = _tokens(messages)
    if n == 0:
        return PruneResult(
            messages=[],
            pinned_tool_results=frozenset(),
            original_tokens=0,
            final_tokens=0,
            messages_in=0,
            messages_out=0,
            notes=[],
        )

    user_indices = [i for i, m in enumerate(messages) if m.role == "user"]
    recent_turns = max(recent_turns, 0)
    if recent_turns > 0 and user_indices:
        keep_from = user_indices[max(0, len(user_indices) - recent_turns)]
    elif recent_turns > 0:
        keep_from = 0
    else:
        keep_from = n

    base_keep = {i for i, m in enumerate(messages) if m.role == "system" or i >= keep_from}

    components = _link_components(messages)
    kept_components = {
        c for c, group in enumerate(components) if group & base_keep
    }
    kept = set()
    for c in kept_components:
        kept |= components[c]

    def priority(c):
        group = components[c]
        if any(messages[i].role == "system" for i in group):
            return 2
        if group & base_keep:
            return 1
        return 0

    order = sorted(kept_components, key=lambda c: (priority(c), min(components[c])))

    notes = []
    dropped_units = 0
    result_messages = _assemble(messages, kept, marker)
    final_tokens = _tokens(result_messages)
    for c in order:
        if final_tokens <= budget:
            break
        kept -= components[c]
        dropped_units += 1
        result_messages = _assemble(messages, kept, marker)
        final_tokens = _tokens(result_messages)

    if dropped_units:
        notes.append(f"budget trimming dropped {dropped_units} linked message group(s)")

    if final_tokens > budget:
        kept = set()
        result_messages = []
        final_tokens = 0
        notes.append("hard-dropped all remaining messages to stay within budget")

    pinned_tool_results = set()
    for i in kept - base_keep:
        pinned_tool_results |= _issued_tool_call_ids(messages[i])
        pinned_tool_results |= _fulfilled_tool_call_ids(messages[i])

    return PruneResult(
        messages=result_messages,
        pinned_tool_results=frozenset(pinned_tool_results),
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        messages_in=n,
        messages_out=len(kept),
        notes=notes,
    )
