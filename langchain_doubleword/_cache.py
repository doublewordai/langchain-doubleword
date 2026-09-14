"""Prompt caching breakpoints for the chat completions payload."""

from typing import Any, Literal, NotRequired

from typing_extensions import TypedDict

MAX_BREAKPOINTS = 4


class CacheControl(TypedDict):
    """A prompt caching breakpoint, as defined in Doubleword's prompt caching docs."""

    type: Literal["ephemeral"]
    ttl: NotRequired[Literal["5m", "1h"]]


def _is_marked(item: Any) -> bool:
    return isinstance(item, dict) and "cache_control" in item


def _mark(message: dict[str, Any], cache_control: CacheControl) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or any(_is_marked(block) for block in content):
        return False
    texts = [
        i
        for i, block in enumerate(content)
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]
    if not texts:
        return False
    # Copy so the caller's message blocks are never mutated.
    content = list(content)
    content[texts[-1]] = {**content[texts[-1]], "cache_control": dict(cache_control)}
    message["content"] = content
    return True


def apply_cache_control(payload: dict[str, Any], cache_control: CacheControl) -> None:
    """Mark the last system message and the latest message within the breakpoint limit."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    blocks = [b for m in messages if isinstance(m.get("content"), list) for b in m["content"]]
    count = sum(_is_marked(item) for item in [*(payload.get("tools") or []), *blocks])
    latest = len(messages) - 1
    systems = [i for i, m in enumerate(messages) if m.get("role") == "system"]
    targets = [systems[-1], latest] if systems and systems[-1] != latest else [latest]
    for index in targets:
        if count < MAX_BREAKPOINTS and _mark(messages[index], cache_control):
            count += 1
