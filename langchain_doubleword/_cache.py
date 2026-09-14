"""Prompt caching helpers for :class:`~langchain_doubleword.ChatDoubleword`.

Doubleword serves Anthropic-style prompt caching over its OpenAI-compatible
``/chat/completions`` endpoint: mark a message content block with
``cache_control`` and everything up to that mark becomes a reusable prefix
(left-anchored, ~1024-token floor, ttl ``5m`` or ``1h``). ``langchain-openai``
never emits ``cache_control`` on its own, so these helpers stamp it onto the
already-assembled request payload, driven by ``ChatDoubleword(prompt_cache=...)``.

Everything here is a pure transform over the request payload dict, so it is
unit-testable without the network.
"""

from typing import Any, Literal, NamedTuple

from typing_extensions import TypedDict

CacheTTL = Literal["5m", "1h"]
"""Cache lifetime. Doubleword accepts ``"5m"`` or ``"1h"``."""

CacheScope = Literal["system", "lastUser"] | list[int]
"""Which messages carry the cache breakpoint.

* ``"system"`` (default): the last system message, caching the system prefix.
* ``"lastUser"``: the last user message.
* ``list[int]``: exact message indices, for full control.
"""


class CacheConfig(TypedDict, total=False):
    """Tuning for the ``prompt_cache`` option."""

    ttl: CacheTTL
    scope: CacheScope


CacheOption = bool | CacheConfig
"""``True`` uses the defaults; ``False`` disables caching."""


class ResolvedCacheConfig(NamedTuple):
    """A ``prompt_cache`` option with every default filled in."""

    ttl: CacheTTL
    scope: CacheScope


def normalize_cache_config(option: CacheOption | None) -> ResolvedCacheConfig | None:
    """Normalize a ``prompt_cache`` option into a concrete config.

    Returns ``None`` when caching is off, which every helper below treats as a
    no-op.
    """
    if option is None:
        return None
    if isinstance(option, bool):
        return ResolvedCacheConfig(ttl="1h", scope="system") if option else None
    return ResolvedCacheConfig(
        ttl=option.get("ttl", "1h"),
        scope=option.get("scope", "system"),
    )


def _resolve_targets(messages: list[Any], scope: CacheScope) -> list[int]:
    """Return the indices of the messages that should carry the breakpoint."""
    if isinstance(scope, list):
        return [i for i in scope if isinstance(i, int) and 0 <= i < len(messages)]
    role = "user" if scope == "lastUser" else "system"
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == role:
            return [index]
    return []


def _mark_message(message: dict[str, Any], ttl: CacheTTL) -> None:
    """Attach ``cache_control`` to one message, converting string content to the
    block form the endpoint expects. A no-op if the message has no text to mark.
    """
    cache_control = {"type": "ephemeral", "ttl": ttl}
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "text":
                part["cache_control"] = cache_control
                return


def apply_cache_control(
    payload: dict[str, Any],
    config: ResolvedCacheConfig | None,
) -> dict[str, Any]:
    """Stamp ``cache_control`` onto the request payload in place and return it.

    Safe to call with a ``None`` config, which returns the payload untouched so
    hand-built ``cache_control`` blocks are forwarded exactly as written.
    """
    if config is None:
        return payload
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    for index in _resolve_targets(messages, config.scope):
        message = messages[index]
        if isinstance(message, dict):
            _mark_message(message, config.ttl)
    return payload
