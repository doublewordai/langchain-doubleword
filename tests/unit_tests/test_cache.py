"""Unit tests for prompt caching.

Two layers: the pure payload transforms in ``_cache``, and the
``ChatDoubleword._get_request_payload`` hook that drives them. Both run
offline: nothing here touches the network.
"""

from typing import Any

import pytest

from langchain_doubleword import ChatDoubleword, ChatDoublewordAsync, ChatDoublewordBatch
from langchain_doubleword._cache import (
    CacheOption,
    ResolvedCacheConfig,
    apply_cache_control,
    normalize_cache_config,
)

EPHEMERAL_1H = {"type": "ephemeral", "ttl": "1h"}
EPHEMERAL_5M = {"type": "ephemeral", "ttl": "5m"}


def _payload(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"model": "m", "messages": list(messages)}


# ---------------------------------------------------------------------------
# normalize_cache_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        (None, None),
        (False, None),
        (True, ResolvedCacheConfig(ttl="1h", scope="system")),
        ({}, ResolvedCacheConfig(ttl="1h", scope="system")),
        ({"ttl": "5m"}, ResolvedCacheConfig(ttl="5m", scope="system")),
        ({"scope": "lastUser"}, ResolvedCacheConfig(ttl="1h", scope="lastUser")),
        ({"ttl": "5m", "scope": [0, 2]}, ResolvedCacheConfig(ttl="5m", scope=[0, 2])),
    ],
)
def test_normalize_fills_defaults_or_disables(
    option: CacheOption | None,
    expected: ResolvedCacheConfig | None,
) -> None:
    assert normalize_cache_config(option) == expected


# ---------------------------------------------------------------------------
# apply_cache_control
# ---------------------------------------------------------------------------


def test_system_scope_marks_last_system_message() -> None:
    payload = _payload(
        {"role": "system", "content": "old instructions"},
        {"role": "system", "content": "big stable prompt"},
        {"role": "user", "content": "hi"},
    )
    apply_cache_control(payload, normalize_cache_config(True))

    # String content is converted to the block form the endpoint expects.
    assert payload["messages"][1]["content"] == [
        {"type": "text", "text": "big stable prompt", "cache_control": EPHEMERAL_1H}
    ]
    # Every other message is left exactly as it arrived.
    assert payload["messages"][0]["content"] == "old instructions"
    assert payload["messages"][2]["content"] == "hi"


def test_explicit_ttl_is_used() -> None:
    payload = _payload({"role": "system", "content": "prefix"})
    apply_cache_control(payload, normalize_cache_config({"ttl": "5m"}))
    assert payload["messages"][0]["content"][0]["cache_control"] == EPHEMERAL_5M


def test_last_user_scope_marks_the_final_user_message() -> None:
    payload = _payload(
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    )
    apply_cache_control(payload, normalize_cache_config({"scope": "lastUser"}))

    assert payload["messages"][2]["content"] == [
        {"type": "text", "text": "second", "cache_control": EPHEMERAL_1H}
    ]
    assert payload["messages"][0]["content"] == "first"


def test_index_scope_marks_the_last_text_block_of_each_index() -> None:
    payload = _payload(
        {"role": "system", "content": "prefix"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        },
    )
    apply_cache_control(payload, normalize_cache_config({"scope": [0, 1]}))

    assert payload["messages"][0]["content"][0]["cache_control"] == EPHEMERAL_1H
    blocks = payload["messages"][1]["content"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == EPHEMERAL_1H


def test_out_of_range_indices_are_dropped() -> None:
    payload = _payload({"role": "user", "content": "hi"})
    apply_cache_control(payload, normalize_cache_config({"scope": [5, -1]}))
    assert payload["messages"][0]["content"] == "hi"


def test_disabled_is_a_no_op() -> None:
    payload = _payload({"role": "system", "content": "prefix"})
    apply_cache_control(payload, normalize_cache_config(False))
    assert payload["messages"][0]["content"] == "prefix"


def test_missing_target_role_is_a_no_op() -> None:
    """A conversation with no system message has nothing to mark."""
    payload = _payload({"role": "user", "content": "hi"})
    apply_cache_control(payload, normalize_cache_config(True))
    assert payload["messages"][0]["content"] == "hi"


def test_message_without_text_is_a_no_op() -> None:
    payload = _payload(
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    )
    apply_cache_control(payload, normalize_cache_config({"scope": [0]}))
    assert payload["messages"][0]["content"] == [{"type": "image_url", "image_url": {"url": "x"}}]


def test_payload_without_messages_is_a_no_op() -> None:
    payload: dict[str, Any] = {"model": "m"}
    assert apply_cache_control(payload, normalize_cache_config(True)) == {"model": "m"}


# ---------------------------------------------------------------------------
# ChatDoubleword._get_request_payload
# ---------------------------------------------------------------------------


def test_payload_is_untouched_by_default() -> None:
    """The hand-built route: no ``prompt_cache``, no rewriting."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatDoubleword(model="m", api_key="x")
    system = SystemMessage(
        content=[{"type": "text", "text": "prefix", "cache_control": EPHEMERAL_5M}]
    )
    payload = llm._get_request_payload([system, HumanMessage(content="hi")])

    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "prefix", "cache_control": EPHEMERAL_5M}
    ]
    assert payload["messages"][1]["content"] == "hi"


def test_user_supplied_markers_survive_an_enabled_cache() -> None:
    """``prompt_cache`` marks its own target and leaves other markers alone."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatDoubleword(model="m", api_key="x", prompt_cache=True)
    human = HumanMessage(content=[{"type": "text", "text": "q", "cache_control": EPHEMERAL_5M}])
    payload = llm._get_request_payload([SystemMessage(content="prefix"), human])

    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "prefix", "cache_control": EPHEMERAL_1H}
    ]
    assert payload["messages"][1]["content"][0]["cache_control"] == EPHEMERAL_5M


def test_payload_honours_ttl_and_scope() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatDoubleword(
        model="m",
        api_key="x",
        prompt_cache={"ttl": "5m", "scope": "lastUser"},
    )
    payload = llm._get_request_payload(
        [SystemMessage(content="prefix"), HumanMessage(content="hi")]
    )

    assert payload["messages"][0]["content"] == "prefix"
    assert payload["messages"][1]["content"] == [
        {"type": "text", "text": "hi", "cache_control": EPHEMERAL_5M}
    ]


def test_prompt_cache_rejects_a_bad_ttl() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatDoubleword(
            model="m",
            api_key="x",
            prompt_cache={"ttl": "2h"},  # type: ignore[typeddict-item]
        )


@pytest.mark.parametrize("model_class", [ChatDoublewordBatch, ChatDoublewordAsync])
def test_batch_variants_inherit_the_hook(
    model_class: type[ChatDoublewordBatch],
) -> None:
    """``ChatDoublewordBatch`` / ``ChatDoublewordAsync`` subclass
    ``ChatDoubleword``, so they pick up ``prompt_cache`` unchanged."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = model_class(model="m", api_key="x", prompt_cache=True)
    payload = llm._get_request_payload(
        [SystemMessage(content="prefix"), HumanMessage(content="hi")]
    )

    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "prefix", "cache_control": EPHEMERAL_1H}
    ]
