"""Unit tests for prompt caching. Pure payload transforms, no network."""

from typing import Any, Literal

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from langchain_doubleword import (
    CacheControl,
    ChatDoubleword,
    ChatDoublewordAsync,
    ChatDoublewordBatch,
)
from langchain_doubleword._cache import apply_cache_control

EPHEMERAL: CacheControl = {"type": "ephemeral"}
EPHEMERAL_1H: CacheControl = {"type": "ephemeral", "ttl": "1h"}


def _text(text: str, cache_control: CacheControl | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache_control is not None:
        block["cache_control"] = cache_control
    return block


def _payload(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"model": "m", "messages": list(messages)}


def _contents(payload: dict[str, Any]) -> list[Any]:
    return [message["content"] for message in payload["messages"]]


def test_marks_last_system_and_latest_message() -> None:
    payload = _payload(
        {"role": "system", "content": "old"},
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    )
    apply_cache_control(payload, EPHEMERAL)
    assert _contents(payload) == [
        "old",
        [_text("prefix", EPHEMERAL)],
        "q1",
        "a1",
        [_text("q2", EPHEMERAL)],
    ]


def test_system_that_is_also_latest_is_marked_once() -> None:
    payload = _payload({"role": "system", "content": "prefix"})
    apply_cache_control(payload, EPHEMERAL)
    assert _contents(payload) == [[_text("prefix", EPHEMERAL)]]


def test_without_system_only_latest_is_marked() -> None:
    payload = _payload(
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    )
    apply_cache_control(payload, EPHEMERAL)
    assert _contents(payload) == ["q1", "a1", [_text("q2", EPHEMERAL)]]


def test_marks_last_text_block_and_skips_targets_without_text() -> None:
    image = {"type": "image_url", "image_url": {"url": "x"}}
    payload = _payload(
        {"role": "system", "content": [_text("a"), _text("b"), image]},
        {"role": "user", "content": [image]},
    )
    apply_cache_control(payload, EPHEMERAL)
    assert _contents(payload) == [[_text("a"), _text("b", EPHEMERAL), image], [image]]


def test_messages_with_existing_markers_are_untouched() -> None:
    payload = _payload(
        {"role": "system", "content": [_text("prefix", EPHEMERAL_1H), _text("rules")]},
        {"role": "user", "content": "q"},
    )
    apply_cache_control(payload, EPHEMERAL)
    assert _contents(payload) == [
        [_text("prefix", EPHEMERAL_1H), _text("rules")],
        [_text("q", EPHEMERAL)],
    ]


def test_never_exceeds_four_breakpoints() -> None:
    payload = _payload(
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": [_text("doc1", EPHEMERAL)]},
        {"role": "user", "content": [_text("doc2", EPHEMERAL)]},
        {"role": "user", "content": "q"},
    )
    payload["tools"] = [{"type": "function", "function": {"name": "t"}, "cache_control": EPHEMERAL}]
    apply_cache_control(payload, EPHEMERAL)
    assert _contents(payload)[0] == [_text("prefix", EPHEMERAL)]
    assert _contents(payload)[3] == "q"


def test_omitted_ttl_sends_no_ttl_key() -> None:
    llm = ChatDoubleword(model="m", api_key="x", cache_control={"type": "ephemeral"})
    payload = llm._get_request_payload([HumanMessage(content="q")])
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_ttl_passes_through(ttl: Literal["5m", "1h"]) -> None:
    llm = ChatDoubleword(model="m", api_key="x", cache_control={"type": "ephemeral", "ttl": ttl})
    payload = llm._get_request_payload([HumanMessage(content="q")])
    assert payload["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": ttl,
    }


def test_per_call_dict_overrides_the_field() -> None:
    llm = ChatDoubleword(model="m", api_key="x", cache_control=EPHEMERAL)
    payload = llm._get_request_payload([HumanMessage(content="q")], cache_control=EPHEMERAL_1H)
    assert _contents(payload) == [[_text("q", EPHEMERAL_1H)]]


def test_per_call_none_skips_caching() -> None:
    llm = ChatDoubleword(model="m", api_key="x", cache_control=EPHEMERAL)
    messages = [SystemMessage(content="prefix"), HumanMessage(content="q")]
    assert _contents(llm._get_request_payload(messages, cache_control=None)) == ["prefix", "q"]


def test_unset_field_leaves_payload_untouched() -> None:
    llm = ChatDoubleword(model="m", api_key="x")
    messages = [SystemMessage(content="prefix"), HumanMessage(content="q")]
    assert _contents(llm._get_request_payload(messages)) == ["prefix", "q"]


@pytest.mark.parametrize("override", [{}, {"cache_control": EPHEMERAL_1H}, {"cache_control": None}])
def test_cache_control_is_never_a_top_level_parameter(override: dict[str, Any]) -> None:
    llm = ChatDoubleword(model="m", api_key="x", cache_control=EPHEMERAL)
    assert "cache_control" not in llm._get_request_payload([HumanMessage(content="q")], **override)


def test_caller_messages_are_not_mutated() -> None:
    llm = ChatDoubleword(model="m", api_key="x", cache_control=EPHEMERAL)
    human = HumanMessage(content=[_text("q")])
    llm._get_request_payload([human])
    assert human.content == [_text("q")]


def test_rejects_unknown_ttl() -> None:
    with pytest.raises(ValidationError):
        ChatDoubleword(
            model="m",
            api_key="x",
            cache_control={"type": "ephemeral", "ttl": "2h"},  # type: ignore[typeddict-item]
        )


@pytest.mark.parametrize("model_class", [ChatDoublewordBatch, ChatDoublewordAsync])
def test_batch_and_async_inherit_the_hook(model_class: type[ChatDoubleword]) -> None:
    llm = model_class(model="m", api_key="x", cache_control=EPHEMERAL)
    payload = llm._get_request_payload([HumanMessage(content="q")])
    assert _contents(payload) == [[_text("q", EPHEMERAL)]]
