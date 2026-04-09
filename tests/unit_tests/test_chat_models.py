"""Unit tests for ChatDoubleword and ChatDoublewordBatch.

Real-time variant goes through ``langchain_tests.unit_tests.ChatModelUnitTests``,
which exercises the standard LangChain chat model contract with no network
access. Batch-variant tests verify the autobatcher wiring and the async-only
contract without spinning up a real BatchOpenAI worker loop.
"""

from typing import Any

import pytest
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_doubleword import ChatDoubleword, ChatDoublewordBatch


class TestChatDoublewordUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatDoubleword]:
        return ChatDoubleword

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {
            "model": "doubleword-test-model",
            "api_key": "test-key",
        }

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"DOUBLEWORD_API_KEY": "env-api-key"},
            {"model": "doubleword-test-model"},
            {"openai_api_key": "env-api-key"},
        )



def test_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOUBLEWORD_API_BASE", raising=False)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    llm = ChatDoubleword(model="m")
    assert llm.openai_api_base == "https://api.doubleword.ai/v1"


def test_base_url_override_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOUBLEWORD_API_BASE", "https://example.com/v1")
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    llm = ChatDoubleword(model="m")
    assert llm.openai_api_base == "https://example.com/v1"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "env-key")
    llm = ChatDoubleword(model="m")
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "env-key"


def test_lc_secrets_maps_to_doubleword_env() -> None:
    llm = ChatDoubleword(model="m", api_key="x")
    assert llm.lc_secrets == {"openai_api_key": "DOUBLEWORD_API_KEY"}


def test_lc_namespace() -> None:
    assert ChatDoubleword.get_lc_namespace() == ["langchain_doubleword", "chat_models"]


def test_llm_type() -> None:
    assert ChatDoubleword(model="m", api_key="x")._llm_type == "doubleword-chat"
    assert (
        ChatDoublewordBatch(model="m", api_key="x")._llm_type
        == "doubleword-chat-batch"
    )


# ---------------------------------------------------------------------------
# Batch variant
# ---------------------------------------------------------------------------


def test_batch_installs_autobatcher_client() -> None:
    """The batch variant should swap async_client for an autobatcher client."""
    from autobatcher import BatchOpenAI

    llm = ChatDoublewordBatch(model="m", api_key="x")
    assert isinstance(llm.root_async_client, BatchOpenAI)
    # Sync clients are intentionally cleared so misuse fails loudly.
    assert llm.root_client is None
    assert llm.client is None


def test_batch_sync_invoke_raises() -> None:
    llm = ChatDoublewordBatch(model="m", api_key="x")
    with pytest.raises(NotImplementedError, match="async-only"):
        llm.invoke("hi")


def test_batch_sync_stream_raises() -> None:
    llm = ChatDoublewordBatch(model="m", api_key="x")
    with pytest.raises(NotImplementedError, match="does not support streaming"):
        list(llm.stream("hi"))


async def test_batch_async_stream_raises() -> None:
    """Streaming over a batch model is fundamentally incompatible with batching.

    Without an explicit override of ``_astream``, ``ChatDoublewordBatch``
    would inherit ``BaseChatOpenAI._astream`` and silently try to stream via
    the autobatcher client (which forces ``stream=False`` internally), causing
    a confusing crash deep in langchain-openai code. We override ``_astream``
    to raise immediately so users get a clear error.
    """
    llm = ChatDoublewordBatch(model="m", api_key="x")
    with pytest.raises(NotImplementedError, match="does not support streaming"):
        async for _ in llm.astream("hi"):
            pass


async def test_batch_ainvoke_routes_through_with_raw_response() -> None:
    """Regression test: ``ChatDoublewordBatch.ainvoke`` must reach the
    underlying enqueue path through autobatcher's ``with_raw_response``
    accessor.

    ``BaseChatOpenAI._agenerate`` calls
    ``self.async_client.with_raw_response.create(**payload)`` (not
    ``.create()`` directly) on its non-streaming async path. Earlier
    autobatcher releases only implemented ``.create()`` on the proxy classes,
    so this code path crashed with ``AttributeError`` — exactly the failure
    mode that ``examples/langgraph-basic/batched.py`` hit and that motivated
    the autobatcher ``with_raw_response`` shim.

    This test exercises the full chain end-to-end inside one process,
    stubbing only the very last hop (``BatchOpenAI._enqueue_request``) so the
    autobatcher proxy classes and their ``with_raw_response`` accessors are
    really exercised.
    """
    from unittest.mock import AsyncMock

    from langchain_core.messages import AIMessage, HumanMessage
    from openai.types.chat import ChatCompletion

    llm = ChatDoublewordBatch(model="test-model", api_key="x")

    # Synthesize a parsed ChatCompletion that mimics what autobatcher would
    # return after a batch poll.
    fake_completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello back"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
    )

    # Stub the bottom of the chain. The full call path being exercised is:
    #   ChatDoublewordBatch.ainvoke
    #     -> BaseChatOpenAI._agenerate
    #         -> self.async_client.with_raw_response.create(**payload)   ← shim
    #             -> autobatcher's _ChatCompletions.create               ← proxy
    #                 -> BatchOpenAI._enqueue_request                    ← stubbed
    llm.root_async_client._enqueue_request = AsyncMock(  # type: ignore[method-assign]
        return_value=fake_completion
    )

    response = await llm.ainvoke([HumanMessage(content="hi")])

    assert isinstance(response, AIMessage)
    assert response.content == "hello back"
    llm.root_async_client._enqueue_request.assert_awaited_once()  # type: ignore[attr-defined]


def test_batch_inherits_doubleword_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOUBLEWORD_API_BASE", raising=False)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    llm = ChatDoublewordBatch(model="m")
    assert llm.openai_api_base == "https://api.doubleword.ai/v1"


def test_batch_default_autobatcher_config() -> None:
    llm = ChatDoublewordBatch(model="m", api_key="x")
    assert llm.batch_size == 1000
    assert llm.batch_window_seconds == 10.0
    assert llm.poll_interval_seconds == 5.0
    assert llm.completion_window == "24h"


def test_batch_autobatcher_config_propagates_to_client() -> None:
    llm = ChatDoublewordBatch(
        model="m",
        api_key="x",
        batch_size=250,
        batch_window_seconds=2.5,
        poll_interval_seconds=1.0,
        completion_window="1h",
    )
    client = llm.root_async_client
    assert client._batch_size == 250
    assert client._batch_window_seconds == 2.5
    assert client._poll_interval_seconds == 1.0
    assert client._completion_window == "1h"


def test_batch_completion_window_validates() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatDoublewordBatch(
            model="m",
            api_key="x",
            completion_window="48h",  # type: ignore[arg-type]
        )
