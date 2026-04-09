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


def test_batch_inherits_doubleword_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOUBLEWORD_API_BASE", raising=False)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    llm = ChatDoublewordBatch(model="m")
    assert llm.openai_api_base == "https://api.doubleword.ai/v1"
