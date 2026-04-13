"""Unit tests for DoublewordEmbeddings and DoublewordEmbeddingsBatch."""

from typing import Any

import pytest
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_doubleword import DoublewordEmbeddings, DoublewordEmbeddingsBatch


class TestDoublewordEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[DoublewordEmbeddings]:
        return DoublewordEmbeddings

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {"model": "doubleword-test-embedding", "api_key": "test-key"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"DOUBLEWORD_API_KEY": "env-api-key"},
            {"model": "doubleword-test-embedding"},
            {"openai_api_key": "env-api-key"},
        )


# ---------------------------------------------------------------------------
# Doubleword-specific defaults that the standard suite does not cover.
# ---------------------------------------------------------------------------


def test_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOUBLEWORD_API_BASE", raising=False)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    emb = DoublewordEmbeddings(model="text-embed")
    assert emb.openai_api_base == "https://api.doubleword.ai/v1"


def test_lc_secrets() -> None:
    emb = DoublewordEmbeddings(model="text-embed", api_key="x")
    assert emb.lc_secrets == {"openai_api_key": "DOUBLEWORD_API_KEY"}


def test_lc_namespace() -> None:
    assert DoublewordEmbeddings.get_lc_namespace() == [
        "langchain_doubleword",
        "embeddings",
    ]


# ---------------------------------------------------------------------------
# Batch variant
# ---------------------------------------------------------------------------


def test_batch_installs_autobatcher_client() -> None:
    from autobatcher import BatchOpenAI

    emb = DoublewordEmbeddingsBatch(model="text-embed", api_key="x")
    parent = getattr(emb.async_client, "_client", None)
    assert isinstance(parent, BatchOpenAI), (
        "DoublewordEmbeddingsBatch.async_client should be a BatchOpenAI embeddings accessor"
    )
    assert emb.client is None


def test_batch_sync_embed_raises() -> None:
    emb = DoublewordEmbeddingsBatch(model="text-embed", api_key="x")
    with pytest.raises(NotImplementedError, match="async-only"):
        emb.embed_query("hi")
    with pytest.raises(NotImplementedError, match="async-only"):
        emb.embed_documents(["hi"])


def test_batch_default_autobatcher_config() -> None:
    emb = DoublewordEmbeddingsBatch(model="text-embed", api_key="x")
    assert emb.batch_size == 1000
    assert emb.batch_window_seconds == 10.0
    assert emb.poll_interval_seconds == 5.0
    assert emb.completion_window == "24h"


def test_batch_autobatcher_config_propagates_to_client() -> None:
    emb = DoublewordEmbeddingsBatch(
        model="text-embed",
        api_key="x",
        batch_size=500,
        batch_window_seconds=3.0,
        poll_interval_seconds=2.0,
        completion_window="1h",
    )
    client = emb.async_client._client
    assert client._batch_size == 500
    assert client._batch_window_seconds == 3.0
    assert client._poll_interval_seconds == 2.0
    assert client._completion_window == "1h"
