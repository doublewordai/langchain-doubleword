"""Doubleword embedding model implementations.

Mirrors :mod:`langchain_doubleword.chat_models`: a real-time
:class:`DoublewordEmbeddings` plus an async-only
:class:`DoublewordEmbeddingsBatch` that routes through
:class:`autobatcher.BatchOpenAI` for transparent batching against
Doubleword's batch embeddings endpoint.
"""

from typing import Any, Literal

from langchain_core.utils import from_env
from langchain_openai.embeddings.base import OpenAIEmbeddings
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_doubleword._credentials import resolve_api_key
from langchain_doubleword.chat_models import DEFAULT_DOUBLEWORD_API_BASE


class DoublewordEmbeddings(OpenAIEmbeddings):
    """Doubleword embeddings (real-time)."""

    model_config = ConfigDict(populate_by_name=True)

    openai_api_key: SecretStr | None = Field(  # type: ignore[assignment]
        alias="api_key",
        default_factory=resolve_api_key,
    )
    openai_api_base: str | None = Field(
        alias="base_url",
        default_factory=from_env(
            "DOUBLEWORD_API_BASE", default=DEFAULT_DOUBLEWORD_API_BASE
        ),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "DOUBLEWORD_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_doubleword", "embeddings"]


class DoublewordEmbeddingsBatch(DoublewordEmbeddings):
    """Doubleword embeddings routed through the batch API via autobatcher.

    Async-only. Synchronous ``embed_query`` / ``embed_documents`` raise.
    Use ``aembed_query`` / ``aembed_documents`` from async code.
    """

    batch_size: int = Field(
        default=1000,
        description=(
            "Submit a batch when this many requests have been queued. "
            "Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    batch_window_seconds: float = Field(
        default=10.0,
        description=(
            "Submit a batch after this many seconds even if `batch_size` "
            "is not reached. Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    poll_interval_seconds: float = Field(
        default=5.0,
        description=(
            "How often autobatcher polls Doubleword's batch endpoint for "
            "completion. Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    completion_window: Literal["24h", "1h"] = Field(
        default="24h",
        description=(
            "Doubleword batch completion window. '1h' is more expensive "
            "but completes faster. Forwarded to autobatcher.BatchOpenAI."
        ),
    )

    @model_validator(mode="after")
    def _install_autobatcher(self) -> "DoublewordEmbeddingsBatch":
        from autobatcher import BatchOpenAI

        api_key: str | None = None
        if self.openai_api_key is not None:
            if isinstance(self.openai_api_key, SecretStr):
                api_key = self.openai_api_key.get_secret_value()
            else:
                api_key = self.openai_api_key  # type: ignore[assignment]

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.openai_api_base,
            "batch_size": self.batch_size,
            "batch_window_seconds": self.batch_window_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "completion_window": self.completion_window,
        }
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        if self.max_retries is not None:
            client_kwargs["max_retries"] = self.max_retries
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        if self.default_query:
            client_kwargs["default_query"] = self.default_query

        # OpenAIEmbeddings stores full clients in `client` / `async_client`
        # (no root_* split, unlike BaseChatOpenAI), and calls
        # `self.async_client.create(...)` against the embeddings accessor.
        # We swap in BatchOpenAI's embeddings accessor.
        batch_client = BatchOpenAI(**client_kwargs)
        self.async_client = batch_client.embeddings
        self.client = None
        return self

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(
            "DoublewordEmbeddingsBatch is async-only. Use `aembed_query`."
        )

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None
    ) -> list[list[float]]:
        raise NotImplementedError(
            "DoublewordEmbeddingsBatch is async-only. Use `aembed_documents`."
        )
