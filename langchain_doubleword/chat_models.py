"""Doubleword chat model implementations.

Two classes are exposed:

* :class:`ChatDoubleword` — a thin subclass of
  :class:`langchain_openai.chat_models.base.BaseChatOpenAI` that targets
  Doubleword's OpenAI-compatible inference endpoint
  (``https://api.doubleword.ai/v1``) by default and reads
  ``DOUBLEWORD_API_KEY`` from the environment.

* :class:`ChatDoublewordBatch` — same surface, but the async client is replaced
  with :class:`autobatcher.BatchOpenAI`, which transparently collects concurrent
  requests and submits them through Doubleword's batch API. This is the only
  way to access models that Doubleword exposes solely via the batch endpoint,
  and it is the recommended choice for LangGraph workflows that fan out many
  parallel calls.
"""

from typing import Any

from langchain_core.utils import from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_doubleword._credentials import resolve_api_key

DEFAULT_DOUBLEWORD_API_BASE = "https://api.doubleword.ai/v1"


class ChatDoubleword(BaseChatOpenAI):
    """Doubleword chat model.

    A real-time chat model targeting Doubleword's OpenAI-compatible API.
    Configure with ``DOUBLEWORD_API_KEY`` (or pass ``api_key=...``).

    Example:
        .. code-block:: python

            from langchain_doubleword import ChatDoubleword

            llm = ChatDoubleword(model="your-model")
            llm.invoke("Hello, world.")
    """

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
        return ["langchain_doubleword", "chat_models"]

    @property
    def _llm_type(self) -> str:
        return "doubleword-chat"

    @property
    def _default_params(self) -> dict[str, Any]:
        params = super()._default_params
        # Doubleword does not (currently) accept the OpenAI ``stream_options``
        # field, and forwards everything else verbatim. Strip nothing for now;
        # this hook exists so future divergences are easy to handle.
        return params


class ChatDoublewordBatch(ChatDoubleword):
    """Doubleword chat model that routes through the batch API via autobatcher.

    Concurrent ``ainvoke`` calls are collected by
    :class:`autobatcher.BatchOpenAI` and submitted as a single batch to
    Doubleword. This is the **only** way to use models that Doubleword exposes
    solely through batch endpoints, and the natural choice for LangGraph
    workflows with parallel branches: collected calls amortize batch pricing
    (~50% cost savings) without changing user code.

    **Async-only.** Synchronous calls (``invoke``, ``stream``, ``batch``)
    raise ``NotImplementedError`` because a 10-second batch collection window
    is incompatible with blocking call sites. Use ``ainvoke`` /
    ``astream`` / ``abatch`` from async LangGraph nodes instead.

    Example:
        .. code-block:: python

            import asyncio
            from langchain_doubleword import ChatDoublewordBatch

            llm = ChatDoublewordBatch(model="batch-only-model")

            async def main():
                results = await asyncio.gather(*[
                    llm.ainvoke(f"Summarize chapter {i}") for i in range(50)
                ])
                for r in results:
                    print(r.content)

            asyncio.run(main())
    """

    @property
    def _llm_type(self) -> str:
        return "doubleword-chat-batch"

    @model_validator(mode="after")
    def _install_autobatcher(self) -> "ChatDoublewordBatch":
        # Imported lazily so that users who only need ChatDoubleword don't pay
        # the autobatcher import cost (it pulls in openai + asyncio internals).
        from autobatcher import BatchOpenAI

        api_key: str | None = None
        if self.openai_api_key is not None:
            if isinstance(self.openai_api_key, SecretStr):
                api_key = self.openai_api_key.get_secret_value()
            else:
                # Field allows callables; let BatchOpenAI handle them directly.
                api_key = self.openai_api_key  # type: ignore[assignment]

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.openai_api_base,
        }
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        if self.max_retries is not None:
            client_kwargs["max_retries"] = self.max_retries
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        if self.default_query:
            client_kwargs["default_query"] = self.default_query

        batch_client = BatchOpenAI(**client_kwargs)

        # BaseChatOpenAI's after-validator already populated root_async_client
        # with a vanilla openai.AsyncOpenAI. Replace it (and the cached
        # .chat.completions accessor) with the autobatcher equivalent.
        self.root_async_client = batch_client
        self.async_client = batch_client.chat.completions

        # Sync clients are intentionally cleared so any sync call site fails
        # loudly via _generate() below rather than silently bypassing batching.
        self.root_client = None
        self.client = None
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ChatDoublewordBatch is async-only. Use `ainvoke` / `astream` / "
            "`abatch` from an async context (e.g. an async LangGraph node). "
            "If you need a synchronous chat model, use `ChatDoubleword` "
            "instead."
        )

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ChatDoublewordBatch is async-only. Use `astream` instead."
        )
