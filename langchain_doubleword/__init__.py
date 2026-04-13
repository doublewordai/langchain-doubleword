"""LangChain integration for Doubleword.

This package wires Doubleword's OpenAI-compatible inference API into
LangChain and LangGraph as both real-time chat / embedding models and
transparently-batched variants powered by ``autobatcher``.
"""

from importlib import metadata

from langchain_doubleword.chat_models import (
    DEFAULT_DOUBLEWORD_API_BASE,
    ChatDoubleword,
    ChatDoublewordBatch,
)
from langchain_doubleword.embeddings import (
    DoublewordEmbeddings,
    DoublewordEmbeddingsBatch,
)

try:
    __version__ = metadata.version(__package__ or "langchain-doubleword")
except metadata.PackageNotFoundError:
    __version__ = "0.1.1"
del metadata

__all__ = [
    "DEFAULT_DOUBLEWORD_API_BASE",
    "ChatDoubleword",
    "ChatDoublewordBatch",
    "DoublewordEmbeddings",
    "DoublewordEmbeddingsBatch",
    "__version__",
]
