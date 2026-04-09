"""Integration tests for DoublewordEmbeddings.

Skipped unless ``DOUBLEWORD_API_KEY`` is set. Override the model with
``DOUBLEWORD_TEST_EMBEDDING_MODEL``.
"""

import os
from typing import Any

import pytest
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_doubleword import DoublewordEmbeddings

pytestmark = pytest.mark.skipif(
    "DOUBLEWORD_API_KEY" not in os.environ,
    reason="DOUBLEWORD_API_KEY not set; skipping integration tests.",
)

TEST_MODEL = os.environ.get(
    "DOUBLEWORD_TEST_EMBEDDING_MODEL", "doubleword-embedding"
)


class TestDoublewordEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[DoublewordEmbeddings]:
        return DoublewordEmbeddings

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {"model": TEST_MODEL}
