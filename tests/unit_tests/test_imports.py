"""Smoke tests: every public symbol is importable from the package root."""

from langchain_doubleword import __all__

EXPECTED = {
    "ChatDoubleword",
    "ChatDoublewordBatch",
    "DoublewordEmbeddings",
    "DoublewordEmbeddingsBatch",
    "DEFAULT_DOUBLEWORD_API_BASE",
    "__version__",
}


def test_all_exports_present() -> None:
    assert EXPECTED.issubset(set(__all__))


def test_public_symbols_importable() -> None:
    import langchain_doubleword as pkg

    for name in EXPECTED:
        assert hasattr(pkg, name), f"missing public symbol: {name}"
