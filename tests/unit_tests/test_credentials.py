"""Unit tests for the credentials resolution chain."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from langchain_doubleword import _credentials


@pytest.fixture
def fake_dw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ~/.dw at a temporary directory and clear DOUBLEWORD_API_KEY."""
    monkeypatch.delenv("DOUBLEWORD_API_KEY", raising=False)
    monkeypatch.setattr(_credentials, "DW_HOME", tmp_path)
    monkeypatch.setattr(_credentials, "CREDENTIALS_FILE", tmp_path / "credentials.toml")
    monkeypatch.setattr(_credentials, "CONFIG_FILE", tmp_path / "config.toml")
    return tmp_path


def _write(path: Path, contents: str) -> None:
    path.write_text(contents)


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------


def test_env_var_takes_precedence_over_file(
    fake_dw_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "from-env")
    _write(fake_dw_home / "config.toml", 'active_account = "work"\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "from-file"\n',
    )

    result = _credentials.resolve_api_key()
    assert result is not None
    assert result.get_secret_value() == "from-env"


def test_reads_inference_key_from_active_account(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "work"\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-from-file"\nplatform_key = "ignored"\n',
    )

    result = _credentials.resolve_api_key()
    assert result is not None
    assert isinstance(result, SecretStr)
    assert result.get_secret_value() == "sk-from-file"


def test_picks_active_account_not_first_account(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "personal"\n')
    _write(
        fake_dw_home / "credentials.toml",
        """\
[accounts.work]
inference_key = "sk-work"

[accounts.personal]
inference_key = "sk-personal"
""",
    )

    result = _credentials.resolve_api_key()
    assert result is not None
    assert result.get_secret_value() == "sk-personal"


# ---------------------------------------------------------------------------
# Quiet failure modes — never raise; always fall through to None.
# ---------------------------------------------------------------------------


def test_returns_none_when_no_files_present(fake_dw_home: Path) -> None:
    assert _credentials.resolve_api_key() is None


def test_returns_none_when_config_missing(fake_dw_home: Path) -> None:
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-work"\n',
    )
    assert _credentials.resolve_api_key() is None


def test_returns_none_when_active_account_missing_from_credentials(
    fake_dw_home: Path,
) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "ghost"\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-work"\n',
    )
    assert _credentials.resolve_api_key() is None


def test_returns_none_when_account_has_no_inference_key(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "work"\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\nplatform_key = "sk-platform"\n',
    )
    assert _credentials.resolve_api_key() is None


def test_returns_none_on_malformed_config_toml(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", "this is = not [valid toml")
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-work"\n',
    )
    assert _credentials.resolve_api_key() is None


def test_returns_none_on_malformed_credentials_toml(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "work"\n')
    _write(fake_dw_home / "credentials.toml", "[accounts.work\ninference_key =")
    assert _credentials.resolve_api_key() is None


def test_returns_none_when_active_account_is_empty_string(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = ""\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-work"\n',
    )
    assert _credentials.resolve_api_key() is None


# ---------------------------------------------------------------------------
# Integration with the chat model — confirms the factory is wired up.
# ---------------------------------------------------------------------------


def test_chat_model_picks_up_credentials_from_file(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "work"\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-from-file"\n',
    )

    from langchain_doubleword import ChatDoubleword

    llm = ChatDoubleword(model="m")
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "sk-from-file"


def test_embeddings_picks_up_credentials_from_file(fake_dw_home: Path) -> None:
    _write(fake_dw_home / "config.toml", 'active_account = "work"\n')
    _write(
        fake_dw_home / "credentials.toml",
        '[accounts.work]\ninference_key = "sk-from-file"\n',
    )

    from langchain_doubleword import DoublewordEmbeddings

    emb = DoublewordEmbeddings(model="m")
    assert emb.openai_api_key is not None
    assert emb.openai_api_key.get_secret_value() == "sk-from-file"
