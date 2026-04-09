"""Credential resolution for langchain-doubleword.

Resolves the Doubleword inference API key by walking the standard chain:

1. Explicit ``api_key=...`` constructor argument (handled by Pydantic before
   the default factory runs).
2. ``DOUBLEWORD_API_KEY`` environment variable.
3. ``~/.dw/credentials.toml``, looking up the account named in
   ``~/.dw/config.toml``'s ``active_account`` field, then returning its
   ``inference_key``.
4. ``None`` — the OpenAI client will surface a clear "no API key" error at
   first call.

The credentials file format matches the one written by the ``dwctl`` /
Doubleword Python tooling:

.. code-block:: toml

    # ~/.dw/config.toml
    active_account = "work"

    # ~/.dw/credentials.toml
    [accounts.work]
    inference_key = "sk-..."
    platform_key  = "sk-..."
    # ...other metadata fields ignored here

All file-system errors and TOML parse errors are swallowed: if anything
goes wrong reading the files, we fall through to the next step in the
resolution chain rather than confusing users who never opted into using
the file at all.
"""

import os
import sys
from pathlib import Path

from pydantic import SecretStr

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DW_HOME = Path.home() / ".dw"
CREDENTIALS_FILE = DW_HOME / "credentials.toml"
CONFIG_FILE = DW_HOME / "config.toml"


def _read_active_account() -> str | None:
    """Return the active account name from ``~/.dw/config.toml``, if any."""
    try:
        with CONFIG_FILE.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    active = data.get("active_account")
    return active if isinstance(active, str) and active else None


def _read_inference_key(account: str) -> str | None:
    """Return the inference key for the named account, if found."""
    try:
        with CREDENTIALS_FILE.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    accounts = data.get("accounts")
    if not isinstance(accounts, dict):
        return None
    entry = accounts.get(account)
    if not isinstance(entry, dict):
        return None
    key = entry.get("inference_key")
    return key if isinstance(key, str) and key else None


def resolve_api_key() -> SecretStr | None:
    """Resolve a Doubleword API key from env var, then credentials file."""
    env_key = os.environ.get("DOUBLEWORD_API_KEY")
    if env_key:
        return SecretStr(env_key)

    account = _read_active_account()
    if account is None:
        return None
    file_key = _read_inference_key(account)
    if file_key is None:
        return None
    return SecretStr(file_key)
