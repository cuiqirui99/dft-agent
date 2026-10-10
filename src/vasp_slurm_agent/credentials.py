"""Optional API-key storage in the operating system keychain.

Keys are stored only when the user asks. The `keyring` package talks to the
macOS Keychain, Windows Credential Manager or a Linux Secret Service.
"""

from __future__ import annotations

from .providers import PROVIDERS
from .settings import load_settings, save_settings

SERVICE = "dft-agent"


class CredentialError(RuntimeError):
    """The keychain is unavailable or refused the request."""


def _keyring():
    try:
        import keyring
        from keyring import errors  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise CredentialError("Install the keyring package to remember API keys: pip install keyring") from exc
    backend = keyring.get_keyring()
    name = type(backend).__name__.lower()
    if "fail" in name or "null" in name or "chainer" in name and not getattr(backend, "backends", None):
        raise CredentialError("No system keychain is available on this computer.")
    return keyring


def keyring_available() -> bool:
    try:
        _keyring()
    except CredentialError:
        return False
    return True


def _check_provider(provider: str) -> None:
    if provider not in PROVIDERS or provider == "codex":
        raise CredentialError("Choose an API provider.")


def load_key(provider: str) -> str | None:
    _check_provider(provider)
    try:
        value = _keyring().get_password(SERVICE, provider)
    except CredentialError:
        raise
    except Exception as exc:  # keyring backends raise their own error types
        raise CredentialError(f"Cannot read the keychain: {exc}") from exc
    return value or None


def save_key(provider: str, key: str) -> None:
    _check_provider(provider)
    key = (key or "").strip()
    if not key or any(c in key for c in "\n\r\x00"):
        raise CredentialError("Enter an API key on a single line.")
    try:
        _keyring().set_password(SERVICE, provider, key)
    except CredentialError:
        raise
    except Exception as exc:
        raise CredentialError(f"Cannot write to the keychain: {exc}") from exc
    save_settings({"keychain": True})


def delete_key(provider: str) -> None:
    _check_provider(provider)
    try:
        backend = _keyring()
        if backend.get_password(SERVICE, provider) is None:
            return
        try:
            backend.delete_password(SERVICE, provider)
        except Exception:
            # Some backends report an error for a key another process removed.
            # The exception type alone does not establish that the key is gone.
            if backend.get_password(SERVICE, provider) is not None:
                raise
        if backend.get_password(SERVICE, provider) is not None:
            raise CredentialError("Cannot remove the key: the keychain still contains it.")
    except CredentialError:
        raise
    except Exception as exc:
        raise CredentialError(f"Cannot remove the key: {exc}") from exc


def stored_key(provider: str) -> str | None:
    """A remembered key, or None when the user never opted in or nothing is stored."""
    if provider not in PROVIDERS or provider == "codex" or not load_settings().get("keychain"):
        return None
    try:
        return load_key(provider)
    except CredentialError:
        return None
