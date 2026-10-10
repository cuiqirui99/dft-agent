"""Opt-in keychain storage never touches a real keyring in tests."""

import sys

import pytest

from vasp_slurm_agent import cli, credentials, settings


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.store:
            raise RuntimeError("not found")
        del self.store[(service, name)]


@pytest.fixture
def keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda: fake)
    return fake


def test_save_load_delete(keyring):
    assert credentials.keyring_available()
    assert credentials.stored_key("anthropic") is None
    credentials.save_key("anthropic", " key-123 ")
    assert keyring.store == {("dft-agent", "anthropic"): "key-123"}
    assert settings.load_settings()["keychain"] is True
    assert credentials.stored_key("anthropic") == "key-123"
    assert credentials.load_key("qwen") is None
    credentials.delete_key("anthropic")
    credentials.delete_key("anthropic")
    assert credentials.stored_key("anthropic") is None


def test_rejects_bad_input(keyring):
    with pytest.raises(credentials.CredentialError):
        credentials.save_key("codex", "x")
    with pytest.raises(credentials.CredentialError):
        credentials.save_key("anthropic", "two\nlines")
    with pytest.raises(credentials.CredentialError):
        credentials.load_key("unknown")


def test_unavailable_keyring_is_reported(monkeypatch):
    def broken():
        raise credentials.CredentialError("No system keychain is available on this computer.")

    monkeypatch.setattr(credentials, "_keyring", broken)
    assert not credentials.keyring_available()
    settings.save_settings({"keychain": True})
    assert credentials.stored_key("anthropic") is None


def test_cli_key_commands(keyring, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "status"])
    assert cli.main() == 0
    assert "No API key" in capsys.readouterr().out
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "secret-key")
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "set", "deepseek"])
    assert cli.main() == 0
    assert keyring.store[("dft-agent", "deepseek")] == "secret-key"
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "status"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DeepSeek" in out and "secret-key" not in out
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "clear", "deepseek"])
    assert cli.main() == 0
    assert keyring.store == {}


@pytest.mark.parametrize("error_name,message", [("PasswordDeleteError", "access denied"), ("RuntimeError", "service not found")])
def test_failed_delete_keeps_key_and_reports_failure(keyring, monkeypatch, capsys, error_name, message):
    credentials.save_key("anthropic", "test-secret")

    def denied(*args):
        raise type(error_name, (Exception,), {})(message)

    monkeypatch.setattr(keyring, "delete_password", denied)
    with pytest.raises(credentials.CredentialError, match="Cannot remove"):
        credentials.delete_key("anthropic")
    monkeypatch.setattr(sys, "argv", ["dft-agent", "key", "clear", "anthropic"])
    assert cli.main() == 1
    assert credentials.load_key("anthropic") == "test-secret"
    captured = capsys.readouterr()
    assert "Cannot remove" in captured.err
    assert "test-secret" not in captured.out + captured.err


def test_delete_verifies_backend_did_remove_key(keyring, monkeypatch):
    credentials.save_key("anthropic", "test-secret")
    monkeypatch.setattr(keyring, "delete_password", lambda *args: None)
    with pytest.raises(credentials.CredentialError, match="still contains"):
        credentials.delete_key("anthropic")
