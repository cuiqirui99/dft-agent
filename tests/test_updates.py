"""Update checks are cached, quiet on failure and easy to disable."""

from datetime import datetime, timedelta, timezone
import json

from vasp_slurm_agent import updates


def test_parse_version():
    assert updates.parse_version("v0.4.2") == (0, 4, 2)
    assert updates.parse_version("0.10.0rc1") == (0, 10, 0)
    assert updates.parse_version("junk") == (0,)
    assert updates.parse_version("0.4.10") > updates.parse_version("0.4.2")


def test_check_uses_cache_and_reports_newer(tmp_path, monkeypatch):
    monkeypatch.delenv("DFT_AGENT_NO_UPDATE_CHECK")
    cache = tmp_path / "update.json"
    calls = []

    def fetch(timeout):
        calls.append(timeout)
        return {"latest": "0.9.0", "url": "https://github.com/cuiqirui99/dft-agent/releases/tag/v0.9.0"}

    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    first = updates.check_for_update("0.4.2", cache_path=cache, fetch=fetch, now=now)
    assert first["newer"] and first["latest"] == "0.9.0"
    assert json.loads(cache.read_text())["latest"] == "0.9.0"
    second = updates.check_for_update("0.4.2", cache_path=cache, fetch=fetch, now=now + timedelta(hours=1))
    assert second["latest"] == "0.9.0" and len(calls) == 1
    updates.check_for_update("0.4.2", cache_path=cache, fetch=fetch, now=now + timedelta(days=2))
    assert len(calls) == 2
    assert updates.available_update("0.4.2", cache)["latest"] == "0.9.0"
    assert updates.available_update("1.0.0", cache) is None


def test_failures_and_disabling_return_none(tmp_path, monkeypatch):
    monkeypatch.delenv("DFT_AGENT_NO_UPDATE_CHECK")
    cache = tmp_path / "update.json"

    def failing(timeout):
        raise OSError("offline")

    assert updates.check_for_update("0.4.2", cache_path=cache, fetch=failing) is None
    assert not cache.exists()
    monkeypatch.setenv("DFT_AGENT_NO_UPDATE_CHECK", "1")
    assert updates.check_for_update("0.4.2", cache_path=cache, fetch=lambda timeout: {"latest": "9", "url": "https://github.com/x"}) is None
    cache.write_text("garbage")
    assert updates.read_cache(cache) is None


def test_background_check_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(updates, "_started", False)
    updates.start_background_check()
    assert updates._started is False
