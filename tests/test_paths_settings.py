"""Settings locations, legacy fallbacks and saved preferences."""

from pathlib import Path
import sys

from vasp_slurm_agent import paths, settings


def test_defaults_live_under_the_app_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("DFT_AGENT_HOME", raising=False)
    monkeypatch.delenv("VASP_AGENT_CONFIG", raising=False)
    assert paths.app_home() == tmp_path / ".dft-agent"
    assert paths.default_config_path() == tmp_path / ".dft-agent" / "cluster.json"
    assert paths.default_runs_root() == tmp_path / "dft-agent-runs"
    assert paths.settings_path().parent == paths.app_home()


def test_environment_overrides_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_AGENT_CONFIG", str(tmp_path / "custom.json"))
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(tmp_path / "old.json"))
    monkeypatch.setenv("DFT_AGENT_RUNS", str(tmp_path / "runs"))
    assert paths.default_config_path() == tmp_path / "custom.json"
    assert paths.default_runs_root() == tmp_path / "runs"
    monkeypatch.delenv("DFT_AGENT_CONFIG")
    assert paths.default_config_path() == tmp_path / "old.json"


def test_legacy_runs_folder_keeps_being_used(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    legacy = tmp_path / "vasp-slurm-agent-runs"
    legacy.mkdir()
    assert paths.default_runs_root() == legacy
    (tmp_path / "dft-agent-runs").mkdir()
    assert paths.default_runs_root() == tmp_path / "dft-agent-runs"


def test_legacy_config_is_copied_once(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("DFT_AGENT_HOME", raising=False)
    monkeypatch.delenv("VASP_AGENT_CONFIG", raising=False)
    legacy = tmp_path / ".config" / "vasp-slurm-agent" / "cluster.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"host": "old"}')
    copied = paths.migrate_legacy_config()
    assert copied == tmp_path / ".dft-agent" / "cluster.json"
    assert copied.read_text() == '{"host": "old"}'
    if sys.platform == "win32":
        with copied.open("r+") as stream:
            assert stream.read() == '{"host": "old"}'
    else:
        assert copied.stat().st_mode & 0o777 == 0o600
    assert legacy.is_file()
    copied.write_text('{"host": "new"}')
    assert paths.migrate_legacy_config() is None
    assert copied.read_text() == '{"host": "new"}'
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(tmp_path / "explicit.json"))
    assert paths.migrate_legacy_config() is None


def test_settings_round_trip_and_cleaning(tmp_path):
    target = tmp_path / "settings.json"
    assert settings.load_settings(target) == settings.DEFAULTS
    saved = settings.save_settings({"language": "zh", "auto_resume": False, "notify_webhook": "https://hooks.example/x",
                                    "unknown": 1}, target)
    assert saved == target
    loaded = settings.load_settings(target)
    assert loaded["language"] == "zh" and loaded["auto_resume"] is False
    assert loaded["notify_webhook"] == "https://hooks.example/x"
    assert "unknown" not in loaded
    settings.save_settings({"language": "fr", "notifications": "yes", "notify_webhook": "ftp://x"}, target)
    loaded = settings.load_settings(target)
    assert loaded["language"] == "en" and loaded["notifications"] is True and loaded["notify_webhook"] == ""
    target.write_text("not json")
    assert settings.load_settings(target) == settings.DEFAULTS
