"""Local file locations, with fallbacks for the names used before 0.4.2."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

APP_DIR_NAME = ".dft-agent"
RUNS_DIR_NAME = "dft-agent-runs"
LEGACY_CONFIG_RELATIVE = Path(".config") / "vasp-slurm-agent" / "cluster.json"
LEGACY_RUNS_NAME = "vasp-slurm-agent-runs"


def app_home() -> Path:
    """Folder for settings and caches. Override with DFT_AGENT_HOME."""
    value = os.environ.get("DFT_AGENT_HOME")
    return Path(value).expanduser() if value else Path.home() / APP_DIR_NAME


def legacy_config_path() -> Path:
    return Path.home() / LEGACY_CONFIG_RELATIVE


def legacy_runs_root() -> Path:
    return Path.home() / LEGACY_RUNS_NAME


def default_config_path() -> Path:
    """Cluster configuration file. DFT_AGENT_CONFIG wins; VASP_AGENT_CONFIG is still honoured."""
    for name in ("DFT_AGENT_CONFIG", "VASP_AGENT_CONFIG"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return app_home() / "cluster.json"


def default_runs_root() -> Path:
    """Run folder. An existing folder from an earlier version keeps being used."""
    value = os.environ.get("DFT_AGENT_RUNS")
    if value:
        return Path(value).expanduser()
    current = Path.home() / RUNS_DIR_NAME
    legacy = legacy_runs_root()
    if not current.exists() and legacy.is_dir():
        return legacy
    return current


def settings_path() -> Path:
    return app_home() / "settings.json"


def update_cache_path() -> Path:
    return app_home() / "update-check.json"


def migrate_legacy_config(target: Path | None = None) -> Path | None:
    """Copy a configuration saved by an earlier version to the new location, once.

    The old file is left in place. Nothing happens when the new file exists or
    when an explicit path is configured through the environment.
    """
    if os.environ.get("DFT_AGENT_CONFIG") or os.environ.get("VASP_AGENT_CONFIG"):
        return None
    target = Path(target).expanduser() if target else default_config_path()
    legacy = legacy_config_path()
    if target.exists() or not legacy.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(legacy, target)
    target.chmod(0o600)
    return target
