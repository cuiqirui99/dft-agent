"""Preferences saved on this computer. Credentials are never stored here."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import settings_path

LANGUAGES = ("en", "zh")
DEFAULTS = {
    "language": "en",
    "auto_resume": True,
    "notifications": True,
    "notify_webhook": "",
    "interactive_viewer": True,
    "check_updates": True,
    "keychain": False,
}


def _clean(values: dict) -> dict:
    result = dict(DEFAULTS)
    if not isinstance(values, dict):
        return result
    for key, default in DEFAULTS.items():
        value = values.get(key, default)
        if isinstance(default, bool):
            result[key] = bool(value) if isinstance(value, bool) else default
        elif key == "language":
            result[key] = value if value in LANGUAGES else default
        elif key == "notify_webhook":
            text = value.strip() if isinstance(value, str) else ""
            result[key] = text if text.startswith(("https://", "http://")) and "\n" not in text else ""
    return result


def load_settings(path: str | Path | None = None) -> dict:
    target = Path(path).expanduser() if path else settings_path()
    try:
        return _clean(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_settings(values: dict, path: str | Path | None = None) -> Path:
    target = Path(path).expanduser() if path else settings_path()
    merged = _clean({**load_settings(target), **(values or {})})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
