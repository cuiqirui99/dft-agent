"""Tell the user when a calculation finishes: desktop notice and optional webhook."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from .settings import load_settings

_run = subprocess.run
_open = urlopen


def _desktop(title: str, message: str) -> bool:
    system = platform.system()
    try:
        if system == "Darwin" and shutil.which("osascript"):
            script = 'display notification {} with title {}'.format(json.dumps(message), json.dumps(title))
            return _run(["osascript", "-e", script], capture_output=True, timeout=10).returncode == 0
        if system == "Linux" and shutil.which("notify-send"):
            return _run(["notify-send", title, message], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def _webhook(url: str, title: str, message: str, run: str) -> bool:
    if not url.startswith(("https://", "http://")):
        return False
    try:
        text = f"{title}: {message}"
        host = (urlsplit(url).hostname or "").lower()
        if host in {"discord.com", "discordapp.com", "canary.discord.com", "ptb.discord.com"}:
            body = {"content": text[:2000], "allowed_mentions": {"parse": []}}
        else:
            body = {"text": text, "title": title, "message": message, "run": run}
        payload = json.dumps(body).encode("utf-8")
        request = Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "dft-agent"}, method="POST")
        with _open(request, timeout=10) as response:  # noqa: S310 - user-configured URL
            return 200 <= response.status < 300
    except Exception:
        return False


def notify(title: str, message: str, *, run: str = "", settings: dict | None = None) -> list[str]:
    """Send on every enabled channel. Failures are ignored; the run is unaffected."""
    settings = settings if settings is not None else load_settings()
    channels = []
    if settings.get("notifications"):
        if _desktop(title, message):
            channels.append("desktop")
        if settings.get("notify_webhook") and _webhook(settings["notify_webhook"], title, message, run):
            channels.append("webhook")
    return channels


STATUS_WORDS = {"succeeded": "completed", "needs_attention": "needs attention", "failed": "failed", "cancelled": "was cancelled"}


def notify_run(state: dict, run: str = "") -> list[str]:
    status = str(state.get("status", ""))
    label = STATUS_WORDS.get(status, status)
    parts = [part for part in (state.get("formula"), state.get("task")) if part]
    return notify("DFT Agent", f"{' '.join(parts) or 'Calculation'} {label}.", run=run)
