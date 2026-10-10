"""Check GitHub Releases for a newer version. Never raises; cached for a day."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from urllib.request import Request, urlopen

from . import __version__
from .paths import update_cache_path

RELEASES_API = "https://api.github.com/repos/cuiqirui99/dft-agent/releases/latest"
RELEASES_PAGE = "https://github.com/cuiqirui99/dft-agent/releases"
MAX_AGE_SECONDS = 24 * 3600
DISABLE_ENV = "DFT_AGENT_NO_UPDATE_CHECK"


def parse_version(text: str) -> tuple[int, ...]:
    """'v0.4.2' -> (0, 4, 2). Trailing labels such as rc1 are ignored."""
    numbers = re.findall(r"\d+", str(text).split("+")[0])
    return tuple(int(part) for part in numbers[:3]) or (0,)


def _fetch(timeout: float) -> dict:
    request = Request(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                             "User-Agent": f"dft-agent/{__version__}"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        payload = json.loads(response.read(200_000).decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tag_name"), str):
        raise ValueError("Unexpected release data.")
    url = payload.get("html_url") if isinstance(payload.get("html_url"), str) else RELEASES_PAGE
    if not url.startswith("https://github.com/"):
        url = RELEASES_PAGE
    return {"latest": payload["tag_name"].lstrip("v")[:40], "url": url}


def read_cache(path: str | Path | None = None) -> dict | None:
    target = Path(path).expanduser() if path else update_cache_path()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("latest"), str):
        return None
    return value


def check_for_update(current: str = __version__, *, cache_path: str | Path | None = None,
                     max_age: float = MAX_AGE_SECONDS, timeout: float = 3.0, fetch=None,
                     now: datetime | None = None) -> dict | None:
    """Return {"latest", "url", "newer", "checked_at"} or None when unavailable."""
    if os.environ.get(DISABLE_ENV):
        return None
    now = now or datetime.now(timezone.utc)
    target = Path(cache_path).expanduser() if cache_path else update_cache_path()
    cached = read_cache(target)
    if cached:
        try:
            age = (now - datetime.fromisoformat(cached["checked_at"])).total_seconds()
        except (KeyError, TypeError, ValueError):
            age = max_age + 1
        if 0 <= age <= max_age:
            cached["newer"] = parse_version(cached["latest"]) > parse_version(current)
            return cached
    try:
        info = (fetch or _fetch)(timeout)
    except Exception:
        return None
    result = {"latest": info["latest"], "url": info["url"], "checked_at": now.isoformat(),
              "newer": parse_version(info["latest"]) > parse_version(current)}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return result


_started = False


def start_background_check(**options) -> None:
    """Run one check in a daemon thread so the app never waits for the network."""
    global _started
    if _started or os.environ.get(DISABLE_ENV):
        return
    _started = True
    threading.Thread(target=check_for_update, kwargs=options, daemon=True, name="dft-agent-update-check").start()


def available_update(current: str = __version__, cache_path: str | Path | None = None) -> dict | None:
    """The cached result when it names a newer version; otherwise None."""
    cached = read_cache(cache_path)
    if cached and parse_version(cached["latest"]) > parse_version(current):
        return cached
    return None
