"""Completed example runs bundled with the package. No cluster or model needed."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import shutil

SAMPLE_PREFIX = "sample-"


def _folder():
    return files("vasp_slurm_agent").joinpath("samples")


def list_samples() -> list[dict]:
    """Bundled samples with their descriptions, in display order."""
    samples = []
    for entry in sorted(_folder().iterdir(), key=lambda item: item.name):
        card = entry.joinpath("sample.json")
        if entry.is_dir() and card.is_file():
            data = json.loads(card.read_text(encoding="utf-8"))
            samples.append({"id": entry.name, **data})
    samples.sort(key=lambda item: item.get("order", 99))
    return samples


def sample_card(run_dir: str | Path) -> dict | None:
    """The sample description when this run folder came from a bundled sample."""
    card = Path(run_dir) / "sample.json"
    if not card.is_file():
        return None
    try:
        data = json.loads(card.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def install_samples(runs_root: str | Path, names: list[str] | None = None) -> list[Path]:
    """Copy bundled samples into the run folder. Existing copies are left untouched."""
    root = Path(runs_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    installed = []
    for sample in list_samples():
        if names is not None and sample["id"] not in names:
            continue
        destination = root / (SAMPLE_PREFIX + sample["id"])
        if destination.exists():
            continue
        source = Path(str(_folder().joinpath(sample["id"])))
        shutil.copytree(source, destination)
        for path in destination.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        installed.append(destination)
    return installed
