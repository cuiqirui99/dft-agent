"""Read verified, relevant saved runs as context for a new plan."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar

from .methods import method_fingerprint, normalize_method, validate_method_output
from .vasp import TASKS, _parameters, _validate_structure


MAX_RUNS = 24
MAX_RESULTS = 8
MAX_RECORD_BYTES = 1_000_000
_TERMINAL = {"succeeded", "failed", "needs_attention", "cancelled"}
_NUMERIC_FACTS = {
    "energy", "energy_per_atom", "force", "fermi_energy", "structure_atoms",
    "structure_volume", "total_moment", "site_moments",
}
_UNITS = {"", "eV", "eV/atom", "eV/Å", "Å³", "atoms", "mu_B"}
_LIMITS = [
    "Use these cases as context, not as settings to copy.",
    "Result checks do not establish cutoff or k-point convergence, or a magnetic ground state.",
]


def _read(path: Path) -> tuple[dict, str]:
    if not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("Invalid saved record.")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Invalid saved record.")
    return value, hashlib.sha256(raw).hexdigest()


def _structure(value: dict | None) -> Structure | None:
    if not isinstance(value, dict):
        return None
    try:
        sites = value["sites"]
        structure = Structure(value["lattice_angstrom"], [site["element"] for site in sites],
                              [site["fractional_coordinates"] for site in sites])
        _validate_structure(structure)
        return structure
    except (KeyError, TypeError, ValueError):
        return None


def _same_geometry(first: Structure, second: Structure) -> bool:
    if len(first) != len(second) or first.species != second.species:
        return False
    if not np.allclose(first.lattice.matrix, second.lattice.matrix, rtol=1e-6, atol=1e-6):
        return False
    delta = first.frac_coords - second.frac_coords
    return bool(np.allclose(delta - np.round(delta), 0, rtol=0, atol=1e-6))


def _same_method(first: dict, second: dict) -> bool:
    for key in ("functional", "spin", "soc", "hubbard_u"):
        if first[key] != second[key]:
            return False
    if not np.allclose(first["saxis"], second["saxis"], rtol=1e-7, atol=1e-8):
        return False
    left, right = first["magmom"], second["magmom"]
    if left is None or right is None:
        return left is right
    return np.shape(left) == np.shape(right) and bool(np.allclose(left, right, rtol=1e-7, atol=1e-8))


def _numeric(value: Any) -> bool:
    if type(value) in (int, float):
        return math.isfinite(value)
    return isinstance(value, list) and len(value) <= 256 and all(_numeric(item) for item in value)


def _recent(root: Path) -> list[Path]:
    with os.scandir(root) as entries:
        candidates = ((entry.stat(follow_symlinks=False).st_mtime_ns, entry.name)
                      for entry in entries if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."))
        return [root / name for _, name in heapq.nlargest(MAX_RUNS, candidates)]


def _source(root: Path, relative: str, sources: dict) -> Structure:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("Invalid saved structure.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sources.get(relative):
        raise ValueError("The saved structure changed.")
    saved = Structure.from_file(path)
    _validate_structure(saved)
    return saved


def _failure(message: str, scheduler: str) -> str:
    if "hybrid-band orbitals" in message:
        return "hybrid_band_consistency_rejected"
    if "do not match the verified solver output" in message:
        return "saved_values_rejected"
    if "current output checks rejected" in message:
        return "output_checks_rejected"
    if scheduler in {"TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "FAILED"}:
        return "scheduler_" + scheduler.lower()
    return "result_unavailable"


def _stage_inputs(root: Path, spec: dict, stage: dict) -> tuple[Structure, dict, dict]:
    folder = str(spec["folder"]) + "/inputs/"
    metadata_path = (root / folder / "metadata.json").resolve()
    if not metadata_path.is_relative_to(root):
        raise ValueError("Invalid stage folder.")
    metadata, _ = _read(metadata_path)
    if metadata != stage.get("metadata"):
        raise ValueError("Stage metadata changed.")
    hashes = metadata["input_sha256"]
    for filename in ("INCAR", "KPOINTS"):
        path = (root / folder / filename).resolve()
        if not path.is_relative_to(root) or path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError("Invalid stage input.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(filename):
            raise ValueError("Stage inputs changed.")
    relative = folder + "POSCAR"
    saved = _source(root, relative, {relative: hashes["POSCAR"]})
    species = [site.specie.symbol for site in saved]
    method = normalize_method(metadata["method"], species)
    if method_fingerprint(method, species, metadata["potcar_labels"]) != metadata["method_fingerprint"]:
        raise ValueError("Stage method changed.")
    validate_method_output(dict(Incar.from_file(root / folder / "INCAR")), metadata)
    settings = _parameters(metadata["parameters"])
    return saved, method, {key: settings[key] for key in ("encut", "mesh", "ediff", "ediffg", "ismear", "sigma", "nelm")}


def retrieve_experience(runs_root: Path | str | None, structure: dict | None, *,
                        parameters: dict | None = None, tasks: list | None = None,
                        limit: int = 4, run_paths: list[Path] | None = None) -> list[dict]:
    """Inspect at most 24 recent run folders. Never copy or change settings.

    Without a method, same-composition cases are advisory examples. With a
    method, all method settings must match; site-dependent settings also require
    the same stage-input cell and ordered sites. Changed results never supply
    success facts. Method settings describe prepared inputs, not unverified outputs.
    """
    from .explanation import load_run_context

    query = _structure(structure)
    if runs_root is None or query is None or type(limit) is not int or limit <= 0:
        return []
    requested = {task for task in (tasks or []) if task in TASKS}
    if tasks and not requested:
        return []
    try:
        root = Path(runs_root).expanduser().resolve()
        if run_paths is None:
            recent = _recent(root)
        else:
            recent = [Path(path).expanduser().resolve() for path in run_paths[:MAX_RUNS]]
            if any(not path.is_dir() or path == root or not path.is_relative_to(root) for path in recent):
                return []
        query_method = normalize_method(parameters, [site.specie.symbol for site in query]) if parameters is not None else None
    except (OSError, TypeError, ValueError):
        return []
    records = []
    for run in recent:
        try:
            state, state_hash = _read(run / "run.json")
            if state.get("status") not in _TERMINAL:
                continue
            plan, plan_hash = _read(run / "plan.json")
            if state.get("plan_sha256") != plan_hash:
                continue
            specifications = plan.get("stages") or []
            sources = plan.get("sources") or {}
            matched = []
            for index, spec in enumerate(specifications):
                task = spec.get("name")
                if task not in TASKS or (requested and task not in requested):
                    continue
                saved = _source(run, spec["source_path"], sources)
                if query.composition.reduced_composition != saved.composition.reduced_composition:
                    continue
                try:
                    saved, method, settings = _stage_inputs(run, spec, state["stages"][index])
                except (OSError, KeyError, IndexError, TypeError, ValueError):
                    continue
                if query.composition.reduced_composition != saved.composition.reduced_composition:
                    continue
                same_geometry = _same_geometry(query, saved)
                if query_method is not None:
                    if not _same_method(query_method, method):
                        continue
                    if (method["magmom"] is not None or method["soc"]) and not same_geometry:
                        continue
                matched.append((index, task, method, settings, same_geometry))
            if not matched:
                continue
            context = load_run_context(run)
            if not context.get("final_plan") or context.get("status") not in _TERMINAL:
                continue
            # Do not mix a fresh context with older run/plan records.
            if _read(run / "run.json")[1] != state_hash or _read(run / "plan.json")[1] != plan_hash:
                continue
            stages = []
            for index, task, method, settings, same_geometry in matched:
                prefix = f"stage_{index + 1}"
                facts = context["facts"]
                accepted = facts.get(prefix + ".accepted", {}).get("value") is True
                status = facts.get(prefix + ".status", {}).get("value")
                if status not in _TERMINAL:
                    continue
                evidence = []
                if accepted:
                    for suffix in sorted(_NUMERIC_FACTS):
                        fact_id = prefix + "." + suffix
                        fact = facts.get(fact_id, {})
                        if _numeric(fact.get("value")) and fact.get("unit", "") in _UNITS:
                            evidence.append({"id": fact_id, "value": fact["value"], "unit": fact.get("unit", "")})
                message = facts.get(prefix + ".availability", {}).get("value", "")
                scheduler = state["stages"][index].get("scheduler_state")
                stages.append({"task": task, "accepted": accepted, "status": status,
                               "method": method, "settings": settings, "same_input_geometry": same_geometry,
                               "settings_basis": "prepared_inputs", "site_order": "stage_POSCAR",
                               "evidence": evidence,
                               "check_evidence": [prefix + ".accepted", prefix + ".status"],
                               "failure": None if accepted else _failure(message, scheduler)})
            if not stages:
                continue
            fingerprint = context["context_sha256"]
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
                continue
            records.append({"case_id": fingerprint[:16], "context_sha256": fingerprint,
                            "formula": query.composition.reduced_formula,
                            "status": context["status"], "stages": stages,
                            "applicability": "matching_method" if query_method is not None else "context_only",
                            "source": "checked_saved_run", "limits": list(_LIMITS)})
        except (OSError, KeyError, IndexError, TypeError, ValueError, AttributeError):
            continue
    records.sort(key=lambda record: any(stage["same_input_geometry"] for stage in record["stages"]), reverse=True)
    return records[:min(limit, MAX_RESULTS)]
