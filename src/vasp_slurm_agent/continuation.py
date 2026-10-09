"""Prepare a new calculation from a verified structure without changing its run."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .explanation import load_run_context
from .methods import METHOD_DEFAULTS
from . import workflow


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_continuation(source_run: str | Path, source_stage: int | str) -> dict:
    """Return a verified local planning source and its inherited settings.

    The stage is a zero-based index or an exact stage folder name. Keep the
    context_sha256 and pass it to prepare_continuation after reviewing the plan.
    """
    root = Path(source_run).expanduser().resolve()
    context = load_run_context(root)
    state = json.loads((root / "run.json").read_text())
    stages = state.get("stages", [])
    index = source_stage
    if isinstance(source_stage, str):
        matches = [i for i, stage in enumerate(stages) if stage.get("folder") == source_stage]
        index = matches[0] if len(matches) == 1 else None
    if type(index) is not int or not 0 <= index < len(stages):
        raise ValueError("Choose an existing stage by index or folder name.")
    stage = stages[index]
    prefix = f"stage_{index + 1}"
    if context["facts"].get(prefix + ".accepted", {}).get("value") is not True:
        raise ValueError("Continue from a stage with accepted results.")
    artifact = next((item for item in context["artifacts"]
                     if item["id"] == prefix + ".final_structure.vasp" and item["status"] == "verified"), None)
    if not artifact:
        raise ValueError("The verified final POSCAR is unavailable; its site order and Cartesian frame are required.")
    structure = root / artifact["path"]
    metadata = stage["metadata"]
    parameters = deepcopy(metadata.get("parameters") or stage.get("parameters") or {})
    parameters.update({key: deepcopy(value) for key, value in (metadata.get("method") or {}).items()
                       if key in METHOD_DEFAULTS})
    # normalize_method generates zero vectors itself for spin=none with SOC.
    if parameters.get("spin", "none") == "none":
        parameters["magmom"] = None
    provenance = {"schema_version": 1, "source_run_id": state["run_id"],
                  "source_stage": stage["folder"], "source_task": stage["name"],
                  "source_context_sha256": context["context_sha256"],
                  "source_structure_sha256": artifact["sha256"],
                  "source_vasprun_sha256": _sha256(root / stage["folder"] / "outputs/vasprun.xml"),
                  "source_plan_sha256": state.get("plan_sha256"),
                  "site_order": "Accepted final POSCAR order; Cartesian cell and SAXIS are preserved.",
                  "electronic_state_reused": False,
                  "note": "Only the structure is reused. Initial moments are seeds; electronic states are recalculated."}
    if load_run_context(root)["context_sha256"] != context["context_sha256"]:
        raise ValueError("The source results changed. Select the stage again.")
    return {"structure_path": str(structure), "parameters": parameters, "provenance": provenance,
            "context_sha256": context["context_sha256"], "source_stage": index}


def _prepare(source_run, source_stage, run_dir, expected_context_sha256, prepare):
    source = Path(source_run).expanduser().resolve()
    destination = Path(run_dir).expanduser().resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Use a separate run folder for the continuation.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("This run already exists. Choose a new run folder.")
    inspected = inspect_continuation(source, source_stage)
    if expected_context_sha256 is not None and expected_context_sha256 != inspected["context_sha256"]:
        raise ValueError("The source results changed after planning. Review a new plan.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dft-continue-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        structure = temporary / "POSCAR"
        shutil.copyfile(inspected["structure_path"], structure)
        if _sha256(structure) != inspected["provenance"]["source_structure_sha256"]:
            raise ValueError("The source structure changed. Review a new plan.")
        prepared = temporary / "run"
        state = prepare(structure, prepared, inspected)
        (prepared / "parent.json").write_text(json.dumps(inspected["provenance"], indent=2) + "\n")
        if load_run_context(source)["context_sha256"] != inspected["context_sha256"]:
            raise ValueError("The source results changed during preparation. Review a new plan.")
        os.replace(prepared, destination)
    return state


def prepare_continuation(source_run, source_stage, run_dir, config, tasks,
                         parameters=None, stage_parameters=None, expected_context_sha256=None):
    """Prepare offline inputs from the selected accepted stage; never submit.

    Parameters override the inherited recipe. Per-stage overrides follow
    workflow.prepare_plan. Requested bands/DOS receive a fresh SCF when needed.
    """
    def prepare(structure, prepared, inspected):
        recipe = {**inspected["parameters"], **deepcopy(dict(parameters or {}))}
        return workflow.prepare_plan(structure, prepared, config, tasks, recipe,
                                     stage_parameters=stage_parameters)
    return _prepare(source_run, source_stage, run_dir, expected_context_sha256, prepare)


def prepare_task_continuation(source_run, source_stage, run_dir, config, plan,
                              expected_context_sha256):
    """Prepare a reviewed task, including structure edits or comparison variants."""
    from .task_agent import prepare_task

    def prepare(structure, prepared, inspected):
        if plan.get("source_sha256") != inspected["provenance"]["source_structure_sha256"]:
            raise ValueError("The selected structure differs from the task plan. Create a new plan.")
        return prepare_task(structure, prepared, config, plan)
    return _prepare(source_run, source_stage, run_dir, expected_context_sha256, prepare)
