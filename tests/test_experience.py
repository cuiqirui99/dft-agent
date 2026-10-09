"""History is advisory and cannot turn a stale result into a successful case."""

from copy import deepcopy
import hashlib
from importlib.resources import files
import json
import os

import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from vasp_slurm_agent import explanation, experience, workflow
from vasp_slurm_agent.config import ClusterConfig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value))


def summary(structure):
    return {"formula": structure.composition.reduced_formula, "number_of_sites": len(structure),
            "lattice_angstrom": structure.lattice.matrix.tolist(),
            "sites": [{"index": index, "element": site.specie.symbol,
                       "fractional_coordinates": site.frac_coords.tolist()} for index, site in enumerate(structure)]}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    structure = Structure.from_file(source)
    results = {}
    monkeypatch.setattr(explanation, "analyze_outputs", lambda path, *args: deepcopy(results[(path / "vasprun.xml").read_text()]))
    monkeypatch.setattr(workflow, "SSHTransport", lambda *args: pytest.fail("History must not use SSH"))

    def create(name="case", parameters=None, status="succeeded", tasks=None, relaxed=None):
        run = root / name
        config = ClusterConfig(host="private-host.example", user="private-user", remote_root="/private/remote",
                               vasp_command="srun vasp_std", vasp_ncl_command="srun vasp_ncl",
                               potcar_root="/private/licensed", partition="cpu")
        state = workflow.prepare_plan(source, run, config, tasks or ["scf"], parameters or {})
        for index, stage in enumerate(state["stages"]):
            workflow._materialize_stage(run, state, config, stage)
            inputs, output = run / stage["folder"] / "inputs", run / stage["folder"] / "outputs"
            output.mkdir()
            for filename in ("POSCAR", "INCAR", "KPOINTS"):
                (output / filename).write_bytes((inputs / filename).read_bytes())
            key = name if index == 0 else f"{name}-{index}"
            (output / "vasprun.xml").write_text(key)
            (output / "OUTCAR").write_text("solver fixture")
            (output / "input_hashes.sha256").write_text("\n".join(f"{sha(inputs / filename)}  {filename}" for filename in ("INCAR", "KPOINTS", "POSCAR")))
            final_structure = relaxed if relaxed is not None else structure
            final_structure.to(filename=output / "final_structure.cif")
            Poscar(final_structure).write_file(output / "final_structure.vasp")
            write(output / "artifact_manifest.json", {path.name: {"sha256": sha(path)} for path in output.iterdir()})
            result = {"success": True, "task": stage["name"], "converged_electronic": True, "converged_ionic": True,
                      "input_identity_verified": True, "scheduler_state": "COMPLETED", "final_energy_ev": -10.0,
                      "final_energy_ev_per_atom": -5.0, "final_max_force_ev_angstrom": 0.001, "fermi_energy_ev": 5.0,
                      "vasprun_sha256": sha(output / "vasprun.xml"), "final_structure_sha256": sha(output / "final_structure.cif"),
                      "final_structure_poscar_sha256": sha(output / "final_structure.vasp"),
                      "method_fingerprint": stage["metadata"]["method_fingerprint"], "method": stage["metadata"]["method"]}
            results[key] = result
            write(output / "result.json", result)
            stage.update(status=status, scheduler_state="TIMEOUT" if status == "failed" else "COMPLETED", result=result)
        state.update(status=status, last_error="password=private-secret /private/remote")
        write(run / "run.json", state)
        plan = json.loads((run / "plan.json").read_text())
        write(run / "proposal.json", {"goal": "private research question; password=private-secret",
                                       "source_sha256": sha(source), "tasks": plan["tasks"], "parameters": plan["parameters"]})
        return run, output

    return root, summary(structure), create, results


def test_retrieves_verified_context_without_private_text(runs):
    root, query, create, _ = runs
    run, _ = create()
    before = {str(path.relative_to(run)): sha(path) for path in run.rglob("*") if path.is_file()}
    record, = experience.retrieve_experience(root, query)
    assert record["applicability"] == "context_only"
    assert record["stages"][0]["accepted"]
    assert record["stages"][0]["same_input_geometry"]
    assert {fact["id"]: fact["value"] for fact in record["stages"][0]["evidence"]}["stage_1.energy"] == -10
    metadata = json.loads((run / "01_scf/inputs/metadata.json").read_text())
    assert record["stages"][0]["settings"]["mesh"] == metadata["parameters"]["mesh"]
    assert record["numerical_convergence_verified"] is False
    assert record["context_sha256"] == explanation.load_run_context(run)["context_sha256"]
    encoded = json.dumps(record)
    for secret in ("private-host", "private-user", "private-secret", "private research", "/private/", str(run)):
        assert secret not in encoded
    assert before == {str(path.relative_to(run)): sha(path) for path in run.rglob("*") if path.is_file()}


@pytest.mark.parametrize("parameters", [
    {"functional": "HSE06"}, {"spin": "collinear", "magmom": [2, -2]},
    {"soc": True}, {"hubbard_u": {"Si": {"l": 1, "u": 4, "j": 0}}},
])
def test_incompatible_methods_are_excluded(runs, parameters):
    root, query, create, _ = runs
    create()
    assert experience.retrieve_experience(root, query, parameters=parameters) == []
    assert experience.retrieve_experience(root, query, parameters={})[0]["applicability"] == "matching_method"


def test_soc_axis_and_site_order_are_not_transferable(runs):
    root, query, create, _ = runs
    params = {"spin": "noncollinear", "soc": True, "magmom": [[0, 0, 2], [0, 0, -2]], "saxis": [0, 0, 1]}
    create(parameters=params)
    assert experience.retrieve_experience(root, query, parameters=params)
    assert experience.retrieve_experience(root, query, parameters={**params, "saxis": [1, 0, 0]}) == []
    assert experience.retrieve_experience(root, query, parameters={**params, "magmom": list(reversed(params["magmom"]))}) == []
    reversed_sites = {**query, "sites": list(reversed(query["sites"]))}
    assert experience.retrieve_experience(root, reversed_sites, parameters=params) == []
    advisory, = experience.retrieve_experience(root, reversed_sites)
    assert not advisory["stages"][0]["same_input_geometry"]


def test_downstream_soc_uses_relaxed_stage_geometry(runs):
    root, query, create, _ = runs
    relaxed = experience._structure(query)
    relaxed.scale_lattice(relaxed.volume * 1.1)
    relaxed.translate_sites([1], [0.03, 0, 0], frac_coords=True)
    params = {"spin": "noncollinear", "soc": True, "magmom": [[0, 0, 2], [0, 0, -2]], "saxis": [0, 0, 1]}
    run, _ = create(parameters=params, tasks=["relax", "scf"], relaxed=relaxed)
    assert experience.retrieve_experience(root, query, parameters=params, tasks=["scf"]) == []
    advisory, = experience.retrieve_experience(root, query, tasks=["scf"])
    assert advisory["stages"][0]["same_input_geometry"] is False
    actual = Structure.from_file(run / "02_scf/inputs/POSCAR")
    record, = experience.retrieve_experience(root, summary(actual), parameters=params, tasks=["scf"])
    assert record["stages"][0]["accepted"]
    assert record["stages"][0]["same_input_geometry"]
    assert record["stages"][0]["site_order"] == "stage_POSCAR"


def test_changed_prepared_inputs_cannot_supply_method_context(runs):
    root, query, create, _ = runs
    run, _ = create()
    (run / "01_scf/inputs/POSCAR").write_text("changed")
    assert experience.retrieve_experience(root, query, parameters={}) == []
    assert experience.retrieve_experience(root, query) == []


def test_composition_task_and_pending_cases_are_filtered(runs):
    root, query, create, _ = runs
    create()
    create("pending", status="running")
    assert len(experience.retrieve_experience(root, query)) == 1
    assert experience.retrieve_experience(root, query, tasks=["relax"]) == []
    other = deepcopy(query)
    for site in other["sites"]:
        site["element"] = "C"
    assert experience.retrieve_experience(root, other) == []


@pytest.mark.parametrize("filename", ["vasprun.xml", "OUTCAR", "result.json"])
def test_changed_output_never_supplies_success(runs, filename):
    root, query, create, _ = runs
    _, output = create()
    (output / filename).write_text("changed")
    record, = experience.retrieve_experience(root, query)
    assert record["status"] == "needs_attention"
    assert record["stages"][0]["accepted"] is False
    assert record["stages"][0]["evidence"] == []
    assert record["stages"][0]["failure"] == "result_unavailable"
    assert record["stages"][0]["settings_basis"] == "prepared_inputs"
    assert record["source"] == "checked_saved_run"


def test_changed_frozen_source_is_excluded(runs):
    root, query, create, _ = runs
    run, _ = create()
    plan = json.loads((run / "plan.json").read_text())
    (run / next(iter(plan["sources"]))).write_text("changed source")
    assert experience.retrieve_experience(root, query) == []


def test_current_rejection_is_useful_negative_evidence(runs):
    root, query, create, results = runs
    create()
    results["case"] = {"success": False, "hybrid_kpoint_consistency": {"passed": False}}
    record, = experience.retrieve_experience(root, query)
    stage = record["stages"][0]
    assert not stage["accepted"] and stage["evidence"] == []
    assert stage["failure"] == "hybrid_band_consistency_rejected"


def test_failed_scheduler_keeps_failure_not_cached_energy(runs):
    root, query, create, _ = runs
    create(status="failed")
    record, = experience.retrieve_experience(root, query)
    assert record["stages"][0]["failure"] == "scheduler_timeout"
    assert record["stages"][0]["evidence"] == []


def test_scans_only_bounded_recent_runs_and_skips_symlinks(runs, monkeypatch):
    root, query, create, _ = runs
    old, _ = create("old")
    new, _ = create("new")
    os.utime(old, ns=(1, 1))
    (root / "link").symlink_to(old, target_is_directory=True)
    original = explanation.load_run_context
    calls = []
    def record(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(explanation, "load_run_context", record)
    monkeypatch.setattr(experience, "MAX_RUNS", 1)
    assert len(experience.retrieve_experience(root, query, limit=100)) == 1
    assert calls == [new]


def test_empty_or_invalid_queries_do_not_read_history(runs, monkeypatch):
    root, query, create, _ = runs
    create()
    monkeypatch.setattr(experience, "_recent", lambda *args: pytest.fail("No history scan expected"))
    assert experience.retrieve_experience(None, query) == []
    assert experience.retrieve_experience(root, None) == []
    assert experience.retrieve_experience(root, query, limit=0) == []
