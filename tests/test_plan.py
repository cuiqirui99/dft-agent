from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent import workflow
from test_workflow import FakeTransport, completed_files


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = ClusterConfig(host="cluster.invalid", user="test", remote_root="/scratch/test",
                           vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu")
    source = tmp_path / "POSCAR"
    Poscar(Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]])).write_file(source)
    monkeypatch.setattr(workflow, "SSHTransport", lambda *a, **k: pytest.fail("No real SSH in plan tests"))
    return source, tmp_path / "run", config


def fake_analysis(output, task, expected, *, failure=False, relaxed=False):
    metadata = json.loads((output / "metadata.json").read_text())
    structure = Structure.from_file(expected)
    if relaxed:
        structure.translate_sites([1], [.01, 0, 0])
    if not failure:
        structure.to(filename=str(output / "final_structure.cif"))
        Poscar(structure).write_file(output / "final_structure.vasp")
    return {"success": not failure, "reason": "Injected parser refusal" if failure else "Accepted by fake parser",
            "converged_electronic": not failure, "final_energy_ev": -float(len(structure)),
            "fermi_energy_ev": 3.5,
            "final_structure_poscar_sha256": workflow._digest(output / "final_structure.vasp") if not failure else None,
            "method_comparison_fingerprint": metadata.get("method_comparison_fingerprint")}


def complete_stage(root, remote):
    state = workflow.read_state(root)
    remote.job_id = str(80721 + state["current_stage"])
    remote.queue, remote.accounting = "PENDING", ""
    submitted = workflow.advance(root, remote)
    assert submitted["status"] == "queued", submitted.get("last_error")
    completed_files(root, remote)
    return workflow.advance(root, remote)


def test_relaxed_structure_is_deferred_and_scf_reused(setup, monkeypatch):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["relax", "bands", "dos"])
    assert [stage["name"] for stage in state["stages"]] == ["relax", "scf", "bands", "dos"]
    assert [stage["materialized"] for stage in state["stages"]] == [True, False, False, False]
    assert not (root / "02_scf" / "inputs").exists()
    monkeypatch.setattr(workflow, "analyze_outputs", lambda out, task, expected: fake_analysis(out, task, expected, relaxed=task == "relax"))
    remote = FakeTransport()
    state = complete_stage(root, remote)
    assert state["current_stage"] == 1
    accepted = Structure.from_file(root / "01_relax/outputs/final_structure.cif")
    assert not (root / "02_scf" / "inputs").exists()
    for _ in range(3):
        state = complete_stage(root, remote)
    assert state["status"] == "succeeded"
    for folder in ("02_scf", "03_bands", "04_dos"):
        assert Structure.from_file(root / folder / "inputs/POSCAR") == accepted
    for folder in ("03_bands", "04_dos"):
        script = (root / folder / "inputs/submit.sh").read_text()
        assert "/02_scf/CHGCAR" in script
        assert "/01_relax/CHGCAR" not in script and "/03_bands/CHGCAR" not in script
        assert json.loads((root / folder / "outputs/metadata.json").read_text())["scf_fermi_energy_ev"] == 3.5
    assert len(remote.remote_jobs) == 4


def test_post_relax_moments_follow_sorted_site_order(setup, monkeypatch):
    source, root, config = setup
    Poscar(Structure(Lattice.cubic(4), ["Ni", "Fe"], [[0, 0, 0], [.5, .5, .5]])).write_file(source)
    state = workflow.prepare_plan(source, root, config, ["relax", "scf"], {"spin": "collinear", "magmom": [1., 4.]})
    sorted_moments = state["stages"][0]["metadata"]["method"]["magmom"]
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    complete_stage(root, remote)
    remote.queue, remote.accounting, remote.job_id = "PENDING", "", "80722"
    state = workflow.advance(root, remote)
    assert state["status"] == "queued"
    assert state["stages"][1]["metadata"]["method"]["magmom"] == sorted_moments


def test_missing_ncl_command_refuses_without_partial_run(setup):
    source, root, config = setup
    with pytest.raises(ValueError, match="vasp_ncl_command"):
        workflow.prepare_plan(source, root, config, ["scf"], {"soc": True})
    assert not root.exists()


def test_ncl_route_and_old_config_default(setup):
    source, root, config = setup
    legacy = config.to_dict()
    legacy.pop("vasp_ncl_command")
    assert ClusterConfig(**legacy).vasp_ncl_command == ""
    with pytest.raises(ValueError, match="single line"):
        replace(config, vasp_ncl_command="vasp_ncl\nrm x")
    config = replace(config, vasp_ncl_command="srun /apps/vasp_ncl")
    state = workflow.prepare_plan(source, root, config, ["scf"], {"soc": True})
    script = workflow._script(config, state["stages"][0], None)
    assert "srun /apps/vasp_ncl" in script and "srun vasp_std" not in script
    assert "cp " not in script


@pytest.mark.parametrize("functional", ["HSE06", "PBE0"])
def test_hybrid_spectra_do_not_copy_charge_or_add_scf(setup, functional):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["bands", "dos"], {"functional": functional})
    assert [stage["name"] for stage in state["stages"]] == ["bands", "dos"]
    for stage in state["stages"]:
        assert stage["charge_from"] is None
        assert not stage["metadata"]["requires_chgcar"]
        assert "CHGCAR" not in workflow._script(config, stage, None)


@pytest.mark.parametrize("changed", ["plan", "source", "stage"])
def test_frozen_plan_or_source_change_refuses_before_network(setup, changed):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["scf"], magnetic_states=[])
    if changed == "plan":
        (root / "plan.json").write_text("{}")
    elif changed == "source":
        (root / state["stages"][0]["source_path"]).write_text("changed")
    else:
        state["stages"][0]["parameters"]["encut"] = 999
        workflow._write(root / "run.json", state)
    remote = FakeTransport()
    result = workflow.advance(root, remote)
    assert result["status"] == "needs_attention"
    assert not remote.commands


def test_magnetic_seeds_share_one_supercell_and_failures_are_excluded(setup, monkeypatch):
    source, root, config = setup
    Poscar(Structure(Lattice.cubic(2.9), ["Fe"], [[0, 0, 0]])).write_file(source)
    state = workflow.prepare_plan(source, root, config, ["scf"], magnetic_states=["NM", "FM", "AFM"])
    assert {stage["metadata"]["number_of_atoms"] for stage in state["stages"]} == {2}
    assert len({stage["metadata"]["input_sha256"]["POSCAR"] for stage in state["stages"]}) == 1
    assert state["stages"][1]["metadata"]["method"]["magmom"] == [3., 3.]
    assert sorted(state["stages"][2]["metadata"]["method"]["magmom"]) == [-3., 3.]
    assert json.loads((root / "plan.json").read_text())["magnetic_seeds"]["supercell"] == [2, 1, 1]
    monkeypatch.setattr(workflow, "analyze_outputs", lambda out, task, expected: fake_analysis(out, task, expected, failure="_fm" in str(out)))
    remote = FakeTransport()
    for _ in range(3):
        state = complete_stage(root, remote)
    assert state["status"] == "needs_attention"
    assert state["comparison"]["status"] == "partial"
    assert {row["seed"] for row in state["comparison"]["ranking"]} == {"NM", "AFM"}
    assert state["comparison"]["excluded"][0]["seed"] == "FM"
    assert len(remote.remote_jobs) == 3


def test_comparison_rejects_changed_method_and_nonfinite_energy(setup):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["scf"], magnetic_states=["NM", "FM", "AFM"])
    for stage in state["stages"]:
        stage["status"] = "succeeded"
        stage["result"] = {"success": True, "converged_electronic": True, "final_energy_ev": -4.,
                           "method_comparison_fingerprint": stage["metadata"]["method_comparison_fingerprint"]}
    state["stages"][1]["metadata"]["parameters"]["encut"] = 999
    state["stages"][2]["result"]["final_energy_ev"] = float("nan")
    workflow._update_comparison(state)
    assert [row["seed"] for row in state["comparison"]["ranking"]] == ["NM"]
    assert not state["comparison"]["comparison_available"]
    assert state["comparison"]["lowest_energy_seed"] is None
    assert len(state["comparison"]["excluded"]) == 2


def test_invalid_sequence_does_not_create_run(setup):
    source, root, config = setup
    with pytest.raises(ValueError, match="SCF only"):
        workflow.prepare_plan(source, root, config, ["relax", "scf"], magnetic_states=["NM", "FM"])
    assert not root.exists()


def test_legacy_state_and_config_still_complete(setup, monkeypatch):
    source, root, config = setup
    state = workflow.prepare_run(source, root, config, "bands")
    state["schema_version"] = 1
    for key in ("plan_sha256", "tasks"):
        state.pop(key)
    for stage in state["stages"]:
        for key in ("materialized", "parameters", "structure_from", "charge_from", "source_path", "source_sha256"):
            stage.pop(key)
        for key in ("method", "method_fingerprint", "method_comparison_fingerprint", "requires_ncl", "spectral_charge_mode"):
            stage["metadata"].pop(key, None)
    old_config = config.to_dict()
    old_config.pop("vasp_ncl_command")
    workflow._write(root / "config.json", old_config)
    state["config_sha256"] = workflow._digest(root / "config.json")
    workflow._write(root / "run.json", state)
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    complete_stage(root, remote)
    state = complete_stage(root, remote)
    assert state["status"] == "succeeded"
    assert "/01_scf/CHGCAR" in (root / "02_bands/inputs/submit.sh").read_text()
    assert len(remote.remote_jobs) == 2


def test_soc_relaxation_preserves_cartesian_orientation(setup, monkeypatch):
    source, root, config = setup
    angle = np.deg2rad(29)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    structure = Structure(Lattice(np.diag([4., 5., 6.]) @ rotation), ["Si", "Si"], [[0, 0, 0], [.3, .3, .3]])
    Poscar(structure).write_file(source)
    config = replace(config, vasp_ncl_command="srun vasp_ncl")
    workflow.prepare_plan(source, root, config, ["relax", "scf"], {"soc": True, "saxis": [1, 0, 0]})
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    state = complete_stage(root, remote)
    assert state["stages"][0]["result"]["final_structure_poscar_sha256"]
    remote.queue, remote.accounting, remote.job_id = "PENDING", "", "80722"
    state = workflow.advance(root, remote)
    assert state["status"] == "queued", state.get("last_error")
    downstream = Structure.from_file(root / "02_scf/inputs/POSCAR")
    assert np.allclose(downstream.lattice.matrix, structure.lattice.matrix)
    assert state["stages"][1]["metadata"]["method"]["saxis"] == [1., 0., 0.]


@pytest.mark.parametrize("change", ["missing_hash", "changed_file"])
def test_soc_deferred_stage_refuses_unbound_structure(setup, monkeypatch, change):
    source, root, config = setup
    config = replace(config, vasp_ncl_command="srun vasp_ncl")
    workflow.prepare_plan(source, root, config, ["relax", "scf"], {"soc": True})
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    state = complete_stage(root, remote)
    if change == "missing_hash":
        state["stages"][0]["result"].pop("final_structure_poscar_sha256")
        workflow._write(root / "run.json", state)
    else:
        (root / "01_relax/outputs/final_structure.vasp").write_text("changed")
    commands_before = len(remote.commands)
    state = workflow.advance(root, remote)
    assert state["status"] == "needs_attention"
    assert len(remote.commands) == commands_before
    assert not (root / "02_scf/inputs").exists()


def test_noncollinear_comparison_keeps_one_spin_basis(setup):
    source, root, config = setup
    config = replace(config, vasp_ncl_command="srun vasp_ncl")
    state = workflow.prepare_plan(source, root, config, ["scf"],
                                  {"spin": "noncollinear", "saxis": [1, 0, 0]}, ["NM", "FM", "AFM"])
    assert all(stage["metadata"]["requires_ncl"] for stage in state["stages"])
    assert state["stages"][0]["metadata"]["method"]["magmom"] == [[0., 0., 0.], [0., 0., 0.]]
    assert len({stage["metadata"]["method_comparison_fingerprint"] for stage in state["stages"]}) == 1
