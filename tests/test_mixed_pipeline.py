"""Mixed methods keep structure, charge and moment dependencies explicit."""

from dataclasses import replace
import json

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Incar, Poscar

from vasp_slurm_agent import workflow
from test_plan import setup, complete_stage, fake_analysis
from test_workflow import FakeTransport


def test_pbe_relaxation_then_soc_bands_uses_new_soc_charge(setup, monkeypatch):
    source, root, config = setup
    config = replace(config, vasp_ncl_command="srun vasp_ncl")
    state = workflow.prepare_plan(source, root, config, ["relax", "bands"],
                                  stage_parameters=[{}, {"soc": True}])
    assert [stage["name"] for stage in state["stages"]] == ["relax", "scf", "bands"]
    assert [stage["charge_from"] for stage in state["stages"]] == [None, None, 1]
    assert [stage["structure_from"] for stage in state["stages"]] == [None, 0, 0]
    assert not (root / "02_scf/inputs").exists()
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    for _ in range(3):
        state = complete_stage(root, remote)
    assert state["status"] == "succeeded"
    assert len(remote.remote_jobs) == 3
    assert not Incar.from_file(root / "01_relax/inputs/INCAR").get("LSORBIT", False)
    assert Incar.from_file(root / "02_scf/inputs/INCAR")["LSORBIT"]
    assert Incar.from_file(root / "03_bands/inputs/INCAR")["LSORBIT"]
    assert state["stages"][1]["metadata"]["method_fingerprint"] == state["stages"][2]["metadata"]["method_fingerprint"]
    assert "/02_scf/CHGCAR" in (root / "03_bands/inputs/submit.sh").read_text()


def test_changed_scf_method_gets_distinct_job_and_dependency(setup):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, replace(config, vasp_ncl_command="srun vasp_ncl"),
                                  ["scf", "bands", "dos"],
                                  stage_parameters=[{}, {"soc": True}, {"soc": True}])
    assert [stage["name"] for stage in state["stages"]] == ["scf", "scf", "bands", "dos"]
    assert len({stage["job_name"] for stage in state["stages"]}) == 4
    assert state["stages"][2]["charge_from"] == state["stages"][3]["charge_from"] == 1


@pytest.mark.parametrize("functional", ["HSE06", "PBE0"])
def test_pbe_relaxation_then_hybrid_is_self_consistent(setup, monkeypatch, functional):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["relax", "bands"],
                                  stage_parameters=[{}, {"functional": functional}])
    assert [stage["name"] for stage in state["stages"]] == ["relax", "bands"]
    assert state["stages"][1]["charge_from"] is None
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    complete_stage(root, remote)
    state = complete_stage(root, remote)
    assert state["status"] == "succeeded"
    incar = Incar.from_file(root / "02_bands/inputs/INCAR")
    assert incar["LHFCALC"] and incar["ICHARG"] == 2
    assert "CHGCAR" not in (root / "02_bands/inputs/submit.sh").read_text()


def test_next_stage_explicit_vectors_survive_collinear_relaxation(setup, monkeypatch):
    source, root, config = setup
    Poscar(Structure(Lattice.cubic(5), ["O", "Fe"], [[0, 0, 0], [.5, .5, .5]])).write_file(source)
    later = [[0, 0, .1], [0, 4, 0]]
    config = replace(config, vasp_ncl_command="srun vasp_ncl")
    workflow.prepare_plan(source, root, config, ["relax", "scf"], stage_parameters=[
        {"spin": "collinear", "magmom": [.2, 3]},
        {"spin": "noncollinear", "soc": True, "magmom": later},
    ])
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    complete_stage(root, remote)
    remote.queue, remote.accounting = "PENDING", ""
    state = workflow.advance(root, remote)
    assert state["status"] == "queued", state.get("last_error")
    assert state["stages"][1]["metadata"]["method"]["magmom"] == [later[1], later[0]]
    assert state["stages"][1]["parameters"]["magmom"] == later


@pytest.mark.parametrize("overrides", [[{}], [{}, {"magmom": [2]}], [{}, {"functional": "UNKNOWN"}], [{}, {"soc": True}]])
def test_invalid_later_recipe_fails_before_any_run_exists(setup, overrides):
    source, root, config = setup
    with pytest.raises(ValueError):
        workflow.prepare_plan(source, root, config, ["relax", "scf"], stage_parameters=overrides)
    assert not root.exists()


def test_failed_relaxation_never_submits_later_method(setup, monkeypatch):
    source, root, config = setup
    workflow.prepare_plan(source, root, config, ["relax", "bands"],
                          stage_parameters=[{}, {"functional": "HSE06"}])
    monkeypatch.setattr(workflow, "analyze_outputs", lambda *args: fake_analysis(*args, failure=True))
    remote = FakeTransport()
    state = complete_stage(root, remote)
    assert state["status"] == "needs_attention" and len(remote.remote_jobs) == 1
    assert not (root / "02_bands/inputs").exists()
    state = workflow.advance(root, remote)
    assert state["status"] == "needs_attention" and len(remote.remote_jobs) == 1


def test_recipe_overrides_are_frozen_and_changed_settings_insert_scf(setup):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["scf", "dos"], parameters={"encut": 400},
                                  stage_parameters=[{}, {"encut": 500}])
    assert [stage["parameters"]["encut"] for stage in state["stages"]] == [400, 500, 500]
    assert state["stages"][2]["charge_from"] == 1
    saved = json.loads((root / "plan.json").read_text())
    assert saved["stage_parameters"] == [{}, {"encut": 500}]
