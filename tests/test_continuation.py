import json
from pathlib import Path

import pytest
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure

from vasp_slurm_agent import continuation, explanation, workflow
from vasp_slurm_agent.config import ClusterConfig
from test_explanation import accepted_run, sha, write


@pytest.fixture
def parent(accepted_run):
    root, output, parser = accepted_run
    structure = Structure.from_file(root / "01_scf/inputs/POSCAR")
    Poscar(structure).write_file(output / "final_structure.vasp")
    state = json.loads((root / "run.json").read_text())
    result = state["stages"][0]["result"]
    result["final_structure_poscar_sha256"] = sha(output / "final_structure.vasp")
    write(output / "result.json", result)
    write(root / "run.json", state)
    original = parser.side_effect
    def parse(*args):
        parsed = original(*args)
        parsed["final_structure_poscar_sha256"] = result["final_structure_poscar_sha256"]
        return parsed
    parser.side_effect = parse
    return root, output


def config():
    return ClusterConfig(host="cluster.example", user="researcher", remote_root="/scratch/test",
                         vasp_command="srun vasp_std", vasp_ncl_command="srun vasp_ncl",
                         potcar_root="/licensed", partition="cpu")


def hashes(root):
    return {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}


def test_continue_spectra_from_accepted_structure_without_repeating_relax(parent, tmp_path):
    root, output = parent
    before = hashes(root)
    inspected = continuation.inspect_continuation(root, "01_scf")
    destination = tmp_path / "continued"
    state = continuation.prepare_continuation(root, 0, destination, config(), ["bands", "dos"],
                                             parameters={"encut": 400},
                                             expected_context_sha256=inspected["context_sha256"])
    assert [stage["name"] for stage in state["stages"]] == ["scf", "bands", "dos"]
    assert state["stages"][0]["parameters"]["encut"] == 400
    assert (destination / "source/input/POSCAR").read_bytes() == (output / "final_structure.vasp").read_bytes()
    assert not list(destination.rglob("CHGCAR")) and not list(destination.rglob("WAVECAR"))
    receipt = json.loads((destination / "parent.json").read_text())
    assert receipt["source_structure_sha256"] == sha(output / "final_structure.vasp")
    assert receipt["electronic_state_reused"] is False
    assert str(root) not in json.dumps(receipt)
    assert hashes(root) == before


def test_stage_overrides_are_forwarded(parent, tmp_path):
    root, _ = parent
    destination = tmp_path / "mixed"
    state = continuation.prepare_continuation(root, 0, destination, config(), ["scf", "bands"],
                                             stage_parameters=[{"encut": 420}, {"encut": 420}])
    assert all(stage["parameters"]["encut"] == 420 for stage in state["stages"])
    assert json.loads((destination / "plan.json").read_text())["stage_parameters"] == [{"encut": 420}, {"encut": 420}]


def test_changed_parent_after_planning_is_rejected(parent, tmp_path):
    root, _ = parent
    selected = continuation.inspect_continuation(root, 0)
    proposal = json.loads((root / "proposal.json").read_text())
    proposal["goal"] = "A revised goal"
    write(root / "proposal.json", proposal)
    destination = tmp_path / "changed"
    with pytest.raises(ValueError, match="after planning"):
        continuation.prepare_continuation(root, 0, destination, config(), ["scf"],
                                          expected_context_sha256=selected["context_sha256"])
    assert not destination.exists()


def test_changed_parent_during_preparation_leaves_no_run(parent, tmp_path, monkeypatch):
    root, output = parent
    original = workflow.prepare_plan
    def changed(*args, **kwargs):
        state = original(*args, **kwargs)
        (output / "OUTCAR").write_text("changed during preparation")
        return state
    monkeypatch.setattr(workflow, "prepare_plan", changed)
    destination = tmp_path / "changed"
    with pytest.raises(ValueError, match="during preparation"):
        continuation.prepare_continuation(root, 0, destination, config(), ["scf"])
    assert not destination.exists()


@pytest.mark.parametrize("target", ["final_structure.vasp", "vasprun.xml"])
def test_changed_structure_or_result_cannot_be_continued(parent, target):
    root, output = parent
    (output / target).write_text("changed")
    with pytest.raises(ValueError, match="accepted results|verified final POSCAR"):
        continuation.inspect_continuation(root, 0)


@pytest.mark.parametrize("stage", [-1, 1, "missing", True])
def test_stage_selection_must_exist(parent, stage):
    with pytest.raises(ValueError, match="existing stage"):
        continuation.inspect_continuation(parent[0], stage)


def test_source_tree_cannot_be_destination(parent):
    root, _ = parent
    with pytest.raises(ValueError, match="separate run"):
        continuation.prepare_continuation(root, 0, root / "continued", config(), ["scf"])


def test_soc_continuation_preserves_cartesian_frame_and_sorted_spin_seeds(tmp_path, monkeypatch):
    # A rotated cell and interleaved species expose both CIF-frame and site-sort errors.
    lattice = [[0, 4, 0], [-4, 0, 0], [0, 0, 4]]
    structure = Structure(lattice, ["Ni", "O", "Ni", "O"],
                          [[0, 0, 0], [.25, .25, .25], [.5, .5, .5], [.75, .75, .75]])
    source = tmp_path / "POSCAR"
    Poscar(structure).write_file(source)
    params = {"spin": "noncollinear", "soc": True, "saxis": [1, 0, 0],
              "magmom": [[1, 2, 3], [0, 0, 0], [-1, -2, -3], [0, 0, 0]]}
    root = tmp_path / "soc"
    state = workflow.prepare_plan(source, root, config(), ["scf"], params)
    stage = state["stages"][0]
    output, inputs = root / "01_scf/outputs", root / "01_scf/inputs"
    output.mkdir()
    for name in ("POSCAR", "INCAR", "KPOINTS"):
        (output / name).write_bytes((inputs / name).read_bytes())
    (output / "vasprun.xml").write_text("accepted SOC fixture")
    (output / "OUTCAR").write_text("accepted moments fixture")
    (output / "input_hashes.sha256").write_text("\n".join(f"{sha(inputs/name)}  {name}" for name in ("INCAR", "KPOINTS", "POSCAR")))
    final = Structure.from_file(inputs / "POSCAR")
    final.to(filename=str(output / "final_structure.cif"))
    Poscar(final).write_file(output / "final_structure.vasp")
    write(output / "artifact_manifest.json", {path.name: {"sha256": sha(path), "size": path.stat().st_size} for path in output.iterdir()})
    result = {"success": True, "converged_electronic": True, "input_identity_verified": True,
              "scheduler_state": "COMPLETED", "vasprun_sha256": sha(output / "vasprun.xml"),
              "final_structure_sha256": sha(output / "final_structure.cif"),
              "final_structure_poscar_sha256": sha(output / "final_structure.vasp")}
    write(output / "result.json", result)
    stage.update(status="succeeded", scheduler_state="COMPLETED", result=result)
    state["status"] = "succeeded"
    write(root / "run.json", state)
    monkeypatch.setattr(explanation, "analyze_outputs", lambda *args: result.copy())
    before = hashes(root)
    inspected = continuation.inspect_continuation(root, 0)
    assert inspected["parameters"]["magmom"] == stage["metadata"]["method"]["magmom"]
    assert inspected["parameters"]["magmom"] != params["magmom"]
    new = continuation.prepare_continuation(root, 0, tmp_path / "soc-next", config(), ["scf"])
    continued = Structure.from_file(tmp_path / "soc-next/01_scf/inputs/POSCAR")
    assert np.allclose(continued.lattice.matrix, final.lattice.matrix)
    assert new["stages"][0]["metadata"]["method"] == stage["metadata"]["method"]
    assert hashes(root) == before


def task_plan(inspected, variants=False):
    from test_task_agent import response, stage
    stages = [stage("scf", encut=400)]
    return {**response(stages), "kind": "task", "goal": "Make a 2x1x1 cell, then SCF without SOC.",
            "source_sha256": inspected["provenance"]["source_structure_sha256"],
            "operations": [{"type": "supercell", "matrix": [2, 1, 1]}],
            "stages": stages,
            "variants": [{"label": "base", "strain": None, "stages": stages},
                         {"label": "expanded", "strain": [.01, .01, .01], "stages": stages}] if variants else []}


@pytest.mark.parametrize("variants", [False, True])
def test_task_continuation_supports_edits_and_comparisons(parent, tmp_path, variants):
    root, _ = parent
    inspected = continuation.inspect_continuation(root, 0)
    destination = tmp_path / "task-next"
    before = hashes(root)
    continuation.prepare_task_continuation(root, 0, destination, config(), task_plan(inspected, variants),
                                          inspected["context_sha256"])
    assert (destination / "parent.json").is_file()
    assert len(Structure.from_file(destination / "structure_edit/POSCAR")) == 4
    assert len(list(destination.rglob("run.json"))) == (2 if variants else 1)
    assert hashes(root) == before


def test_task_continuation_rejects_plan_for_another_structure(parent, tmp_path):
    root, _ = parent
    inspected = continuation.inspect_continuation(root, 0)
    plan = task_plan(inspected)
    plan["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the task plan"):
        continuation.prepare_task_continuation(root, 0, tmp_path / "bad-task", config(), plan,
                                              inspected["context_sha256"])
    assert not (tmp_path / "bad-task").exists()


def test_task_continuation_rechecks_source_after_edits(parent, tmp_path, monkeypatch):
    from vasp_slurm_agent import task_agent
    root, output = parent
    inspected = continuation.inspect_continuation(root, 0)
    original = task_agent.prepare_task
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (output / "OUTCAR").write_text("changed")
        return result
    monkeypatch.setattr(task_agent, "prepare_task", mutate)
    with pytest.raises(ValueError, match="during preparation"):
        continuation.prepare_task_continuation(root, 0, tmp_path / "bad-task", config(), task_plan(inspected),
                                              inspected["context_sha256"])
    assert not (tmp_path / "bad-task").exists()
