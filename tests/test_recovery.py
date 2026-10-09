"""Repair contracts use simulated solver output; no SSH, model or VASP calls."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Poscar

from vasp_slurm_agent import agent, explanation, recovery, vasp, workflow
from vasp_slurm_agent.agent import AgentError, ModelSettings
from vasp_slurm_agent.config import ClusterConfig


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "Si.cif"
SETTINGS = ModelSettings("responses", "test-model", api_key="private-test-key")


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    config = ClusterConfig(host="private.example", user="private-user", remote_root="/private/runs",
                           vasp_command="srun vasp_std", vasp_ncl_command="srun vasp_ncl", potcar_root="/licensed/pbe", partition="cpu")
    calls = []
    structures = {}

    def parser(path, **kwargs):
        directory = Path(path).parent
        metadata = json.loads((directory / "metadata.json").read_text())
        params = metadata["parameters"]
        incar = {"EDIFFG": params["ediffg"], "NSW": params["nsw"] if metadata["task"] == "relax" else 0,
                 "IBRION": 2 if metadata["task"] == "relax" else -1, "ICHARG": 2,
                 **metadata["method_incar_expected"]}
        kind = Path(path).read_text()
        if kind == "unknown":
            raise ValueError("truncated XML")
        structure = structures[metadata["formula"]].get_sorted_structure()
        return SimpleNamespace(vasp_version="test-stub", initial_structure=structure, final_structure=structure,
                               final_energy=-10.0, efermi=0.0, converged_electronic=kind != "electronic",
                               converged_ionic=kind != "ionic", incar=incar,
                               ionic_steps=[{"forces": [[0, 0, 0] for _ in structure], "e_0_energy": -10.0}])

    def model(payload, schema, settings, **kwargs):
        request = json.loads(payload)
        calls.append(request)
        return json.dumps({"action": request["allowed_actions"][0], "diagnosis": "More iterations may allow convergence.", "questions": []})

    monkeypatch.setattr(vasp, "Vasprun", parser)
    monkeypatch.setattr(agent, "_request_structured", model)
    monkeypatch.setattr(workflow, "SSHTransport", lambda *args: pytest.fail("No remote access allowed"))
    explanation._current_result.cache_clear()

    def collect(root, kind="electronic", scheduler="COMPLETED"):
        state = workflow.read_state(root)
        index = state["current_stage"]
        stage = state["stages"][index]
        inputs = root / stage["folder"] / "inputs"
        output = root / stage["folder"] / "outputs"
        output.mkdir(exist_ok=True)
        for name in ("POSCAR", "INCAR", "KPOINTS", "metadata.json"):
            (output / name).write_bytes((inputs / name).read_bytes())
        (output / "vasprun.xml").write_text(kind)
        (output / "OUTCAR").write_text("test solver log")
        (output / "execution.json").write_text(json.dumps({"job_id": "456", "started_at": "2026-10-09"}))
        (output / "input_hashes.sha256").write_text("\n".join(f"{recovery._sha(inputs / name)}  {name}" for name in ("INCAR", "KPOINTS", "POSCAR")))
        manifest = {name: {"sha256": recovery._sha(output / name)} for name in workflow.FILES if (output / name).is_file()}
        workflow._write(output / "artifact_manifest.json", manifest)
        result = vasp.analyze_outputs(output, stage["name"], inputs / "POSCAR")
        result.update(input_identity_verified=True, scheduler_state=scheduler)
        if kind == "accepted":
            assert result["success"]
            result["final_structure_sha256"] = recovery._sha(output / "final_structure.cif")
        else:
            result["success"] = False
        workflow._write(output / "result.json", result)
        stage.update(status="succeeded" if kind == "accepted" else "needs_attention", staged=True, job_id="456", scheduler_state=scheduler, result=result)
        state.update(status="needs_attention", last_error=None, remote_failures=0)
        if kind == "accepted" and index + 1 < len(state["stages"]):
            state.update(current_stage=index + 1, status="planned")
            workflow._materialize_stage(root, state, ClusterConfig.load(root / "config.json"), state["stages"][index + 1])
        workflow._write(root / "run.json", state)
        return root

    def create(task="scf", kind="electronic", scheduler="COMPLETED", parameters=None, source=EXAMPLE):
        root = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
        structure = Structure.from_file(source)
        structures[structure.composition.reduced_formula] = structure
        workflow.prepare_plan(source, root, config, task if isinstance(task, list) else [task], parameters or {})
        return collect(root, kind, scheduler)

    return create, collect, calls


def test_convergence_repair_changes_only_nelm_and_preserves_parent(failed_run, tmp_path):
    create, _, calls = failed_run
    root = create(parameters={"spin": "collinear", "magmom": [1, -1], "functional": "HSE06"})
    before = {str(path.relative_to(root)): recovery._sha(path) for path in root.rglob("*") if path.is_file()}
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["status"] == "ready"
    assert proposal["changes"] == [{"scope": "parameters", "key": "nelm", "before": 120, "after": 240}]
    child = tmp_path / "repair"
    state = recovery.prepare_repair(root, child, proposal)
    assert state["status"] == "planned" and state["stages"][0]["job_id"] is None
    assert state["repair"]["attempt"] == 1 and state["repair"]["parent_run_id"] == workflow.read_state(root)["run_id"]
    assert state["remote_root"] != workflow.read_state(root)["remote_root"]
    old = Incar.from_file(root / "01_scf/inputs/INCAR")
    new = Incar.from_file(child / "01_scf/inputs/INCAR")
    assert {key for key in old if old[key] != new[key]} == {"NELM"}
    assert before == {str(path.relative_to(root)): recovery._sha(path) for path in root.rglob("*") if path.is_file()}
    assert calls and calls[0]["scientific_context"]["evidence"]
    payload = json.dumps(calls[0])
    assert "private.example" not in payload and "/private/runs" not in payload and "private-test-key" not in payload


def test_ionic_repair_keeps_force_tolerance(failed_run, tmp_path):
    create, _, _ = failed_run
    root = create("relax", "ionic")
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["action"] == "increase_nsw"
    state = recovery.prepare_repair(root, tmp_path / "repair", proposal)
    assert state["parameters"]["nsw"] == 200 and state["parameters"]["ediffg"] == -0.03


def test_nonmagnetic_soc_and_u_survive_repair(failed_run, tmp_path):
    root = failed_run[0](parameters={"soc": True, "spin": "none", "hubbard_u": {"Si": {"l": 1, "u": 2, "j": 0}}})
    state = recovery.prepare_repair(root, tmp_path / "repair", recovery.draft_repair(root, SETTINGS))
    assert state["parameters"]["spin"] == "none" and state["parameters"]["magmom"] is None
    assert state["stages"][0]["metadata"]["method"] == workflow.read_state(root)["stages"][0]["metadata"]["method"]
    assert str(root) not in json.dumps(state["repair"])


def test_small_iteration_limits_increase_to_default(failed_run):
    root = failed_run[0](parameters={"nelm": 1})
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["changes"][0]["after"] == 120


def test_sorted_moments_match_repair_and_experience_geometry(failed_run, tmp_path, monkeypatch):
    from vasp_slurm_agent import experience
    structure = Structure([[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]], ["O", "Mg"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    source = tmp_path / "POSCAR.mgo"
    Poscar(structure, sort_structure=False).write_file(source)
    root = failed_run[0](parameters={"spin": "collinear", "magmom": [-0.2, 2.0]}, source=source)
    queries = []
    def retrieve(runs_root, summary, **kwargs):
        queries.append((summary, kwargs["parameters"]))
        return []
    monkeypatch.setattr(experience, "retrieve_experience", retrieve)
    proposal = recovery.draft_repair(root, SETTINGS)
    assert [site["element"] for site in queries[0][0]["sites"]] == ["Mg", "O"]
    assert queries[0][1]["magmom"] == [2.0, -0.2]
    assert failed_run[2][0]["parameters"]["magmom"] == [2.0, -0.2]
    child = tmp_path / "repair"
    state = recovery.prepare_repair(root, child, proposal)
    assert state["stages"][0]["metadata"]["method"]["magmom"] == [2.0, -0.2]


def test_timeout_changes_only_configuration(failed_run, tmp_path):
    create, _, _ = failed_run
    root = create(kind="unknown", scheduler="TIMEOUT")
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["action"] == "increase_walltime"
    child = tmp_path / "repair"
    recovery.prepare_repair(root, child, proposal)
    assert ClusterConfig.load(child / "config.json").walltime == "01:00:00"
    for name in ("POSCAR", "KPOINTS", "INCAR"):
        assert (root / "01_scf/inputs" / name).read_bytes() == (child / "01_scf/inputs" / name).read_bytes()


@pytest.mark.parametrize("target", ["01_scf/outputs/OUTCAR", "01_scf/inputs/INCAR", "plan.json", "config.json"])
def test_stale_evidence_refuses_preparation(failed_run, tmp_path, target):
    root = failed_run[0]()
    proposal = recovery.draft_repair(root, SETTINGS)
    path = root / target
    path.write_text(path.read_text() + " ")
    with pytest.raises(AgentError):
        recovery.prepare_repair(root, tmp_path / "repair", proposal)
    assert not (tmp_path / "repair").exists()


def test_matching_mutated_metadata_cannot_override_frozen_plan(failed_run):
    root = failed_run[0]()
    state = workflow.read_state(root)
    stage = state["stages"][0]
    inputs, output = root / "01_scf/inputs", root / "01_scf/outputs"
    incar = Incar.from_file(inputs / "INCAR")
    incar["NELM"] = 10
    incar.write_file(inputs / "INCAR")
    (output / "INCAR").write_bytes((inputs / "INCAR").read_bytes())
    metadata = stage["metadata"]
    metadata["parameters"]["nelm"] = 10
    metadata["input_sha256"]["INCAR"] = recovery._sha(inputs / "INCAR")
    workflow._write(inputs / "metadata.json", metadata)
    workflow._write(output / "metadata.json", metadata)
    (output / "input_hashes.sha256").write_text("\n".join(f"{recovery._sha(inputs / name)}  {name}" for name in ("INCAR", "KPOINTS", "POSCAR")))
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name in ("INCAR", "input_hashes.sha256"):
        manifest[name]["sha256"] = recovery._sha(output / name)
    workflow._write(output / "artifact_manifest.json", manifest)
    workflow._write(root / "run.json", state)
    with pytest.raises(AgentError, match="frozen plan"):
        recovery.draft_repair(root, SETTINGS)


@pytest.mark.parametrize("changes", [{"last_error": "SSH outcome uncertain"}, {"remote_failures": 1}, {"status": "succeeded"}, {"cancel_requested": True}])
def test_uncertain_or_completed_run_is_not_repaired(failed_run, changes):
    root = failed_run[0]()
    state = workflow.read_state(root)
    state.update(changes)
    workflow._write(root / "run.json", state)
    with pytest.raises(AgentError):
        recovery.draft_repair(root, SETTINGS)
    assert not failed_run[2]


@pytest.mark.parametrize("change", ["missing_job", "active", "missing_receipt", "wrong_receipt"])
def test_receipt_and_terminal_status_are_required(failed_run, change):
    root = failed_run[0]()
    state = workflow.read_state(root)
    if change == "missing_job":
        state["stages"][0]["job_id"] = None
    elif change == "active":
        state["stages"][0]["scheduler_state"] = "RUNNING"
    elif change == "missing_receipt":
        (root / "01_scf/outputs/execution.json").unlink()
    else:
        (root / "01_scf/outputs/execution.json").write_text('{"job_id":"999"}')
    workflow._write(root / "run.json", state)
    with pytest.raises(AgentError):
        recovery.draft_repair(root, SETTINGS)


def test_unknown_failure_and_reached_limit_need_input(failed_run):
    for root in (failed_run[0](kind="unknown"), failed_run[0](parameters={"nelm": 480})):
        proposal = recovery.draft_repair(root, SETTINGS)
        assert proposal["status"] == "needs_input" and not proposal["changes"]
    assert not failed_run[2]


def test_inherited_attempt_cap_cannot_be_raised(failed_run, tmp_path):
    create, collect, _ = failed_run
    root = create()
    proposal = recovery.draft_repair(root, SETTINGS, max_attempts=1)
    child = tmp_path / "repair"
    recovery.prepare_repair(root, child, proposal)
    collect(child)
    second = recovery.draft_repair(child, SETTINGS, max_attempts=2)
    assert second["status"] == "needs_input" and second["max_attempts"] == 1
    with pytest.raises(AgentError, match="limit"):
        recovery.draft_repair(child, SETTINGS, max_attempts=3)


def test_two_attempt_chain_then_stops(failed_run, tmp_path):
    create, collect, _ = failed_run
    root = create()
    first = tmp_path / "repair-1"
    recovery.prepare_repair(root, first, recovery.draft_repair(root, SETTINGS))
    collect(first)
    second = tmp_path / "repair-2"
    state = recovery.prepare_repair(first, second, recovery.draft_repair(first, SETTINGS))
    assert state["repair"]["attempt"] == 2
    assert state["repair"]["root_run_id"] == workflow.read_state(root)["run_id"]
    collect(second)
    assert recovery.draft_repair(second, SETTINGS)["status"] == "needs_input"


def test_duplicate_prepare_and_tampered_action_refused(failed_run, tmp_path):
    root = failed_run[0]()
    proposal = recovery.draft_repair(root, SETTINGS)
    invalid = deepcopy(proposal)
    invalid["changes"][0].update(key="ediff", after=0.1)
    with pytest.raises(AgentError):
        recovery.prepare_repair(root, tmp_path / "invalid", invalid)
    recovery.prepare_repair(root, tmp_path / "first", proposal)
    with pytest.raises(AgentError, match="already prepared"):
        recovery.prepare_repair(root, tmp_path / "second", proposal)


def test_repair_continues_after_accepted_relaxation(failed_run, tmp_path):
    create, collect, _ = failed_run
    root = create(["relax", "scf", "dos"], "accepted")
    collect(root)
    child = tmp_path / "repair"
    state = recovery.prepare_repair(root, child, recovery.draft_repair(root, SETTINGS))
    assert state["tasks"] == ["scf", "dos"]
    assert state["current_stage"] == 0 and len(state["stages"]) == 2
    assert state["repair"]["accepted_predecessors"][0]["stage"] == "01_relax"
    assert not (child / "01_relax").exists()
    assert state["stages"][0]["parameters"]["nelm"] == 240
    assert state["stages"][1]["parameters"].get("nelm", 120) == 120
    assert (child / "01_scf/inputs/POSCAR").read_bytes() == (root / "02_scf/inputs/POSCAR").read_bytes()


def test_pbe_spectrum_needing_old_charge_density_needs_input(failed_run):
    root = failed_run[0]("bands", "accepted")
    failed_run[1](root, "unknown")
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["status"] == "needs_input" and "charge density" in proposal["diagnosis"]


def test_model_cannot_add_commands_or_choose_unsupported_action(failed_run, monkeypatch):
    root = failed_run[0]()
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps({"action": "change_functional", "diagnosis": "Try PBE.", "questions": []}))
    with pytest.raises(AgentError):
        recovery.draft_repair(root, SETTINGS)
