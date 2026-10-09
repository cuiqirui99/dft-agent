"""Result explanations use saved evidence, never a solver or SSH connection."""

from copy import deepcopy
import hashlib
from importlib.resources import files
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pymatgen.core import Structure

from vasp_slurm_agent import agent, explanation, workflow
from vasp_slurm_agent.agent import AgentError, ModelSettings
from vasp_slurm_agent.config import ClusterConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def accepted_run(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    root = tmp_path / "run"
    config = ClusterConfig(host="private-host.example", user="private-user", remote_root="/private/remote",
                           vasp_command="srun vasp_std", potcar_root="/private/licensed", partition="cpu")
    state = workflow.prepare_run(source, root, config, "scf", {})
    stage = state["stages"][0]
    output = root / stage["folder"] / "outputs"
    output.mkdir()
    inputs = root / stage["folder"] / "inputs"
    for name in ("POSCAR", "INCAR", "KPOINTS"):
        (output / name).write_bytes((inputs / name).read_bytes())
    (output / "vasprun.xml").write_text("verified solver fixture")
    (output / "OUTCAR").write_text("verified moments fixture")
    (output / "input_hashes.sha256").write_text("\n".join(f"{sha(inputs/name)}  {name}" for name in ("INCAR", "KPOINTS", "POSCAR")))
    (output / "final_structure.cif").write_bytes(source.read_bytes())
    manifest = {path.name: {"sha256": sha(path), "size": path.stat().st_size} for path in output.iterdir()}
    write(output / "artifact_manifest.json", manifest)
    result = {"success": True, "task": "scf", "converged_electronic": True, "converged_ionic": True,
              "input_identity_verified": True, "scheduler_state": "COMPLETED", "final_energy_ev": -10.0,
              "final_energy_ev_per_atom": -5.0, "final_max_force_ev_angstrom": 0.001, "fermi_energy_ev": 5.0,
              "vasprun_sha256": sha(output / "vasprun.xml"), "final_structure_sha256": sha(output / "final_structure.cif"),
              "method_fingerprint": stage["metadata"]["method_fingerprint"], "method": stage["metadata"]["method"],
              "magnetization": {"total_moment": 0.0, "site_moments": [1.69, -1.69]},
              "scientific_accuracy_validated": False, "magnetic_ground_state_validated": False}
    write(output / "result.json", result)
    stage.update(status="succeeded", scheduler_state="COMPLETED", result=result)
    state["status"] = "succeeded"
    write(root / "run.json", state)
    plan = json.loads((root / "plan.json").read_text())
    write(root / "proposal.json", {"goal": "SCF only; no relaxation.", "source_sha256": sha(source),
                                   "tasks": plan["tasks"], "parameters": plan["parameters"],
                                   "dialogue": [{"role": "user", "content": "Keep SOC off."}]})
    def parser(temp, task, expected):
        assert Path(temp).resolve() != output.resolve()
        assert (Path(temp) / "vasprun.xml").read_text() == "verified solver fixture"
        return deepcopy(result)
    parsed = Mock(side_effect=parser)
    monkeypatch.setattr(explanation, "analyze_outputs", parsed)
    monkeypatch.setattr(workflow, "SSHTransport", Mock(side_effect=AssertionError("No remote connection allowed")))
    return root, output, parsed


def model_reply(answer="The SCF output passed the result checks.", evidence=None, **changes):
    return {"answer": answer, "evidence": evidence or ["stage_1.accepted"], "limits": [], "next_steps": [], **changes}


def stub(monkeypatch, reply):
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps(reply))


def test_context_whitelist_preserves_goal_without_private_config(accepted_run):
    root, output, _ = accepted_run
    before = {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}
    context = explanation.load_run_context(root)
    assert context["status"] == "succeeded"
    assert context["goal"] == "SCF only; no relaxation."
    assert context["dialogue"][0]["content"] == "Keep SOC off."
    assert context["facts"]["stage_1.energy"]["value"] == -10.0
    assert context["facts"]["stage_1.soc"]["value"] is False
    assert context["facts"]["stage_1.site_moments"]["value"] == [1.69, -1.69]
    payload = json.dumps(context)
    for private in ("private-host", "private-user", "/private/remote", "/private/licensed", str(root)):
        assert private not in payload
    assert before == {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("tamper", [None, "plan_sha256", "source_sha256", "source_path", "tasks", "parameters", "stage_parameters"])
def test_task_goal_is_bound_to_its_executed_plan(accepted_run, tamper):
    root, _, _ = accepted_run
    plan = json.loads((root / "plan.json").read_text())
    execution = {"plan_sha256": sha(root / "plan.json"), "source_sha256": next(iter(plan["sources"].values())),
                 "source_path": next(iter(plan["sources"])), "stage_parameters": plan.get("stage_parameters"),
                 "parameters": plan["parameters"], "tasks": plan["tasks"]}
    if tamper:
        execution[tamper] = "changed"
    write(root / "proposal.json", {"kind": "task", "goal": "Edit then calculate.", "source_sha256": "original-upload",
                                   "parameters": {"encut": 520}, "operations": [{"type": "supercell", "matrix": [2, 1, 1]}],
                                   "execution": execution, "dialogue": [{"role": "user", "content": "Keep SOC off."}]})
    context = explanation.load_run_context(root)
    assert bool(context["goal"]) is (tamper is None)
    assert bool(context["dialogue"]) is (tamper is None)


def test_parser_cache_uses_artifact_identity(accepted_run):
    root, output, parser = accepted_run
    first = explanation.load_run_context(root)
    explanation.load_run_context(root)
    assert parser.call_count == 1
    (output / "OUTCAR").write_text("new verified moments")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    manifest["OUTCAR"]["sha256"] = sha(output / "OUTCAR")
    write(output / "artifact_manifest.json", manifest)
    second = explanation.load_run_context(root)
    assert parser.call_count == 2 and first["context_sha256"] != second["context_sha256"]


@pytest.mark.parametrize("target", ["vasprun.xml", "OUTCAR", "final_structure.cif", "INCAR", "result.json"])
def test_changed_artifact_is_unavailable(accepted_run, target):
    root, output, _ = accepted_run
    (output / target).write_text("changed content")
    context = explanation.load_run_context(root)
    assert context["status"] == "needs_attention"
    assert context["facts"]["stage_1.accepted"]["value"] is False
    assert context["facts"]["stage_1.status"]["value"] == "needs_attention"
    assert "stage_1.energy" not in context["facts"]


def test_changed_source_invalidates_acceptance(accepted_run):
    root, _, _ = accepted_run
    plan = json.loads((root / "plan.json").read_text())
    (root / next(iter(plan["sources"]))).write_text("changed structure")
    context = explanation.load_run_context(root)
    assert not context["facts"]["stage_1.accepted"]["value"]


def test_result_edit_cannot_override_raw_solver_values(accepted_run):
    root, output, _ = accepted_run
    state = json.loads((root / "run.json").read_text())
    state["stages"][0]["result"]["final_energy_ev_per_atom"] = -999.123
    write(output / "result.json", state["stages"][0]["result"])
    write(root / "run.json", state)
    context = explanation.load_run_context(root)
    assert not context["facts"]["stage_1.accepted"]["value"]
    assert "stage_1.energy_per_atom" not in context["facts"]


def test_current_science_rejection_overrides_historical_success(accepted_run):
    root, _, parser = accepted_run
    parser.side_effect = lambda *args: {"success": False, "hybrid_kpoint_consistency": {"passed": False}}
    context = explanation.load_run_context(root)
    assert context["status"] == "needs_attention"
    assert "hybrid-band" in context["facts"]["stage_1.availability"]["value"]


def test_failed_scheduler_is_never_an_accepted_result(accepted_run):
    root, output, _ = accepted_run
    state = json.loads((root / "run.json").read_text())
    state["status"] = "failed"
    state["stages"][0].update(status="failed", scheduler_state="TIMEOUT")
    write(root / "run.json", state)
    context = explanation.load_run_context(root)
    assert context["status"] == "failed"
    assert not context["facts"]["stage_1.accepted"]["value"]
    assert "stage_1.energy" not in context["facts"]


@pytest.mark.parametrize("status", ["planned", "submitting", "queued", "running", "collecting"])
def test_pending_workflow_status_is_preserved(accepted_run, status):
    root, _, _ = accepted_run
    state = json.loads((root / "run.json").read_text())
    state["status"] = state["stages"][0]["status"] = status
    write(root / "run.json", state)
    context = explanation.load_run_context(root)
    assert context["status"] == status
    assert not context["facts"]["stage_1.accepted"]["value"]


def test_mismatched_proposal_does_not_supply_user_goal(accepted_run):
    root, _, _ = accepted_run
    proposal = json.loads((root / "proposal.json").read_text())
    proposal["source_sha256"] = "0" * 64
    write(root / "proposal.json", proposal)
    assert explanation.load_run_context(root)["goal"] == ""


def test_explanation_history_does_not_invalidate_context(accepted_run):
    root, _, _ = accepted_run
    before = explanation.load_run_context(root)["context_sha256"]
    write(root / "explanations.json", {"answer": "saved separately"})
    assert explanation.load_run_context(root)["context_sha256"] == before
    proposal = json.loads((root / "proposal.json").read_text())
    proposal["dialogue"].append({"role": "user", "content": "Explain the energy."})
    write(root / "proposal.json", proposal)
    assert explanation.load_run_context(root)["context_sha256"] != before


def test_explanation_receives_verified_facts_and_returns_provenance(accepted_run, monkeypatch):
    root, _, _ = accepted_run
    def request(payload, schema, settings, *, instructions, name):
        data = json.loads(payload)
        assert data["context"]["facts"]["stage_1.energy"]["value"] == -10.0
        assert data["question"] == "Explain the energy."
        assert "Never run tools" in instructions and name == "dft_explanation"
        return json.dumps(model_reply("The SCF energy is −10.0 eV.", ["stage_1.energy"]))
    monkeypatch.setattr(agent, "_request_structured", request)
    result = explanation.explain_run(root, ModelSettings(model="test"), "Explain the energy.")
    assert result["answer"] == "The SCF energy is −10.0 eV."
    assert result["provenance"]["model"] == "test"
    assert result["context_sha256"] == explanation.load_run_context(root)["context_sha256"]
    assert result["limits"] == []


def test_exact_fact_placeholder_is_optional(accepted_run, monkeypatch):
    stub(monkeypatch, model_reply("The energy is {{stage_1.energy}}.", ["stage_1.energy"]))
    result = explanation.explain_run(accepted_run[0], ModelSettings(model="test"))
    assert result["answer"] == "The energy is -10.0 eV."


def test_available_artifacts_can_be_cited(accepted_run, monkeypatch):
    stub(monkeypatch, model_reply("The final structure is available.", ["stage_1.final_structure.cif"]))
    result = explanation.explain_run(accepted_run[0], ModelSettings(model="test"))
    assert result["evidence"] == ["stage_1.final_structure.cif"]


@pytest.mark.parametrize("reply", [
    model_reply(evidence=["unknown.energy"]),
    model_reply("The energy is -999.123 eV.", ["stage_1.energy"]),
    model_reply("The band gap is {{stage_1.fermi_energy}}, without a convergence study.", ["stage_1.fermi_energy"]),
    model_reply("FM is the magnetic ground state, not NM.", ["stage_1.energy"]),
    model_reply("The ground state is FM.", ["stage_1.energy"]),
])
def test_unverified_claims_are_rejected(accepted_run, monkeypatch, reply):
    stub(monkeypatch, reply)
    with pytest.raises(AgentError, match="could not be verified"):
        explanation.explain_run(accepted_run[0], ModelSettings(model="test"))


def test_missing_gap_answer_and_followup_suggestion_are_allowed(accepted_run, monkeypatch):
    stub(monkeypatch, model_reply("The band gap is unavailable from these facts.", ["stage_1.accepted"],
                                 next_steps=["Extract a band gap from the accepted spectra."]))
    assert explanation.explain_run(accepted_run[0], ModelSettings(model="test"))["next_steps"]


def add_gap(parser, **overrides):
    original = parser.side_effect
    def parse(*args):
        result = original(*args)
        result["band_gap"] = {"available": True, "status": "gapped", "scope": "sampled_kpoints",
                              "gap_ev": 1.25, "direct_gap_ev": 1.8, "vbm_ev": 4.0, "cbm_ev": 5.25,
                              "reason": "Only the retained k-points were checked.", **overrides}
        return result
    parser.side_effect = parse


def test_explanation_can_use_only_verified_sampled_gap(accepted_run, monkeypatch):
    root, _, parser = accepted_run
    add_gap(parser)
    context = explanation.load_run_context(root)
    assert context["facts"]["stage_1.band_gap"]["value"] == 1.25
    assert context["facts"]["stage_1.band_gap_scope"]["value"] == "sampled_kpoints"
    assert not any("not been quantified" in item for item in context["limits"])
    stub(monkeypatch, model_reply("The sampled band gap is {{stage_1.band_gap}}.",
                                 ["stage_1.band_gap", "stage_1.band_gap_scope"]))
    assert "1.25 eV" in explanation.explain_run(root, ModelSettings(model="test"))["answer"]


@pytest.mark.parametrize("answer", ["The band gap is 1.25 eV.",
                                     "The sampled full-zone band gap is 1.25 eV.",
                                     "The sampled optical gap is 1.25 eV."])
def test_gap_claim_cannot_drop_sampling_scope_or_claim_optical_gap(accepted_run, monkeypatch, answer):
    root, _, parser = accepted_run
    add_gap(parser)
    stub(monkeypatch, model_reply(answer, ["stage_1.band_gap"]))
    with pytest.raises(AgentError, match="could not be verified"):
        explanation.explain_run(root, ModelSettings(model="test"))


def test_partial_occupations_do_not_supply_gap_fact(accepted_run):
    root, _, parser = accepted_run
    add_gap(parser, available=False, status="metallic_or_partially_occupied", gap_ev=None,
            direct_gap_ev=None, vbm_ev=None, cbm_ev=None)
    context = explanation.load_run_context(root)
    assert "stage_1.band_gap" not in context["facts"]
    assert context["facts"]["stage_1.band_gap_status"]["value"] == "metallic_or_partially_occupied"


def test_cached_gap_is_not_evidence_without_fresh_extraction(accepted_run):
    root, output, _ = accepted_run
    state = json.loads((root / "run.json").read_text())
    state["stages"][0]["result"]["band_gap"] = {"available": True, "gap_ev": 999, "scope": "sampled_kpoints"}
    write(output / "result.json", state["stages"][0]["result"])
    write(root / "run.json", state)
    context = explanation.load_run_context(root)
    assert context["facts"]["stage_1.accepted"]["value"]
    assert "stage_1.band_gap" not in context["facts"]


def test_a_cited_fermi_energy_cannot_be_substituted_for_a_gap(accepted_run, monkeypatch):
    root, _, parser = accepted_run
    add_gap(parser)
    stub(monkeypatch, model_reply("The sampled band gap is {{stage_1.fermi_energy}}.",
                                 ["stage_1.fermi_energy", "stage_1.band_gap"]))
    with pytest.raises(AgentError, match="could not be verified"):
        explanation.explain_run(root, ModelSettings(model="test"))


def test_context_changed_during_model_request_is_rejected(accepted_run, monkeypatch):
    root, output, _ = accepted_run
    def request(*args, **kwargs):
        (output / "OUTCAR").write_text("changed during request")
        return json.dumps(model_reply())
    monkeypatch.setattr(agent, "_request_structured", request)
    with pytest.raises(AgentError, match="results changed"):
        explanation.explain_run(root, ModelSettings(model="test"))


def test_credentials_are_removed_before_json_serialization(accepted_run, monkeypatch):
    key = 'private-"key'
    def request(payload, *args, **kwargs):
        assert key not in json.loads(payload)["question"]
        return json.dumps(model_reply(answer=key))
    monkeypatch.setattr(agent, "_request_structured", request)
    with pytest.raises(AgentError) as caught:
        explanation.explain_run(accepted_run[0], ModelSettings(model="test", api_key=key), "Explain " + key)
    assert key not in str(caught.value)
