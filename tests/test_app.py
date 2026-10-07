"""Exercise user actions through Streamlit; all remote execution is stubbed."""

from importlib.resources import files
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import matplotlib.pyplot as plt

from pymatgen.core import Structure
import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import workflow
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent import agent
from test_cli_plan import model_response


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    from vasp_slurm_agent import explanation

    config = ClusterConfig(
        host="cluster.example.invalid", user="researcher", remote_root="/scratch/test",
        vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu",
    )
    config_path = config.save(tmp_path / "cluster.json")
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(config_path))

    def no_remote(*args, **kwargs):
        pytest.fail("UI smoke tests must never contact an SSH server")

    monkeypatch.setattr(workflow, "SSHTransport", no_remote)
    worker = Mock(return_value=43210)
    monkeypatch.setattr(workflow, "start_worker", worker)

    def fixture_context(run_dir):
        state = workflow.read_state(run_dir)
        return {"status": state["status"], "context_sha256": hashlib.sha256(json.dumps(state).encode()).hexdigest(),
                "facts": {f"stage_{index}.accepted": {"value": bool((stage.get("result") or {}).get("success"))}
                          for index, stage in enumerate(state["stages"], 1)}, "goal": "", "limits": []}

    monkeypatch.setattr(explanation, "load_run_context", fixture_context)
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    widget(app.text_input, "Run folder").set_value(str(tmp_path / "runs")).run()
    widget(app.radio, "Mode").set_value("Manual").run()
    assert not app.exception
    return app, worker, tmp_path / "runs"


def silicon_bytes():
    return files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()


@pytest.mark.parametrize("file_format", ["cif", "poscar"])
def test_structure_import_and_review_gate(workbench, file_format):
    app, worker, runs_root = workbench
    assert widget(app.button, "Prepare inputs").disabled
    if file_format == "cif":
        filename, payload = "Si.cif", silicon_bytes()
    else:
        structure = Structure.from_str(silicon_bytes().decode(), fmt="cif")
        filename, payload = "POSCAR", structure.to(fmt="poscar").encode()
    app.file_uploader[0].upload(filename, payload).run()
    assert not app.exception
    assert widget(app.metric, "Formula").value == "Si"
    assert not widget(app.button, "Prepare inputs").disabled
    assert not runs_root.exists()
    worker.assert_not_called()

    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["status"] == "planned"
    assert state["stages"][0]["job_id"] is None
    assert (run_dir / "01_relax" / "inputs" / "POSCAR").is_file()
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()

    widget(app.checkbox, "Structure, settings and resources reviewed.").check().run()
    assert not widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()
    widget(app.button, "Submit calculation").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    assert any("43210" in message.value for message in app.success)


def test_invalid_structure_cannot_be_prepared(workbench):
    app, worker, runs_root = workbench
    app.file_uploader[0].upload("POSCAR", b"This is not a crystal structure").run()
    assert not app.exception
    assert any("Cannot read structure" in message.value for message in app.error)
    assert widget(app.button, "Prepare inputs").disabled
    assert not runs_root.exists()
    worker.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False])
def test_completed_result_is_visible_without_resubmission(workbench, accepted):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state["status"] = "succeeded" if accepted else "needs_attention"
    stage = state["stages"][0]
    stage.update(status=state["status"], job_id="12345", result={
        "success": accepted, "reason": "Electronic convergence was not reached." if not accepted else "Accepted test fixture",
        "final_energy_ev": -10.123456, "final_max_force_ev_angstrom": 0.002,
    })
    (run_dir / "run.json").write_text(json.dumps(state))
    output = run_dir / stage["folder"] / "outputs"
    output.mkdir()
    # Rejected output must not appear as an accepted result.
    (output / "final_structure.cif").write_bytes(silicon_bytes())
    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    fig.savefig(output / "result.png")
    plt.close(fig)
    app.run()
    assert not app.exception
    assert not any(button.label == "Submit calculation" for button in app.button)
    if accepted:
        assert widget(app.metric, "Final total energy (eV)").value == "-10.123456"
        assert widget(app.download_button, "Download structure (.cif)").proto.url
        assert any("Si" in item.value and "atoms" in item.value for item in app.markdown)
    else:
        assert any("Electronic convergence" in item.value for item in app.warning)
        assert not any(button.label == "Download structure (.cif)" for button in app.download_button)
    captions = [image.caption for block in app.get("image") for image in block.proto.imgs]
    assert ("relax · result.png" in captions) is accepted
    widget(app.button, "Prepare download").click().run()
    assert not app.exception
    assert (run_dir / "results.zip").is_file()
    assert widget(app.download_button, "Download results (.zip)").proto.url
    worker.assert_not_called()


def test_installed_examples_are_readable_crystal_inputs():
    examples = files("vasp_slurm_agent").joinpath("examples")
    expected = {"Si", "C", "Ge", "Al", "Cu", "MgO", "NaCl", "SiC"}
    structures = {sample.name[:-4]: Structure.from_str(sample.read_text(), fmt="cif")
                  for sample in examples.iterdir() if sample.name.endswith(".cif")}
    assert structures.keys() == expected
    for structure in structures.values():
        assert structure.is_ordered and len(structure) > 0 and structure.volume > 0


def test_paused_job_can_reconnect_from_the_result_panel(workbench):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state.update(status="needs_attention", last_error="Connection unavailable", remote_failures=3)
    state["stages"][0].update(status="running", job_id="12345", staged=True)
    (run_dir / "run.json").write_text(json.dumps(state))
    app.run()
    worker.assert_not_called()
    widget(app.button, "Reconnect").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    restored = workflow.read_state(run_dir)
    assert restored["stages"][0]["job_id"] == "12345"
    assert restored["status"] not in workflow.TERMINAL


@pytest.mark.parametrize("task", ["bands", "dos"])
@pytest.mark.parametrize("scf_fermi", [None, 5.25])
def test_spectral_results_use_scf_reference_not_ground_state_metrics(workbench, task, scf_fermi):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Calculation").select(task)
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["task"] == task
    state.update(status="succeeded", current_stage=1)
    state["stages"][0].update(status="succeeded", job_id="12345", result={
        "success": True, "final_energy_ev": -10.0, "final_max_force_ev_angstrom": 0.002,
        "fermi_energy_ev": scf_fermi,
    })
    state["stages"][1].update(status="succeeded", job_id="12346", result={
        "success": True, "final_energy_ev": 123.0, "final_max_force_ev_angstrom": 99.0,
        "fermi_energy_ev": 8.88,
    })
    state_path = run_dir / "run.json"
    state_path.write_text(json.dumps(state))
    original = state_path.read_bytes()
    app.run()
    assert not app.exception
    assert [item.value for item in app.metric if item.label == "Final total energy (eV)"] == ["-10.000000"]
    assert [item.value for item in app.metric if item.label == "Maximum atomic force (eV/Å)"] == ["0.002000"]
    references = [item.value for item in app.metric if item.label == "SCF Fermi energy (eV)"]
    assert references == (["5.250000"] if scf_fermi is not None else [])
    assert any("Fixed-charge spectrum" in item.value for item in app.caption)
    assert state_path.read_bytes() == original
    worker.assert_not_called()


@pytest.mark.parametrize("scheduler_state", [None, "RUNNING", "COMPLETED", "CANCELLED", "FAILED"])
def test_paused_job_can_request_cancel_only_before_confirmed_scheduler_end(workbench, monkeypatch, scheduler_state):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state.update(status="needs_attention", last_error="Connection unavailable", remote_failures=3)
    state["stages"][0].update(status="running", job_id="12345", staged=True, scheduler_state=scheduler_state)
    state_path = run_dir / "run.json"
    state_path.write_text(json.dumps(state))
    original = state_path.read_bytes()
    cancellation = Mock(return_value=state)
    monkeypatch.setattr(workflow, "cancel", cancellation)
    app.run()
    assert not app.exception
    if scheduler_state in {None, "RUNNING"}:
        widget(app.button, "Cancel calculation").click().run()
        assert not app.exception
        cancellation.assert_called_once_with(run_dir)
    else:
        assert not any(button.label == "Cancel calculation" for button in app.button)
        cancellation.assert_not_called()
    assert state_path.read_bytes() == original
    worker.assert_not_called()


def open_agent(app):
    widget(app.radio, "Mode").set_value("Agent").run()
    widget(app.selectbox, "Provider").select("codex").run()
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.text_area, "Goal").set_value("Relax, then SCF.").run()


def test_agent_proposal_requires_review_and_keeps_runtime_plan(workbench, monkeypatch):
    app, worker, runs_root = workbench
    provider = Mock(return_value=json.dumps(model_response()))
    monkeypatch.setattr(agent, "_request_plan", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert not app.exception
    assert app.session_state["agent_proposal"]["plan"]["status"] == "ready"
    assert not runs_root.exists()
    worker.assert_not_called()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    proposal = json.loads((root / "proposal.json").read_text())
    frozen = json.loads((root / "plan.json").read_text())
    assert proposal["status"] == "ready" and frozen["stages"]
    assert "status" not in frozen
    assert state["plan_sha256"] == hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    workflow._check_config(root, state)
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()
    widget(app.checkbox, "Structure, settings and resources reviewed.").check().run()
    worker.assert_not_called()
    widget(app.button, "Submit calculation").click().run()
    worker.assert_called_once_with(root)
    assert provider.call_count == 1


def test_agent_questions_block_preparation(workbench, monkeypatch):
    app, worker, runs_root = workbench
    monkeypatch.setattr(agent, "_request_plan", lambda *args: json.dumps(model_response(["scf"], missing_moments=True)))
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert not app.exception
    proposal = app.session_state["agent_proposal"]["plan"]
    assert proposal["status"] == "needs_input"
    assert any("moment" in message.value for message in app.info)
    assert widget(app.button, "Prepare inputs").disabled
    assert not runs_root.exists()
    worker.assert_not_called()


def test_agent_revision_failure_preserves_feedback_and_blocks_stale_prepare(workbench, monkeypatch):
    app, worker, runs_root = workbench
    provider = Mock(side_effect=[json.dumps(model_response()), agent.AgentError("Provider unavailable"),
                                 json.dumps(model_response(["scf"]))])
    monkeypatch.setattr(agent, "_request_plan", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    widget(app.text_input, "Change the plan").set_value("SCF only. Do not relax.").run()
    assert widget(app.button, "Prepare inputs").disabled
    widget(app.button, "Plan").click().run()
    assert not app.exception
    assert any("Provider unavailable" in message.value for message in app.error)
    assert not any(button.label == "Prepare inputs" for button in app.button)
    assert not runs_root.exists()
    feedback = {"role": "user", "content": "SCF only. Do not relax."}
    assert app.session_state["agent_history"][-1] == feedback
    app.run()
    assert widget(app.text_input, "Change the plan").value == feedback["content"]
    widget(app.button, "Plan").click().run()
    assert not app.exception
    retry_payload = json.loads(provider.call_args.args[0])
    assert retry_payload["history"].count(feedback) == 1
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    proposal = json.loads((root / "proposal.json").read_text())
    assert proposal["goal"] == "Relax, then SCF."
    assert proposal["dialogue"][1] == feedback
    assert len(proposal["dialogue"]) == 3
    assert proposal["tasks"] == ["scf"]
    assert workflow.read_state(root)["tasks"] == ["scf"]
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


def test_agent_clarification_is_saved_with_accepted_plan(workbench, monkeypatch):
    app, worker, _ = workbench
    reply = model_response(["scf"], missing_moments=True)
    resolved = model_response(["scf"], missing_moments=True)
    resolved["parameters"]["magmom"] = [1, -1]
    provider = Mock(side_effect=[json.dumps(reply), json.dumps(resolved)])
    monkeypatch.setattr(agent, "_request_plan", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert widget(app.button, "Prepare inputs").disabled
    widget(app.text_input, "Change the plan").set_value("Use site moments [1, -1].").run()
    widget(app.button, "Plan").click().run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    proposal = json.loads((root / "proposal.json").read_text())
    assert proposal["parameters"]["magmom"] == [1, -1]
    assert json.loads(proposal["dialogue"][0]["content"])["status"] == "needs_input"
    assert proposal["dialogue"][1]["content"] == "Use site moments [1, -1]."
    worker.assert_not_called()


@pytest.mark.parametrize("changed", ["goal", "structure"])
def test_agent_cannot_prepare_stale_proposal(workbench, monkeypatch, changed):
    app, worker, runs_root = workbench
    provider = Mock(return_value=json.dumps(model_response()))
    monkeypatch.setattr(agent, "_request_plan", provider)
    open_agent(app)
    widget(app.button, "Plan").click().run()
    assert not widget(app.button, "Prepare inputs").disabled
    if changed == "goal":
        widget(app.text_area, "Goal").set_value("SCF only.").run()
    else:
        payload = files("vasp_slurm_agent").joinpath("examples", "Al.cif").read_bytes()
        app.file_uploader[0].upload("Al.cif", payload).run()
    assert not app.exception
    assert not any(button.label == "Prepare inputs" for button in app.button)
    assert provider.call_count == 1
    assert not runs_root.exists()
    worker.assert_not_called()


def test_agent_preparation_rechecks_source_hash(workbench, monkeypatch):
    app, worker, runs_root = workbench
    proposal = {"status": "ready", "summary": "SCF", "tasks": ["scf"], "parameters": {},
                "source_sha256": "0" * 64, "questions": []}
    monkeypatch.setattr(agent, "draft_plan", Mock(return_value=proposal))
    open_agent(app)
    widget(app.button, "Plan").click().run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    assert any("structure changed" in message.value for message in app.error)
    assert not runs_root.exists()
    worker.assert_not_called()


@pytest.mark.parametrize("method", ["hybrid", "u", "soc"])
def test_manual_method_controls_reach_frozen_inputs(workbench, method):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Calculation").select("scf")
    if method == "hybrid":
        widget(app.selectbox, "Functional").select("HSE06")
    elif method == "u":
        widget(app.selectbox, "Spin").select("collinear")
        widget(app.text_area, "Site moments (JSON)").set_value("[1, -1]")
        widget(app.text_area, "Hubbard U (JSON)").set_value('{"Si":{"l":1,"u":2,"j":0}}')
    else:
        widget(app.text_input, "SOC / noncollinear command").set_value("srun vasp_ncl")
        widget(app.button, "Save cluster settings").click().run()
        widget(app.selectbox, "Calculation").select("scf")
        widget(app.selectbox, "Spin").select("noncollinear")
        widget(app.checkbox, "SOC").check()
        widget(app.text_area, "Site moments (JSON)").set_value("[[0,0,1],[0,0,-1]]")
        widget(app.text_input, "Spin axis").set_value("1 0 0")
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    metadata = state["stages"][0]["metadata"]
    if method == "hybrid":
        assert metadata["method"]["functional"] == "HSE06"
    elif method == "u":
        assert metadata["method"]["hubbard_u"] == {"Si": {"l": 1, "u": 2., "j": 0.}}
        assert metadata["method"]["magmom"] == [1., -1.]
    else:
        assert metadata["requires_ncl"] and metadata["method"]["soc"]
        assert metadata["method"]["saxis"] == [1., 0., 0.]
    assert state["status"] == "planned"
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


@pytest.mark.parametrize("task", ["bands", "dos"])
def test_hybrid_spectrum_uses_own_fermi_reference(workbench, task):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "Calculation").select(task)
    widget(app.selectbox, "Functional").select("HSE06")
    widget(app.button, "Prepare inputs").click().run()
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    spectrum = state["stages"][0]
    spectrum.update(status="succeeded", job_id="12346", result={"success": True, "energy_reference_ev": 6.75,
                    "fermi_energy_ev": 8.88, "energy_reference_source": "hybrid_scf_mesh"})
    state["stages"].insert(0, {"name": "scf", "folder": "00_scf", "status": "succeeded", "job_id": "12345",
                              "result": {"success": True, "fermi_energy_ev": 1.25, "final_energy_ev": -10.}})
    state.update(status="succeeded", current_stage=1)
    (root / "run.json").write_text(json.dumps(state))
    app.run()
    assert not app.exception
    assert [item.value for item in app.metric if item.label == "Fermi energy (eV)"] == ["6.750000"]
    assert not any(item.label == "SCF Fermi energy (eV)" for item in app.metric)
    worker.assert_not_called()


def explanation_run(workbench, monkeypatch):
    from vasp_slurm_agent import explanation

    app, worker, _ = workbench
    context = {"context_sha256": "a" * 64, "goal": "Relax silicon, then inspect its structure.",
               "status": "succeeded", "outcome": "Relaxation completed.",
               "facts": {"energy": {"label": "Final energy", "value": -10.0, "unit": "eV", "source": "01_relax/outputs/result.json"}},
               "limits": [], "artifacts": [], "final_plan": {"tasks": ["relax"]}, "dialogue": []}
    monkeypatch.setattr(explanation, "load_run_context", lambda _: deepcopy(context))

    def answer(*args, **kwargs):
        return {"answer": "The requested relaxation completed.", "evidence": ["energy"],
                "limits": ["Material-specific accuracy is not established."], "next_steps": ["Download the structure."],
                "context_sha256": context["context_sha256"], "provenance": {"provider": "codex", "model": "test"}}

    explain = Mock(side_effect=answer)
    monkeypatch.setattr(explanation, "explain_run", explain)
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    return app, worker, root, context, explain, answer


def test_explanation_is_explicit_grounded_and_persists_across_sessions(workbench, monkeypatch):
    app, worker, root, context, explain, _ = explanation_run(workbench, monkeypatch)
    original = (root / "run.json").read_bytes()
    assert any(context["goal"] == item.value for item in app.markdown)
    assert any("Relaxation completed." == item.value for item in app.markdown)
    explain.assert_not_called()
    app.run()
    explain.assert_not_called()
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert explain.call_args.kwargs == {"question": "Explain the results.", "history": []}
    assert any("energy · Final energy: -10.0 eV" in item.value for item in app.markdown)
    record = json.loads((root / "explanations.json").read_text())
    assert record["context_sha256"] == context["context_sha256"]
    assert len(record["exchanges"]) == 1
    app.run()
    assert explain.call_count == 1
    widget(app.text_input, "Ask about this run").set_value("Where is the structure?").run()
    widget(app.button, "Ask").click().run()
    assert not app.exception
    assert explain.call_args.kwargs["history"] == record["history"]
    assert explain.call_args.kwargs["question"] == "Where is the structure?"
    assert len(json.loads((root / "explanations.json").read_text())["exchanges"]) == 2
    reopened = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    widget(reopened.text_input, "Or enter a run folder").set_value(str(root)).run()
    widget(reopened.button, "Open run").click().run()
    assert not reopened.exception
    assert any("The requested relaxation completed." == item.value for item in reopened.markdown)
    assert explain.call_count == 2
    assert (root / "run.json").read_bytes() == original
    worker.assert_not_called()


def test_changed_result_context_hides_old_answers_and_resets_model_history(workbench, monkeypatch):
    app, worker, root, context, explain, _ = explanation_run(workbench, monkeypatch)
    widget(app.button, "Explain results").click().run()
    old_record = (root / "explanations.json").read_bytes()
    context.update(context_sha256="b" * 64, status="needs_attention", outcome="Output checks failed.")
    app.run()
    assert not app.exception
    assert any("Results changed" in item.value for item in app.info)
    assert not any("The requested relaxation completed." == item.value for item in app.markdown)
    assert (root / "explanations.json").read_bytes() == old_record
    assert explain.call_count == 1
    widget(app.text_input, "Ask about this run").set_value("Why did it fail?").run()
    widget(app.button, "Ask").click().run()
    assert explain.call_args.kwargs["history"] == []
    new_record = json.loads((root / "explanations.json").read_text())
    assert new_record["context_sha256"] == "b" * 64
    assert len(new_record["exchanges"]) == 1
    worker.assert_not_called()


def test_explanation_provider_failure_can_retry_without_submission(workbench, monkeypatch):
    app, worker, root, _, explain, answer = explanation_run(workbench, monkeypatch)
    explain.side_effect = agent.AgentError("Provider unavailable")
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert any("Provider unavailable" in item.value for item in app.error)
    assert not (root / "explanations.json").exists()
    app.run()
    assert explain.call_count == 1
    explain.side_effect = answer
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert (root / "explanations.json").is_file()
    assert explain.call_count == 2
    worker.assert_not_called()


def test_current_evidence_overrules_cached_success_in_results_panel(workbench, monkeypatch):
    app, worker, root, context, explain, _ = explanation_run(workbench, monkeypatch)
    state = workflow.read_state(root)
    state["status"] = "succeeded"
    stage = state["stages"][0]
    stage.update(status="succeeded", job_id="12345", scheduler_state="COMPLETED",
                 result={"success": True, "final_energy_ev": -10.0})
    (root / "run.json").write_text(json.dumps(state))
    saved = (root / "run.json").read_bytes()
    output = root / stage["folder"] / "outputs"
    output.mkdir()
    (output / "final_structure.cif").write_bytes(silicon_bytes())
    fig, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    fig.savefig(output / "result.png")
    plt.close(fig)
    context.update(status="needs_attention", outcome="No accepted results.")
    context["facts"].update({"stage_1.accepted": {"value": False},
                             "stage_1.availability": {"value": "Current output checks rejected this result."}})
    app.run()
    assert not app.exception
    assert any("Needs attention" in item.value for item in app.markdown)
    assert any("Current output checks rejected" in item.value for item in app.warning)
    assert not any("Complete. Results are ready" in item.value for item in app.info)
    assert not any(item.label == "Final total energy (eV)" for item in app.metric)
    assert not any(item.label == "Download structure (.cif)" for item in app.download_button)
    assert not any("relax · result.png" in block.captions for block in app.get("image"))
    assert (root / "run.json").read_bytes() == saved
    explain.assert_not_called()
    worker.assert_not_called()


def test_explanation_does_not_save_answer_if_results_change_during_request(workbench, monkeypatch):
    app, worker, root, context, explain, answer = explanation_run(workbench, monkeypatch)

    def changed(*args, **kwargs):
        response = answer(*args, **kwargs)
        context["context_sha256"] = "b" * 64
        return response

    explain.side_effect = changed
    widget(app.button, "Explain results").click().run()
    assert not app.exception
    assert any("Results changed" in item.value for item in app.error)
    assert not (root / "explanations.json").exists()
    assert not any("The requested relaxation completed." == item.value for item in app.markdown)
    worker.assert_not_called()
