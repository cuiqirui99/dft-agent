"""Scientific context reaches the model and survives preparation."""

from importlib.resources import files
import json
from unittest.mock import Mock

from vasp_slurm_agent import agent, experience
from vasp_slurm_agent.agent import ModelSettings
from test_cli_plan import local_cli, model_response


def test_plan_uses_sources_and_experience_without_changing_method(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    cases = [{"case_id": "prior-case", "applicability": "Context only", "stages": []}]
    retrieve = Mock(return_value=cases)
    monkeypatch.setattr(experience, "retrieve_experience", retrieve)
    reply = model_response(["bands"])
    reply["parameters"]["functional"] = "HSE06"
    reply["intent"].update(functional="HSE06", forbidden_tasks=["relax"], soc=False)
    requests = []

    def provider(payload, schema, settings):
        requests.append(json.loads(payload))
        return json.dumps(reply)

    monkeypatch.setattr(agent, "_request_plan", provider)
    plan = agent.draft_plan("HSE06 bands, no relaxation or SOC.", source,
                            ModelSettings("codex"), runs_root=tmp_path / "runs")
    assert len(requests) == 1
    science = requests[0]["scientific_context"]
    assert science["evidence"] and science["experience"] == cases
    assert all(item["sources"] for item in science["evidence"])
    assert str(tmp_path) not in json.dumps(requests)
    assert plan["parameters"]["functional"] == "HSE06"
    assert plan["tasks"] == ["bands"] and plan["parameters"]["soc"] is False
    assert plan["scientific_report"]["method_decisions"]
    assert plan["scientific_report"]["experience"] == cases
    assert retrieve.call_args.kwargs["parameters"]["functional"] == "HSE06"


def test_cli_saves_guidance_in_the_reviewed_proposal(local_cli, monkeypatch, capsys):
    source, config, run, invoke = local_cli
    proposal = run.parent / "proposal.json"
    monkeypatch.setattr(agent, "_request_plan", lambda *args: json.dumps(model_response()))
    assert invoke("plan", source, "Relax then SCF.", "--provider", "codex",
                  "--runs", run.parent / "history", "--output", proposal) == 0
    capsys.readouterr()
    assert invoke("prepare", source, run, "--config", config, "--plan", proposal) == 0
    report = json.loads((run / "proposal.json").read_text())["scientific_report"]
    assert report["evidence"] and report["validation_plan"]
    state = json.loads((run / "run.json").read_text())
    assert state["status"] == "planned" and all(s["job_id"] is None for s in state["stages"])


def test_large_history_report_does_not_block_clarification(local_cli, monkeypatch):
    source, _, root, _ = local_cli
    moments = [[1.234567, -0.123456, 2.345678] for _ in range(128)]
    cases = [{"case_id": str(index), "applicability": "context_only", "stages": [
        {"task": task, "accepted": True, "method": {"spin": "noncollinear", "magmom": moments},
         "evidence": [{"id": "stage_1.site_moments", "value": moments, "unit": "mu_B"}]}
        for task in ("relax", "scf")]}
        for index in range(4)]
    monkeypatch.setattr(experience, "retrieve_experience", Mock(return_value=cases))
    requests = []

    def provider(payload, schema, settings):
        requests.append(json.loads(payload))
        reply = model_response(["scf"], missing_moments=True)
        if len(requests) == 2:
            reply["parameters"]["magmom"] = [1, -1]
        return json.dumps(reply)

    monkeypatch.setattr(agent, "_request_plan", provider)
    goal = "Spin-polarized SCF only. Keep relaxation off."
    first = agent.draft_plan(goal, source, ModelSettings("codex"), runs_root=root.parent)
    assert first["status"] == "needs_input" and first["questions"]
    assert len(json.dumps(first["scientific_report"])) > 20000
    history = first["dialogue"] + [{"role": "user", "content": "Use moments [1, -1] in uploaded site order."}]
    second = agent.draft_plan(goal, source, ModelSettings("codex"), history, runs_root=root.parent)
    assert second["status"] == "ready" and second["parameters"]["magmom"] == [1, -1]
    assert second["scientific_report"]["experience"] == cases
    assert requests[1]["scientific_context"]["experience"] == cases
    previous = json.loads(requests[1]["history"][0]["content"])
    assert previous["questions"] == first["questions"] and previous["tasks"] == ["scf"]
    assert "scientific_report" not in previous
    assert len(first["dialogue"][-1]["content"]) <= 20000


from test_recovery import failed_run


def test_cli_repair_prepares_without_submitting(failed_run, local_cli, capsys):
    from vasp_slurm_agent import workflow
    import zipfile

    _, _, child, invoke = local_cli
    parent = failed_run[0](parameters={"nelm": 1})
    proposal = child.parent / "repair.json"
    assert invoke("repair", parent, "--provider", "codex", "--output", proposal) == 0
    capsys.readouterr()
    report = json.loads(proposal.read_text())
    assert report["changes"] == [{"scope": "parameters", "key": "nelm", "before": 1, "after": 120}]
    assert invoke("prepare-repair", parent, child, "--plan", proposal) == 0
    capsys.readouterr()
    state = workflow.read_state(child)
    assert state["status"] == "planned" and state["stages"][0]["job_id"] is None
    with zipfile.ZipFile(workflow.bundle_run(child)) as archive:
        assert "repair.json" in archive.namelist()
        assert str(parent) not in archive.read("plan.json").decode()
