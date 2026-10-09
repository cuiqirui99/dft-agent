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
