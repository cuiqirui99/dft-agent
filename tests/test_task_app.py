from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path
from unittest.mock import Mock

from pymatgen.core import Structure
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import agent, batch, continuation, workflow
from test_app import open_agent, task_response, widget, workbench
from test_continuation import parent
from test_explanation import accepted_run


def test_unified_edit_and_mixed_methods_prepare_one_reviewed_task(workbench, monkeypatch):
    app, worker, runs_root = workbench
    reply = task_response(["relax", "bands"])
    reply["operations"] = [{"type": "supercell", "matrix": [2, 1, 1]},
                           {"type": "replace_sites", "indices": [0], "species": "Ge"}]
    reply["stages"][1]["parameters"]["functional"] = "HSE06"
    reply["stages"][1]["requirements"]["functional"] = "HSE06"
    request = Mock(return_value=json.dumps(reply))
    monkeypatch.setattr(agent, "_request_structured", request)
    open_agent(app)
    widget(app.text_area, "Goal").set_value("Make 2x1x1, replace site 0 with Ge, relax with PBE, then HSE06 bands.").run()
    widget(app.button, "Plan").click().run()
    assert not app.exception
    proposal = app.session_state["agent_proposal"]["plan"]
    assert proposal["preview"]["output"]["number_of_sites"] == 4
    assert proposal["preview"]["output"]["sites"][0]["element"] == "Ge"
    assert not runs_root.exists()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    state = workflow.read_state(root)
    assert [item["name"] for item in state["stages"]] == ["relax", "bands"]
    assert [item["parameters"]["functional"] for item in state["stages"]] == ["PBE", "HSE06"]
    assert len(Structure.from_file(root / "structure_edit/POSCAR")) == 4
    assert widget(app.button, "Submit calculation").disabled
    assert request.call_count == 1
    worker.assert_not_called()


def test_batch_review_is_required_before_starting_any_member(workbench, monkeypatch):
    app, worker, runs_root = workbench
    reply = task_response(["scf"])
    for label, moments in (("FM", [2, 2]), ("AFM", [2, -2])):
        stages = deepcopy(reply["stages"])
        stages[0]["parameters"].update(spin="collinear", magmom=moments)
        stages[0]["requirements"]["spin"] = "collinear"
        reply["variants"].append({"label": label, "strain": None, "stages": stages})
    monkeypatch.setattr(agent, "_request_structured", Mock(return_value=json.dumps(reply)))
    batch_worker = Mock(return_value=1234)
    monkeypatch.setattr(batch, "start_batch_worker", batch_worker)
    open_agent(app)
    widget(app.text_area, "Goal").set_value("Compare PBE SCF with collinear moments [2,2] and [2,-2].").run()
    widget(app.button, "Plan").click().run()
    assert not runs_root.exists()
    widget(app.button, "Prepare inputs").click().run()
    assert not app.exception
    root = Path(app.session_state["active_run"])
    state = batch.read_batch(root)
    assert [item["label"] for item in state["runs"]] == ["FM", "AFM"]
    assert all(workflow.read_state(root / item["folder"])["status"] == "planned" for item in state["runs"])
    assert widget(app.download_button, "Download table").proto.url
    assert widget(app.button, "Submit batch").disabled
    batch_worker.assert_not_called()
    worker.assert_not_called()
    widget(app.checkbox, "Structures, settings and resources reviewed.").check().run()
    widget(app.button, "Submit batch").click().run()
    assert not app.exception
    batch_worker.assert_called_once_with(root)
    worker.assert_not_called()


def continue_app(parent, monkeypatch):
    root, _ = parent
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(root / "config.json"))
    worker = Mock(return_value=1234)
    monkeypatch.setattr(workflow, "start_worker", worker)
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    widget(app.selectbox, "Provider").select("codex").run()
    widget(app.text_input, "Or enter a run folder").set_value(str(root)).run()
    widget(app.button, "Open run").click().run()
    assert not app.exception
    return app, worker


def test_continue_prepares_new_run_from_checked_structure_with_no_submission(parent, monkeypatch):
    root, _ = parent
    original = (root / "run.json").read_bytes()
    inherited = continuation.inspect_continuation(root, 0)
    request = Mock(return_value=json.dumps(task_response(["bands"])))
    monkeypatch.setattr(agent, "_request_structured", request)
    app, worker = continue_app(parent, monkeypatch)
    widget(app.text_area, "Next calculation").set_value("Calculate PBE bands from this result.").run()
    widget(app.button, "Plan next calculation").click().run()
    assert not app.exception
    assert json.loads(request.call_args.args[0])["previous_parameters"] == inherited["parameters"]
    worker.assert_not_called()
    widget(app.button, "Prepare continuation").click().run()
    assert not app.exception
    child = Path(app.session_state["active_run"])
    assert child != root
    assert [item["name"] for item in workflow.read_state(child)["stages"]] == ["scf", "bands"]
    assert json.loads((child / "parent.json").read_text())["source_structure_sha256"] == inherited["provenance"]["source_structure_sha256"]
    assert (root / "run.json").read_bytes() == original
    assert widget(app.button, "Submit calculation").disabled
    worker.assert_not_called()


def test_continue_rejects_changed_source_after_plan(parent, monkeypatch):
    root, _ = parent
    monkeypatch.setattr(agent, "_request_structured", Mock(return_value=json.dumps(task_response(["bands"]))))
    app, worker = continue_app(parent, monkeypatch)
    widget(app.text_area, "Next calculation").set_value("PBE bands.").run()
    widget(app.button, "Plan next calculation").click().run()
    before = {path for path in root.parent.iterdir() if path.is_dir()}
    inspect = continuation.inspect_continuation

    def changed(*args, **kwargs):
        result = inspect(*args, **kwargs)
        result["context_sha256"] = "changed-after-review"
        return result

    monkeypatch.setattr(continuation, "inspect_continuation", changed)
    widget(app.button, "Prepare continuation").click().run()
    assert not app.exception
    assert any("source results changed" in item.value for item in app.error)
    assert {path for path in root.parent.iterdir() if path.is_dir()} == before
    assert Path(app.session_state["active_run"]) == root
    worker.assert_not_called()
