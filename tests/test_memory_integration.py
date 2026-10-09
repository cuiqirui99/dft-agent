"""Memory reaches planning and repair without changing calculation settings."""

import hashlib
from importlib.resources import files
import json

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import agent, knowledge, recovery
from vasp_slurm_agent.agent import ModelSettings
from test_app import widget
from test_cli_plan import local_cli, model_response
from test_experience import runs
from test_recovery import failed_run, SETTINGS


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def legacy_records(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"record_id": "old-si", "status": "production", "step_kind": "scf",
         "material_context": {"reduced_formula": "Si"},
         "method_context": {"functional": "PBE"}, "result_quality": {"converged": True},
         "workflow_root": "/private/unpublished-study", "notes": ["password=private-secret"]},
        {"record_id": "old-mgo", "status": "rejected", "step_kind": "scf",
         "material_context": {"reduced_formula": "MgO"},
         "notes": ["Rejected instruction: change the functional without review."]},
        {"record_id": "old-carbon", "status": "candidate", "step_kind": "scf",
         "material_context": {"reduced_formula": "C"}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_bundled_cases_reach_plan_and_report_without_changing_settings(local_cli, monkeypatch):
    source, _, root, _ = local_cli
    reply = model_response(["scf"])
    reply["parameters"].update(encut=467, mesh=[5, 5, 5], ediff=2e-6)
    requests = []

    def model(payload, *_):
        requests.append(json.loads(payload))
        return json.dumps(reply)

    monkeypatch.setattr(agent, "_request_plan", model)
    goal = "Nonmagnetic PBE SCF. Use ENCUT 467 eV, a 5x5x5 mesh and EDIFF 2e-6."
    plan = agent.draft_plan(goal, source, ModelSettings("codex"), runs_root=root)
    assert plan["status"] == "ready"
    for references in (requests[0]["scientific_context"]["knowledge"],
                       plan["scientific_report"]["knowledge"]):
        ids = {record["id"] for record in references}
        assert {"si-pbe", "cutoff"}.issubset(ids)
        case = next(record for record in references if record["id"] == "si-pbe")
        assert case["evidence"] and case["conditions"] and case["limitations"]
        assert all(record["status"] == "verified" for record in references)
        assert "si-hse06" not in ids and "repair-electronic" not in ids
    assert plan["tasks"] == ["scf"]
    assert {key: plan["parameters"][key] for key in ("encut", "mesh", "ediff")} == {
        "encut": 467, "mesh": [5, 5, 5], "ediff": 2e-6}
    assert plan["parameters"]["functional"] == "PBE"
    assert not root.exists()


def test_bundled_repair_references_do_not_expand_allowed_change(failed_run, tmp_path):
    create, _, calls = failed_run
    root = create(parameters={"encut": 467, "mesh": [5, 5, 5], "nelm": 40})
    before = hashes(root)
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["status"] == "ready"
    for references in (calls[-1]["scientific_context"]["knowledge"],
                       proposal["guidance"]["knowledge"]):
        record = next(record for record in references if record["id"] == "repair-electronic")
        assert record["status"] == "verified" and record["evidence"]
        assert record["conditions"] and record["limitations"]
    assert proposal["changes"] == [{"scope": "parameters", "key": "nelm", "before": 40, "after": 120}]
    assert calls[-1]["allowed_actions"] == ["increase_nelm", "none"]
    assert calls[-1]["current_checks"]["result_reparsed"] is True
    assert calls[-1]["current_checks"]["converged_electronic"] is False
    assert "soc" not in {item["id"] for item in proposal["guidance"]["evidence"]}
    child = tmp_path / "repaired"
    state = recovery.prepare_repair(root, child, proposal)
    assert state["parameters"]["encut"] == 467
    assert state["parameters"]["mesh"] == [5, 5, 5]
    assert state["status"] == "planned" and state["stages"][0]["job_id"] is None
    assert hashes(root) == before


def test_cli_preview_selective_import_and_repeat_are_local(local_cli, legacy_records, capsys):
    _, _, root, invoke = local_cli
    before = legacy_records.read_bytes()
    assert invoke("memory", "import", legacy_records, "--runs", root) == 0
    preview = json.loads(capsys.readouterr().out)
    assert len(preview) == 3 and not root.exists()
    selected = [record["id"] for record in preview if record["legacy_id"] in {"old-si", "old-mgo"}]
    statuses = {record["legacy_id"]: record["status"] for record in preview}
    assert statuses == {"old-si": "candidate", "old-mgo": "rejected", "old-carbon": "candidate"}
    assert "private-secret" not in json.dumps(preview)
    assert invoke("memory", "import", legacy_records, "--ids", *selected, "--runs", root) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["imported"] == 2 and imported["skipped"] == 0
    stored = hashes(root)
    assert invoke("memory", "import", legacy_records, "--ids", *selected, "--runs", root) == 0
    repeat = json.loads(capsys.readouterr().out)
    assert repeat["imported"] == 0 and repeat["skipped"] == 2
    assert hashes(root) == stored
    assert legacy_records.read_bytes() == before
    assert invoke("memory", "list", "--runs", root, "--status", "rejected") == 0
    listed = json.loads(capsys.readouterr().out)
    local = [record for record in listed if record["origin"] == "local"]
    assert len(local) == 1 and local[0]["id"] in selected
    assert invoke("memory", "show", local[0]["id"], "--runs", root) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"


def test_imported_legacy_records_never_become_model_advice(local_cli, legacy_records, monkeypatch):
    source, _, root, _ = local_cli
    selected = [record["id"] for record in knowledge.preview_import(legacy_records)]
    knowledge.import_selected(legacy_records, selected, root)
    requests = []

    def model(payload, *_):
        requests.append(json.loads(payload))
        return json.dumps(model_response(["scf"]))

    monkeypatch.setattr(agent, "_request_plan", model)
    plan = agent.draft_plan("Nonmagnetic PBE SCF.", source, ModelSettings("codex"), runs_root=root)
    for references in (requests[0]["scientific_context"]["knowledge"],
                       plan["scientific_report"]["knowledge"]):
        assert references and not set(selected).intersection(record["id"] for record in references)
    text = json.dumps(requests)
    for forbidden in ("private-secret", "unpublished-study", "Rejected instruction", *selected):
        assert forbidden not in text
    assert plan["parameters"]["functional"] == "PBE"


def test_selected_saved_run_reaches_planning_and_is_rechecked(runs, tmp_path, monkeypatch):
    old_root, _, create, _ = runs
    saved, output = create()
    memory_root = tmp_path / "new-runs"
    preview = knowledge.preview_import(saved)
    knowledge.import_selected(saved, [preview[0]["id"]], memory_root)
    source = tmp_path / "new-Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples/Si.cif").read_bytes())
    requests = []

    def model(payload, *_):
        requests.append(json.loads(payload))
        return json.dumps(model_response(["scf"]))

    monkeypatch.setattr(agent, "_request_plan", model)
    plan = agent.draft_plan("Nonmagnetic PBE SCF.", source, ModelSettings("codex"), runs_root=memory_root)
    for cases in (requests[-1]["scientific_context"]["experience"], plan["scientific_report"]["experience"]):
        assert len(cases) == 1 and cases[0]["source"] == "checked_imported_run"
        assert cases[0]["stages"][0]["accepted"]
    assert str(old_root) not in json.dumps(requests)
    assert "private-secret" not in json.dumps(requests)
    (output / "OUTCAR").write_text("changed after import")
    later = agent.draft_plan("Nonmagnetic PBE SCF.", source, ModelSettings("codex"), runs_root=memory_root)
    for cases in (requests[-1]["scientific_context"]["experience"], later["scientific_report"]["experience"]):
        assert cases and not cases[0]["stages"][0]["accepted"]
        assert cases[0]["stages"][0]["evidence"] == []


def test_memory_ui_preview_selects_one_record_and_keeps_rejected_visible(tmp_path, legacy_records):
    root = tmp_path / "ui-runs"
    app = AppTest.from_string(
        "from pathlib import Path\nfrom vasp_slurm_agent.app import _memory_panel\n"
        f"_memory_panel(Path({str(root)!r}))\n", default_timeout=20).run()
    assert not app.exception
    widget(app.text_input, "Search memory").set_value("SOC").run()
    assert not app.exception and len(app.dataframe[0].value) > 0
    widget(app.text_input, "Local file or run folder").set_value(str(legacy_records)).run()
    widget(app.button, "Preview import").click().run()
    assert not app.exception and not root.exists()
    preview = app.session_state["memory_preview"]
    rejected = next(record["id"] for record in preview if record["status"] == "rejected")
    widget(app.multiselect, "Records").select(rejected).run()
    widget(app.button, "Import selected").click().run()
    assert not app.exception
    assert any("Imported 1" in message.value for message in app.success)
    widget(app.text_input, "Search memory").set_value("MgO").run()
    widget(app.selectbox, "Status").set_value("rejected").run()
    assert not app.exception
    rows = app.dataframe[0].value
    assert set(rows["Status"]) == {"rejected"}
    assert list(rows[rows["Source"] == "local"]["Title"]) == ["MgO SCF record"]
    assert any("not used as verified advice" in message.value for message in app.info)
    local = [record for record in knowledge.load_catalog(root) if record["origin"] == "local"]
    assert [record["id"] for record in local] == [rejected]
