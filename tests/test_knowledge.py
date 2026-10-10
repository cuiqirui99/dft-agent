"""Lessons stay advisory, scoped and tied to their evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from vasp_slurm_agent import knowledge
from test_experience import runs  # Reuse solver-backed run fixtures.


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(identifier="si-pbe", **changes):
    value = {"id": identifier, "kind": "case", "title": "Si SCF", "status": "verified",
             "topics": ["convergence"], "tasks": ["scf"], "formulas": ["Si"],
             "methods": ["PBE", "nonmagnetic"], "summary": "A checked SCF example.",
             "conditions": ["Use the same method."], "limitations": ["Not a convergence study."],
             "outcome": "succeeded", "sources": [{"title": "VASP", "url": "https://www.vasp.at/wiki/index.php/ENCUT"}],
             "evidence": []}
    return {**value, **changes}


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    package = tmp_path / "package"
    root = package / "knowledge"
    root.mkdir(parents=True)
    monkeypatch.setattr(knowledge, "files", lambda name: package)

    def save(records):
        for value in records:
            proof = root / "evidence" / (value["id"] + ".json")
            write(proof, {"checked": True})
            value["evidence"] = [{"path": str(proof.relative_to(root)), "sha256": digest(proof),
                                  "description": "Checked result receipt."}]
        write(root / "catalog.json", {"schema_version": 1, "records": records})
        return root

    return save


def retrieve(**kwargs):
    return knowledge.retrieve_knowledge("Si SCF", {"formula": "Si"}, parameters={}, tasks=["scf"], **kwargs)


def test_verified_catalogue_retains_sources_and_limits(catalogue):
    root = catalogue([record()])
    before = digest(root / "catalog.json")
    item, = retrieve()
    assert item["status"] == "verified"
    assert item["conditions"] == ["Use the same method."]
    assert item["limitations"] == ["Not a convergence study."]
    assert item["sources"][0]["url"].endswith("ENCUT")
    assert "path" not in item["evidence"][0]
    assert digest(root / "catalog.json") == before


@pytest.mark.parametrize("status", ["candidate", "shadow_verified", "rejected", "deprecated"])
def test_inactive_records_are_visible_but_not_recommended(catalogue, status):
    catalogue([record(status=status)])
    assert knowledge.load_catalog()[0]["status"] == status
    assert retrieve() == []


def test_changed_missing_and_escaping_evidence_are_rejected(catalogue):
    root = catalogue([record()])
    proof = root / "evidence/si-pbe.json"
    proof.write_text("changed")
    assert knowledge.load_catalog()[0]["evidence_valid"] is False
    assert retrieve() == []
    proof.unlink()
    assert retrieve() == []
    outside = root.parent / "outside.json"
    outside.write_text('{"checked": true}')
    proof.symlink_to(outside)
    value = json.loads((root / "catalog.json").read_text())
    value["records"][0]["evidence"][0]["sha256"] = digest(outside)
    write(root / "catalog.json", value)
    assert retrieve() == []


@pytest.mark.parametrize("path", ["../secret", "/private/secret", "a/../../secret", "a\\..\\secret"])
def test_unsafe_evidence_paths_are_invalid(path):
    with pytest.raises(ValueError, match="evidence"):
        knowledge.validate_record(record(evidence=[{"path": path, "sha256": "a" * 64}]))


def test_material_and_method_scope_are_hard_filters(catalogue):
    catalogue([record()])
    assert not knowledge.retrieve_knowledge("Fe SCF", {"formula": "Fe"}, parameters={}, tasks=["scf"])
    assert not knowledge.retrieve_knowledge("Si bands", {"formula": "Si"}, parameters={}, tasks=["bands"])
    for parameters in ({"functional": "HSE06"}, {"soc": True}, {"hubbard_u": {"Si": {"u": 2}}}, {"spin": "collinear"}):
        assert not knowledge.retrieve_knowledge("Si SCF", {"formula": "Si"}, parameters=parameters, tasks=["scf"])
    assert retrieve()


def test_formula_uses_sites_and_reduced_composition(catalogue):
    catalogue([record(formulas=["Fe2O2"])])
    assert knowledge.retrieve_knowledge("SCF", {"formula": "Si", "sites": [{"element": "Fe"}, {"element": "O"}]}, parameters={}, tasks=["scf"])
    assert not retrieve()


def test_functional_alternatives_and_required_features(catalogue):
    catalogue([record("hybrid", kind="guidance", topics=["hybrid"], tasks=[], formulas=[],
                      methods=["HSE06", "PBE0"], outcome="guidance")])
    assert knowledge.retrieve_knowledge("HSE06 bands", None)
    assert knowledge.retrieve_knowledge("PBE0 bands", None)
    assert not knowledge.retrieve_knowledge("PBE bands", None)


@pytest.mark.parametrize("goal,history", [
    ("Si without SOC", None), ("Si SOC", [{"role": "user", "content": "No SOC"}]),
    ("SOC off", None), ("不要自旋轨道耦合", None),
])
def test_explicit_soc_exclusions_override_topic_match(catalogue, goal, history):
    catalogue([record("soc", kind="guidance", topics=["soc"], tasks=[], formulas=[], methods=["SOC"], outcome="guidance")])
    assert not knowledge.retrieve_knowledge(goal, None, parameters={"soc": True}, history=history)


def test_repairs_need_a_failure_context(catalogue):
    catalogue([record("repair", kind="repair", topics=["recovery", "electronic"], formulas=[], methods=[])])
    assert not retrieve()
    assert knowledge.retrieve_knowledge("Electronic convergence failed", {"formula": "Si"}, tasks=["scf"])


def test_repairs_match_the_reported_cause(catalogue):
    catalogue([record("repair-electronic", kind="repair", topics=["recovery", "electronic"], formulas=[], methods=[]),
               record("repair-walltime", kind="repair", topics=["recovery", "walltime"], formulas=[], methods=[]),
               record("missing-output", topics=["recovery", "missing_output"], methods=[])])
    def ids(goal):
        return {item["id"] for item in knowledge.retrieve_knowledge(goal, {"formula": "Si"}, tasks=["scf"])}
    assert ids("Electronic convergence not reached; repair failed calculation") == {"repair-electronic"}
    assert ids("Repair SCF timeout") == {"repair-walltime"}
    assert ids("Missing output from failed SCF") == {"missing-output"}


def test_prompt_has_no_paths_or_credentials(catalogue):
    catalogue([record(summary="password=secret123 api_key=sk-confidentialsecret /private/work user@example.org Normal SCF.")])
    encoded = json.dumps(retrieve())
    for private in ("secret123", "confidentialsecret", "/private/work", "user@example.org"):
        assert private not in encoded
    assert "Normal SCF" in encoded


def legacy(identifier="old-1", **changes):
    return {"record_id": identifier, "status": "verified", "step_kind": "scf",
            "material_context": {"reduced_formula": "Si"}, "method_context": {"functional": "PBE"},
            "workflow_root": "/private/work", "notes": ["password=secret123; unpublished plan"],
            "result_quality": {"converged": True}, **changes}


def test_jsonl_preview_and_selective_idempotent_import(tmp_path, catalogue):
    catalogue([])
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(value) for value in [legacy(), legacy("old-2", status="rejected")]))
    before = digest(source)
    runs_root = tmp_path / "runs"
    preview = knowledge.preview_import(source)
    assert [value["status"] for value in preview] == ["candidate", "rejected"]
    assert not runs_root.exists()
    assert "/private" not in json.dumps(preview)
    assert "secret123" not in json.dumps(preview)
    result = knowledge.import_selected(source, [preview[0]["id"]], runs_root)
    assert result["imported"] == 1
    assert result["skipped"] == 0
    records = knowledge.load_catalog(runs_root)
    assert len(records) == 1
    assert records[0]["legacy_id"] == "old-1"
    assert records[0]["status"] == "candidate"
    assert records[0]["evidence_valid"]
    assert not retrieve(runs_root=runs_root)
    evidence = Path(result["catalog_path"]).parent / records[0]["evidence"][0]["path"]
    assert json.loads(evidence.read_text())["workflow_root"] == "/private/work"
    if sys.platform == "win32":
        with evidence.open("r+") as stream:
            assert json.load(stream)["workflow_root"] == "/private/work"
    else:
        assert evidence.stat().st_mode & 0o077 == 0
    assert knowledge.import_selected(source, [preview[0]["id"]], runs_root)["skipped"] == 1
    assert digest(source) == before


def test_empty_or_stale_selection_does_not_write(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(legacy()))
    destination = tmp_path / "runs"
    assert knowledge.import_selected(source, [], destination)["imported"] == 0
    assert not destination.exists()
    with pytest.raises(ValueError, match="selection changed"):
        knowledge.import_selected(source, ["not-from-preview"], destination)
    assert not destination.exists()


def test_import_cannot_write_through_memory_symlink(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(legacy()))
    preview, = knowledge.preview_import(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "runs"
    destination.mkdir()
    (destination / ".dft-agent-memory").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        knowledge.import_selected(source, [preview["id"]], destination)
    assert list(outside.iterdir()) == []


def test_imported_self_asserted_verified_never_becomes_public_advice(tmp_path, catalogue):
    catalogue([])
    root = tmp_path / "runs"
    memory = root / ".dft-agent-memory"
    write(memory / "catalog.json", {"schema_version": 1, "records": [record(kind="guidance")]})
    assert knowledge.load_catalog(root)[0]["status"] == "candidate"
    assert not retrieve(runs_root=root)


def test_sqlite_import_is_read_only_and_retains_legacy_statuses(tmp_path, catalogue):
    catalogue([])
    source = tmp_path / "platform.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE lessons (id TEXT, lesson_key TEXT, current_status TEXT, current_version_id TEXT)")
        db.execute("CREATE TABLE lesson_versions (id TEXT, status TEXT, scope TEXT, lesson_type TEXT, recommended_action TEXT)")
        for index, status in enumerate(["production", "shadow_verified", "rejected", "deprecated", "candidate"]):
            db.execute("INSERT INTO lessons VALUES (?, ?, ?, ?)", (str(index), "DFT lesson", status, str(index)))
            db.execute("INSERT INTO lesson_versions VALUES (?, ?, ?, ?, ?)", (str(index), status, '{"method":"dft","task":"scf"}', "failure_repair", '{"password":"secret123"}'))
    before = digest(source)
    records = knowledge.preview_import(source)
    assert [item["status"] for item in records] == ["candidate", "shadow_verified", "rejected", "deprecated", "candidate"]
    assert records[0]["legacy_status"] == "production"
    assert "secret123" not in json.dumps(records)
    selected = [records[0]["id"], records[2]["id"]]
    destination = tmp_path / "runs"
    assert knowledge.import_selected(source, selected, destination)["imported"] == 2
    assert {value["status"] for value in knowledge.load_catalog(destination)} == {"candidate", "rejected"}
    assert not retrieve(runs_root=destination)
    assert digest(source) == before


def test_import_current_run_rechecks_outputs(runs, tmp_path, catalogue):
    catalogue([])
    _, query, create, _ = runs
    source, output = create()
    before = {str(path): digest(path) for path in source.rglob("*") if path.is_file()}
    preview, = knowledge.preview_import(source)
    assert preview["import_type"] == "run"
    destination = tmp_path / "elsewhere"
    knowledge.import_selected(source, [preview["id"]], destination)
    case, = knowledge.retrieve_imported_runs(destination, query, parameters={}, tasks=["scf"])
    assert case["source"] == "checked_imported_run"
    assert case["stages"][0]["accepted"]
    assert str(source) not in json.dumps(case)
    assert not knowledge.retrieve_imported_runs(destination, query, parameters={"soc": True})
    assert before == {str(path): digest(path) for path in source.rglob("*") if path.is_file()}
    (output / "vasprun.xml").write_text("changed solver output")
    changed, = knowledge.retrieve_imported_runs(destination, query)
    assert not changed["stages"][0]["accepted"]
    assert changed["stages"][0]["evidence"] == []


def test_imported_run_reference_cannot_escape_evidence_folder(runs, tmp_path, catalogue):
    catalogue([])
    _, query, create, _ = runs
    source, _ = create()
    preview, = knowledge.preview_import(source)
    destination = tmp_path / "destination"
    report = knowledge.import_selected(source, [preview["id"]], destination)
    catalog_path = Path(report["catalog_path"])
    stored = json.loads(catalog_path.read_text())
    path = catalog_path.parent / stored["records"][0]["evidence"][0]["path"]
    path.write_text('{"source_run":"/other/private/run"}')
    assert not knowledge.retrieve_imported_runs(destination, query)


def test_selected_old_run_does_not_depend_on_recent_history_limit(runs):
    root, query, create, _ = runs
    selected, _ = create("selected")
    from vasp_slurm_agent.experience import retrieve_experience
    assert retrieve_experience(root, query, run_paths=[selected])[0]["stages"][0]["accepted"]
    assert not retrieve_experience(root, query, run_paths=[root.parent])


def test_bundled_catalogue_has_checked_cases_and_inactive_lessons():
    records = knowledge.load_catalog()
    assert records
    assert all(item["evidence_valid"] for item in records)
    assert any(item["kind"] == "case" and item["outcome"] == "failed" for item in records)
    assert any(item["kind"] == "repair" and item["status"] == "verified" for item in records)
    assert any(item["status"] == "rejected" for item in records)
    references = retrieve(limit=8)
    assert any(item["kind"] == "case" for item in references)
    assert all(item["status"] == "verified" for item in references)
