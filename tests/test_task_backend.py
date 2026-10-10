"""The prepared executor retains the task's intent and selected recipes."""

from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from vasp_slurm_agent import agent, batch, task_agent, workflow
from vasp_slurm_agent.explanation import load_run_context
from test_task_agent import mock_response, response, setup, stage
from test_plan import complete_stage, fake_analysis
from test_workflow import FakeTransport


def read(path):
    return json.loads(path.read_text())


def test_edited_task_binds_goal_and_dialogue_to_frozen_execution(setup, monkeypatch):
    source, root, config = setup
    mock_response(monkeypatch, response(operations=[{"type": "supercell", "matrix": [2, 1, 1]}]))
    history = [{"role": "user", "content": "Keep relaxation without SOC; use SOC for bands, axis z."}]
    plan = task_agent.draft_task("Relax a 2x1x1 supercell, then SOC bands, axis z.", source,
                                 agent.ModelSettings("codex"), history)
    original = deepcopy(plan)
    state = task_agent.prepare_task(source, root, config, plan)
    assert plan == original
    saved, frozen = read(root / "proposal.json"), read(root / "plan.json")
    binding = saved["execution"]
    assert saved["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert binding["source_sha256"] != saved["source_sha256"]
    assert binding["source_sha256"] == hashlib.sha256((root / binding["source_path"]).read_bytes()).hexdigest()
    assert binding["plan_sha256"] == state["plan_sha256"]
    assert binding["parameters"] == frozen["parameters"] == {}
    assert binding["stage_parameters"] == frozen["stage_parameters"]
    context = load_run_context(root)
    assert context["goal"] == plan["goal"]
    assert context["dialogue"] == plan["dialogue"]
    frozen["stage_parameters"][0]["encut"] = 123
    (root / "plan.json").write_text(json.dumps(frozen))
    assert load_run_context(root)["goal"] == ""


def test_complete_variants_ignore_unused_incomplete_base_and_bind_each_child(setup, monkeypatch):
    source, root, config = setup
    base = stage("scf", hubbard_u=[{"element": "Si", "l": 1, "u": None, "j": None}])
    variants = [{"label": f"U{value}", "strain": None,
                 "stages": [stage("scf", hubbard_u=[{"element": "Si", "l": 1, "u": value, "j": 0}])]}
                for value in (1, 2)]
    mock_response(monkeypatch, response(stages=[base], variants=variants))
    plan = task_agent.draft_task("Compare Si l=1 with U=1 and 2 eV, J=0.", source, agent.ModelSettings("codex"))
    assert plan["status"] == "ready", plan["questions"]
    assert plan["stages"] == plan["variants"][0]["stages"]
    assert {item["variant"] for item in plan["scientific_reports"]} == {"U1", "U2"}
    state = task_agent.prepare_task(source, root, config, plan)
    saved = read(root / "proposal.json")
    assert saved["execution"]["plan_sha256"] == state["plan_sha256"]
    frozen = read(root / "batch-plan.json")
    assert saved["execution"]["source_path"] == frozen["source"] == "source/POSCAR"
    assert saved["execution"]["source_sha256"] == frozen["source_sha256"] == hashlib.sha256((root / frozen["source"]).read_bytes()).hexdigest()
    assert len(saved["variants"]) == 2
    for member, variant in zip(state["runs"], plan["variants"]):
        child = root / member["folder"]
        proposal = read(child / "proposal.json")
        assert proposal["variants"] == []
        assert proposal["selected_variant"] == {"label": variant["label"], "strain": None}
        assert proposal["stages"] == variant["stages"]
        assert proposal["execution"]["plan_sha256"] == member["plan_sha256"]
        assert {report["variant"] for report in proposal["scientific_reports"]} == {variant["label"]}
        context = load_run_context(child)
        assert context["goal"] == plan["goal"]
        assert context["dialogue"] == plan["dialogue"]
        assert all(row["context_sha256"] == load_run_context(root / row["folder"])["context_sha256"]
                   for row in read(root / "summary.json")["rows"])
    with zipfile.ZipFile(batch.bundle_batch(root)) as archive:
        assert "proposal.json" in archive.namelist()
        assert "structure_edit/source.cif" in archive.namelist()
    for member in state["runs"]:
        with zipfile.ZipFile(root / member["folder"] / "results.zip") as archive:
            assert json.loads(archive.read("proposal.json"))["selected_variant"]["label"] == member["label"]


def test_variant_default_does_not_inherit_another_variants_override(setup, monkeypatch):
    source, root, config = setup
    variants = [{"label": "override", "strain": None, "stages": [stage("scf", encut=700)]},
                {"label": "default", "strain": None, "stages": [stage("scf")]}]
    mock_response(monkeypatch, response(stages=[stage("scf")], variants=variants))
    plan = task_agent.draft_task("Compare ENCUT 700 eV and the default.", source, agent.ModelSettings("codex"))
    state = task_agent.prepare_task(source, root, config, plan)
    children = [workflow.read_state(root / member["folder"]) for member in state["runs"]]
    assert children[0]["stages"][0]["metadata"]["parameters"]["encut"] == 700
    assert children[1]["stages"][0]["metadata"]["parameters"]["encut"] == agent.DEFAULTS["encut"]


def test_preparation_preserves_reviewed_local_guidance(setup, monkeypatch):
    source, root, config = setup
    mock_response(monkeypatch, response(stages=[stage("scf")]))
    plan = task_agent.draft_task("Run SCF.", source, agent.ModelSettings("codex"))
    case = {"case_id": "local-imported-case-17", "source": "selected local case"}
    plan["scientific_report"]["knowledge"].append(case)
    plan["scientific_reports"][0]["knowledge"].append(case)
    task_agent.prepare_task(source, root, config, plan)
    saved = read(root / "proposal.json")
    assert saved["scientific_report"] == plan["scientific_report"]
    assert saved["scientific_reports"] == plan["scientific_reports"]
    assert case in saved["scientific_report"]["knowledge"]


@pytest.mark.parametrize("change", ["sequence", "requirement", "exclusion", "unsupported"])
def test_saved_ready_task_is_rechecked_before_preparation(setup, monkeypatch, change):
    source, root, config = setup
    variants = [{"label": "point", "strain": None, "stages": [stage("scf")]}]
    mock_response(monkeypatch, response(stages=[stage("scf")], variants=variants))
    plan = task_agent.draft_task("Run SCF.", source, agent.ModelSettings("codex"))
    assert plan["status"] == "ready"
    if change == "sequence":
        plan["variants"][0]["stages"][0]["task"] = "dos"
    elif change == "requirement":
        plan["variants"][0]["stages"][0]["requirements"]["functional"] = "HSE06"
    elif change == "exclusion":
        plan["intent"]["forbidden_tasks"] = ["scf"]
    else:
        plan["intent"]["unsupported"] = ["phonons"]
    with pytest.raises(ValueError):
        task_agent.prepare_task(source, root, config, plan)
    assert not root.exists()


def test_explicit_task_moments_follow_edited_site_order_across_relaxation(setup, monkeypatch):
    source, root, config = setup
    vectors = [[0, 0, .1], [0, 4, 0]]
    stages = [stage("relax", spin="collinear", magmom=[.2, 3]),
              stage("scf", spin="noncollinear", soc=True, magmom=vectors)]
    operations = [{"type": "replace_sites", "indices": [0], "species": "O"},
                  {"type": "replace_sites", "indices": [1], "species": "Fe"}]
    mock_response(monkeypatch, response(stages=stages, operations=operations))
    plan = task_agent.draft_task("O/Fe sites: collinear relaxation, then these SOC vectors, axis z.",
                                 source, agent.ModelSettings("codex"))
    assert plan["status"] == "ready", plan["questions"]
    task_agent.prepare_task(source, root, config, plan)
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    remote = FakeTransport()
    complete_stage(root, remote)
    remote.queue, remote.accounting = "PENDING", ""
    state = workflow.advance(root, remote)
    assert state["status"] == "queued", state.get("last_error")
    assert state["stages"][1]["parameters"]["magmom"] == vectors
    assert state["stages"][1]["metadata"]["method"]["magmom"] == [vectors[1], vectors[0]]


def test_summary_failure_leaves_no_batch_folder(setup, monkeypatch):
    source, root, config = setup
    def fail(*args, **kwargs):
        raise ValueError("Unable to verify the prepared batch")
    monkeypatch.setattr(batch, "_summarize_batch", fail)
    with pytest.raises(ValueError, match="verify"):
        batch.prepare_batch(source, root, config, ["scf"], [{"label": "one"}])
    assert not root.exists()
