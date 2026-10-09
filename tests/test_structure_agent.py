"""Structure dialogue boundary checks with actual local geometry operations."""

from __future__ import annotations

import hashlib
import json

import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import agent, structure_agent
from vasp_slurm_agent.agent import AgentError, ModelSettings
from vasp_slurm_agent.structure_agent import draft_structure


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "private-source.cif"
    Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]]).to(filename=path)
    return path


def response(operations=None, **changes):
    value = {"status": "ready", "summary": "Edit the structure.",
             "operations": operations or [], "output_format": "both", "questions": [], "notes": []}
    return {**value, **changes}


def stub(monkeypatch, value):
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps(value))


def test_conversion_uses_original_geometry_without_writing(monkeypatch, source):
    before = source.read_bytes()
    def request(payload, schema, settings, **kwargs):
        data = json.loads(payload)
        assert str(source) not in payload and source.name not in payload
        assert data["original_structure"]["number_of_sites"] == 2
        assert [site["index"] for site in data["original_structure"]["sites"]] == [0, 1]
        assert kwargs["name"] == "dft_structure"
        assert "ORIGINAL upload" in kwargs["instructions"]
        return json.dumps(response(output_format="poscar"))
    monkeypatch.setattr(agent, "_request_structured", request)
    result = draft_structure("Convert to POSCAR.", source, ModelSettings("codex", "test"))
    assert result["status"] == "ready" and result["operations"] == []
    assert result["output_format"] == "poscar"
    assert result["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert source.read_bytes() == before and list(source.parent.iterdir()) == [source]
    assert result["provenance"] == {"provider": "codex", "model": "test"}


def test_revision_replaces_supercell_and_replays_original(monkeypatch, source):
    from vasp_slurm_agent.structures import preview_structure
    replies = [response([{"type": "supercell", "matrix": [2, 2, 2]}]),
               response([{"type": "supercell", "matrix": [3, 1, 1]},
                         {"type": "replace_sites", "indices": [0], "species": "Ge"}])]
    payloads = []
    def request(payload, *args, **kwargs):
        payloads.append(json.loads(payload))
        return json.dumps(replies.pop(0))
    monkeypatch.setattr(agent, "_request_structured", request)
    goal = "Make a 2x2x2 supercell."
    first = draft_structure(goal, source, ModelSettings("codex", "test"))
    followup = {"role": "user", "content": "Change to 3x1x1, then replace site 0 with Ge."}
    second = draft_structure(goal, source, ModelSettings("codex", "test"), first["dialogue"] + [followup])
    assert second["status"] == "ready" and len(second["operations"]) == 2
    output = preview_structure(source, second["operations"])["output"]
    assert output["number_of_sites"] == 6
    assert output["sites"][0]["element"] == "Ge"
    assert payloads[1]["original_structure"]["number_of_sites"] == 2
    assert payloads[1]["history"][-1] == followup
    assert len(second["dialogue"]) == 3
    assert first["source_sha256"] == second["source_sha256"]


def test_large_preview_is_retained_without_filling_dialogue(monkeypatch, source):
    stub(monkeypatch, response([{"type": "supercell", "matrix": [8, 4, 4]}]))
    first = draft_structure("Make an 8x4x4 supercell.", source, ModelSettings("codex", "test"))
    assert len(json.dumps(first["preview"])) > 20000
    assert len(first["dialogue"][-1]["content"]) < 20000
    assert "preview" not in json.loads(first["dialogue"][-1]["content"])
    stub(monkeypatch, response([{"type": "supercell", "matrix": [6, 4, 4]}]))
    second = draft_structure(first["goal"], source, ModelSettings("codex", "test"),
                             first["dialogue"] + [{"role": "user", "content": "Use 6x4x4 instead."}])
    assert second["status"] == "ready" and second["preview"]["output"]["number_of_sites"] == 192


@pytest.mark.parametrize("operation,missing", [
    ({"type": "replace_sites", "indices": None, "species": "Ge"}, "indices"),
    ({"type": "remove_sites", "indices": None}, "indices"),
    ({"type": "slab", "miller_index": [0, 0, 1], "min_slab_size": None,
      "min_vacuum_size": None, "termination": None}, "min_slab_size"),
    ({"type": "translate_sites", "indices": [0], "vector": [0, 0, .1], "cartesian": None}, "cartesian"),
])
def test_unresolved_geometry_cannot_be_ready(monkeypatch, source, operation, missing):
    stub(monkeypatch, response([operation]))
    result = draft_structure("Edit the structure.", source, ModelSettings("codex", "test"))
    assert result["status"] == "needs_input"
    assert any(missing in question for question in result["questions"])


def test_missing_site_keeps_the_models_specific_question(monkeypatch, source):
    question = "Which Si site should become Ge: 0 or 1?"
    stub(monkeypatch, response([{"type": "replace_sites", "indices": None, "species": "Ge"}],
                              status="needs_input", questions=[question]))
    result = draft_structure("Replace one Si with Ge.", source, ModelSettings("codex", "test"))
    assert result["status"] == "needs_input" and result["questions"] == [question]


def test_missing_source_keeps_existing_upload_question(monkeypatch):
    question = "Please upload a source structure."
    stub(monkeypatch, response(status="unsupported", questions=[question]))
    result = draft_structure("Create silicon from its formula.", None, ModelSettings("codex", "test"))
    assert result["status"] == "unsupported" and result["questions"] == [question]


def test_source_required_and_no_crystal_is_invented(monkeypatch):
    stub(monkeypatch, response())
    result = draft_structure("Create silicon.", None, ModelSettings("codex", "test"))
    assert result["status"] == "needs_input" and result["source_sha256"] is None
    assert result["operations"] == []
    assert any("Upload" in question for question in result["questions"])


def test_unsupported_edit_remains_unsupported(monkeypatch, source):
    stub(monkeypatch, response(status="unsupported", summary="Direct lattice edits are not supported."))
    result = draft_structure("Change a to 7 angstroms.", source, ModelSettings("codex", "test"))
    assert result["status"] == "unsupported"


@pytest.mark.parametrize("operation", [
    {"type": "remove_sites", "indices": [100]},
    {"type": "set_site", "index": 1, "fractional_coordinates": [0, 0, 0]},
    {"type": "supercell", "matrix": [0, 1, 1]},
])
def test_geometry_checker_blocks_invalid_edit(monkeypatch, source, operation):
    stub(monkeypatch, response([operation]))
    result = draft_structure("Edit the structure.", source, ModelSettings("codex", "test"))
    assert result["status"] == "needs_input" and result["questions"]


def test_slab_choices_feed_the_next_turn(monkeypatch, source):
    from vasp_slurm_agent import structures
    preview = structures.preview_structure
    def checked(path, operations):
        if operations and operations[0]["termination"] is None:
            raise structures.StructureReviewError("Choose a termination.", [
                {"index": 0, "shift": .125, "number_of_sites": 8},
                {"index": 1, "shift": .375, "number_of_sites": 8}])
        return preview(path, []) | {"operations": operations}
    monkeypatch.setattr(structures, "preview_structure", checked)
    slab = {"type": "slab", "miller_index": [1, 1, 1], "min_slab_size": 10,
            "min_vacuum_size": 15, "termination": None}
    stub(monkeypatch, response([slab]))
    first = draft_structure("Make a (111) slab, 10 A thick and 15 A vacuum.", source, ModelSettings("codex", "test"))
    assert first["status"] == "needs_input"
    assert any('"index":1' in note for note in first["notes"])
    def request(payload, *args, **kwargs):
        prior = json.loads(json.loads(payload)["history"][0]["content"])
        assert "Choose a termination." in prior["questions"]
        assert json.loads(prior["notes"][-1].removeprefix("Choices: "))[1]["index"] == 1
        return json.dumps(response([{**slab, "termination": 1}]))
    monkeypatch.setattr(agent, "_request_structured", request)
    second = draft_structure("Make a (111) slab, 10 A thick and 15 A vacuum.", source,
                             ModelSettings("codex", "test"), first["dialogue"] + [{"role": "user", "content": "Use termination 1."}])
    assert second["status"] == "ready" and second["operations"][0]["termination"] == 1


@pytest.mark.parametrize("bad", [
    "not JSON", "NaN", '{"status":"ready"}',
    json.dumps(response([{"type": "shell", "command": "anything"}])),
    json.dumps(response([{"type": "remove_sites", "indices": [0], "extra": True}])),
])
def test_invalid_model_output_is_rejected(monkeypatch, source, bad):
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: bad)
    with pytest.raises(AgentError, match="invalid"):
        draft_structure("Edit the structure.", source, ModelSettings("codex", "test"))


def test_source_change_during_model_call_rejected(monkeypatch, source):
    def request(*args, **kwargs):
        source.write_bytes(source.read_bytes() + b"\n")
        return json.dumps(response())
    monkeypatch.setattr(agent, "_request_structured", request)
    with pytest.raises(AgentError, match="source changed"):
        draft_structure("Convert to POSCAR.", source, ModelSettings("codex", "test"))


def test_secrets_redacted_and_provider_echo_rejected(monkeypatch, source):
    secret = "private-api-key-structure-test"
    settings = ModelSettings("responses", "test", api_key=secret)
    def request(payload, *args, **kwargs):
        assert secret not in payload
        return json.dumps(response(notes=[secret]))
    monkeypatch.setattr(agent, "_request_structured", request)
    with pytest.raises(AgentError, match="invalid"):
        draft_structure("Convert " + secret, source, settings,
                        [{"role": "user", "content": secret}])


def test_questions_prevent_ready_status(monkeypatch, source):
    stub(monkeypatch, response(questions=["Which output format?"]))
    result = draft_structure("Convert.", source, ModelSettings("codex", "test"))
    assert result["status"] == "needs_input"
