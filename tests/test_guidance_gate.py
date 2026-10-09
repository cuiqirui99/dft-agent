import json

from pymatgen.core import Structure

from vasp_slurm_agent import agent
from test_cli_plan import model_response


def test_required_scientific_inputs_gate_the_saved_plan(tmp_path, monkeypatch):
    source = tmp_path / "Fe.cif"
    Structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["Fe", "Fe"],
              [[0, 0, 0], [.5, .5, .5]]).to(filename=source)
    response = model_response(["scf"])
    response["parameters"].update(spin="collinear", magmom=None, hubbard_u={"Fe": None})
    response["intent"]["spin"] = "collinear"

    def request(payload, *args):
        evidence = json.loads(payload)["scientific_context"]["evidence"]
        magnetic = next(item for item in evidence if item["id"] == "magnetism")
        assert magnetic["applies_to"] and magnetic["next_check"] and magnetic["sources"]
        return json.dumps(response)

    monkeypatch.setattr(agent, "_request_plan", request)
    result = agent.draft_plan("Magnetic SCF; ask for the initial moments.", source, agent.ModelSettings("codex"))
    assert result["status"] == "needs_input"
    request = next(item for item in result["scientific_report"]["required_inputs"]
                   if item["id"] == "initial_moments")
    assert request["question"] in result["questions"]
    assert result["parameters"]["magmom"] is None
    assert json.loads(result["dialogue"][-1]["content"])["status"] == "needs_input"


def test_vacuum_review_does_not_block_the_requested_relaxation(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    Structure([[5, 0, 0], [0, 5, 0], [0, 0, 20]], ["Si", "Si"],
              [[0, 0, .4], [.5, .5, .5]]).to(filename=source)
    response = model_response(["relax"])
    response["parameters"]["cell_relax"] = True
    monkeypatch.setattr(agent, "_request_plan", lambda *args: json.dumps(response))
    result = agent.draft_plan("Relax this structure and the cell.", source, agent.ModelSettings("codex"))
    assert result["status"] == "ready"
    assert result["parameters"]["cell_relax"] is True
    assert not result["scientific_report"]["required_inputs"]
    assert any(item["id"] == "vacuum_cell" for item in result["scientific_report"]["action_checks"])
