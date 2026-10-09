"""Scientific context remains specific, sourced and separate from execution."""

from copy import deepcopy
import json

import pytest

from vasp_slurm_agent.guidance import build_context, review_plan


def cell(elements=("Si", "Si"), lattice=None, coordinates=None):
    return {"formula": "example", "number_of_sites": len(elements),
            "lattice_angstrom": lattice or [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            "sites": [{"index": index, "element": element,
                       "fractional_coordinates": (coordinates or [[0, 0, 0], [.25, .25, .25]])[index % 2]}
                      for index, element in enumerate(elements)]}


def ids(context):
    return {entry["id"] for entry in context["evidence"]}


def plan(**parameters):
    return {"tasks": ["scf", "bands"], "parameters": {"functional": "PBE", "spin": "none", **parameters},
            "intent": {}, "magnetic_states": []}


def test_plain_silicon_does_not_retrieve_unrelated_methods():
    context = build_context("Relax this structure", cell())
    assert ids(context) == {"convergence"}
    assert context["material_hints"] == []


def test_context_is_specific_and_every_rule_has_source_provenance():
    context = build_context("Calculate HSE06 bands with SOC", cell())
    assert {"hybrid", "soc", "smearing"} <= ids(context)
    assert not {"pbe_spectra", "hubbard_u", "structure"} & ids(context)
    for item in context["evidence"]:
        assert item["summary"] and item["title"]
        assert item["sources"]
        assert all(source["url"].startswith("https://www.vasp.at/wiki/index.php/")
                   and source["checked"] == "2026-10-09" for source in item["sources"])


@pytest.mark.parametrize("word", ["DFT+U", "LDA+U", "+U", "Hubbard U"])
def test_u_aliases_retrieve_the_dudarev_convention(word):
    context = build_context(f"Calculate {word} SCF", cell())
    text = json.dumps(context)
    assert "hubbard_u" in ids(context)
    assert "U-J" in text and "LDAUTYPE=2" in text


def test_composition_only_adds_uncertain_hints():
    context = build_context("SCF", cell(("Ni", "O")))
    assert {"magnetism", "hubbard_u"} <= ids(context)
    assert any("does not establish" in text for text in context["material_hints"])
    assert any("does not justify" in text for text in context["material_hints"])
    assert "parameters" not in context


def test_exclusions_override_chemistry_and_prior_user_turns():
    context = build_context("Try HSE06 SOC and DFT+U with magnetic seeds", cell(("Fe", "O")),
        history=[{"role": "user", "content": "Use PBE instead. Nonmagnetic SCF only, no SOC or +U. Skip relaxation."},
                 {"role": "assistant", "content": "Add SOC, AFM and HSE06."}])
    assert not {"soc", "hybrid", "magnetism", "hubbard_u"} & ids(context)
    assert context["material_hints"] == []
    assert {"soc", "hybrid", "magnetism", "hubbard_u", "relax"} <= set(context["constraints"]["detected_exclusions"])


def test_later_explicit_user_request_can_enable_an_excluded_topic():
    context = build_context("No SOC", cell(), history=[{"role": "user", "content": "Now use SOC."}])
    assert "soc" in ids(context)
    assert "soc" not in context["constraints"]["detected_exclusions"]


def test_pbe_prestep_does_not_hide_a_later_hybrid_request():
    context = build_context("Use PBE then HSE06 bands", cell())
    assert "hybrid" in ids(context)
    assert "pbe_spectra" not in ids(context)


def test_hybrid_and_pbe_spectra_keep_distinct_charge_routes():
    pbe = review_plan(plan(), cell())
    hybrid = review_plan(plan(functional="HSE06"), cell())
    assert "pbe_spectra" in ids(pbe) and "hybrid" not in ids(pbe)
    assert "hybrid" in ids(hybrid) and "pbe_spectra" not in ids(hybrid)
    decision = next(item for item in hybrid["method_decisions"] if item["topic"] == "spectra")
    assert "self-consistent" in decision["decision"] and "ICHARG=11 is not valid" in decision["decision"]
    assert any("path" in text for text in hybrid["validation_plan"])


def test_afm_zero_net_moment_is_not_a_failure_and_plan_is_unchanged():
    original = plan(spin="collinear", magmom=[2, -2])
    original["goal"] = "AFM SCF"
    before = deepcopy(original)
    report = review_plan(original, cell(("Fe", "Fe")))
    assert original == before
    assert any("zero total moment alone is not magnetic collapse" in text for text in report["validation_plan"])
    assert any("do not establish a global ground state" in text for text in report["risks"])
    assert not report["questions"]


def test_explicit_nonmagnetic_and_no_u_do_not_trigger_chemistry_questions():
    original = plan()
    original["intent"] = {"spin": "none", "hubbard_u": False}
    report = review_plan(original, cell(("Ni", "O")))
    assert "hubbard_u" not in ids(report)
    assert not report["questions"]
    assert not any("Composition suggests" in text or "Correlation may" in text for text in report["risks"])
    assert any("explicitly requested nonmagnetic" in item["decision"] for item in report["method_decisions"])


def test_incomplete_u_and_soc_are_questions_not_parameter_edits():
    original = plan(spin="noncollinear", soc=True, saxis=None, hubbard_u={"Ni": {"l": 2, "u": 5, "j": None}})
    before = deepcopy(original)
    report = review_plan(original, cell(("Ni", "O")))
    assert original == before
    assert any("SAXIS" in text for text in report["questions"])
    assert any("reference" in text for text in report["questions"])


def test_vacuum_warning_handles_skew_cells_without_rejecting_a_structure():
    structure = cell(lattice=[[5, 0, 0], [1, 5, 0], [12, 0, 20]], coordinates=[[0, 0, .4], [.5, .5, .5]])
    context = build_context("Relax", structure)
    assert "structure" in ids(context)
    original = plan(cell_relax=True)
    original["tasks"] = ["relax"]
    report = review_plan(original, structure)
    assert any("cell stay fixed" in text for text in report["questions"])
    assert "status" not in report


def test_fixed_cell_slab_does_not_ask_to_fix_the_cell_again():
    structure = cell(lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 20]], coordinates=[[0, 0, .4], [.5, .5, .5]])
    original = plan(cell_relax=False)
    original["tasks"] = ["relax"]
    report = review_plan(original, structure)
    assert "structure" in ids(report)
    assert not any("cell stay fixed" in text for text in report["questions"])
    assert original["parameters"]["cell_relax"] is False


def test_missing_structure_and_large_queries_produce_bounded_context():
    context = build_context("SOC HSE06 DFT+U AFM bands DOS slab " * 5000, None)
    assert len(context["evidence"]) <= 8
    assert len(json.dumps(context)) < 6500
    assert review_plan(plan(), None)["questions"] == ["Which structure should be calculated?"]


def test_returned_context_cannot_modify_the_bundled_rules():
    first = build_context("SOC", cell())
    first["evidence"][0]["summary"] = "changed"
    first["evidence"][0]["sources"][0]["url"] = "changed"
    second = build_context("SOC", cell())
    assert "changed" not in json.dumps(second)


def test_actions_have_applicability_sources_and_a_time_to_check():
    original = plan(functional="HSE06", spin="noncollinear", soc=True,
                    saxis=[0, 0, 1], magmom=[[0, 0, 2], [0, 0, -2]],
                    hubbard_u={"Fe": {"l": 2, "u": 4, "j": 0}})
    before = deepcopy(original)
    report = review_plan(original, cell(("Fe", "Fe")))
    assert original == before
    assert not report["required_inputs"]
    actions = {item["id"]: item for item in report["action_checks"]}
    assert {"hybrid_charge", "hybrid_path", "spin_basis", "hubbard_tags", "local_moments"} <= actions.keys()
    assert "scf_charge" not in actions
    for item in actions.values():
        assert item["applies_to"] and item["action"] and item["when"]
        assert set(item["evidence_ids"]) <= ids(report)
        assert item["status"] == "pending"
    assert all(item["applies_to"] and item["next_check"] for item in report["evidence"])


def test_required_inputs_only_cover_missing_selected_method_fields():
    original = plan(spin="noncollinear", magmom=None, soc=True, saxis=None,
                    hubbard_u={"Ni": {"l": 2, "u": 5, "j": None}})
    report = review_plan(original, cell(("Ni", "O")))
    requests = report["required_inputs"]
    assert {item["id"] for item in requests} == {"initial_moments", "spin_axis", "hubbard_values"}
    assert all(item["status"] == "needs_input" and item["when"] == "before_prepare" for item in requests)
    assert all(item["question"] in report["questions"] and item["fields"] for item in requests)
    baseline = plan()
    baseline["intent"] = {"spin": "none", "hubbard_u": False}
    report = review_plan(baseline, cell(("Ni", "O")))
    assert not report["required_inputs"]
    assert not {"initial_moments", "hubbard_values", "hubbard_tags", "local_moments"} & {
        item["id"] for item in report["action_checks"]}


def test_vacuum_hint_is_review_only_not_a_required_cell_change():
    structure = cell(lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 20]])
    original = plan(cell_relax=True)
    original["tasks"] = ["relax"]
    report = review_plan(original, structure)
    action = next(item for item in report["action_checks"] if item["id"] == "vacuum_cell")
    assert action["status"] == "review"
    assert not report["required_inputs"]
    assert original["parameters"]["cell_relax"] is True
