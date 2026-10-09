from copy import deepcopy
import hashlib
import json

from vasp_slurm_agent.guidance import build_context
from vasp_slurm_agent.prompt_context import (
    compact_explanation_context,
    compact_history,
    compact_scientific_context,
    compact_structure,
)


def serialized(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def test_history_keeps_every_constraint_choice_question_and_latest_plan():
    goal = "Use HSE06. No SOC or relaxation."
    old = {"status": "needs_input", "tasks": ["bands"], "parameters": {"functional": "HSE06", "encut": 400},
           "questions": ["Use ENCUT=400 eV?"], "notes": ["No relaxation or SOC."],
           "goal": goal, "scientific_report": {"long": "evidence " * 10000},
           "model_usage": {"input_tokens": 5000}, "source_sha256": "a" * 64}
    latest = {**old, "status": "ready", "parameters": {"functional": "HSE06", "encut": 520}, "questions": []}
    history = [{"role": "assistant", "content": json.dumps(old)},
               {"role": "user", "content": "Yes, but use 520 eV. Keep every exclusion."},
               {"role": "assistant", "content": json.dumps(latest)},
               {"role": "user", "content": "Add DOS. 不要改变之前的限制。"}]
    before = deepcopy(history)
    result = compact_history(history, goal=goal)
    assert len(result) == len(history)
    assert result[1] == history[1] and result[3] == history[3]
    for actual, expected in zip((result[0], result[2]), (old, latest)):
        plan = json.loads(actual["content"])
        assert plan == {key: expected[key] for key in ("status", "tasks", "parameters", "questions", "notes")}
    assert history == before
    assert len(serialized(result)) < len(serialized(history)) / 20


def test_structure_history_keeps_complete_operations_and_termination_choices():
    plan = {"status": "needs_input", "operations": [
        {"type": "supercell", "matrix": [3, 1, 1]},
        {"type": "replace_sites", "indices": [0], "species": "Ge"},
        {"type": "slab", "miller_index": [0, 0, 1], "min_slab_size": 10,
         "min_vacuum_size": 15, "termination": None}],
        "notes": ["Choices: [{\"index\":0},{\"index\":1}]"], "questions": ["Which termination?"],
        "output_format": "both", "preview": {"sites": ["bulky"] * 500},
        "goal": "Original goal not present elsewhere."}
    history = [{"role": "assistant", "content": json.dumps(plan)},
               {"role": "user", "content": "Use termination 1."}]
    result = compact_history(history, goal="A different goal")
    saved = json.loads(result[0]["content"])
    assert saved == {key: value for key, value in plan.items() if key != "preview"}
    assert result[1] == history[1]


def test_unknown_assistant_text_and_json_are_unchanged():
    history = [{"role": "assistant", "content": "Use the second site?"},
               {"role": "assistant", "content": '{"status":"ready","answer":"A result"}'},
               {"role": "user", "content": "  Keep  spaces, δ and 1/3 exactly.  "}]
    assert compact_history(history) == history


def test_long_history_keeps_early_constraints_and_last_plan():
    history = [{"role": "user", "content": "No SOC, no relaxation. Keep Ni U=5.0 J=0.5 l=2."}]
    for index in range(40):
        history += [{"role": "assistant", "content": json.dumps({
            "status": "ready", "tasks": ["scf"], "parameters": {"encut": 400 + index,
            "hubbard_u": {"Ni": {"l": 2, "u": 5.0, "j": .5}}}, "questions": [], "notes": []})},
            {"role": "user", "content": f"Set ENCUT to {401 + index}; keep earlier constraints."}]
    result = compact_history(history)
    assert len(result) == 81
    assert [item for item in result if item["role"] == "user"] == [item for item in history if item["role"] == "user"]
    assert json.loads(result[-2]["content"]) == json.loads(history[-2]["content"])


def test_site_table_roundtrip_is_complete_and_exact():
    structure = {"formula": "FeO", "number_of_sites": 64,
                 "lattice_angstrom": [[4.12345678912345, 0, 0], [.1, 5, 0], [.2, .3, 8]],
                 "sites": [{"index": index, "element": "Fe" if index % 2 else "O",
                            "fractional_coordinates": [index/64, .3333333429999996, .6666666870000029],
                            "source_index": index % 2} for index in range(64)]}
    original = deepcopy(structure)
    result = compact_structure(structure)
    assert result["lattice_angstrom"] == structure["lattice_angstrom"]
    restored = [dict(zip(result["site_columns"], row)) for row in result["sites"]]
    assert restored == structure["sites"]
    assert len(result["sites"]) == structure["number_of_sites"]
    assert structure == original
    assert len(serialized(result)) < len(serialized(structure))


def test_small_or_irregular_site_records_are_not_reinterpreted():
    assert compact_structure(None) is None
    tiny = {"sites": [{"index": 0, "element": "Si", "fractional_coordinates": [0, 0, 0]}]}
    assert compact_structure(tiny) == tiny
    irregular = {"sites": [{"index": i, "element": "Si"} for i in range(16)]}
    irregular["sites"][3]["special"] = True
    assert compact_structure(irregular) == irregular


def test_advisory_vectors_become_hash_references_and_full_report_survives():
    moments = [[1.234567, -.123456, 2.345678] for _ in range(128)]
    science = {"evidence": [{"id": "magnetism", "title": "Magnetic states", "summary": "Zero net moment can be AFM.",
                              "sources": [{"url": "https://www.vasp.at/wiki/MAGMOM", "checked": "2026-10-09"}]}],
               "material_hints": ["Composition does not establish order."],
               "constraints": {"detected_exclusions": ["soc"]},
               "experience": [{"case_id": "case1", "context_sha256": "a" * 64, "applicability": "context_only",
                    "stages": [{"task": "scf", "accepted": True, "same_input_geometry": False,
                         "method": {"functional": "PBE", "spin": "noncollinear", "magmom": moments},
                         "evidence": [{"id": "stage_1.site_moments", "value": moments, "unit": "mu_B"},
                                      {"id": "stage_1.energy", "value": -31.1, "unit": "eV"}]}]}]}
    original = deepcopy(science)
    result = compact_scientific_context(science)
    stage = result["experience"][0]["stages"][0]
    expected_hash = hashlib.sha256(serialized(moments).encode()).hexdigest()
    assert stage["method"]["magmom_ref"]["sha256"] == expected_hash
    assert stage["evidence"][0]["value_ref"]["sha256"] == expected_hash
    assert stage["evidence"][0]["id"] == "stage_1.site_moments"
    assert stage["method"]["magmom_ref"]["items"] == 128
    assert stage["method"]["magmom_ref"]["available_to_model"] is False
    assert stage["evidence"][1] == science["experience"][0]["stages"][0]["evidence"][1]
    assert stage["same_input_geometry"] is False
    assert result["constraints"] == science["constraints"]
    assert result["evidence"][0]["sources"][0]["url"] == science["evidence"][0]["sources"][0]["url"]
    assert science == original
    assert len(serialized(result)) < len(serialized(science)) / 3


def test_explanation_retains_all_fact_values_and_limits():
    context = {"goal": "Keep SOC off.", "dialogue": [{"role": "user", "content": "Do not infer a band gap."}],
               "context_sha256": "a" * 64, "final_plan": {"parameters": {"magmom": [3, -3]}},
               "facts": {"stage_1.site_moments": {"label": "Site moments", "value": [[i, 0, -i] for i in range(100)],
                           "unit": "mu_B", "source": "01_scf/outputs/result.json#/magnetization"}},
               "artifacts": [{"id": "stage_1.bands.png", "label": "Bands", "status": "present_unverified",
                              "path": "01_bands/outputs/bands.png", "sha256": "b" * 64}],
               "limits": ["A band gap has not been quantified."]}
    before = deepcopy(context)
    result = compact_explanation_context(context)
    assert result["facts"]["stage_1.site_moments"]["value"] == context["facts"]["stage_1.site_moments"]["value"]
    assert result["final_plan"] == context["final_plan"]
    assert result["dialogue"] == context["dialogue"]
    assert result["limits"] == context["limits"]
    assert result["artifacts"][0]["status"] == "present_unverified"
    assert context == before


def test_guidance_remembers_early_exclusions_after_many_turns():
    history = [{"role": "user", "content": "Use PBE instead. No SOC or +U."}]
    history += [{"role": "user", "content": "Keep ENCUT at 500 eV."} for _ in range(15)]
    context = build_context("HSE06 bands with SOC and +U.", None, history=history)
    ids = {item["id"] for item in context["evidence"]}
    assert not ids & {"soc", "hubbard_u", "hybrid"}
    assert {"soc", "hubbard_u", "hybrid"} <= set(context["constraints"]["detected_exclusions"])
