"""Small, source-linked scientific context. No network calls or parameter changes."""

from __future__ import annotations

import math
import re


VERSION = "2"
CHECKED = "2026-10-09"
_WIKI = "https://www.vasp.at/wiki/index.php/"
_RULES = {
    "convergence": (
        "Numerical settings",
        "Keep ENCUT consistent across energy comparisons. Choose it for the POTCAR set; "
        "a default cutoff does not establish property convergence.",
        ("ENCUT",),
    ),
    "magnetism": (
        "Magnetic states",
        "MAGMOM sets initial moments and affects symmetry. Different seeds can reach different minima. "
        "Compare plausible states with the same method and cell. Zero total moment can be valid AFM; "
        "inspect local moments. A few seeds do not establish the ground state.",
        ("MAGMOM",),
    ),
    "hubbard_u": (
        "DFT+U",
        "The runner uses Dudarev DFT+U (LDAUTYPE=2), with U-J as the effective interaction. "
        "Confirm element, orbital, U, J and their source; do not infer values from composition. "
        "Use sufficient LMAXMIX for charge reuse. Do not rank energies across different U/J choices.",
        ("LDAUTYPE",),
    ),
    "soc": (
        "SOC and spin coordinates",
        "SOC uses vasp_ncl. Specify vector moments and SAXIS together: moments use its spinor basis. "
        "Preserve the lattice frame and check magnetic symmetry. Nondefault SAXIS with hybrid "
        "functionals requires compatible symmetry settings.",
        ("LSORBIT", "SAXIS"),
    ),
    "hybrid": (
        "Hybrid spectra",
        "Hybrid calculations need orbitals on a regular k mesh and a self-consistent charge density; "
        "never use ICHARG=11. Zero-weight band-path orbitals need separate convergence checks. "
        "Total-energy convergence alone does not establish path-state convergence.",
        ("Band-structure_calculation_using_hybrid_functionals",),
    ),
    "pbe_spectra": (
        "PBE spectra",
        "PBE bands and DOS use a converged SCF CHGCAR with ICHARG=11 in this runner. "
        "Keep structure and method consistent. This fixed-charge route does not apply to hybrid functionals.",
        ("ICHARG",),
    ),
    "smearing": (
        "Occupations",
        "Choose smearing for the electronic state. Positive ISMEAR can give unphysical occupations "
        "in insulators. Tetrahedron integration needs a suitable mesh; a line path is not a DOS mesh.",
        ("ISMEAR",),
    ),
    "structure": (
        "Vacuum and surfaces",
        "For a slab or molecule, check vacuum separation, cell constraints and any dipole correction. "
        "A long cell alone does not prove a slab. Charged slabs need a separate electrostatic treatment.",
        ("Electrostatic_corrections",),
    ),
}
_PATTERNS = {
    "magnetism": r"\b(?:magnet\w*|afm|fm|ferrimagnet\w*|spin[- ]polar\w*)\b|磁性|磁序|反铁磁|铁磁",
    "soc": r"\b(?:soc|spin[- ]orbit\w*|noncollinear|saxis|lsorbit)\b|自旋轨道|非共线",
    "hubbard_u": r"\b(?:dft|lda)\s*\+\s*u\b|(?<!\w)\+\s*u\b|\b(?:hubbard|ldau\w*)\b|强关联",
    "hybrid": r"\b(?:hse(?:06)?|pbe0|hybrid\w*)\b|杂化",
    "relax": r"\b(?:relax\w*|optimi[sz]e)\b|弛豫|结构优化",
    "bands": r"\bbands?\b|能带",
    "dos": r"\bdos\b|density of states|态密度",
    "structure": r"\b(?:slabs?|surfaces?|vacuum|monolayers?|molecules?|2d)\b|表面|真空|单层|二维",
}
_NEGATION = re.compile(r"\b(?:no|not|without|skip|avoid|exclude|disable|off)\b|不要|不用|不加|关闭", re.I)
_MAGNETIC_ELEMENTS = set("Ti V Cr Mn Fe Co Ni Cu Ru Rh Pd Os Ir Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb U Np Pu".split())
_LIGANDS = set("O S Se Te F Cl Br I N P As".split())
_APPLICATION = {
    "convergence": ("All calculations", "Use one cutoff and k mesh for an energy comparison."),
    "magnetism": ("Selected magnetic calculations", "Specify site seeds, then inspect final local moments."),
    "hubbard_u": ("Explicit DFT+U requests", "Ask for missing l/U/J; check charge-reuse LMAXMIX."),
    "soc": ("SOC or noncollinear spin", "Confirm vectors and SAXIS, then use vasp_ncl."),
    "hybrid": ("HSE06 or PBE0 spectra", "Use a weighted mesh and self-consistent charge; check path states."),
    "pbe_spectra": ("PBE bands or DOS", "Run matching SCF before fixed-charge spectra."),
    "smearing": ("Electronic spectra", "Check occupations and the spectral k-point sampling."),
    "structure": ("Confirmed vacuum or surface geometry", "Review cell relaxation before changing vacuum."),
}


def _evidence(ids: list[str]) -> list[dict]:
    return [{"id": key, "title": _RULES[key][0], "summary": _RULES[key][1],
             "applies_to": _APPLICATION[key][0], "next_check": _APPLICATION[key][1],
             "sources": [{"url": _WIKI + page, "checked": CHECKED} for page in _RULES[key][2]]}
            for key in dict.fromkeys(ids)]


def _signals(goal: str, history: list | None) -> dict[str, bool]:
    """Select context, not executable intent. Later user mentions take precedence."""
    turns = [str(goal)] + [str(item.get("content", ""))
        for item in (history or []) if isinstance(item, dict) and item.get("role") == "user"]
    state: dict[str, bool] = {}
    for text in turns:
        for clause in re.split(r"[.!?;\n。；！？]|\bbut\b|但是|但", text, flags=re.I):
            events = []
            for key, pattern in _PATTERNS.items():
                for match in re.finditer(pattern, clause, flags=re.I):
                    before = " ".join(clause[:match.start()].split()[-8:])
                    after = clause[match.end():match.end() + 12]
                    enabled = not bool(_NEGATION.search(before) or re.match(r"\s*(?:off|disabled)\b", after, re.I))
                    events.append((match.start(), key, enabled))
            for match in re.finditer(r"\bnon[- ]?magnetic\b|无磁|非磁", clause, re.I):
                events.append((match.end(), "magnetism", False))
            for match in re.finditer(r"\b(?:use|with|switch to)\s+pbe\b|\bpbe\s+(?:instead|only)\b", clause, re.I):
                events.append((match.end(), "hybrid", False))
            for _, key, enabled in sorted(events):
                state[key] = enabled
    return state


def _elements(structure: dict | None) -> set[str]:
    if not isinstance(structure, dict):
        return set()
    return {site["element"] for site in structure.get("sites", [])
            if isinstance(site, dict) and isinstance(site.get("element"), str)
            and re.fullmatch(r"[A-Z][a-z]?", site["element"])}


def _vacuum_hint(structure: dict | None) -> bool:
    """Look for a large empty interval normal to a lattice plane, including skew cells."""
    if not isinstance(structure, dict):
        return False
    try:
        lattice = structure["lattice_angstrom"]
        sites = structure["sites"]
        if not sites or len(lattice) != 3 or any(len(row) != 3 for row in lattice):
            return False
        rows = [[float(value) for value in row] for row in lattice]
        for axis in range(3):
            a, b = rows[(axis + 1) % 3], rows[(axis + 2) % 3]
            normal = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
            norm = math.sqrt(sum(value*value for value in normal))
            height = abs(sum(x*y for x, y in zip(rows[axis], normal))) / norm
            fractions = sorted(float(site["fractional_coordinates"][axis]) % 1 for site in sites)
            gap = max(right-left for left, right in zip(fractions, fractions[1:] + [fractions[0] + 1]))
            if math.isfinite(height) and height >= 12 and gap * height >= 7:
                return True
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False
    return False


def build_context(goal: str, structure: dict | None, *, history: list | None = None) -> dict:
    """Retrieve a bounded local context; hints cannot override the user's choices."""
    signals = _signals(goal, history)
    elements = _elements(structure)
    possible_magnetic = bool(elements & _MAGNETIC_ELEMENTS)
    ids = ["convergence"]
    hints = []
    for key in ("magnetism", "hubbard_u", "soc", "hybrid", "structure"):
        if signals.get(key):
            ids.append(key)
    if possible_magnetic and signals.get("magnetism") is not False:
        ids.append("magnetism")
        hints.append("Composition suggests checking magnetism; it does not establish magnetic order or moments.")
    if possible_magnetic and elements & _LIGANDS and signals.get("hubbard_u") is not False:
        ids.append("hubbard_u")
        hints.append("Correlation may matter in this compound. Composition does not justify adding +U or choosing U/J.")
    if signals.get("bands") or signals.get("dos"):
        if not signals.get("hybrid"):
            ids.append("pbe_spectra")
        ids.append("smearing")
    if _vacuum_hint(structure):
        ids.append("structure")
        hints.append("The geometry may contain vacuum. Confirm its purpose before changing the cell.")
    return {
        "version": VERSION, "evidence": _evidence(ids), "material_hints": hints,
        "constraints": {
            "detected_exclusions": [key for key, enabled in signals.items() if not enabled],
            "rule": "Follow explicit user choices. Hints are uncertain and never authorize extra methods or tasks. "
                    "This selector retrieves guidance; it does not determine intent or validate a calculation.",
        },
    }


def review_plan(plan: dict, structure: dict | None) -> dict:
    """Explain method choices and checks without changing the plan."""
    parameters = plan.get("parameters") or {}
    intent = plan.get("intent") or {}
    tasks = set(plan.get("tasks") or [])
    functional = parameters.get("functional", "PBE")
    hybrid = functional in {"HSE06", "PBE0"}
    spin = parameters.get("spin", "none")
    soc = bool(parameters.get("soc"))
    hubbard = {key: value for key, value in (parameters.get("hubbard_u") or {}).items() if value is not None}
    comparison = bool(plan.get("magnetic_states"))
    context = build_context(plan.get("goal", ""), structure, history=plan.get("dialogue"))
    ids = [item["id"] for item in context["evidence"]]
    decisions, risks, questions = [], list(context["material_hints"]), []
    if intent.get("spin") == "none":
        risks = [item for item in risks if not item.startswith("Composition suggests")]
    if intent.get("hubbard_u") is False:
        risks = [item for item in risks if not item.startswith("Correlation may")]
        if not hubbard:
            ids = [key for key in ids if key != "hubbard_u"]
    checks = ["Verify the executed inputs and electronic convergence.",
              "Keep workflow acceptance separate from property convergence."]
    actions, required_inputs = [], []

    def action(key: str, text: str, evidence_id: str, *, when: str = "after_run",
               status: str = "pending", fields: list[str] | None = None):
        ids.append(evidence_id)
        item = {"id": key, "when": when, "status": status,
                "applies_to": _APPLICATION[evidence_id][0], "action": text,
                "evidence_ids": [evidence_id]}
        if fields:
            item["fields"] = fields
        actions.append(item)
        return item

    def require(key: str, question: str, fields: list[str], evidence_id: str):
        questions.append(question)
        item = action(key, question, evidence_id, when="before_prepare", status="needs_input", fields=fields)
        covered = {"initial_moments": (r"moments?",), "spin_axis": (r"saxis|spin axis",),
                   "hubbard_values": (r"\bU\b", r"\bJ\b"), "structure_input": (r"structure",)}[key]
        existing = next((text for text in plan.get("questions", []) if isinstance(text, str)
                         and all(re.search(pattern, text, re.I) for pattern in covered)), None)
        required_inputs.append({**item, "question": existing or question})

    action("executed_method", "Compare executed INCAR, KPOINTS and structure with the reviewed inputs.", "convergence")
    action("electronic_convergence", "Require electronic convergence before accepting a result or reusing its charge.", "convergence")

    def decision(topic: str, text: str, evidence_id: str):
        ids.append(evidence_id)
        decisions.append({"topic": topic, "decision": text, "evidence_ids": [evidence_id]})

    decision("functional", f"Use {functional} with the reviewed numerical settings.", "hybrid" if hybrid else "convergence")
    if tasks & {"bands", "dos"}:
        if hybrid:
            ids = [key for key in ids if key != "pbe_spectra"]
            decision("spectra", "Use self-consistent hybrid spectra; ICHARG=11 is not valid.", "hybrid")
            if "bands" in tasks:
                checks.append("Check zero-weight path orbitals and equivalent k-point energies.")
                risks.append("A hybrid total-energy check does not prove band-path convergence.")
                action("hybrid_path", "Check zero-weight path states separately from total-energy convergence.", "hybrid")
            action("hybrid_charge", "Use self-consistent hybrid inputs; do not freeze a PBE CHGCAR with ICHARG=11.",
                   "hybrid", when="before_prepare")
        else:
            decision("spectra", "Use the accepted SCF charge with the same structure and method.", "pbe_spectra")
            action("scf_charge", "Use an accepted SCF CHGCAR with matching structure, method and spin settings.",
                   "pbe_spectra", when="before_spectra")
        ids.append("smearing")
        checks.append("Check the k mesh, energy reference and occupations for the requested spectrum.")
        action("spectral_sampling", "Check the k-point sampling and energy reference before interpreting a gap or DOS.", "smearing")
    if spin != "none" or comparison:
        decision("magnetism", "Treat moments as starting seeds; inspect final local moments and ordering.", "magnetism")
        checks.append("Assess local moments; zero total moment alone is not magnetic collapse.")
        action("local_moments", "Inspect site moments and ordering; a zero total moment can be AFM.", "magnetism")
        risks.append("The tested magnetic seeds do not establish a global ground state.")
        if comparison:
            checks.append("Rank accepted magnetic seeds only with matching cell, method and numerical settings.")
            action("magnetic_ranking", "Rank only accepted seeds with the same cell, method, U/J and numerical settings.", "magnetism")
        elif parameters.get("magmom") is None:
            require("initial_moments", "What initial moments and site ordering should be used?", ["magmom"], "magnetism")
    elif intent.get("spin") == "none":
        decision("magnetism", "Keep the explicitly requested nonmagnetic baseline.", "magnetism")
    if soc or spin == "noncollinear":
        decision("spin_coordinates", "Use vasp_ncl and preserve the SAXIS spinor basis and lattice frame.", "soc")
        checks.append("Check vector moments, SAXIS and the executed noncollinear method.")
        action("spin_basis", "Verify vasp_ncl, vector moments, SAXIS and the preserved lattice frame.", "soc", when="before_prepare")
        if parameters.get("saxis") is None:
            require("spin_axis", "Which SAXIS should define the spinor basis?", ["saxis"], "soc")
    if hubbard or intent.get("hubbard_u") is True:
        decision("hubbard_u", "Use explicit Dudarev l/U/J values in the POTCAR species order.", "hubbard_u")
        checks.append("Verify LDAUL/LDAUU/LDAUJ, LMAXMIX and the source of U/J.")
        action("hubbard_tags", "Check l/U/J in POTCAR order; charge reuse needs LMAXMIX=4 for d or 6 for f shells.",
               "hubbard_u", when="before_prepare")
        risks.append("Total energies from different U/J choices are not a magnetic-state ranking.")
        if not hubbard or any(not isinstance(value, dict) or any(value.get(key) is None for key in ("l", "u", "j"))
                              for value in hubbard.values()):
            require("hubbard_values", "Which elements, orbitals, U and J should be used, and from which reference?",
                    ["hubbard_u"], "hubbard_u")
    if "relax" in tasks:
        checks.append("Require ionic convergence and the requested force threshold before reusing the structure.")
        action("relaxed_structure", "Require ionic convergence and the requested force threshold before reusing the structure.", "convergence")
    if "structure" in ids:
        risks.append("Slab electrostatics and vacuum convergence are not validated by the bulk workflow.")
        if "relax" in tasks and parameters.get("cell_relax", True):
            questions.append("Should the cell stay fixed to preserve the vacuum?")
            action("vacuum_cell", "Confirm whether cell relaxation should preserve the vacuum; geometry alone cannot decide.",
                   "structure", when="before_prepare", status="review", fields=["cell_relax"])
    if not isinstance(structure, dict):
        require("structure_input", "Which structure should be calculated?", ["structure"], "convergence")
    return {"version": VERSION, "evidence": _evidence(ids), "method_decisions": decisions,
            "validation_plan": list(dict.fromkeys(checks)), "risks": list(dict.fromkeys(risks)),
            "questions": list(dict.fromkeys(questions)), "action_checks": actions,
            "required_inputs": required_inputs}
