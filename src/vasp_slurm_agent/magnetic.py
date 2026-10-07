"""Explicit trial magnetic seeds on one shared cell."""

from copy import deepcopy
import math

from pymatgen.core import Structure


def make_candidates(structure_path, states, parameters=None, initial_moment=3.0):
    """Return a common structure, labelled parameters and seed provenance."""
    if not isinstance(states, (list, tuple)) or not states:
        raise ValueError("Choose magnetic states from NM, FM and AFM.")
    labels = [str(label).upper() for label in states]
    if len(set(labels)) != len(labels) or any(label not in {"NM", "FM", "AFM"} for label in labels):
        raise ValueError("Magnetic states must be unique NM, FM or AFM labels.")
    if isinstance(initial_moment, bool) or not math.isfinite(float(initial_moment)) or float(initial_moment) <= 0:
        raise ValueError("initial_moment must be finite and positive.")
    structure = Structure.from_file(structure_path).get_sorted_structure()
    if not structure.is_ordered or not len(structure):
        raise ValueError("Magnetic seeds require an ordered structure.")

    def active_sites(cell):
        sites = [i for i, site in enumerate(cell) if any(
            getattr(site.specie, attr, False)
            for attr in ("is_transition_metal", "is_lanthanoid", "is_actinoid"))]
        return sites or list(range(len(cell)))

    active = active_sites(structure)
    doubled = "AFM" in labels and len(active) % 2 != 0
    if doubled:
        structure.make_supercell([2, 1, 1])
        structure = structure.get_sorted_structure()
        active = active_sites(structure)
    base = deepcopy(parameters or {})
    vectors = bool(base.get("soc")) or base.get("spin") == "noncollinear"
    candidates = []
    for label in labels:
        settings = deepcopy(base)
        settings.pop("magmom", None)
        settings["spin"] = "noncollinear" if vectors else "none" if label == "NM" else "collinear"
        if label != "NM" or vectors:
            moments = [0.0] * len(structure)
            for position, site in enumerate(active):
                if label != "NM":
                    moments[site] = float(initial_moment) * (-1 if label == "AFM" and position % 2 else 1)
            settings["magmom"] = [[0.0, 0.0, moment] for moment in moments] if vectors else moments
        candidates.append({"label": label, "parameters": settings})
    provenance = {
        "states": labels, "initial_moment": float(initial_moment),
        "supercell": [2, 1, 1] if doubled else [1, 1, 1],
        "seed_site_indices": active,
        "seed_rule": "Transition/f-block sites, or all sites when none are present; AFM alternates signs.",
        "scope": "Trial magnetic seeds, not a global ground-state search.",
    }
    return structure, candidates, provenance
