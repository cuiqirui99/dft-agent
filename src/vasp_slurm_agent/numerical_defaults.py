"""Resolve input defaults from geometry and explicitly supplied material type."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


# MPRelaxSet, pymatgen 2026.9.24. Keep f electrons active unless a user chooses
# a fixed-valence potential; W_sv replaces W_pv in newer VASP PAW releases.
POTCAR_RECOMMENDATIONS = dict(item.split(":") for item in (
    "Ba:Ba_sv Be:Be_sv Ca:Ca_sv Cr:Cr_pv Cs:Cs_sv Cu:Cu_pv Fe:Fe_pv "
    "Ga:Ga_d Ge:Ge_d Hf:Hf_pv In:In_d K:K_sv Li:Li_sv Mg:Mg_pv Mn:Mn_pv "
    "Mo:Mo_pv Na:Na_pv Nb:Nb_pv Ni:Ni_pv Os:Os_pv Pb:Pb_d Rb:Rb_sv "
    "Re:Re_pv Rh:Rh_pv Ru:Ru_pv Sc:Sc_sv Sn:Sn_d Sr:Sr_sv Ta:Ta_pv "
    "Tc:Tc_pv Ti:Ti_pv Tl:Tl_d V:V_pv W:W_sv Y:Y_sv Zr:Zr_sv"
).split())
POTCAR_REFERENCE = "https://pymatgen.org/pymatgen.io.vasp.html#pymatgen.io.vasp.sets.MPRelaxSet"
PAW_REFERENCE = "https://vasp.at/wiki/Choosing_pseudopotentials"
KPOINT_REFERENCE = "https://vasp.at/wiki/KSPACING"
SMEARING_REFERENCE = "https://vasp.at/wiki/Smearing_technique"


def vacuum_axes(structure):
    """Find large empty slabs between periodic atomic planes, in any cell basis."""
    reciprocal = structure.lattice.reciprocal_lattice_crystallographic
    heights = 1 / np.asarray(reciprocal.abc)
    fractions = np.mod(np.asarray(structure.frac_coords), 1)
    axes = []
    for axis, height in enumerate(heights):
        positions = np.sort(fractions[:, axis])
        largest = float(np.max(np.diff(np.r_[positions, positions[0] + 1])))
        gap = largest * height
        if gap >= 10 and largest >= 0.5:
            axes.append({"axis": axis, "empty_plane_gap_angstrom": float(gap),
                         "cell_height_angstrom": float(height), "empty_fraction": largest})
    return axes


def resolve_numerics(structure, task, settings):
    """Return explicit settings and a short record of how defaults were chosen."""
    result = deepcopy(settings)
    vacuum = vacuum_axes(structure)
    explicit_mesh = result["mesh"] is not None
    if not explicit_mesh:
        result["mesh"] = [max(1, math.ceil(length / result["kspacing"]))
                          for length in structure.lattice.reciprocal_lattice.abc]
        for item in vacuum:
            result["mesh"][item["axis"]] = 1
    notes = []
    if vacuum and not explicit_mesh:
        notes.append("One k point on each inferred vacuum axis; check the geometric vacuum assignment.")
    electronic_type = result["electronic_type"]
    explicit_smearing = result["ismear"] is not None
    if not explicit_smearing:
        result["ismear"] = 0
        if task in {"relax", "scf"} and electronic_type == "metal":
            result["ismear"] = 1
        elif task == "dos" and electronic_type == "insulator":
            if min(result["mesh"]) >= 2 and not vacuum:
                result["ismear"] = -5
            else:
                notes.append("Gaussian DOS: tetrahedra need a three-dimensional mesh; no automatic -5 for vacuum or a one-point axis.")
    if result["sigma"] is None:
        result["sigma"] = 0.2 if result["ismear"] in {1, 2} else 0.05
    if result["ismear"] == -5 and (task == "bands" or min(result["mesh"]) < 2):
        raise ValueError("ISMEAR=-5 requires a three-dimensional regular mesh; use ISMEAR=0 for a band path or one-point axis.")
    if electronic_type == "auto" and not explicit_smearing:
        notes.append("Gaussian smearing: metallicity was not specified and is not inferred from composition.")
    if electronic_type == "insulator" and result["ismear"] > 0:
        notes.append("Explicit Methfessel-Paxton smearing can give incorrect results for an insulator.")
    if result["ismear"] in {1, 2}:
        notes.append("Check the smearing entropy term and converge SIGMA for metallic forces.")
    record = {"mesh_mode": "explicit" if explicit_mesh else "reciprocal_spacing",
              "mesh": result["mesh"], "kspacing_inv_angstrom": result["kspacing"],
              "reciprocal_convention": "Physical reciprocal vectors include 2*pi; N_i=ceil(|G_i|/kspacing).",
              "vacuum_axes": vacuum, "smearing_mode": "explicit" if explicit_smearing else "material_type",
              "electronic_type": electronic_type, "ismear": result["ismear"], "sigma_ev": result["sigma"],
              "notes": notes, "sources": [KPOINT_REFERENCE, SMEARING_REFERENCE]}
    return result, record


def potcar_choices(elements, overrides=None):
    overrides = overrides or {}
    labels = [overrides.get(symbol, POTCAR_RECOMMENDATIONS.get(symbol, symbol)) for symbol in elements]
    return labels, {"policy": "MPRelaxSet 2026.9.24 with active f shells and W_sv; explicit labels take precedence.",
                    "overrides": {symbol: overrides[symbol] for symbol in elements if symbol in overrides},
                    "sources": [POTCAR_REFERENCE, PAW_REFERENCE]}
