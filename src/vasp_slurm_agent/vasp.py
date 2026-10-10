"""VASP inputs and output checks, adapted from AI_AGNET.

Defaults need material-specific convergence checks. POTCAR stays on the host.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Vasprun

from .methods import METHOD_DEFAULTS, HYBRID_BAND_NELMIN, method_fingerprint, method_incar, normalize_method, validate_method_output
from .numerical_defaults import potcar_choices, resolve_numerics


TASKS = frozenset({"relax", "scf", "bands", "dos"})
PLOT_ENERGY_WINDOW_EV = (-15.0, 10.0)
HYBRID_KPOINT_TOLERANCE_EV = 0.05
DEFAULTS = {
    "encut": 520.0,
    "ediff": 1e-5,
    "ediffg": -0.03,
    "nsw": 100,
    "mesh": None,
    "kspacing": 0.25,
    "ismear": None,
    "sigma": None,
    "electronic_type": "auto",
    "cell_relax": False,
    "nelm": 120,
    "line_density": 20,
    "nedos": 2001,
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_structure(structure: Structure) -> None:
    if not len(structure) or not structure.is_ordered:
        raise ValueError("A nonempty, fully ordered crystal structure is required.")
    if not np.isfinite(structure.lattice.matrix).all() or structure.volume <= 1e-6:
        raise ValueError("Structure has a nonfinite or singular unit cell.")
    if not np.isfinite(structure.frac_coords).all():
        raise ValueError("Structure contains nonfinite coordinates.")
    if not structure.is_valid(tol=0.5) or min(structure.lattice.abc) < 0.5:
        raise ValueError("Structure contains overlapping sites or an invalid short cell.")
    # Dummy species and partially occupied sites cannot select a licensed PAW file.
    if any(not getattr(site.specie, "Z", 0) for site in structure):
        raise ValueError("Every site must identify a chemical element.")


def _parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(parameters)
    if "kpoint_grid" in parameters:
        if parameters.get("mesh") is not None and parameters["mesh"] != parameters["kpoint_grid"]:
            raise ValueError("mesh and kpoint_grid disagree.")
        parameters["mesh"] = parameters.pop("kpoint_grid")
    unknown = set(parameters) - set(DEFAULTS) - set(METHOD_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported VASP parameters: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **parameters}
    for key in ("encut", "ediff", "kspacing", "sigma"):
        if key == "sigma" and result[key] is None:
            continue
        if isinstance(result[key], bool) or not math.isfinite(float(result[key])) or float(result[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
        result[key] = float(result[key])
    if isinstance(result["ediffg"], bool) or not math.isfinite(float(result["ediffg"])) or float(result["ediffg"]) >= 0:
        raise ValueError("ediffg must be negative: this release uses a force convergence criterion.")
    result["ediffg"] = float(result["ediffg"])
    for key in ("nsw", "nelm", "line_density", "nedos"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if result["nedos"] < 2:
        raise ValueError("nedos must be at least 2.")
    if type(result["cell_relax"]) is not bool:
        raise ValueError("cell_relax must be a boolean.")
    if result["ismear"] is not None and (type(result["ismear"]) is not int or result["ismear"] not in {-5, -1, 0, 1, 2}):
        raise ValueError("ismear must be one of -5, -1, 0, 1, 2.")
    if result["electronic_type"] not in {"auto", "metal", "insulator"}:
        raise ValueError("electronic_type must be auto, metal or insulator.")
    mesh = result["mesh"]
    if mesh is not None and (not isinstance(mesh, (list, tuple)) or len(mesh) != 3 or any(type(n) is not int or n <= 0 for n in mesh)):
        raise ValueError("mesh must contain three positive integers.")
    result["mesh"] = list(mesh) if mesh is not None else None
    return result


def prepare_inputs(
    structure_path: str | Path,
    destination: str | Path,
    task: str,
    parameters: dict[str, Any],
    potcar_symbols: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write inputs without changing the cell.

    PBE spectra require a compatible SCF CHGCAR. Hybrid spectra are self-consistent.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    source = Path(structure_path)
    settings = _parameters(parameters)
    structure = Structure.from_file(source)
    _validate_structure(structure)
    settings, numerical_choices = resolve_numerics(structure, task, settings)
    method = normalize_method(settings, [site.specie.symbol for site in structure])
    # Match species, counts and POTCAR order.
    site_order = sorted(range(len(structure)), key=lambda index: structure[index])
    structure = Structure.from_sites([structure[index] for index in site_order])
    if method["magmom"] is not None:
        method["magmom"] = [method["magmom"][index] for index in site_order]
    poscar = Poscar(structure)
    hybrid = method["functional"] != "PBE"
    fixed_charge = task in {"bands", "dos"} and not hybrid
    requires_ncl = method["soc"] or method["spin"] == "noncollinear"
    if hybrid and task == "bands" and settings["nelm"] <= HYBRID_BAND_NELMIN:
        raise ValueError(f"Hybrid bands require nelm > {HYBRID_BAND_NELMIN}.")
    labels, potential_choices = potcar_choices(poscar.site_symbols, potcar_symbols)
    if any(not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) for label in labels):
        raise ValueError("POTCAR labels must be simple symbols such as Si or Na_pv.")

    incar_data: dict[str, Any] = {
        "SYSTEM": f"dft-agent {structure.composition.reduced_formula} {task}",
        "GGA": "PE", "ISPIN": 1, "ENCUT": settings["encut"],
        "EDIFF": settings["ediff"], "NELM": settings["nelm"],
        "PREC": "Accurate", "ALGO": "Normal", "LREAL": False, "LASPH": True,
        "ISMEAR": settings["ismear"], "SIGMA": settings["sigma"],
        "IBRION": 2 if task == "relax" else -1,
        "NSW": settings["nsw"] if task == "relax" else 0,
        "ISIF": 3 if task == "relax" and settings["cell_relax"] else 2,
        "LWAVE": False, "LCHARG": task == "scf", "ISTART": 0,
        "ICHARG": 11 if task in {"bands", "dos"} else 2,
    }
    if task == "relax":
        incar_data["EDIFFG"] = settings["ediffg"]
    if task == "dos":
        incar_data.update(NEDOS=settings["nedos"], LORBIT=11)
    if task == "bands":
        incar_data["ISYM"] = 0
    incar_data.update(method_incar(method, poscar.site_symbols, task))
    expected_method = {key: incar_data[key] for key in method_incar(method, poscar.site_symbols, task)}
    expected_method.update({key: incar_data.get(key, False) for key in ("LSORBIT", "LNONCOLLINEAR", "LDAU", "LHFCALC")})
    expected_method["GGA"] = incar_data["GGA"]
    site_species = [site.specie.symbol for site in structure]

    metadata: dict[str, Any] = {
        "schema_version": 1, "task": task,
        "formula": structure.composition.reduced_formula,
        "number_of_atoms": len(structure), "parameters": settings,
        "numerical_choices": numerical_choices, "potential_choices": potential_choices,
        "source_sha256": _sha256(source), "potcar_labels": labels,
        "potcar_elements": poscar.site_symbols,
        "requires_chgcar": fixed_charge, "requires_ncl": requires_ncl,
        "spectral_charge_mode": "fixed_charge" if fixed_charge else "self_consistent",
        "method": method, "method_incar_expected": expected_method,
        "method_fingerprint": method_fingerprint(method, site_species, labels),
        "method_comparison_fingerprint": method_fingerprint(method, site_species, labels, comparison=True),
        "input_site_order": site_order,
        "scope": "Check convergence and the magnetic state for your material.",
    }
    if task == "bands":
        import seekpath

        # orig_cell maps both the basis and rotation back to the input cell.
        path = seekpath.get_explicit_k_path_orig_cell(
            (structure.lattice.matrix, structure.frac_coords, [site.specie.Z for site in structure]),
            reference_distance=1.0 / settings["line_density"],
            with_time_reversal=method["spin"] == "none",
        )
        fractional = np.asarray(path["explicit_kpoints_rel"])
        cartesian = structure.lattice.reciprocal_lattice.get_cartesian_coords(fractional)
        point_labels = path["explicit_kpoints_labels"]
        if not len(fractional) or not np.isfinite(fractional).all():
            raise ValueError("Could not construct a finite high-symmetry path.")
        mesh_points = [list(point) for point in product(*(np.arange(n) / n for n in settings["mesh"]))] if hybrid else []
        all_points = mesh_points + fractional.tolist()
        kpoints = Kpoints(
            comment="High-symmetry path in the actual POSCAR reciprocal basis",
            num_kpts=len(all_points), style=Kpoints.supported_modes.Reciprocal,
            kpts=all_points, kpts_weights=[1.0] * len(mesh_points) + [0.0 if hybrid else 1.0] * len(fractional),
            labels=[""] * len(mesh_points) + point_labels,
        )
        metadata["band_path_offset"] = len(mesh_points)
        metadata["band_path_count"] = len(fractional)
        if hybrid:
            metadata["hybrid_mesh"] = {"reciprocal_fractional": mesh_points, "weights": [1.0] * len(mesh_points)}
            metadata["band_convergence_note"] = "Zero-weight path convergence needs a separate check; SCF convergence alone does not prove it."
        metadata["band_path"] = {
            "method": "seekpath_hpkot_orig_cell",
            "symmetry_basis": "structural; magnetic space group not inferred",
            "labels": point_labels, "reciprocal_fractional": fractional.tolist(),
            "cartesian_inv_angstrom": np.asarray(cartesian).tolist(),
            "segments": path["explicit_segments"],
            "distance_inv_angstrom": np.asarray(path["explicit_kpoints_linearcoord"]).tolist(),
            "is_supercell": bool(path["is_supercell"]),
        }
        # Tetrahedron integration requires a regular mesh, not a symmetry line.
        if settings["ismear"] == -5:
            raise ValueError("ISMEAR=-5 is not supported on a band path; use ISMEAR=0.")
    else:
        kpoints = Kpoints.gamma_automatic(tuple(settings["mesh"]))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "POSCAR").write_text(poscar.get_str(), encoding="utf-8", newline="\n")
    (destination / "INCAR").write_text(str(Incar(incar_data)), encoding="utf-8", newline="\n")
    (destination / "KPOINTS").write_text(str(kpoints), encoding="utf-8", newline="\n")
    metadata["input_sha256"] = {name: _sha256(destination / name) for name in ("POSCAR", "INCAR", "KPOINTS")}
    _write_json(destination / "metadata.json", metadata)
    return metadata


def _same_structure(actual: Structure, expected: Structure) -> bool:
    if [site.species for site in actual] != [site.species for site in expected]:
        return False
    if not np.allclose(actual.lattice.matrix, expected.lattice.matrix, rtol=1e-6, atol=1e-5):
        return False
    delta = actual.frac_coords - expected.frac_coords
    return bool(np.allclose(delta - np.rint(delta), 0, atol=1e-5))


def _write_csv(path: Path, header: list[str], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _magnetization(output: Path, count: int, metadata: dict[str, Any]) -> dict[str, Any]:
    """Read final OUTCAR spin moments without treating sphere sums as cell totals."""
    method = metadata.get("method", {})
    ncl = bool(metadata.get("requires_ncl", False))
    result: dict[str, Any] = {
        "available": False, "unit": "mu_B", "site_order": "POSCAR",
        "basis": "SAXIS spinor basis" if ncl else "collinear spin axis",
        "saxis": method.get("saxis", [0.0, 0.0, 1.0]),
        "site_moments": None, "total_moment": None,
        "note": "Site moments are PAW sphere projections; their sum is not the full cell moment.",
    }
    path = output / "OUTCAR"
    if not path.is_file():
        return result
    # OUTCAR tables are scalar for collinear runs and x/y/z in spinor space for NCL.
    tables: dict[str, list[float]] = {}
    axis = None
    values: list[float] = []
    number = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
    with path.open(errors="replace") as stream:
        for line in stream:
            match = re.fullmatch(r"\s*magnetization \(([xyz])\)\s*", line)
            if match:
                axis, values = match[1], []
                if axis == "x":
                    tables = {}
                else:
                    tables.pop(axis, None)
            elif axis is not None:
                row = re.fullmatch(r"\s*(\d+)\s+((?:" + number + r"\s+)*" + number + r")\s*", line)
                if row and int(row[1]) == len(values) + 1:
                    values.append(float(row[2].split()[-1]))
                    if len(values) == count:
                        tables[axis] = values
                        axis = None
            if "number of electron" in line and "magnetization" in line:
                raw = re.findall(number, line.split("magnetization", 1)[1])
                if len(raw) == (3 if ncl else 1):
                    moment = [float(value) for value in raw]
                    if np.isfinite(moment).all():
                        result["total_moment"] = moment if ncl else moment[0]
    if ncl and all(key in tables for key in ("x", "y", "z")):
        result["site_moments"] = np.asarray([tables[key] for key in ("x", "y", "z")]).T.tolist()
    elif not ncl and "x" in tables:
        result["site_moments"] = tables["x"]
    if result["site_moments"] is not None:
        result["available"] = True
    return result


def _hybrid_kpoint_consistency(run: Vasprun) -> dict[str, Any]:
    """Check orbital energies independently of the weighted total-energy stop."""
    points = np.asarray(run.actual_kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Hybrid k-points are unavailable or nonfinite.")
    if not run.eigenvalues:
        raise ValueError("Band eigenvalues are unavailable.")
    pairs = []
    for first in range(len(points)):
        difference = points[first + 1:] - points[first]
        equivalent = np.all(np.abs(difference - np.rint(difference)) <= 1e-6, axis=1)
        pairs.extend((first, first + 1 + int(index)) for index in np.flatnonzero(equivalent))
    check: dict[str, Any] = {
        "tolerance_ev": HYBRID_KPOINT_TOLERANCE_EV,
        "scope": "All returned bands, energy-sorted within each spin channel.",
        "equivalent_pair_count": len(pairs), "compared_eigenvalue_count": 0,
        "max_energy_deviation_ev": 0.0, "worst_pair": None,
        "passed": bool(pairs),
    }
    for spin, eigenvalues in run.eigenvalues.items():
        values = np.asarray(eigenvalues, dtype=float)
        if values.ndim != 3 or values.shape[0] != len(points) or values.shape[1] == 0 or values.shape[2] != 2 or not np.isfinite(values).all():
            raise ValueError("Band eigenvalues do not match the finite k-point path.")
        energies = np.sort(values[:, :, 0], axis=1)
        for first, second in pairs:
            deviations = np.abs(energies[first] - energies[second])
            check["compared_eigenvalue_count"] += len(deviations)
            band = int(np.argmax(deviations))
            maximum = float(deviations[band])
            if maximum > check["max_energy_deviation_ev"]:
                check["max_energy_deviation_ev"] = maximum
                check["worst_pair"] = {"kpoint_indices_zero_based": [first, second], "spin": int(spin), "band": band + 1}
    check["passed"] = bool(pairs) and check["max_energy_deviation_ev"] <= HYBRID_KPOINT_TOLERANCE_EV
    return check


def _export_plots(run: Vasprun, output: Path, task: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        if task == "relax":
            energies = [float(step["e_0_energy"]) for step in run.ionic_steps]
            if not energies or not np.isfinite(energies).all():
                raise ValueError("No finite ionic energy history.")
            name = "relax_energy"
            _write_csv(output / f"{name}.csv", ["ionic_step", "energy_ev"], enumerate(energies, 1))
            ax.plot(range(1, len(energies) + 1), energies, marker="o", markersize=3)
            ax.set(xlabel="Ionic step", ylabel="Energy (eV)")
        elif task == "dos":
            dos = run.complete_dos
            if dos is None or not np.isfinite(dos.energies).all() or not math.isfinite(float(run.efermi)):
                raise ValueError("No finite total DOS or Fermi energy.")
            name = "dos"
            energies = np.asarray(dos.energies) - run.efermi
            densities = list(dos.densities.items())
            if not densities or any(not np.isfinite(values).all() for _, values in densities):
                raise ValueError("Nonfinite DOS densities.")
            header = ["energy_minus_fermi_ev"] + [f"dos_spin_{int(spin)}" for spin, _ in densities]
            _write_csv(output / f"{name}.csv", header, zip(energies, *(values for _, values in densities)))
            for spin, values in densities:
                ax.plot(energies, values, label=f"spin {int(spin)}")
            ax.set(xlabel="Energy − Fermi energy (eV)", ylabel="DOS (states/eV)")
            ax.set_xlim(*PLOT_ENERGY_WINDOW_EV)
            visible = (energies >= PLOT_ENERGY_WINDOW_EV[0]) & (energies <= PLOT_ENERGY_WINDOW_EV[1])
            if visible.any():
                peak = max(float(np.max(values[visible])) for _, values in densities)
                ax.set_ylim(0, peak * 1.05 if peak > 0 else 1.0)
            ax.axvline(0, color="gray", linewidth=0.7)
        elif task == "bands":
            if not run.eigenvalues:
                raise ValueError("Band eigenvalues are unavailable.")
            all_kpts = np.asarray(run.actual_kpoints, dtype=float)
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            hybrid = metadata.get("method", {}).get("functional", "PBE") != "PBE"
            offset = int(metadata.get("band_path_offset", 0))
            kpts = all_kpts[offset:]
            if hybrid:
                mesh = np.asarray(metadata.get("hybrid_mesh", {}).get("reciprocal_fractional", []))
                if offset <= 0 or mesh.shape != all_kpts[:offset].shape or not np.allclose(mesh - all_kpts[:offset] - np.rint(mesh - all_kpts[:offset]), 0, atol=1e-6):
                    raise ValueError("Hybrid output does not match the frozen regular mesh.")
                weights = np.asarray(getattr(run, "actual_kpoints_weights", []))
                if weights.shape != (len(all_kpts),) or not np.isfinite(weights).all() or np.any(weights[:offset] <= 0) or not np.allclose(weights[offset:], 0):
                    raise ValueError("Hybrid bands require positive mesh weights and zero path weights.")
                if not np.allclose(weights[:offset] / sum(weights[:offset]), np.ones(offset) / offset, atol=1e-7):
                    raise ValueError("Hybrid regular-mesh weights differ from the prepared inputs.")
            fermi = run.efermi if hybrid else metadata.get("scf_fermi_energy_ev")
            if fermi is None or not math.isfinite(float(fermi)):
                raise ValueError("A finite hybrid-run Fermi energy is required." if hybrid else "A finite preceding SCF Fermi energy is required for band analysis.")
            fermi = float(fermi)
            path_metadata = metadata.get("band_path", {})
            if not path_metadata:
                raise ValueError("Frozen band path metadata.json is required for band analysis.")
            labels = path_metadata.get("labels", [])
            if len(labels) != len(kpts) or int(metadata.get("band_path_count", len(labels))) != len(kpts):
                raise ValueError("Band path metadata does not match output k-points.")
            planned = np.asarray(path_metadata.get("reciprocal_fractional", []))
            if planned.shape != kpts.shape or not np.allclose(planned, kpts, atol=1e-6, rtol=0):
                raise ValueError("Output k-points differ from the frozen band path.")
            distance = np.asarray(path_metadata.get("distance_inv_angstrom", []))
            segments = path_metadata.get("segments", [])
            if distance.shape != (len(kpts),) or not np.isfinite(distance).all() or not segments:
                raise ValueError("Frozen band path is missing its distance or segment definition.")
            rows = []
            mesh_rows = []
            for spin, eigenvalues in run.eigenvalues.items():
                all_values = np.asarray(eigenvalues)
                if len(all_values) != len(all_kpts) or not np.isfinite(all_values).all():
                    raise ValueError("Band eigenvalues do not match the finite k-point path.")
                for ik, point in enumerate(all_kpts[:offset]):
                    for ib in range(all_values.shape[1]):
                        mesh_rows.append([ik, *point, int(spin), ib + 1, all_values[ik, ib, 0] - fermi, all_values[ik, ib, 1]])
                values = all_values[offset:]
                energies = values[:, :, 0] - fermi
                for start, stop in segments:
                    ax.plot(distance[start:stop], energies[start:stop], color="tab:blue" if int(spin) == 1 else "tab:orange", linewidth=0.6)
                for ik, point in enumerate(kpts):
                    for ib in range(values.shape[1]):
                        rows.append([ik, distance[ik], *point, labels[ik], int(spin), ib + 1, energies[ik, ib], values[ik, ib, 1]])
            name = "bands"
            _write_csv(output / f"{name}.csv", ["kpoint", "distance_inv_angstrom", "kx", "ky", "kz", "label", "spin", "band", "energy_minus_fermi_ev", "occupation"], rows)
            if mesh_rows:
                _write_csv(output / "bands_mesh.csv", ["kpoint", "kx", "ky", "kz", "spin", "band", "energy_minus_fermi_ev", "occupation"], mesh_rows)
            ticks: dict[float, str] = {}
            for x, label in zip(distance, labels):
                if label:
                    previous = ticks.get(float(x))
                    ticks[float(x)] = label if not previous or previous == label else f"{previous}|{label}"
            ax.set_xticks(list(ticks), [label.replace("GAMMA", "Γ") for label in ticks.values()])
            for position in ticks:
                ax.axvline(position, color="0.85", linewidth=0.6, zorder=0)
            ax.set(xlabel="Reciprocal path", ylabel="Energy − Fermi energy (eV)")
            ax.set_ylim(*PLOT_ENERGY_WINDOW_EV)
            ax.set_xlim(float(distance[0]), float(distance[-1]))
            ax.axhline(0, color="gray", linewidth=0.7)
        else:
            return []
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        return [f"{name}.csv", f"{name}.png"] + (["bands_mesh.csv"] if task == "bands" and mesh_rows else [])
    finally:
        plt.close(fig)


def _band_gap(run, task: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Occupation-based edges on the retained k-points, never a full-zone claim."""
    tolerance = 1e-3
    result = {"available": False, "status": "unavailable",
              "scope": "sampled_path" if task == "bands" else "sampled_kpoints",
              "gap_ev": None, "direct_gap_ev": None, "vbm_ev": None, "cbm_ev": None,
              "partial_occupations": False, "occupation_tolerance": tolerance,
              "full_bz_validated": False, "reason": "Eigenvalues and occupations are unavailable."}
    if task not in {"scf", "bands", "dos"}:
        return result
    eigenvalues = getattr(run, "eigenvalues", None)
    if not isinstance(eigenvalues, dict) or not eigenvalues:
        return result
    try:
        channels = [np.asarray(value, dtype=float) for value in eigenvalues.values()]
        count = len(getattr(run, "actual_kpoints", []))
        if not count or any(value.ndim != 3 or value.shape[0] != count or value.shape[2] != 2
                            or not np.isfinite(value).all() for value in channels):
            return result
        if task == "bands":
            offset, length = metadata.get("band_path_offset", 0), metadata.get("band_path_count")
            if type(offset) is not int or type(length) is not int or offset < 0 or length < 1 or offset + length != count:
                result["reason"] = "The eigenvalues are not linked to the prepared band path."
                return result
            channels = [value[offset:offset + length] for value in channels]
        # vasprun.xml reports occupations per state in [0, 1], including ISPIN=1.
        values = np.concatenate(channels, axis=1)
        energies, occupations = values[:, :, 0], values[:, :, 1]
        if np.any(occupations < -tolerance) or np.any(occupations > 1 + tolerance):
            result["reason"] = "Occupations are outside the supported zero-to-one convention."
            return result
        result["kpoint_count"] = len(values)
        partial = (occupations > tolerance) & (occupations < 1 - tolerance)
        if np.any(partial):
            result.update(status="metallic_or_partially_occupied", partial_occupations=True,
                          reason="Partial occupations prevent a resolved gap; smearing can also cause them.")
            return result
        occupied, empty = occupations >= 1 - tolerance, occupations <= tolerance
        if not np.all(occupied.any(axis=1)) or not np.all(empty.any(axis=1)):
            result["reason"] = "Both occupied and empty states are needed at every sampled k-point."
            return result
        valence = np.where(occupied, energies, -np.inf).max(axis=1)
        conduction = np.where(empty, energies, np.inf).min(axis=1)
        vbm, cbm = float(valence.max()), float(conduction.min())
        # A different number of occupied states at different k-points signals a
        # Fermi surface even if sparse sampling leaves a positive edge separation.
        crossing = any(np.ptp((channel[:, :, 1] >= 1 - tolerance).sum(axis=1)) > 0 for channel in channels)
        metallic = bool(crossing or cbm <= vbm)
        result.update(available=True, status="metallic" if metallic else "gapped",
                      gap_ev=0.0 if metallic else cbm - vbm,
                      direct_gap_ev=None if metallic else float((conduction - valence).min()),
                      vbm_ev=vbm, cbm_ev=cbm,
                      reason="Only the retained k-points were checked; this is not a converged full-zone or optical gap.")
    except (TypeError, ValueError):
        pass
    return result


def analyze_outputs(output_dir: str | Path, task: str, expected_structure_path: str | Path) -> dict[str, Any]:
    """Reject incomplete, unconverged or mismatched XML.

    Acceptance does not establish scientific accuracy.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Remove stale results before checking new output.
    generated = ["final_structure.cif", "final_structure.vasp", "relax_energy.csv", "relax_energy.png", "bands.csv", "bands.png", "bands_mesh.csv", "dos.csv", "dos.png", "magnetization.json"]
    for filename in generated:
        (output / filename).unlink(missing_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1, "task": task, "success": False,
        "converged_electronic": False, "converged_ionic": False,
        "valid_structure": False, "final_energy_ev": None,
        "final_max_force_ev_angstrom": None, "vasp_version": None,
        "forces_available": False,
        "ionic_steps_count": None, "ionic_iteration_limit_reached": False,
        "ionic_force_limit_ev_angstrom": None,
        "fermi_energy_ev": None,
        "final_energy_ev_per_atom": None,
        "reason": "Results have not been checked yet.", "artifacts": [],
        "scientific_accuracy_validated": False,
    }
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        hybrid = metadata.get("method", {}).get("functional", "PBE") != "PBE"
        expected = Structure.from_file(expected_structure_path).get_sorted_structure()
        _validate_structure(expected)
        xml = output / "vasprun.xml"
        if not xml.is_file() or not xml.stat().st_size:
            raise ValueError("vasprun.xml is missing or empty. The job may have ended before VASP finished.")
        run = Vasprun(xml, parse_potcar_file=False, parse_eigen=task in {"scf", "bands", "dos"}, parse_dos=task in {"scf", "dos"} or (hybrid and task == "bands"), exception_on_bad_xml=True)
        result["vasprun_sha256"] = _sha256(xml)
        result["vasp_version"] = str(run.vasp_version)
        fermi = getattr(run, "efermi", None)
        if fermi is not None and math.isfinite(float(fermi)):
            result["fermi_energy_ev"] = float(fermi)
        if task == "scf" and result["fermi_energy_ev"] is None:
            raise ValueError("SCF Fermi energy is unavailable from the complete XML.")
        result["converged_electronic"] = bool(run.converged_electronic)
        result["converged_ionic"] = bool(run.converged_ionic)
        if task == "relax":
            result["ionic_steps_count"] = len(run.ionic_steps)
            step_limit = int(run.incar.get("NSW", 0))
            result["ionic_iteration_limit_reached"] = step_limit > 0 and len(run.ionic_steps) >= step_limit
            ediffg = float(run.incar.get("EDIFFG", 0))
            if ediffg < 0:
                result["ionic_force_limit_ev_angstrom"] = abs(ediffg)
        validate_method_output(run.incar, metadata, vasp_version=str(run.vasp_version))
        if metadata.get("method"):
            result.update(method=metadata["method"], method_fingerprint=metadata["method_fingerprint"], method_comparison_fingerprint=metadata["method_comparison_fingerprint"])
        if task in {"bands", "dos"} and not hybrid and int(run.incar.get("ICHARG", 0)) != 11:
            raise ValueError("Bands/DOS output did not use the required SCF charge density (ICHARG=11).")
        if hybrid and int(run.incar.get("ICHARG", 0)) >= 10:
            raise ValueError("Hybrid output used a fixed charge density.")
        if hybrid and task == "bands" and len(run.ionic_steps[-1].get("electronic_steps", [])) < HYBRID_BAND_NELMIN:
            raise ValueError("Hybrid bands ended before the minimum electronic iterations.")
        if task == "scf" and int(run.incar.get("ICHARG", 0)) >= 10:
            raise ValueError("SCF output used a fixed charge density instead of self-consistency.")
        if task == "relax" and (int(run.incar.get("NSW", 0)) <= 0 or int(run.incar.get("IBRION", -1)) != 2):
            raise ValueError("Relaxation output does not match the requested ionic relaxation recipe.")
        if task != "relax" and int(run.incar.get("NSW", 0)) != 0:
            raise ValueError("Static task output has NSW != 0.")
        _validate_structure(run.final_structure)
        if not _same_structure(run.initial_structure, expected):
            raise ValueError("XML initial structure differs from the expected input structure.")
        if len(run.final_structure) != len(expected) or run.final_structure.composition != expected.composition:
            raise ValueError("Final structure atom count or composition differs from the input.")
        if task != "relax" and not _same_structure(run.final_structure, expected):
            raise ValueError("A static task unexpectedly changed the crystal structure.")
        result["valid_structure"] = True
        energy = float(run.final_energy)
        if not math.isfinite(energy):
            raise ValueError("Final energy is nonfinite or unavailable.")
        result["final_energy_ev"] = energy
        result["final_energy_ev_per_atom"] = energy / len(expected)
        forces = np.asarray(run.ionic_steps[-1].get("forces", []), dtype=float)
        if forces.size:
            if forces.shape != (len(expected), 3):
                raise ValueError("Final atomic forces are nonfinite or inconsistent with the structure.")
            if not np.isfinite(forces).all():
                if hybrid and task == "bands":
                    result["force_warning"] = "Forces are unavailable for this zero-weight hybrid band calculation; do not use it for force or stress analysis."
                else:
                    raise ValueError("Final atomic forces are nonfinite or inconsistent with the structure.")
            else:
                result["final_max_force_ev_angstrom"] = float(np.max(np.linalg.norm(forces, axis=1)))
                result["forces_available"] = True
        elif task == "relax":
            raise ValueError("Final atomic forces are missing for a relaxation.")
        if not result["converged_electronic"]:
            raise ValueError("Electronic convergence was not reached.")
        if task == "relax":
            if not result["converged_ionic"]:
                raise ValueError("Ionic convergence was not reached.")
            force_limit = float(run.incar.get("EDIFFG", 0))
            if force_limit >= 0 or result["final_max_force_ev_angstrom"] > abs(force_limit):
                result["converged_ionic"] = False
                raise ValueError("Final atomic force exceeds the requested negative EDIFFG criterion.")
        if hybrid and task == "bands":
            check = _hybrid_kpoint_consistency(run)
            result["hybrid_kpoint_consistency"] = check
            if not check["passed"]:
                if not check["equivalent_pair_count"]:
                    raise ValueError("Hybrid bands have no equivalent k-points for the orbital consistency check.")
                raise ValueError(
                    f"Hybrid band orbitals disagree at equivalent k-points: {check['max_energy_deviation_ev']:.4f} eV "
                    f"exceeds {HYBRID_KPOINT_TOLERANCE_EV:.2f} eV. Increase orbital iterations or change the electronic optimizer and rerun."
                )
        if metadata.get("warm_start"):
            from .restart import validate_warm_start
            result["warm_start"] = validate_warm_start(output, run, metadata)
        artifacts = _export_plots(run, output, task)
        if task in {"scf", "bands", "dos"}:
            result["band_gap"] = _band_gap(run, task, metadata)
        if task in {"bands", "dos"}:
            result["plot_settings"] = {"energy_window_ev": list(PLOT_ENERGY_WINDOW_EV), "csv_contains_full_data": True}
        if task == "bands":
            result["energy_reference_ev"] = float(run.efermi if hybrid else metadata["scf_fermi_energy_ev"])
            result["energy_reference_source"] = "hybrid_scf_mesh" if hybrid else "preceding_scf"
            if hybrid:
                result["band_path_convergence_independently_validated"] = False
        result["magnetization"] = _magnetization(output, len(expected), metadata)
        result["magnetic_ground_state_validated"] = False
        if result["magnetization"]["available"] or result["magnetization"]["total_moment"] is not None:
            _write_json(output / "magnetization.json", result["magnetization"])
            artifacts.append("magnetization.json")
        run.final_structure.to(filename=str(output / "final_structure.cif"))
        # CIF stores cell lengths/angles, losing the Cartesian frame needed by SOC.
        Poscar(run.final_structure).write_file(output / "final_structure.vasp")
        result["final_structure_poscar_sha256"] = _sha256(output / "final_structure.vasp")
        result["artifacts"] = ["final_structure.cif", "final_structure.vasp", *artifacts]
        result["success"] = True
        result["reason"] = "The calculation finished and its output passed the checks for this task."
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    _write_json(output / "result.json", result)
    return result
