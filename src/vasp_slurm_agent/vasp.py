"""Explicit, nonmagnetic PBE VASP recipes and analysis of complete solver output.

Adapted from AI_AGNET's deterministic DFT input and validation routines.  These
defaults are starting parameters, not material-specific convergence guarantees.
Licensed POTCAR data is resolved on the execution host, never distributed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Vasprun


TASKS = frozenset({"relax", "scf", "bands", "dos"})
PLOT_ENERGY_WINDOW_EV = (-15.0, 10.0)
DEFAULTS = {
    "encut": 520.0,
    "ediff": 1e-5,
    "ediffg": -0.03,
    "nsw": 100,
    "mesh": [4, 4, 4],
    "ismear": 0,
    "sigma": 0.05,
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
        if "mesh" in parameters and parameters["mesh"] != parameters["kpoint_grid"]:
            raise ValueError("mesh and kpoint_grid disagree.")
        parameters["mesh"] = parameters.pop("kpoint_grid")
    unknown = set(parameters) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported VASP parameters: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **parameters}
    for key in ("encut", "ediff", "sigma"):
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
    if type(result["ismear"]) is not int or result["ismear"] not in {-5, -1, 0, 1, 2}:
        raise ValueError("ismear must be one of -5, -1, 0, 1, 2.")
    mesh = result["mesh"]
    if not isinstance(mesh, (list, tuple)) or len(mesh) != 3 or any(type(n) is not int or n <= 0 for n in mesh):
        raise ValueError("mesh must contain three positive integers.")
    result["mesh"] = list(mesh)
    return result


def prepare_inputs(
    structure_path: str | Path,
    destination: str | Path,
    task: str,
    parameters: dict[str, Any],
    potcar_symbols: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write deterministic inputs without changing the supplied crystal cell.

    Bands and DOS require a compatible preceding SCF CHGCAR.  The workflow
    engine is responsible for that dependency and must not run them standalone.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
    source = Path(structure_path)
    settings = _parameters(parameters)
    structure = Structure.from_file(source)
    _validate_structure(structure)
    # Group the atoms consistently so that species, counts, and POTCAR order agree.
    structure = structure.get_sorted_structure()
    poscar = Poscar(structure)
    symbols = potcar_symbols or {}
    labels = [symbols.get(symbol, symbol) for symbol in poscar.site_symbols]
    if any(not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) for label in labels):
        raise ValueError("POTCAR labels must be simple symbols such as Si or Na_pv.")

    incar_data: dict[str, Any] = {
        "SYSTEM": f"vasp-slurm-agent {structure.composition.reduced_formula} {task}",
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

    metadata: dict[str, Any] = {
        "schema_version": 1, "task": task,
        "formula": structure.composition.reduced_formula,
        "number_of_atoms": len(structure), "parameters": settings,
        "source_sha256": _sha256(source), "potcar_labels": labels,
        "potcar_elements": poscar.site_symbols,
        "requires_chgcar": task in {"bands", "dos"},
        "scope": "Nonmagnetic PBE. Check convergence for your material.",
    }
    if task == "bands":
        import seekpath

        # The orig_cell API maps the standardized path back through both the
        # primitive/conventional basis change AND Cartesian cell rotation.
        # Merely converting standardized Cartesian points to our reciprocal
        # basis is wrong for a rotated cell, including many CIF imports.
        path = seekpath.get_explicit_k_path_orig_cell(
            (structure.lattice.matrix, structure.frac_coords, [site.specie.Z for site in structure]),
            reference_distance=1.0 / settings["line_density"],
        )
        fractional = np.asarray(path["explicit_kpoints_rel"])
        cartesian = structure.lattice.reciprocal_lattice.get_cartesian_coords(fractional)
        point_labels = path["explicit_kpoints_labels"]
        if not len(fractional) or not np.isfinite(fractional).all():
            raise ValueError("Could not construct a finite high-symmetry path.")
        kpoints = Kpoints(
            comment="High-symmetry path in the actual POSCAR reciprocal basis",
            num_kpts=len(fractional), style=Kpoints.supported_modes.Reciprocal,
            kpts=fractional.tolist(), kpts_weights=[1.0] * len(fractional),
            labels=point_labels,
        )
        metadata["band_path"] = {
            "method": "seekpath_hpkot_orig_cell",
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
    poscar.write_file(destination / "POSCAR")
    Incar(incar_data).write_file(destination / "INCAR")
    kpoints.write_file(destination / "KPOINTS")
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
            kpts = np.asarray(run.actual_kpoints, dtype=float)
            cartesian = run.final_structure.lattice.reciprocal_lattice.get_cartesian_coords(kpts)
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            fermi = metadata.get("scf_fermi_energy_ev")
            if fermi is None or not math.isfinite(float(fermi)):
                raise ValueError("A finite preceding SCF Fermi energy is required for band analysis.")
            fermi = float(fermi)
            path_metadata = metadata.get("band_path", {})
            if not path_metadata:
                raise ValueError("Frozen band path metadata.json is required for band analysis.")
            labels = path_metadata.get("labels", [])
            if len(labels) != len(kpts):
                raise ValueError("Band path metadata does not match output k-points.")
            planned = np.asarray(path_metadata.get("reciprocal_fractional", []))
            if planned.shape != kpts.shape or not np.allclose(planned, kpts, atol=1e-6, rtol=0):
                raise ValueError("Output k-points differ from the frozen band path.")
            distance = np.asarray(path_metadata.get("distance_inv_angstrom", []))
            segments = path_metadata.get("segments", [])
            if distance.shape != (len(kpts),) or not np.isfinite(distance).all() or not segments:
                raise ValueError("Frozen band path is missing its distance or segment definition.")
            rows = []
            for spin, eigenvalues in run.eigenvalues.items():
                values = np.asarray(eigenvalues)
                if len(values) != len(kpts) or not np.isfinite(values).all():
                    raise ValueError("Band eigenvalues do not match the finite k-point path.")
                energies = values[:, :, 0] - fermi
                for start, stop in segments:
                    ax.plot(distance[start:stop], energies[start:stop], color="tab:blue", linewidth=0.6)
                for ik, point in enumerate(kpts):
                    for ib in range(values.shape[1]):
                        rows.append([ik, distance[ik], *point, labels[ik], int(spin), ib + 1, energies[ik, ib], values[ik, ib, 1]])
            name = "bands"
            _write_csv(output / f"{name}.csv", ["kpoint", "distance_inv_angstrom", "kx", "ky", "kz", "label", "spin", "band", "energy_minus_fermi_ev", "occupation"], rows)
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
        return [f"{name}.csv", f"{name}.png"]
    finally:
        plt.close(fig)


def analyze_outputs(output_dir: str | Path, task: str, expected_structure_path: str | Path) -> dict[str, Any]:
    """Parse complete XML and refuse incomplete, unconverged or mismatched output.

    A successful result is numerical run acceptance, not a claim of scientific
    accuracy: a separate convergence/validation campaign is still required.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # A fresh refusal must not leave a previous successful analysis looking current.
    generated = ["final_structure.cif", "relax_energy.csv", "relax_energy.png", "bands.csv", "bands.png", "dos.csv", "dos.png"]
    for filename in generated:
        (output / filename).unlink(missing_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1, "task": task, "success": False,
        "converged_electronic": False, "converged_ionic": False,
        "valid_structure": False, "final_energy_ev": None,
        "final_max_force_ev_angstrom": None, "vasp_version": None,
        "fermi_energy_ev": None,
        "reason": "Results have not been checked yet.", "artifacts": [],
        "scientific_accuracy_validated": False,
    }
    try:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        expected = Structure.from_file(expected_structure_path).get_sorted_structure()
        _validate_structure(expected)
        xml = output / "vasprun.xml"
        if not xml.is_file() or not xml.stat().st_size:
            raise ValueError("vasprun.xml is missing or empty. The job may have ended before VASP finished.")
        run = Vasprun(xml, parse_potcar_file=False, parse_eigen=task == "bands", parse_dos=task in {"scf", "dos"}, exception_on_bad_xml=True)
        result["vasprun_sha256"] = _sha256(xml)
        result["vasp_version"] = str(run.vasp_version)
        fermi = getattr(run, "efermi", None)
        if fermi is not None and math.isfinite(float(fermi)):
            result["fermi_energy_ev"] = float(fermi)
        if task == "scf" and result["fermi_energy_ev"] is None:
            raise ValueError("SCF Fermi energy is unavailable from the complete XML.")
        result["converged_electronic"] = bool(run.converged_electronic)
        result["converged_ionic"] = bool(run.converged_ionic)
        if task in {"bands", "dos"} and int(run.incar.get("ICHARG", 0)) != 11:
            raise ValueError("Bands/DOS output did not use the required SCF charge density (ICHARG=11).")
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
        forces = np.asarray(run.ionic_steps[-1].get("forces", []), dtype=float)
        if forces.size:
            if forces.shape != (len(expected), 3) or not np.isfinite(forces).all():
                raise ValueError("Final atomic forces are nonfinite or inconsistent with the structure.")
            result["final_max_force_ev_angstrom"] = float(np.max(np.linalg.norm(forces, axis=1)))
        elif task == "relax":
            raise ValueError("Final atomic forces are missing for a relaxation.")
        if not result["converged_electronic"]:
            raise ValueError("Electronic convergence was not reached.")
        if task == "relax":
            if not result["converged_ionic"]:
                raise ValueError("Ionic convergence was not reached.")
            force_limit = float(run.incar.get("EDIFFG", 0))
            if force_limit >= 0 or result["final_max_force_ev_angstrom"] > abs(force_limit):
                raise ValueError("Final atomic force exceeds the requested negative EDIFFG criterion.")
        artifacts = _export_plots(run, output, task)
        if task in {"bands", "dos"}:
            result["plot_settings"] = {"energy_window_ev": list(PLOT_ENERGY_WINDOW_EV), "csv_contains_full_data": True}
        if task == "bands":
            result["energy_reference_ev"] = float(json.loads((output / "metadata.json").read_text())["scf_fermi_energy_ev"])
            result["energy_reference_source"] = "preceding_scf"
        run.final_structure.to(filename=str(output / "final_structure.cif"))
        result["artifacts"] = ["final_structure.cif", *artifacts]
        result["success"] = True
        result["reason"] = "The calculation finished and its output passed the checks for this task."
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    _write_json(output / "result.json", result)
    return result
