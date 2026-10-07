#!/usr/bin/env python3
"""Independent ASE/OUTCAR/XML audit and frozen-tolerance reference comparison.

This module intentionally imports neither vasp_slurm_agent nor pymatgen. A
same-model pass checks implementation agreement, not experimental accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase.io import read


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value: str):
    value = value.strip().upper()
    if value in {"T", "TRUE", ".TRUE."}:
        return True
    if value in {"F", "FALSE", ".FALSE."}:
        return False
    try:
        return float(value.replace("D", "E"))
    except ValueError:
        return value


def read_incar(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        for field in re.split(r"[!#]", line)[0].split(";"):
            if "=" in field:
                key, value = field.split("=", 1)
                result[key.strip().upper()] = scalar(value)
    return result


def read_mesh(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 4 or int(lines[1]) != 0 or not lines[2].lower().startswith("g"):
        raise ValueError("Reference comparison accepts an automatic Gamma mesh only.")
    return {"grid": [int(x) for x in lines[3].split()], "shift": [float(x) for x in lines[4].split()] if len(lines) > 4 else [0.0, 0.0, 0.0]}


def potcar_hash(directory: Path) -> str:
    value = (directory / "potcar_hash.sha256").read_text().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Missing or malformed licensed-potential hash receipt.")
    return value


def artifact_receipts(directory: Path) -> dict:
    manifest_path = directory / "artifact_manifest.json"
    if not manifest_path.exists():
        return {"verified": False, "reason": "artifact_manifest.json missing", "file_count": 0}
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for name, expected in manifest.items():
        if Path(name).name != name:
            failures.append(name)
            continue
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            failures.append(name)
    required = {"INCAR", "KPOINTS", "POSCAR", "OUTCAR", "vasprun.xml", "CONTCAR", "potcar_hash.sha256", "execution.json"}
    failures += sorted(required - set(manifest))
    return {"verified": not failures, "file_count": len(manifest), "failed_files": failures}


def xml_structure_matches(node: ET.Element | None, atoms) -> bool:
    if node is None:
        return False
    basis = np.array([[float(x) for x in (item.text or "").split()] for item in node.findall("./crystal/varray[@name='basis']/v")])
    positions = np.array([[float(x) for x in (item.text or "").split()] for item in node.findall("./varray[@name='positions']/v")])
    if basis.shape != (3, 3) or positions.shape != (len(atoms), 3):
        return False
    delta = positions - atoms.get_scaled_positions(wrap=False)
    delta -= np.rint(delta)
    return bool(np.allclose(basis, atoms.cell.array, atol=1e-6, rtol=0) and np.allclose(delta, 0, atol=1e-6, rtol=0))


def audit(directory: Path, require_internal_relaxation: bool = False) -> dict:
    directory = Path(directory)
    root = ET.parse(directory / "vasprun.xml").getroot()
    if root.tag != "modeling":
        raise ValueError("Not a complete VASP modeling XML document.")
    xml_images = read(directory / "vasprun.xml", index=":", format="vasp-xml")
    outcar_images = read(directory / "OUTCAR", index=":", format="vasp-out")
    if not xml_images or len(xml_images) != len(outcar_images):
        raise ValueError("Independent XML and OUTCAR trajectories are missing or disagree in length.")
    trajectory = []
    parser_energy_difference = 0.0
    parser_force_difference = 0.0
    for index, (xml_atoms, out_atoms) in enumerate(zip(xml_images, outcar_images)):
        energy = float(out_atoms.get_potential_energy())
        forces = np.asarray(out_atoms.get_forces())
        xml_forces = np.asarray(xml_atoms.get_forces())
        if not math.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError("Nonfinite energy or forces.")
        if xml_atoms.get_chemical_symbols() != out_atoms.get_chemical_symbols():
            raise ValueError("Independent parsers disagree on site identity.")
        parser_energy_difference = max(parser_energy_difference, abs(energy - float(xml_atoms.get_potential_energy())))
        parser_force_difference = max(parser_force_difference, float(np.max(np.linalg.norm(forces - xml_forces, axis=1))))
        trajectory.append({"ionic_step": index + 1, "energy_ev": energy, "max_force_ev_angstrom": float(max(np.linalg.norm(forces, axis=1))), "volume_angstrom3": float(out_atoms.get_volume())})
    incar = read_incar(directory / "INCAR")
    outcar = (directory / "OUTCAR").read_text(errors="strict")
    calculations = root.findall("calculation")
    scsteps = [len(calculation.findall("scstep")) for calculation in calculations]
    parameters = {item.attrib["name"]: scalar(item.text or "") for item in root.findall(".//parameters//i") if "name" in item.attrib}
    nelm = int(parameters.get("NELM", incar.get("NELM", 60)))
    electronic_converged = bool(scsteps) and all(0 < steps < nelm for steps in scsteps) and outcar.count("aborting loop because EDIFF is reached") >= len(calculations)
    task_relax = int(incar.get("NSW", 0)) > 0
    ionic_converged = "reached required accuracy - stopping structural energy minimisation" in outcar if task_relax else None
    finished = "General timing and accounting informations for this job" in outcar
    initial = read(directory / "POSCAR", format="vasp")
    final = read(directory / "CONTCAR", format="vasp")
    same_elements = initial.get_chemical_symbols() == final.get_chemical_symbols() == outcar_images[-1].get_chemical_symbols()
    initial_geometry_matches = xml_structure_matches(root.find("./structure[@name='initialpos']"), initial)
    final_geometry_matches = xml_structure_matches(root.find("./structure[@name='finalpos']"), final)
    fractional_delta = final.get_scaled_positions(wrap=False) - initial.get_scaled_positions(wrap=False)
    fractional_delta -= np.rint(fractional_delta)
    displacement = fractional_delta @ initial.cell.array
    execution = json.loads((directory / "execution.json").read_text())
    exit_code = execution.get("exit_code")
    receipts = artifact_receipts(directory)
    passed = finished and electronic_converged and same_elements and initial_geometry_matches and final_geometry_matches and exit_code == 0 and receipts["verified"] and parser_energy_difference <= 1e-6 and parser_force_difference <= 1e-6
    if task_relax:
        passed = passed and ionic_converged and trajectory[-1]["max_force_ev_angstrom"] <= abs(float(incar["EDIFFG"]))
    report = {
        "schema_version": 1, "method": "ASE_OUTCAR_and_ASE_XML_plus_stdlib_XML_checks", "passed": bool(passed),
        "atom_count": len(initial), "species": initial.get_chemical_symbols(),
        "solver_exit_code": exit_code, "complete_output_footer": finished,
        "electronic_converged": electronic_converged, "electronic_steps_per_ionic_step": scsteps,
        "ionic_converged": ionic_converged, "input_final_site_identity": same_elements,
        "xml_initial_matches_poscar": initial_geometry_matches, "xml_final_matches_contcar": final_geometry_matches,
        "max_parser_energy_difference_ev": parser_energy_difference,
        "max_parser_force_difference_ev_angstrom": parser_force_difference,
        "initial_volume_angstrom3": float(initial.get_volume()),
        "final_volume_angstrom3": float(final.get_volume()),
        "max_fractional_coordinate_motion_in_initial_cell_angstrom": float(np.max(np.linalg.norm(displacement, axis=1))),
        "trajectory": trajectory,
        "potcar_sha256": potcar_hash(directory), "artifact_receipts": receipts,
        "source_sha256": {name: sha256(directory / name) for name in ("INCAR", "KPOINTS", "POSCAR", "OUTCAR", "vasprun.xml", "CONTCAR")},
        "scientific_accuracy_validated": False,
    }
    if require_internal_relaxation:
        gates = {
            "fixed_cell_relaxation": task_relax and int(incar.get("ISIF", 0)) == 2 and np.allclose(initial.cell.array, final.cell.array, atol=1e-6, rtol=0),
            "initial_force_exceeds_frozen_threshold": trajectory[0]["max_force_ev_angstrom"] > 0.03,
            "at_least_two_ionic_steps": len(trajectory) >= 2,
            "actual_internal_coordinate_motion": report["max_fractional_coordinate_motion_in_initial_cell_angstrom"] > 0.01,
            "energy_decreased": trajectory[-1]["energy_ev"] < trajectory[0]["energy_ev"] - 1e-6,
            "final_force_within_frozen_threshold": trajectory[-1]["max_force_ev_angstrom"] <= 0.03,
        }
        report["internal_relaxation_gates"] = {key: bool(value) for key, value in gates.items()}
        report["passed"] = report["passed"] and all(gates.values())
    result_path = directory / "result.json"
    if result_path.exists():
        application = json.loads(result_path.read_text())
        report["application_comparison"] = {
            "reported_success": application.get("success"),
            "energy_difference_ev": abs(float(application["final_energy_ev"]) - trajectory[-1]["energy_ev"]) if application.get("final_energy_ev") is not None else None,
            "max_force_difference_ev_angstrom": abs(float(application["final_max_force_ev_angstrom"]) - trajectory[-1]["max_force_ev_angstrom"]) if application.get("final_max_force_ev_angstrom") is not None else None,
        }
        comparison = report["application_comparison"]
        comparison["agrees"] = bool(comparison["reported_success"] and comparison["energy_difference_ev"] is not None and comparison["energy_difference_ev"] <= 1e-6 and comparison["max_force_difference_ev_angstrom"] is not None and comparison["max_force_difference_ev_angstrom"] <= 1e-6)
        report["passed"] = report["passed"] and comparison["agrees"]
    return report


MODEL_KEYS = ("GGA", "ISPIN", "ENCUT", "EDIFF", "NELM", "PREC", "ALGO", "LREAL", "LASPH", "ISMEAR", "SIGMA", "IBRION", "NSW", "ISIF", "ICHARG", "ISTART")


def compare(agent: Path, reference: Path, protocol_path: Path, mode: str) -> dict:
    if agent.resolve() == reference.resolve():
        raise ValueError("A run cannot serve as its own independently executed reference.")
    protocol = json.loads(protocol_path.read_text())
    a, b = audit(agent), audit(reference)
    a_input, b_input = read(agent / "POSCAR", format="vasp"), read(reference / "POSCAR", format="vasp")
    same_structure = (a_input.get_chemical_symbols() == b_input.get_chemical_symbols() and np.allclose(a_input.cell.array, b_input.cell.array, atol=1e-9, rtol=0) and np.allclose(a_input.get_scaled_positions(), b_input.get_scaled_positions(), atol=1e-9, rtol=0))
    a_incar, b_incar = read_incar(agent / "INCAR"), read_incar(reference / "INCAR")
    model_differences = {key: [a_incar.get(key), b_incar.get(key)] for key in MODEL_KEYS if a_incar.get(key) != b_incar.get(key)}
    mesh_a, mesh_b = read_mesh(agent / "KPOINTS"), read_mesh(reference / "KPOINTS")
    same_potcar = a["potcar_sha256"] == b["potcar_sha256"]
    scf_identity = all(incar.get("NSW") == 0 and incar.get("IBRION") == -1 and incar.get("ICHARG") == 2 for incar in (a_incar, b_incar))
    a_atoms, b_atoms = read(agent / "OUTCAR", format="vasp-out"), read(reference / "OUTCAR", format="vasp-out")
    if len(a_atoms) != len(b_atoms):
        raise ValueError("Cannot compare force vectors with different atom counts.")
    energy_difference = abs(a_atoms.get_potential_energy() - b_atoms.get_potential_energy()) / len(a_atoms)
    force_difference = float(np.max(np.linalg.norm(a_atoms.get_forces() - b_atoms.get_forces(), axis=1)))
    tolerance = protocol["reference_protocol"]
    common = a["passed"] and b["passed"] and same_structure and same_potcar and scf_identity
    if mode == "same-model":
        passed = common and not model_differences and mesh_a == mesh_b and energy_difference <= tolerance["energy_tolerance_ev_per_atom"] and force_difference <= tolerance["force_tolerance_ev_per_angstrom"]
        status = "passed" if passed else "failed"
    else:
        # Sensitivity is a measured difference, not a convergence certificate.
        passed = None
        allowed_changes = set(model_differences) <= {"ENCUT"}
        status = "measured" if common and allowed_changes else "invalid_comparison"
    return {
        "schema_version": 1, "mode": mode, "status": status, "passed": passed,
        "protocol_sha256": sha256(protocol_path), "independent_runs_passed": [a["passed"], b["passed"]],
        "same_input_structure": same_structure, "same_potcar_sha256": same_potcar,
        "both_runs_are_scf": scf_identity,
        "model_differences": model_differences, "kpoint_grids": [mesh_a, mesh_b],
        "energy_difference_ev_per_atom": energy_difference, "max_force_vector_difference_ev_angstrom": force_difference,
        "energy_tolerance_ev_per_atom": tolerance["energy_tolerance_ev_per_atom"] if mode == "same-model" else None,
        "force_tolerance_ev_per_angstrom": tolerance["force_tolerance_ev_per_angstrom"] if mode == "same-model" else None,
        "agent_audit": a, "reference_audit": b,
        "scientific_accuracy_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("audit")
    one.add_argument("--run", type=Path, required=True)
    one.add_argument("--require-internal-relaxation", action="store_true")
    two = commands.add_parser("compare")
    two.add_argument("--agent", type=Path, required=True)
    two.add_argument("--reference", type=Path, required=True)
    two.add_argument("--mode", choices=["same-model", "sensitivity"], default="same-model")
    two.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocol.v1.json")
    for command in (one, two):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.run, args.require_internal_relaxation) if args.command == "audit" else compare(args.agent, args.reference, args.protocol, args.mode)
    except Exception as exc:
        report = {"schema_version": 1, "passed": False, "status": "error", "reason": f"{type(exc).__name__}: {exc}", "scientific_accuracy_validated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "status", "reason") if key in report}))
    if report.get("passed") is False or report.get("status") == "invalid_comparison":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
