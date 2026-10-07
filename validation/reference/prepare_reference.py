#!/usr/bin/env python3
"""Author VASP reference inputs independently of the application and pymatgen.

Only ASE reads the already frozen POSCAR; INCAR/KPOINTS/POSCAR are handwritten
here. No solver, network, scheduler, or licensed POTCAR data is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read


ELEMENT_ORDER = {"Si": ["Si"], "Al": ["Al"], "MgO": ["Mg", "O"]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, destination: Path, material: str, task: str, variant: str, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["parameters"]
    atoms = read(source, format="vasp")
    symbols = atoms.get_chemical_symbols()
    if set(symbols) != set(ELEMENT_ORDER[material]):
        raise ValueError("Structure elements differ from the declared reference material.")
    if not np.isfinite(atoms.positions).all() or atoms.get_volume() <= 0:
        raise ValueError("Invalid input structure.")
    # Preserve the already frozen input order so force-vector comparison is direct.
    groups = list(dict.fromkeys(symbols))
    if symbols != [s for group in groups for s in symbols if s == group]:
        raise ValueError("Source POSCAR species must be grouped.")
    displacement = None
    if task == "displaced-relax":
        if material not in {"Si", "MgO"} or len(atoms) < 2:
            raise ValueError("Internal displacement acceptance is defined only for Si/MgO.")
        displacement = {"atom_index_zero_based": len(atoms) - 1, "cartesian_angstrom": [0.10, 0.0, 0.0]}
        atoms.positions[-1] += np.array(displacement["cartesian_angstrom"])
    encut = 520.0 if variant in {"higher-cutoff", "combined"} else float(settings["encut"])
    mesh = [6, 6, 6] if variant in {"denser-mesh", "combined"} else settings["kpoint_grid"]
    relax = task == "displaced-relax"
    # Explicitly authored model: match the frozen nonmagnetic PBE SCF baseline.
    incar = {
        "SYSTEM": f"Independent reference {material} {task} {variant}",
        "GGA": "PE", "ISPIN": 1, "ENCUT": encut,
        "EDIFF": settings["ediff"], "NELM": 120,
        "PREC": "Accurate", "ALGO": "Normal", "LREAL": ".FALSE.",
        "LASPH": ".TRUE.", "ISMEAR": settings["ismear"], "SIGMA": settings["sigma"],
        "IBRION": 2 if relax else -1, "NSW": settings["nsw"] if relax else 0,
        "ISIF": 2, "ICHARG": 2, "ISTART": 0,
        "LCHARG": ".FALSE." if relax else ".TRUE.", "LWAVE": ".FALSE.",
    }
    if relax:
        incar["EDIFFG"] = settings["ediffg"]
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("OUTCAR", "vasprun.xml", "POTCAR")):
        raise ValueError("Refusing to prepare over an executed or licensed-data directory.")
    poscar = [f"Independent reference {material}", "1.0"]
    poscar += [" ".join(f"{value:.16f}" for value in vector) for vector in atoms.cell]
    poscar += [" ".join(groups), " ".join(str(symbols.count(group)) for group in groups), "Direct"]
    poscar += [" ".join(f"{value:.16f}" for value in vector) for vector in atoms.get_scaled_positions(wrap=False)]
    (destination / "POSCAR").write_text("\n".join(poscar) + "\n")
    (destination / "INCAR").write_text("\n".join(f"{key} = {value}" for key, value in incar.items()) + "\n")
    (destination / "KPOINTS").write_text("Independent Gamma mesh\n0\nGamma\n" + " ".join(map(str, mesh)) + "\n0 0 0\n")
    receipt = {
        "schema_version": 1, "method": "independently_authored_plain_text_recipe_ase_structure_reader",
        "material": material, "task": task, "variant": variant,
        "source_poscar_sha256": sha256(source), "protocol_sha256": sha256(protocol_path),
        "potcar_elements": groups, "requires_identical_remote_potcar_sha256": True,
        "encut_ev": encut, "kpoint_grid": mesh, "displacement": displacement,
        "inputs": {name: {"sha256": sha256(destination / name), "size": (destination / name).stat().st_size} for name in ("INCAR", "KPOINTS", "POSCAR")},
        "execution_status": "not_run", "scientific_accuracy_validated": False,
    }
    (destination / "reference_input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-poscar", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--material", choices=ELEMENT_ORDER, required=True)
    parser.add_argument("--task", choices=["scf", "displaced-relax"], default="scf")
    parser.add_argument("--variant", choices=["baseline", "higher-cutoff", "denser-mesh", "combined"], default="baseline")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "protocol.v1.json")
    args = parser.parse_args()
    receipt = prepare(args.source_poscar, args.destination, args.material, args.task, args.variant, args.protocol)
    print(json.dumps({"material": receipt["material"], "task": receipt["task"], "variant": receipt["variant"], "execution_status": "not_run"}))


if __name__ == "__main__":
    main()
