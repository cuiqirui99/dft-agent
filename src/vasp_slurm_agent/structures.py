"""Deterministic edits of an uploaded crystal. No model calls or job submission."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.vasp import Poscar

from .vasp import _validate_structure


MAX_ATOMS = 4096
MAX_OPERATIONS = 32
_SOURCE_INDEX = "_dft_source_index"
_FIELDS = {
    "supercell": {"matrix"},
    "replace_species": {"from", "to"},
    "replace_sites": {"indices", "species"},
    "remove_sites": {"indices"},
    "translate_sites": {"indices", "vector", "cartesian"},
    "set_site": {"index", "fractional_coordinates"},
    "slab": {"miller_index", "min_slab_size", "min_vacuum_size", "termination"},
}


class StructureReviewError(ValueError):
    """A structure request needs a concrete choice or correction."""

    def __init__(self, question: str, choices: list | None = None):
        super().__init__(question)
        self.question = question
        self.choices = choices or []


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cif_structure(text: str) -> Structure:
    parser = CifParser.from_str(text, frac_tolerance=0)
    parsed = parser.parse_structures(primitive=False, on_error="raise")
    if len(parsed) != 1:
        raise StructureReviewError("Choose a CIF containing one crystal structure.")
    structure = parsed[0]
    blocks = [block for block in parser.as_dict().values() if "_atom_site_label" in block]
    if len(blocks) == 1:
        labels = blocks[0]["_atom_site_label"]
        order = {label: index for index, label in enumerate(labels)}
        if len(order) == len(labels) and all(site.label in order for site in structure):
            structure = Structure.from_sites(sorted(structure, key=lambda site: order[site.label]))
    return structure


def _check(structure: Structure) -> None:
    if len(structure) > MAX_ATOMS:
        raise StructureReviewError(f"Use a structure with at most {MAX_ATOMS} atoms.")
    try:
        _validate_structure(structure)
    except ValueError as exc:
        raise StructureReviewError(str(exc)) from None


def _load(source_path) -> tuple[Path, bytes, Structure, str]:
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise StructureReviewError("Upload a CIF or POSCAR first.")
    data = source.read_bytes()
    try:
        text = data.decode("utf-8-sig")
        is_cif = source.suffix.lower() == ".cif" or text.lstrip().startswith(("data_", "# generated using pymatgen"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = _cif_structure(text) if is_cif else Poscar.from_str(text, read_velocities=False).structure
        _check(structure)
        structure.add_site_property(_SOURCE_INDEX, list(range(len(structure))))
        return source, data, structure, "cif" if is_cif else "vasp"
    except StructureReviewError:
        raise
    except Exception:
        raise StructureReviewError("Upload a valid, fully occupied CIF or POSCAR.") from None


def _vector(value, name: str, *, integer=False) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise StructureReviewError(f"Supply three values for {name}.")
    valid = all(type(x) is int for x in value) if integer else all(type(x) in (int, float) and math.isfinite(x) for x in value)
    if not valid:
        raise StructureReviewError(f"Use {'integers' if integer else 'finite numbers'} for {name}.")
    return list(value) if integer else [float(x) for x in value]


def _indices(value, count: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise StructureReviewError("Which zero-based site indices should change?")
    if any(type(index) is not int or not 0 <= index < count for index in value):
        raise StructureReviewError(f"Use site indices from 0 to {count - 1} in the current preview.")
    if len(set(value)) != len(value):
        raise StructureReviewError("List each site index once.")
    return list(value)


def _element(value) -> str:
    try:
        if not isinstance(value, str) or Element(value).symbol != value:
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise StructureReviewError("Specify a chemical element, such as Si or Ge.") from None
    return value


def _matrix(value, count: int) -> list[list[int]]:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(type(x) is int for x in value):
        value = [[value[0], 0, 0], [0, value[1], 0], [0, 0, value[2]]]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise StructureReviewError("Supply three repeats or a 3 by 3 integer supercell matrix.")
    matrix = [_vector(row, "supercell matrix", integer=True) for row in value]
    if any(abs(x) > MAX_ATOMS for row in matrix for x in row):
        raise StructureReviewError("Use a smaller supercell matrix.")
    a, b, c = matrix
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
    if determinant == 0:
        raise StructureReviewError("Use a nonsingular supercell matrix.")
    if count * abs(determinant) > MAX_ATOMS:
        raise StructureReviewError(f"The supercell exceeds {MAX_ATOMS} atoms. Choose smaller repeats.")
    candidates = math.prod(1 + sum(abs(row[column]) for row in matrix) for column in range(3))
    if candidates > MAX_ATOMS * 64:
        raise StructureReviewError("Use a supercell matrix with less shear.")
    return matrix


def _size(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 200:
        raise StructureReviewError(f"Specify {name} in angstrom, greater than 0 and at most 200.")
    return float(value)


def _summary(structure: Structure) -> dict:
    return {"formula": structure.composition.reduced_formula, "number_of_sites": len(structure),
            "lattice_angstrom": structure.lattice.matrix.tolist(), "volume_angstrom3": float(structure.volume),
            "sites": [{"index": index, "element": site.specie.symbol,
                       "fractional_coordinates": site.frac_coords.tolist(),
                       "source_index": int(site.properties[_SOURCE_INDEX])}
                      for index, site in enumerate(structure)]}


def _slab(structure: Structure, operation: dict) -> tuple[Structure, dict, list[dict]]:
    miller = _vector(operation.get("miller_index"), "Miller index", integer=True)
    if not any(miller):
        raise StructureReviewError("Choose a nonzero Miller index.")
    divisor = math.gcd(*miller)
    miller = [x // divisor for x in miller]
    thickness = _size(operation.get("min_slab_size"), "slab thickness")
    vacuum = _size(operation.get("min_vacuum_size"), "vacuum thickness")
    if max(abs(x) for x in miller) > 12 or len(structure) * sum(abs(x) for x in miller)**2 > MAX_ATOMS:
        raise StructureReviewError("Use a smaller input cell or lower Miller index for this slab.")
    generator = SlabGenerator(structure, tuple(miller), thickness, vacuum,
                              center_slab=True, primitive=False, reorient_lattice=False)
    oriented = generator.oriented_unit_cell
    normal = np.cross(oriented.lattice.matrix[0], oriented.lattice.matrix[1])
    height = abs(float(np.dot(oriented.lattice.matrix[2], normal))) / float(np.linalg.norm(normal))
    if len(oriented) * math.ceil(thickness / height) > MAX_ATOMS:
        raise StructureReviewError(f"The slab exceeds {MAX_ATOMS} atoms. Reduce its thickness or input cell.")
    slabs = sorted(generator.get_slabs(symmetrize=False, repair=False), key=lambda slab: float(slab.shift))
    if not slabs:
        raise StructureReviewError("No slab termination was generated. Review the cell and Miller index.")
    choices = []
    for index, slab in enumerate(slabs):
        _check(slab)
        normal = np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])
        normal /= np.linalg.norm(normal)
        positions = np.dot(slab.cart_coords, normal)
        top = sorted({site.specie.symbol for site, z in zip(slab, positions) if positions.max() - z < 0.1})
        bottom = sorted({site.specie.symbol for site, z in zip(slab, positions) if z - positions.min() < 0.1})
        choices.append({"index": index, "shift": float(slab.shift), "formula": slab.composition.reduced_formula,
                        "number_of_sites": len(slab), "top_elements": top, "bottom_elements": bottom})
    selected = operation.get("termination")
    if selected is None and len(slabs) == 1:
        selected = 0
    if selected is None:
        raise StructureReviewError("Which slab termination should be used? Choose its zero-based index.", choices)
    if type(selected) is not int or not 0 <= selected < len(slabs):
        raise StructureReviewError("Choose one of the available slab termination indices.", choices)
    normalized = {"type": "slab", "miller_index": miller, "min_slab_size": thickness,
                  "min_vacuum_size": vacuum, "termination": selected}
    return slabs[selected], normalized, choices


def _preview(source_path, operations: list[dict]) -> tuple[dict, Structure, bytes, str]:
    source, data, structure, source_format = _load(source_path)
    original = _summary(structure)
    if not isinstance(operations, list) or len(operations) > MAX_OPERATIONS:
        raise StructureReviewError(f"Supply a list of at most {MAX_OPERATIONS} structure edits.")
    normalized, messages, terminations = [], [], []
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("type") not in _FIELDS:
            raise StructureReviewError("Choose a supported structure edit.", list(_FIELDS))
        kind = operation["type"]
        if set(operation) - {"type"} - _FIELDS[kind]:
            raise StructureReviewError(f"Remove unsupported fields from the {kind} edit.")
        required = _FIELDS[kind] - ({"termination"} if kind == "slab" else set())
        if any(key not in operation or operation[key] is None for key in required):
            raise StructureReviewError(f"Supply {', '.join(sorted(required))} for the {kind} edit.")
        item = deepcopy(operation)
        if kind == "supercell":
            item["matrix"] = _matrix(item["matrix"], len(structure))
            structure.make_supercell(item["matrix"])
        elif kind == "replace_species":
            old, new = _element(item["from"]), _element(item["to"])
            indices = [i for i, site in enumerate(structure) if site.specie.symbol == old]
            if not indices:
                raise StructureReviewError(f"The current structure has no {old} sites. Choose an element it contains.")
            for index in indices:
                structure.replace(index, new, properties=structure[index].properties)
        elif kind in {"replace_sites", "remove_sites", "translate_sites"}:
            indices = item["indices"] = _indices(item["indices"], len(structure))
            if kind == "replace_sites":
                element = _element(item["species"])
                for index in indices:
                    structure.replace(index, element, properties=structure[index].properties)
            elif kind == "remove_sites":
                if len(indices) == len(structure):
                    raise StructureReviewError("Keep at least one atom in the structure.")
                structure.remove_sites(indices)
            else:
                item["vector"] = _vector(item["vector"], "translation")
                if type(item["cartesian"]) is not bool:
                    raise StructureReviewError("Specify whether the translation is Cartesian (angstrom) or fractional.")
                structure.translate_sites(indices, item["vector"], frac_coords=not item["cartesian"], to_unit_cell=True)
        elif kind == "set_site":
            index = _indices([item["index"]], len(structure))[0]
            item["fractional_coordinates"] = _vector(item["fractional_coordinates"], "fractional coordinates")
            site = structure[index]
            structure.replace(index, site.species, coords=np.mod(item["fractional_coordinates"], 1), properties=site.properties)
        elif kind == "slab":
            structure, item, choices = _slab(structure, item)
            terminations.append({"operation_index": len(normalized), "selected": item["termination"], "choices": choices})
            messages.append("A geometric slab is not a validated surface model. Check termination, polarity, vacuum and fixed-cell relaxation.")
        _check(structure)
        normalized.append(item)
    if any(item["type"] in {"replace_species", "replace_sites", "remove_sites"} for item in normalized):
        messages.append("Composition changed. Review charge, magnetic moments, POTCAR choices and relaxation before calculating.")
    if source_format == "cif":
        messages.append("CIF defines lattice lengths and angles; it does not retain an arbitrary Cartesian frame.")
    for index, site in enumerate(structure):
        site.label = f"{site.specie.symbol}{index}"
    output = _summary(structure)
    fingerprint = _digest(json.dumps(output, sort_keys=True, allow_nan=False).encode())
    preview = {"status": "ready", "source_sha256": _digest(data), "input": original, "output": output,
               "output_sha256": fingerprint, "operations": normalized, "warnings": list(dict.fromkeys(messages)),
               "slab_terminations": terminations,
               "provenance": {"engine": "pymatgen", "version": version("pymatgen"),
                   "reference": "https://pymatgen.org/pymatgen.core.html#pymatgen.core.surface.SlabGenerator",
                   "checks": ["ordered periodic structure", "finite cell and coordinates", "minimum site separation 0.5 angstrom", "atom limit"]}}
    if source.read_bytes() != data:
        raise StructureReviewError("The source changed during the preview. Review it again.")
    return preview, structure, data, source_format


def preview_structure(source_path, operations: list[dict]) -> dict:
    """Apply all requested edits to the original input, without writing files."""
    try:
        return _preview(source_path, operations)[0]
    except StructureReviewError:
        raise
    except Exception as exc:
        raise StructureReviewError(f"The structure edit could not be applied ({type(exc).__name__}). Review its settings.") from None


def _check_export(actual: Structure, expected: Structure, *, cif=False):
    _check(actual)
    if [x.specie.symbol for x in actual] != [x.specie.symbol for x in expected]:
        raise StructureReviewError("Export changed the atom order. No files were saved.")
    left = actual.lattice.metric_tensor if cif else actual.lattice.matrix
    right = expected.lattice.metric_tensor if cif else expected.lattice.matrix
    if not np.allclose(left, right, rtol=1e-9, atol=1e-8):
        raise StructureReviewError("Export changed the lattice. No files were saved.")
    difference = actual.frac_coords - expected.frac_coords
    if not np.allclose(difference - np.rint(difference), 0, atol=1e-9):
        raise StructureReviewError("Export changed site positions. No files were saved.")


def apply_structure_plan(source_path, output_dir, plan: dict) -> dict:
    """Save a source-bound, reviewed edit without overwriting an existing result."""
    if not isinstance(plan, dict) or plan.get("status") != "ready":
        raise StructureReviewError("Complete and review the structure plan before applying it.")
    if plan.get("output_format", "both") not in {"cif", "poscar", "both"}:
        raise StructureReviewError("Choose CIF, POSCAR or both formats.")
    preview, structure, data, source_format = _preview(source_path, plan.get("operations"))
    if plan.get("source_sha256") != preview["source_sha256"]:
        raise StructureReviewError("The source differs from the reviewed plan. Preview it again.")
    prior = plan.get("preview") or plan
    if prior.get("output_sha256") and prior["output_sha256"] != preview["output_sha256"]:
        raise StructureReviewError("The edited structure differs from the preview. Review it again.")
    target = Path(output_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if target == source or target in source.parents:
        raise StructureReviewError("Choose an output folder separate from the original source.")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise StructureReviewError("Choose a new or empty output folder.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dft-structure-", dir=target.parent))
    names = {"source": f"source.{source_format}", "cif": "edited.cif", "poscar": "POSCAR", "metadata": "structure.json"}
    try:
        (temporary / names["source"]).write_bytes(data)
        CifWriter(structure, symprec=None, significant_figures=12, refine_struct=False).write_file(temporary / names["cif"])
        Poscar(structure, sort_structure=False).write_file(temporary / names["poscar"], significant_figures=16)
        _check_export(Poscar.from_file(temporary / names["poscar"], read_velocities=False).structure, structure)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _check_export(_cif_structure((temporary / names["cif"]).read_text()), structure, cif=True)
        metadata = {**preview, "output_format": plan.get("output_format", "both"),
                    "files": {key: {"name": name, "sha256": _digest((temporary / name).read_bytes())}
                              for key, name in names.items() if key != "metadata"}}
        (temporary / names["metadata"]).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if source.read_bytes() != data:
            raise StructureReviewError("The source changed while saving. Preview it again.")
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
        return {**preview, "output_format": metadata["output_format"],
                "files": {key: str(target / name) for key, name in names.items()}}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
