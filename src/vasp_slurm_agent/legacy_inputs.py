"""Reproduce inputs approved before versioned input policies were introduced."""

from copy import deepcopy
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Incar

from .vasp import _sha256, _write_json, prepare_inputs


LEGACY_DEFAULTS = {
    "encut": 520.0, "ediff": 1e-5, "ediffg": -0.03, "nsw": 100,
    "mesh": [4, 4, 4], "ismear": 0, "sigma": 0.05, "cell_relax": False,
    "nelm": 120, "line_density": 20, "nedos": 2001,
}


def legacy_parameters(parameters):
    values = deepcopy(parameters)
    values.pop("kspacing", None)
    values.pop("electronic_type", None)
    if "kpoint_grid" in values and "mesh" not in values:
        values["mesh"] = values.pop("kpoint_grid")
    if any(key in values and values[key] is None for key in ("mesh", "ismear", "sigma")):
        raise ValueError("Legacy input recipes require concrete mesh and smearing settings.")
    return {**deepcopy(LEGACY_DEFAULTS), **values}


def prepare_legacy_inputs(source, destination, task, parameters, potcar_symbols=None):
    """Apply the fixed v0.4.0 policy, never settings copied from mutable metadata."""
    symbols = {site.specie.symbol for site in Structure.from_file(source)}
    potentials = {symbol: (potcar_symbols or {}).get(symbol, symbol) for symbol in symbols}
    metadata = prepare_inputs(source, destination, task, legacy_parameters(parameters), potentials)
    path = Path(destination)
    incar = Incar.from_file(path / "INCAR")
    method = metadata["method"]
    method_symmetry = (method["magmom"] is not None or metadata["requires_ncl"]
                       or (method["functional"] != "PBE" and task == "bands"))
    if method_symmetry:
        incar["ISYM"] = metadata["method_incar_expected"]["ISYM"] = -1
    else:
        metadata["method_incar_expected"].pop("ISYM", None)
        if task == "bands":
            incar["ISYM"] = 0
        else:
            incar.pop("ISYM", None)
    incar.write_file(path / "INCAR")
    metadata["input_sha256"]["INCAR"] = _sha256(path / "INCAR")
    for key in ("kspacing", "electronic_type"):
        metadata["parameters"].pop(key, None)
    metadata.pop("numerical_choices", None)
    metadata.pop("potential_choices", None)
    _write_json(path / "metadata.json", metadata)
    return metadata
