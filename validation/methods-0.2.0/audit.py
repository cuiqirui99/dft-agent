"""Independent local audit: stdlib XML/OUTCAR parsing, no product imports or remote calls."""
from pathlib import Path
import csv
import sys
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

BASE = Path(sys.argv[1]).expanduser().resolve()
CASES = ["fe-soc", "nio-u", "si-pbe0", "wheel-agent-hse", "fe-seeds", "si-chain", "wheel-agent-soc"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat(value):
    if isinstance(value, (list, tuple)):
        return [part for item in value for part in flat(item)]
    return [value]


def xml_value(element):
    tokens = (element.text or "").split()
    if element.attrib.get("type") == "logical":
        values = [token == "T" for token in tokens]
    elif element.attrib.get("type") == "string":
        values = [" ".join(tokens)]
    else:
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            values = [" ".join(tokens)]
    return values if element.tag == "v" else values[0]


def equivalent(actual, expected):
    a, b = flat(actual), flat(expected)
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if right is None:
            if left is not None:
                return False
            continue
        if isinstance(right, str):
            if str(left).lower() != right.lower():
                return False
        elif isinstance(right, bool):
            if type(left) is not bool or left != right:
                return False
        elif not isinstance(left, (int, float)) or not math.isclose(left, right, abs_tol=1e-8, rel_tol=1e-7):
            return False
    return True


def last_moments(text, atoms, ncl):
    tables = {}
    for match in re.finditer(r"magnetization \(([xyz])\)(.*?)(?=magnetization \([xyz]\)|\Z)", text, re.S):
        if match[1] == "x":
            tables = {}
        number = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
        rows = re.findall(r"^[ \t]*(\d+)[ \t]+(" + number + r")[ \t]+(" + number + r")[ \t]+(" + number + r")[ \t]+(" + number + r")(?:[ \t]+(" + number + r"))?[ \t]*$", match[2], re.M)
        selected = []
        for row in rows:
            if int(row[0]) != len(selected) + 1:
                if selected:
                    break
                continue
            selected.append(float(row[-1] or row[-2]))
            if len(selected) == atoms:
                break
        if len(selected) == atoms:
            tables[match[1]] = selected
    sites = [[tables[key][i] for key in "xyz"] for i in range(atoms)] if ncl and all(key in tables for key in "xyz") else tables.get("x") if not ncl else None
    totals = re.findall(r"number of electron[^\n]*?magnetization[ \t]+([^\n]+)", text)
    total = None
    if totals:
        values = [float(x) for x in totals[-1].split() if re.fullmatch(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", x)]
        if len(values) == (3 if ncl else 1):
            total = values if ncl else values[0]
    return sites, total


def duplicate_orbital_check(points, channels):
    maximum, count, compared = 0., 0, 0
    for i, left_point in enumerate(points):
        for j in range(i + 1, len(points)):
            if not all(abs(a-b-round(a-b)) <= 1e-6 for a, b in zip(left_point, points[j])):
                continue
            count += 1
            for channel in channels:
                left = sorted(float(row.text.split()[0]) for row in channel[i].findall("r"))
                right = sorted(float(row.text.split()[0]) for row in channel[j].findall("r"))
                assert len(left) == len(right)
                compared += len(left)
                maximum = max(maximum, *(abs(a-b) for a, b in zip(left, right)))
    return {"tolerance_ev": .05, "equivalent_pair_count": count, "compared_eigenvalue_count": compared,
            "max_energy_deviation_ev": maximum, "passed": count > 0 and maximum <= .05,
            "scope": "All returned bands per spin, including reciprocal-equivalent mesh/path points."}


def audit(case, stage):
    directory = BASE / case / stage["folder"]
    output = directory / "outputs"
    root = ET.parse(output / "vasprun.xml").getroot()
    calculations = root.findall("calculation")
    final = calculations[-1]
    meta = json.loads((directory / "inputs/metadata.json").read_text())
    result = json.loads((output / "result.json").read_text())
    incar = {item.attrib["name"]: xml_value(item) for item in root.findall("incar/*") if "name" in item.attrib}
    errors = []
    for tag, expected in meta.get("method_incar_expected", {}).items():
        actual = incar.get(tag, False if expected is False else None)
        if not equivalent(actual, expected):
            errors.append(f"XML method mismatch: {tag}")
    atoms = int(root.findtext("atominfo/atoms"))
    energy = float(final.findtext("energy/i[@name='e_0_energy']"))
    if not math.isfinite(energy) or not equivalent(energy, result.get("final_energy_ev")):
        errors.append("Reported energy differs from XML")
    recorded = {line.split()[-1]: line.split()[0] for line in (output / "input_hashes.sha256").read_text().splitlines()}
    input_checks = {name: digest(directory / "inputs" / name) == digest(output / name) == meta["input_sha256"][name] == recorded[name] for name in ("POSCAR", "INCAR", "KPOINTS")}
    if not all(input_checks.values()):
        errors.append("Input identity mismatch")
    text = (output / "OUTCAR").read_text(errors="replace")
    ediff_reached = "aborting loop because EDIFF is reached" in text
    if not ediff_reached:
        errors.append("OUTCAR lacks electronic convergence receipt")
    if len(final.findall("scstep")) >= int(incar.get("NELM", 60)):
        errors.append("Last electronic cycle reached NELM")
    sites, total = last_moments(text, atoms, bool(meta.get("requires_ncl")))
    declared = result.get("magnetization", {})
    if sites is not None and not equivalent(sites, declared.get("site_moments")):
        errors.append("Reported site moments differ from OUTCAR")
    if total is not None and not equivalent(total, declared.get("total_moment")):
        errors.append("Reported total moment differs from OUTCAR")
    forces = [[float(x) for x in value.text.split()] for value in final.findall("varray[@name='forces']/v")]
    force_finite = len(forces) == atoms and all(math.isfinite(x) for row in forces for x in row)
    spectral_force_exception = stage["name"] == "bands" and meta.get("method", {}).get("functional") in {"HSE06", "PBE0"}
    if not force_finite and not spectral_force_exception:
        errors.append("Forces are missing/nonfinite outside hybrid bands")
    report = {
        "case": case, "stage": stage["folder"], "task": stage["name"], "job_id": stage.get("job_id"),
        "reported_status": stage["status"], "reported_success": result.get("success"),
        "xml_complete": True, "vasp_version": root.findtext("generator/i[@name='version']").strip(),
        "atoms": atoms, "ionic_steps": len(calculations), "electronic_steps_last": len(final.findall("scstep")),
        "outcar_ediff_reached": ediff_reached, "energy_ev": energy, "energy_ev_per_atom": energy / atoms,
        "site_moments_mu_B": sites, "total_moment_mu_B": total,
        "moment_basis": "SAXIS spinor" if meta.get("requires_ncl") else "collinear spin axis",
        "input_hashes_verified": input_checks, "method": meta.get("method"),
        "poscar_sha256": meta["input_sha256"]["POSCAR"],
        "potcar_sha256": (output / "potcar_hash.sha256").read_text().split()[0],
        "xml_sha256": digest(output / "vasprun.xml"), "outcar_sha256": digest(output / "OUTCAR"),
        "forces_finite": force_finite, "spectral_force_exception": spectral_force_exception,
        "errors": errors,
    }
    if stage["name"] == "bands":
        points = [[float(x) for x in v.text.split()] for v in root.findall("kpoints/varray[@name='kpointlist']/v")]
        weights = [float(v.text) for v in root.findall("kpoints/varray[@name='weights']/v")]
        eigen = [[[float(x) for x in row.text.split()] for row in node.findall("r")] for node in final.findall("eigenvalues/array/set/set/set")]
        count = meta.get("band_path_count", len(meta["band_path"]["labels"]))
        offset = meta.get("band_path_offset", 0)
        eigen_finite = bool(eigen) and all(math.isfinite(x) for point in eigen for row in point for x in row)
        if not eigen_finite or len(points) != count + offset:
            errors.append("Band eigenvalues or point count invalid")
        if offset and (any(w <= 0 for w in weights[:offset]) or any(w != 0 for w in weights[offset:])):
            errors.append("Hybrid mesh/path weights invalid")
        if not equivalent(points[offset:], meta["band_path"]["reciprocal_fractional"]):
            errors.append("Path differs from frozen coordinates")
        report["spectrum"] = {"points": len(points), "path_points": count, "mesh_points": offset, "eigenvalues_finite": eigen_finite,
                              "bands_per_point": sorted(set(map(len, eigen))), "zero_weight_path_convergence_independently_validated": False if offset else None}
        if offset:
            channels = [channel.findall("set") for channel in final.findall("eigenvalues/array/set/set")]
            check = duplicate_orbital_check(points, channels)
            report["spectrum"]["equivalent_kpoint_check"] = check
            if not check["passed"]:
                errors.append("Hybrid orbitals disagree at equivalent k-points or no equivalent pairs are available")
            declared = result.get("hybrid_kpoint_consistency", {})
            if not declared.get("passed") or declared.get("equivalent_pair_count") != check["equivalent_pair_count"] or not equivalent(declared.get("max_energy_deviation_ev"), check["max_energy_deviation_ev"]):
                errors.append("Reported hybrid consistency check differs from independent XML check")
        if result.get("success"):
            with (output / "bands.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            expected_rows = sum(len(point) for point in eigen) * count // len(points)
            if len(rows) != expected_rows or any(not math.isfinite(float(row["energy_minus_fermi_ev"])) for row in rows):
                errors.append("Band CSV does not cover the finite path spectrum")
            report["spectrum"]["csv_rows"] = len(rows)
    if stage["name"] == "dos":
        blocks = [[[float(x) for x in row.text.split()] for row in block.findall("r")] for block in final.findall("dos/total/array/set/set")]
        finite = bool(blocks) and all(len(row) >= 3 and all(math.isfinite(x) for x in row) for block in blocks for row in block)
        if not finite:
            errors.append("DOS values are missing/nonfinite")
        if any(len(block) != int(incar.get("NEDOS", 301)) for block in blocks):
            errors.append("DOS does not have the requested NEDOS")
        report["spectrum"] = {"dos_finite": finite, "channels": len(blocks), "points_per_channel": sorted(set(map(len, blocks)))}
        if result.get("success"):
            with (output / "dos.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            if not blocks or len(rows) != len(blocks[0]):
                errors.append("DOS CSV row count differs from XML")
            report["spectrum"]["csv_rows"] = len(rows)
    report["audit_passed"] = not errors
    return report


reports, pending, links = [], [], []
for case in CASES:
    state_path = BASE / case / "run.json"
    if not state_path.is_file():
        pending.append({"case": case, "reason": "not prepared"})
        continue
    state = json.loads(state_path.read_text())
    for stage in state["stages"]:
        path = BASE / case / stage["folder"] / "outputs"
        if not (path / "vasprun.xml").is_file() or not (path / "result.json").is_file():
            pending.append({"case": case, "stage": stage["folder"], "status": stage["status"]})
            continue
        try:
            reports.append(audit(case, stage))
        except Exception as error:
            reports.append({"case": case, "stage": stage["folder"], "audit_passed": False, "errors": [type(error).__name__]})
        previous_index = stage.get("structure_from")
        if previous_index is not None:
            previous = state["stages"][previous_index]
            original = ET.parse(BASE / case / previous["folder"] / "outputs/vasprun.xml").getroot().find("structure[@name='finalpos']")
            current = ET.parse(path / "vasprun.xml").getroot().find("structure[@name='initialpos']")
            def coordinates(node, query):
                return [[float(x) for x in item.text.split()] for item in node.findall(query)]
            basis_ok = equivalent(coordinates(original, "crystal/varray[@name='basis']/v"), coordinates(current, "crystal/varray[@name='basis']/v"))
            a, b = coordinates(original, "varray[@name='positions']/v"), coordinates(current, "varray[@name='positions']/v")
            position_ok = len(a) == len(b) and all(abs(x - y - round(x - y)) < 1e-6 for row1, row2 in zip(a, b) for x, y in zip(row1, row2))
            moments_ok = equivalent(previous["metadata"].get("method", {}).get("magmom"), stage["metadata"].get("method", {}).get("magmom"))
            links.append({"case":case,"from":previous["folder"],"to":stage["folder"],"cartesian_basis_preserved":basis_ok,"site_positions_preserved":position_ok,"seed_moments_preserved":moments_ok})
            if not (basis_ok and position_ok and moments_ok):
                reports[-1]["errors"].append("Relaxed structure or moments changed during propagation")
                reports[-1]["audit_passed"] = False
seeds = [report for report in reports if report["case"] == "fe-seeds"]
rank = sorted(seeds, key=lambda row: row.get("energy_ev_per_atom", math.inf))
ranking = {"complete": len(seeds) == 3, "identical_poscar": len({r.get("poscar_sha256") for r in seeds}) == 1,
           "identical_potcar": len({r.get("potcar_sha256") for r in seeds}) == 1,
           "identical_solver": len({r.get("vasp_version") for r in seeds}) == 1,
           "order": [{"stage": r["stage"], "energy_ev_per_atom": r.get("energy_ev_per_atom")} for r in rank],
           "scope": "Trial seeds on one fixed cell; not a global magnetic ground-state proof."}
crosscheck = None
scf_xml = BASE / "wheel-agent-hse/01_scf/outputs/vasprun.xml"
band_xml = BASE / "wheel-agent-hse/02_bands/outputs/vasprun.xml"
if scf_xml.is_file() and band_xml.is_file():
    def mesh_spectrum(path):
        root = ET.parse(path).getroot()
        final = root.findall("calculation")[-1]
        points = [[float(x) for x in row.text.split()] for row in root.findall("kpoints/varray[@name='kpointlist']/v")]
        eigen = [[float(row.text.split()[0]) for row in block.findall("r")] for block in final.findall("eigenvalues/array/set/set/set")]
        return points, eigen, float(final.findtext("energy/i[@name='e_0_energy']"))
    sp, se, energy_s = mesh_spectrum(scf_xml)
    bp, be, energy_b = mesh_spectrum(band_xml)
    offset = json.loads((band_xml.parent / "metadata.json").read_text())["band_path_offset"]
    first6, all_states, matches = [], [], 0
    for i, left in enumerate(sp):
        for j, right in enumerate(bp[:offset]):
            if all(abs(x-y-round(x-y)) < 1e-6 for x, y in zip(left, right)):
                differences = [abs(x-y) for x,y in zip(se[i], be[j])]
                first6.extend(differences[:6])
                all_states.extend(differences)
                matches += 1
    crosscheck = {"matched_points":matches,"total_energy_delta_ev":energy_b-energy_s,
                  "max_first6_eigen_delta_ev":max(first6),"max_all_common_eigen_delta_ev":max(all_states),
                  "scope":"Bounded comparison to the separate SCF mesh. Independent convergence of all zero-weight path points is not established."}
payload = {"schema_version": 1, "scope": "Independent stdlib XML and OUTCAR audit of live method cases; no product parser imports.",
           "reports": reports, "pending": pending, "fe_seed_comparison": ranking, "relaxation_propagation":links, "hse_matched_mesh_crosscheck":crosscheck,
           "rejected_hybrid_diagnostics": "hybrid-orbital-independent-audit.v1.json preserves the original 26-step and 60-step Damped failures and the successful Normal diagnostic.",
           "limitations": ["Small acceptance cases; no material-specific physical convergence claim.", "Equivalent-point consistency and SCF convergence do not prove independent convergence at every zero-weight path point."]}
destination = BASE / "independent-audit.json"
destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(json.dumps({"checked": len(reports), "passed": sum(r["audit_passed"] for r in reports), "pending": pending,
                  "failures": [{"case":r["case"],"stage":r["stage"],"errors":r["errors"]} for r in reports if not r["audit_passed"]]}, indent=2))
