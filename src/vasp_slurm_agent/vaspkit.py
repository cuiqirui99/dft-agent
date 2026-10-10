"""Run an installed VASPKIT in separate copies of completed VASP outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time


REFERENCE = "https://doi.org/10.1016/j.cpc.2021.108033"
DOCUMENTATION = "https://vaspkit.com/tutorials.html"
RECEIPT_PREFIX = "__DFT_AGENT_VASPKIT__"
TASKS = {
    "bands": {"id": 211, "required": ("INCAR", "POSCAR", "KPOINTS", "EIGENVAL", "DOSCAR"),
              "primary": r"BAND(?:_UP|_DW)?\.dat"},
    "projected_bands": {"id": 213, "required": ("INCAR", "POSCAR", "KPOINTS", "EIGENVAL", "DOSCAR", "PROCAR"),
                        "primary": r"PBAND_[A-Za-z0-9_]+\.dat"},
    "total_dos": {"id": 111, "required": ("INCAR", "POSCAR", "DOSCAR"),
                  "primary": r"TDOS(?:_UP|_DW)?\.dat"},
    "projected_dos": {"id": 113, "required": ("INCAR", "POSCAR", "DOSCAR", "PROCAR"),
                      "primary": r"PDOS_[A-Za-z0-9_]+\.dat"},
}
_OUTPUT = re.compile(r"(?:BAND(?:_[A-Za-z0-9_]+)?\.dat|PBAND_[A-Za-z0-9_]+\.dat|"
                     r"I?TDOS(?:_[A-Za-z0-9_]+)?\.dat|I?PDOS_[A-Za-z0-9_]+\.dat|"
                     r"REFORMATTED_BAND(?:_[A-Za-z0-9_]+)?\.dat|KLINES\.dat|KLABELS|BAND_GAP)")
_CONFIG = "VASP5 .TRUE.\nSET_FERMI_ENERGY_ZERO .TRUE.\nPLOT_MATPLOTLIB .FALSE.\nADVANCED_USER .FALSE.\n"


def _write_text(path, text, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="\n") as target:
        target.write(text)


def _hash(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"size": size, "sha256": digest.hexdigest()}


def _incar(folder):
    result = {}
    path = folder / "INCAR"
    if path.is_file() and not path.is_symlink():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = re.split(r"[#!]", line, maxsplit=1)[0]
            for part in line.split(";"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key.strip().upper()] = value.strip().upper()
    return result


def _path_metadata(folder, band_path=None):
    if band_path is None and (folder / "metadata.json").is_file():
        band_path = json.loads((folder / "metadata.json").read_text(encoding="utf-8")).get("band_path")
    if not isinstance(band_path, dict):
        raise ValueError("Explicit band exports require the original band-path metadata.")
    points, labels, segments = (band_path.get(key) for key in ("reciprocal_fractional", "labels", "segments"))
    if not isinstance(points, list) or not points or not isinstance(labels, list) or len(labels) != len(points):
        raise ValueError("Band-path coordinates or labels are missing.")
    if any(not isinstance(point, list) or len(point) != 3 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point) for point in points):
        raise ValueError("Band-path coordinates are invalid.")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Band-path segments are missing.")
    if any(not isinstance(item, (list, tuple)) or len(item) != 2 or any(not isinstance(index, int) for index in item)
           or not 0 <= item[0] < item[1] - 1 < len(points) for item in segments):
        raise ValueError("Band-path segments are invalid.")
    covered = {index for start, stop in segments for index in range(start, stop)}
    if covered != set(range(len(points))):
        raise ValueError("Band-path segments do not cover every k-point.")
    for start, stop in segments:
        for index in range(start, stop):
            ratio = (index - start) / (stop - start - 1)
            expected = [points[start][axis] + ratio * (points[stop - 1][axis] - points[start][axis]) for axis in range(3)]
            if any(abs(points[index][axis] - expected[axis]) > 1e-7 for axis in range(3)):
                raise ValueError("VASPKIT requires evenly sampled straight path segments.")
    if any(not isinstance(label, str) or any(char in label for char in "\r\n\x00") for label in labels):
        raise ValueError("Band-path labels are invalid.")
    return band_path


def available_tasks(source_dir, band_path=None):
    """Return eligibility and a reason for each supported task."""
    folder = Path(source_dir).expanduser().resolve()
    incar = _incar(folder)
    noncollinear = any(incar.get(key, "").strip(".") in {"TRUE", "T"}
                      for key in ("LSORBIT", "LNONCOLLINEAR"))
    hybrid = incar.get("LHFCALC", "").strip(".") in {"TRUE", "T"}
    result = {}
    for name, task in TASKS.items():
        missing = [item for item in task["required"] if not (folder / item).is_file()
                   or (folder / item).is_symlink() or not (folder / item).stat().st_size]
        reason = "Missing input: " + ", ".join(missing) if missing else ""
        if not reason and noncollinear:
            reason = "SOC and noncollinear VASPKIT exports are not validated. Use the built-in results."
        if not reason and "bands" in name:
            lines = (folder / "KPOINTS").read_text(encoding="utf-8", errors="replace").splitlines()
            if hybrid:
                reason = "VASPKIT band exports require non-hybrid data. Use the built-in hybrid results."
            elif len(lines) < 3:
                reason = "KPOINTS is incomplete."
            elif not lines[2].strip().lower().startswith("l"):
                try:
                    _path_metadata(folder, band_path)
                except (OSError, ValueError) as exc:
                    reason = str(exc)
        if not reason and name.startswith("projected") and incar.get("LORBIT") not in {"10", "11"}:
            reason = "Projected data require a completed calculation with LORBIT = 10 or 11."
        result[name] = {"available": not reason, "reason": reason, "task_id": task["id"]}
    return result


def _explicit_band_copy(folder, band_path, segment):
    """Copy one original segment to line mode without interpolating values."""
    lines = (folder / "KPOINTS").read_text(encoding="utf-8").splitlines()
    if lines[2].strip().lower().startswith("l"):
        return
    if not lines[2].strip().lower().startswith("r"):
        raise ValueError("Explicit VASPKIT band exports require reciprocal coordinates.")
    path = _path_metadata(folder, band_path)
    points, segments, labels = path["reciprocal_fractional"], path["segments"], path["labels"]
    if int(lines[1].strip()) != len(points) or len(lines[3:]) < len(points):
        raise ValueError("KPOINTS and band-path metadata have different lengths.")
    for line, point in zip(lines[3:], points):
        fields = line.split()
        if len(fields) < 4 or any(abs(float(fields[index]) - point[index]) > 1e-7 for index in range(3)):
            raise ValueError("KPOINTS and band-path coordinates do not match.")
    eigen = (folder / "EIGENVAL").read_text(encoding="utf-8").splitlines(keepends=True)
    if len(eigen) < 6:
        raise ValueError("EIGENVAL is incomplete.")
    counts = eigen[5].split()
    number, bands = int(counts[1]), int(counts[2])
    if number != len(points) or bands < 1:
        raise ValueError("EIGENVAL and band-path metadata have different lengths.")
    blocks, cursor = [], 6
    for point in points:
        while cursor < len(eigen) and not eigen[cursor].strip():
            cursor += 1
        block = eigen[cursor:cursor + bands + 1]
        if len(block) != bands + 1:
            raise ValueError("EIGENVAL has an incomplete k-point block.")
        coords = block[0].split()
        if len(coords) < 4 or any(abs(float(coords[index]) - point[index]) > 1e-7 for index in range(3)):
            raise ValueError("EIGENVAL and band-path coordinates do not match.")
        blocks.append(block)
        cursor += bands + 1
    if any(line.strip() for line in eigen[cursor:]):
        raise ValueError("EIGENVAL has unexpected trailing data.")
    start, stop = segments[segment]
    mapping = list(range(start, stop))
    prepared = ["Original sampled segment; no interpolation\n", str(len(mapping)) + "\n", "Line-mode\n", "Reciprocal\n"]
    for index in (start, stop - 1):
        prepared.append(" ".join(format(value, ".14g") for value in points[index]) + " ! " + (labels[index] or "-") + "\n")
    _write_text(folder / "KPOINTS", "".join(prepared))
    counts[1] = str(len(mapping))
    eigen[5] = " ".join(counts) + "\n"
    with (folder / "EIGENVAL").open("w", encoding="utf-8", newline="\n") as target:
        target.writelines(eigen[:6])
        for index in mapping:
            target.write("\n")
            target.writelines(blocks[index])
    procar = folder / "PROCAR"
    if procar.is_file():
        text = procar.read_text(encoding="utf-8")
        headers = list(re.finditer(r"(?m)^.*# of k-points:\s*\d+.*# of bands:.*$", text))
        if not headers:
            raise ValueError("PROCAR has no k-point header.")
        pieces = [text[:headers[0].start()]]
        for ih, header in enumerate(headers):
            body = text[header.end():headers[ih + 1].start() if ih + 1 < len(headers) else len(text)]
            starts = list(re.finditer(r"(?m)^[ \t]*k-point[ \t]+\d+[ \t]*:", body))
            if len(starts) != len(points):
                raise ValueError("PROCAR and band-path metadata have different lengths.")
            pieces.append(re.sub(r"(# of k-points:)(\s*\d+)",
                                 lambda match: match.group(1) + str(len(mapping)).rjust(len(match.group(2))), header.group()))
            pieces.append(body[:starts[0].start()])
            for new_index, old_index in enumerate(mapping, 1):
                block = body[starts[old_index].start():starts[old_index + 1].start() if old_index + 1 < len(starts) else len(body)]
                coords = block.split(":", 1)[1].split()[:3]
                if len(coords) != 3 or any(abs(float(coords[index]) - points[old_index][index]) > 1e-7 for index in range(3)):
                    raise ValueError("PROCAR and band-path coordinates do not match.")
                pieces.append(re.sub(r"(k-point)(\s+\d+)",
                                     lambda match: match.group(1) + str(new_index).rjust(len(match.group(2))), block, count=1))
        _write_text(procar, "".join(pieces).rstrip("\n") + "\n\n")
    _write_text(folder / "kpoint_mapping.json", json.dumps({
        "method": "original_segment", "interpolation": "none", "original_kpoint_count": len(points),
        "original_kpoint_indices_zero_based": mapping, "segments": segments,
    }, indent=2) + "\n")


def _band_blocks(path):
    blocks = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.lstrip().startswith("# Band-Index"):
                blocks.append([])
            elif line.strip() and not line.lstrip().startswith("#"):
                if not blocks:
                    raise ValueError("VASPKIT band output has no band index.")
                fields = line.split()
                if len(fields) < 2 or not all(math.isfinite(float(value)) for value in fields):
                    raise ValueError("VASPKIT returned invalid band data.")
                blocks[-1].append(fields)
    if not blocks or any(not block for block in blocks):
        raise ValueError("VASPKIT returned empty band data.")
    # VASPKIT reverses alternate bands for continuous plotting.
    for block in blocks:
        if float(block[0][0]) > float(block[-1][0]):
            block.reverse()
    return blocks


def _merge_segments(folder, children, path, primary_pattern):
    file_sets = [{item.name for item in child.iterdir() if re.fullmatch(primary_pattern, item.name)
                  and item.is_file() and not item.is_symlink()} for child in children]
    if not file_sets[0] or any(names != file_sets[0] for names in file_sets):
        raise ValueError("VASPKIT did not return matching files for every path segment.")
    distances = path.get("distance_inv_angstrom")
    if not isinstance(distances, list) or len(distances) != len(path["labels"]) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in distances):
        raise ValueError("Original band-path distances are required to join segment exports.")
    mapping = [index for start, stop in path["segments"] for index in range(start, stop)]
    for name in sorted(file_sets[0]):
        all_blocks = [_band_blocks(child / name) for child in children]
        count = len(all_blocks[0])
        if any(len(blocks) != count for blocks in all_blocks):
            raise ValueError("VASPKIT returned different band counts for path segments.")
        with (folder / name).open("w", encoding="utf-8", newline="\n") as target:
            header = (children[0] / name).read_text(encoding="utf-8").splitlines()[0]
            target.write(header + "\n# NKPTS & NBANDS: " + str(len(mapping)) + " " + str(count) + "\n")
            for ib in range(count):
                target.write("# Band-Index: " + str(ib + 1) + "\n")
                for blocks, (start, stop) in zip(all_blocks, path["segments"]):
                    block = blocks[ib]
                    if len(block) != stop - start:
                        raise ValueError("VASPKIT changed the number of sampled k-points.")
                    for index, row in zip(range(start, stop), block):
                        target.write(format(distances[index], ".10g") + " " + " ".join(row[1:]) + "\n")
                    target.write("\n")
    labels = ["# K-Label  K-Coordinate (1/Angstrom)\n"]
    for index, label in enumerate(path["labels"]):
        if label:
            labels.append(label + " " + format(distances[index], ".10g") + "\n")
    _write_text(folder / "KLABELS", "".join(labels))
    _write_text(folder / "kpoint_mapping.json", json.dumps({
        "method": "original_segments", "interpolation": "none", "original_kpoint_count": len(distances),
        "original_kpoint_indices_zero_based": mapping, "segments": path["segments"],
        "distance_source": "original_band_path", "band_order": "forward_with_segment_breaks",
    }, indent=2) + "\n")


def _run_segments(executable, folder, stdin, timeout, environment, path, task):
    children, receipts = [], []
    deadline = time.monotonic() + timeout
    try:
        for index in range(len(path["segments"])):
            child = folder / ("segment_" + str(index + 1).zfill(3))
            child.mkdir()
            children.append(child)
            for filename in task["required"]:
                shutil.copyfile(folder / filename, child / filename)
            if (folder / "FERMI_ENERGY.in").exists():
                shutil.copyfile(folder / "FERMI_ENERGY.in", child / "FERMI_ENERGY.in")
            _explicit_band_copy(child, path, index)
            prepared = {filename: _hash(child / filename) for filename in task["required"]}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("VASPKIT timed out.")
            code, error = _run(executable, child, stdin, remaining, environment)
            receipts.append({"segment": index, "prepared_inputs": prepared, "returncode": code})
            if error:
                raise ValueError(error)
        _merge_segments(folder, children, path, task["primary"])
        return 0, ""
    finally:
        for logname in ("stdout.txt", "stderr.txt"):
            with (folder / logname).open("w", encoding="utf-8", newline="\n") as target:
                for child in children:
                    if (child / logname).is_file():
                        target.write("# " + child.name + "\n" + (child / logname).read_text(encoding="utf-8", errors="replace") + "\n")
        _write_text(folder / "segments.json", json.dumps(receipts, indent=2) + "\n")


def _run(executable, folder, stdin, timeout, environment):
    """Bound execution and retain logs even after a timeout."""
    kwargs = {"start_new_session": True} if os.name != "nt" else {}
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        with subprocess.Popen([executable], cwd=folder, stdin=subprocess.PIPE,
                              stdout=stdout, stderr=stderr, env=environment, **kwargs) as process:
            try:
                process.communicate(stdin.encode("ascii"), timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.communicate(timeout=5)
                return process.returncode, "VASPKIT timed out."
    return process.returncode, "" if process.returncode == 0 else "VASPKIT returned an error."


def _write_fermi_reference(path, fermi_energy_ev):
    if fermi_energy_ev is None:
        return
    _write_text(path.parent / "FERMI_ENERGY.in",
        "# Fermi energy reference (eV)\n" + format(fermi_energy_ev, ".15g") + "\n",
        encoding="ascii")


def _numeric_data(path):
    rows = 0
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            try:
                if len(fields) < 2 or not all(math.isfinite(float(value.replace("D", "E"))) for value in fields):
                    return False
            except ValueError:
                return False
            rows += 1
    return rows > 0


def run_vaspkit(source_dir, destination_dir, tasks=("bands",), *, executable="vaspkit", timeout=120,
                fermi_energy_ev=None, band_path=None):
    """Export named tasks into a new directory and return its receipt.

    ``destination_dir`` must not already exist. Each task has a separate folder;
    ``outputs`` lists only checked result files, with size and SHA-256.
    """
    if isinstance(tasks, str) or not tasks or len(set(tasks)) != len(tasks) or any(name not in TASKS for name in tasks):
        raise ValueError("Choose unique VASPKIT tasks: " + ", ".join(TASKS))
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError("VASPKIT timeout must be between 1 and 600 seconds per task.")
    if fermi_energy_ev is not None and (isinstance(fermi_energy_ev, bool) or not isinstance(fermi_energy_ev, (int, float))
                                      or not math.isfinite(fermi_energy_ev)):
        raise ValueError("The Fermi-energy reference must be finite.")
    if not isinstance(executable, str) or not executable.strip() or any(char in executable for char in "\r\n\x00"):
        raise ValueError("Enter a VASPKIT executable name or path.")
    source = Path(source_dir).expanduser().resolve(strict=True)
    destination = Path(destination_dir).expanduser().resolve()
    if destination == source or destination in source.parents:
        raise ValueError("VASPKIT exports need a separate output directory.")
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "tool": "VASPKIT", "citation": REFERENCE,
               "documentation": DOCUMENTATION, "status": "unsupported", "tasks": [],
               "fermi_energy_ev": fermi_energy_ev,
               "energy_reference": "provided_fermi" if fermi_energy_ev is not None else "DOSCAR_fermi",
               "warnings": [] if fermi_energy_ev is not None else ["Energy zero is read from DOSCAR. For bands, use the SCF Fermi energy."],
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    binary = shutil.which(os.path.expanduser(executable))
    eligibility = available_tasks(source, band_path)
    if binary is None:
        receipt.update(status="unavailable", reason="VASPKIT was not found. Install it or set its executable path.")
    else:
        binary = str(Path(binary).resolve())
        receipt["executable"] = {"name": Path(binary).name, **_hash(binary)}
    for name in tasks:
        entry = {"name": name, "task_id": TASKS[name]["id"], "folder": name,
                 "status": "unsupported", "outputs": {}, "inputs": {}}
        receipt["tasks"].append(entry)
        if not eligibility[name]["available"]:
            entry["reason"] = eligibility[name]["reason"]
            continue
        if binary is None:
            entry.update(status="unavailable", reason=receipt["reason"])
            continue
        folder = destination / name
        folder.mkdir()
        try:
            for filename in TASKS[name]["required"]:
                shutil.copyfile(source / filename, folder / filename)
                entry["inputs"][filename] = _hash(folder / filename)
            _write_fermi_reference(folder / "DOSCAR", fermi_energy_ev)
            entry["prepared_inputs"] = {filename: _hash(folder / filename) for filename in TASKS[name]["required"]}
            home = folder / ".home"
            home.mkdir()
            _write_text(home / ".vaspkit", _CONFIG, encoding="ascii")
            entry["settings"] = _CONFIG
            environment = os.environ.copy()
            environment.update(HOME=str(home), USERPROFILE=str(home), OMP_NUM_THREADS="1")
            stdin = str(entry["task_id"]) + "\n0\n"
            _write_text(folder / "stdin.txt", stdin, encoding="ascii")
            entry["stdin"] = stdin
            explicit = "bands" in name and not (folder / "KPOINTS").read_text(encoding="utf-8").splitlines()[2].strip().lower().startswith("l")
            if explicit:
                code, error = _run_segments(binary, folder, stdin, timeout, environment, _path_metadata(source, band_path), TASKS[name])
            else:
                code, error = _run(binary, folder, stdin, timeout, environment)
            entry["returncode"] = code
            log = (folder / "stdout.txt").read_text(encoding="utf-8", errors="replace")
            version = re.search(r"VASPKIT(?:\s+Version|\s+Standard\s+Edition)?\s*[:vV]?\s*(\d+\.\d+(?:\.\d+)?)", log, re.I)
            entry["version"] = version.group(1) if version else None
            for item in sorted(folder.iterdir()):
                if _OUTPUT.fullmatch(item.name) and item.is_file() and not item.is_symlink() and item.stat().st_size:
                    entry["outputs"][item.name] = _hash(item)
            primary_files = [filename for filename in entry["outputs"] if re.fullmatch(TASKS[name]["primary"], filename)]
            primary = bool(primary_files) and all(_numeric_data(folder / filename) for filename in primary_files)
            entry["status"] = "complete" if not error and primary else "failed"
            if entry["status"] == "failed":
                entry["reason"] = error or "VASPKIT did not produce the expected data files. See stdout.txt and stderr.txt."
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            entry.update(status="failed", reason=str(exc))
        entry["logs"] = {filename: _hash(folder / filename) for filename in ("stdin.txt", "stdout.txt", "stderr.txt", "FERMI_ENERGY.in", "kpoint_mapping.json", "segments.json")
                         if (folder / filename).is_file() and not (folder / filename).is_symlink()}
    if binary is not None:
        statuses = {entry["status"] for entry in receipt["tasks"]}
        receipt["status"] = "complete" if statuses == {"complete"} else "partial" if "complete" in statuses else "failed" if "failed" in statuses else "unsupported"
    receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_text(destination / "receipt.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


def remote_command(source_dir, destination_dir, tasks, *, executable="vaspkit", timeout=120, fermi_energy_ev=None, band_path=None):
    """Use the same adapter on a cluster with Python 3; no package install needed."""
    source = Path(__file__).read_text(encoding="utf-8")
    arguments = {"source_dir": str(source_dir), "destination_dir": str(destination_dir), "tasks": list(tasks),
                 "executable": executable, "timeout": timeout, "fermi_energy_ev": fermi_energy_ev, "band_path": band_path}
    script = source + "\nprint(" + repr(RECEIPT_PREFIX) + " + json.dumps(run_vaspkit(**" + repr(arguments) + ")))\n"
    return "python3 -c " + shlex.quote(script)
