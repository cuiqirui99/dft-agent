"""Collect additional outputs and run optional analysis on a completed stage."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import tempfile
import uuid

from .config import ClusterConfig
from .transport import DOWNLOAD_ALLOWLIST, SSHTransport
from .workflow import SLURM_TERMINAL, _command, _lock, _save, _write, read_state


def _stage(root, identifier):
    state = read_state(root)
    stage = next((item for item in state["stages"] if item["folder"] == identifier), None)
    if stage is None or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Choose a saved stage.")
    if not stage.get("job_id") or stage.get("scheduler_state") not in SLURM_TERMINAL:
        raise ValueError("Wait for the stage to finish before collecting more files.")
    return state, stage, state["remote_root"].rstrip("/") + "/" + identifier


def _inventory(transport, remote):
    code = (
        "from pathlib import Path; import json\n"
        f"root=Path({remote!r})\n"
        f"names={sorted(DOWNLOAD_ALLOWLIST)!r}\n"
        "print(json.dumps([{'name':n,'bytes':(root/n).stat().st_size} for n in names "
        "if not (root/n).is_symlink() and (root/n).is_file()]))\n"
    )
    rows = json.loads(_command(transport, "python3 -c " + shlex.quote(code)))
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or row.get("name") not in DOWNLOAD_ALLOWLIST
        or type(row.get("bytes")) is not int or row["bytes"] < 0 for row in rows
    ) or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Invalid remote file list.")
    return rows


def list_remote_outputs(run_dir, stage_folder, transport=None):
    root = Path(run_dir).expanduser().resolve()
    _, _, remote = _stage(root, stage_folder)
    owned = transport is None
    transport = transport or SSHTransport(ClusterConfig.load(root / "config.json"))
    try:
        return _inventory(transport, remote)
    finally:
        if owned:
            transport.close()


def fetch_outputs(run_dir, stage_folder, filenames, transport=None):
    """Retrieve only selected solver outputs, with size and hash verification."""
    names = list(filenames)
    if not names or len(set(names)) != len(names) or any(name not in DOWNLOAD_ALLOWLIST for name in names):
        raise ValueError("Select output files from the list. POTCAR is not exported.")
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state, stage, remote = _stage(root, stage_folder)
        owned = transport is None
        transport = transport or SSHTransport(ClusterConfig.load(root / "config.json"))
        try:
            available = {row["name"] for row in _inventory(transport, remote)}
            if not set(names) <= available:
                raise ValueError("Some selected files are no longer available on the cluster.")
            output = root / stage_folder / "outputs"
            output.mkdir(exist_ok=True)
            manifest_path = output / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            # Keep the accepted raw evidence immutable. New copies must match it.
            from tempfile import TemporaryDirectory
            from .vasp import _sha256
            with TemporaryDirectory(prefix=".extra-", dir=output) as directory:
                staging = Path(directory)
                fetched = transport.download_many(remote, staging, names)
                for name, metadata in fetched.items():
                    if name in manifest and metadata != manifest[name]:
                        raise ValueError(f"Remote {name} changed since collection; saved results were kept.")
                    if _sha256(staging / name) != metadata["sha256"]:
                        raise ValueError(f"Checksum mismatch for {name}.")
                for name in fetched:
                    (staging / name).replace(output / name)
            manifest.update(fetched)
            _write(manifest_path, manifest)
            _save(root, state, "Collected additional files: " + ", ".join(names))
            return fetched
        finally:
            if owned:
                transport.close()


def postprocess_run(run_dir, stage_folder, tasks=None, transport=None, executable=None):
    """Run VASPKIT on a copied stage and retrieve its verified exports and logs."""
    from .vaspkit import RECEIPT_PREFIX, TASKS, remote_command

    root = Path(run_dir).expanduser().resolve()
    config = ClusterConfig.load(root / "config.json")
    owned = transport is None
    transport = transport or SSHTransport(config)
    try:
        state, stage, remote = _stage(root, stage_folder)
        if not (stage.get("result") or {}).get("success") or stage["name"] not in {"bands", "dos"}:
            raise ValueError("VASPKIT analysis needs a completed bands or DOS stage.")
        allowed = ("bands", "projected_bands") if stage["name"] == "bands" else ("total_dos", "projected_dos")
        tasks = list(allowed if tasks is None else tasks)
        if not tasks or len(set(tasks)) != len(tasks) or any(name not in allowed for name in tasks):
            raise ValueError("Choose VASPKIT tasks that match the completed stage.")
        output = root / stage_folder / "outputs"
        manifest_path = output / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        available = {row["name"] for row in _inventory(transport, remote)}
        needed = sorted({name for task in tasks for name in TASKS[task]["required"]}
                        & available - set(manifest))
        if needed:
            fetch_outputs(root, stage_folder, needed, transport)
            manifest = json.loads(manifest_path.read_text())
        with _lock(root):
            state, stage, remote = _stage(root, stage_folder)
            stamp = uuid.uuid4().hex[:12]
            remote_destination = remote + "/dft-agent-vaspkit-" + stamp
            result = stage["result"]
            reference = result.get("energy_reference_ev", result.get("fermi_energy_ev"))
            command = remote_command(remote, remote_destination, tasks,
                                     executable=executable or config.vaspkit_executable,
                                     fermi_energy_ev=reference, band_path=stage["metadata"].get("band_path"))
            setup = "\n".join(config.setup_commands).replace("\r\n", "\n")
            text = _command(transport, "set -e\n" + setup + "\n" + command, timeout=60 + 120 * len(tasks))
            records = [line[len(RECEIPT_PREFIX):] for line in text.splitlines() if line.startswith(RECEIPT_PREFIX)]
            if len(records) != 1:
                raise ValueError("VASPKIT returned no valid report.")
            receipt = json.loads(records[0])
            if not isinstance(receipt, dict) or receipt.get("tool") != "VASPKIT" or not isinstance(receipt.get("tasks"), list):
                raise ValueError("Invalid VASPKIT report.")
            entries = receipt["tasks"]
            if [item.get("name") for item in entries] != tasks:
                raise ValueError("Unexpected VASPKIT tasks.")
            target_parent = output / "vaspkit"
            target_parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".vaspkit-", dir=output) as temporary:
                staging = Path(temporary)
                for entry in entries:
                    name = entry["name"]
                    if entry.get("folder") != name:
                        raise ValueError("Invalid VASPKIT output folder.")
                    for filename, metadata in entry.get("inputs", {}).items():
                        if filename not in manifest or metadata != manifest[filename]:
                            raise ValueError(f"VASPKIT used a changed or unverified {filename}; saved results were kept.")
                        from .vasp import _sha256
                        if not (output / filename).is_file() or _sha256(output / filename) != metadata["sha256"]:
                            raise ValueError(f"Saved {filename} no longer matches the collected evidence.")
                    files = {**entry.get("outputs", {}), **entry.get("logs", {})}
                    for filename, metadata in files.items():
                        from .vaspkit import _OUTPUT
                        logs = {"stdin.txt", "stdout.txt", "stderr.txt", "FERMI_ENERGY.in", "kpoint_mapping.json", "segments.json"}
                        if (not isinstance(filename, str) or (filename not in logs and not _OUTPUT.fullmatch(filename))
                                or Path(filename).name != filename):
                            raise ValueError("Unexpected VASPKIT output file.")
                        local = staging / name / filename
                        actual = transport.download(remote_destination + "/" + name + "/" + filename, local)
                        if actual != metadata:
                            raise ValueError(f"VASPKIT export checksum mismatch: {filename}.")
                _write(staging / "receipt.json", receipt)
                target = target_parent / stamp
                staging.replace(target)
            message = "; ".join(entry["reason"] for entry in entries if entry.get("reason"))
            summary = {"status": receipt.get("status", "unknown"), "message": message,
                       "folder": target.relative_to(root).as_posix(), "tasks": tasks}
            stage["postprocessing"] = summary
            _save(root, state, "VASPKIT: " + summary["status"])
            return {**summary, "receipt": receipt}
    finally:
        if owned:
            transport.close()
