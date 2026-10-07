from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid
import zipfile

from .config import ClusterConfig
from .transport import SSHTransport, TransportError
from .vasp import prepare_inputs, analyze_outputs

TERMINAL = {"succeeded", "needs_attention", "failed", "cancelled"}
MAX_REMOTE_FAILURES = 3
SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
FILES = ("INCAR", "KPOINTS", "POSCAR", "OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR", "EIGENVAL", "DOSCAR", "slurm.out", "slurm.err", "execution.json", "input_hashes.sha256", "potcar_hash.sha256", "potcar_titles.txt")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def _lock(root):
    with (Path(root) / ".state.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def read_state(run_dir):
    return json.loads((Path(run_dir) / "run.json").read_text())


def _save(root, state, event=None):
    state["updated_at"] = utc_now()
    if event:
        state["history"].append({"at": state["updated_at"], "event": event})
    _write(Path(root) / "run.json", state)
    return state


def prepare_run(structure_path, run_dir, config, task="relax", parameters=None):
    if task not in {"relax", "scf", "bands", "dos"}:
        raise ValueError("Supported tasks: relax, scf, bands, dos")
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run.json").exists():
        raise FileExistsError("This run already exists. Resume it or choose a new run directory.")
    parameters = dict(parameters or {})
    run_id = "vsa-" + uuid.uuid4().hex[:16]
    sequence = ["scf", task] if task in {"bands", "dos"} else [task]
    stages = []
    for i, name in enumerate(sequence):
        folder = f"{i+1:02d}_{name}"
        metadata = prepare_inputs(Path(structure_path), root / folder / "inputs", name, parameters, config.potcar_symbols)
        stages.append({"name": name, "folder": folder, "job_name": f"{run_id}-{name}", "status": "planned", "job_id": None, "metadata": metadata, "staged": False, "result": None})
    config.save(root / "config.json")
    state = {"schema_version": 1, "run_id": run_id, "task": task, "formula": stages[0]["metadata"].get("formula", ""), "status": "planned", "parameters": parameters, "current_stage": 0, "stages": stages, "created_at": utc_now(), "history": [], "last_error": None, "cancel_requested": False, "remote_root": config.remote_root.rstrip("/") + "/" + run_id}
    state["config_sha256"] = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    return _save(root, state, "Prepared locally; awaiting user submission")


def _command(transport, command, timeout=60):
    result = transport.run(command, timeout=timeout)
    if result.returncode:
        raise TransportError((result.stderr or result.stdout or f"Remote exit {result.returncode}")[-3000:])
    return result.stdout.strip()


def _check_config(root, state):
    if state.get("config_sha256") and hashlib.sha256((root / "config.json").read_bytes()).hexdigest() != state["config_sha256"]:
        raise ValueError("Run cluster configuration changed. Restore the approved snapshot to reconnect safely.")


def _script(config, stage, previous):
    labels = stage["metadata"]["potcar_labels"]
    paths = []
    for label in labels:
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", label):
            raise ValueError("Unsafe POTCAR label")
        paths.append(shlex.quote(config.potcar_root.rstrip("/") + "/" + label + "/POTCAR"))
    account = f"#SBATCH --account={config.account}\n" if config.account else ""
    dependency = ""
    if previous:
        dependency = f"test -s {shlex.quote(previous + '/CHGCAR')}\ncp {shlex.quote(previous + '/CHGCAR')} CHGCAR\n"
    setup = "\n".join(config.setup_commands)
    return f'''#!/bin/bash
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks={config.tasks}
#SBATCH --time={config.walltime}
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
{account}set -e
{setup}
{dependency}cat {' '.join(paths)} > POTCAR
test -s POTCAR
sha256sum POSCAR INCAR KPOINTS > input_hashes.sha256
sha256sum POTCAR > potcar_hash.sha256
grep TITEL POTCAR > potcar_titles.txt
python3 - <<'PY'
import json,os,datetime
json.dump({{"job_id":os.environ.get("SLURM_JOB_ID"),"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tasks":os.environ.get("SLURM_NTASKS")}},open("execution.json","w"))
PY
set +e
{config.vasp_command}
code=$?
set -e
python3 - "$code" <<'PY'
import json,sys,datetime
p=json.load(open("execution.json"));p.update(exit_code=int(sys.argv[1]),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(p,open("execution.json","w"))
PY
exit "$code"
'''


def _submit(root, state, config, transport):
    index = state["current_stage"]
    stage = state["stages"][index]
    remote = state["remote_root"] + "/" + stage["folder"]
    if stage["status"] == "planned":
        stage["status"] = state["status"] = "submitting"
        _save(root, state, f"Submitting {stage['name']}")
    if not stage["staged"]:
        inputs = root / stage["folder"] / "inputs"
        for name in ("POSCAR", "INCAR", "KPOINTS"):
            if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != stage["metadata"]["input_sha256"][name]:
                raise ValueError(f"Prepared {name} changed. Prepare and review a new run before submitting.")
        labels = stage["metadata"]["potcar_labels"]
        checks = ["command -v sbatch >/dev/null", "command -v squeue >/dev/null", "command -v sacct >/dev/null", "command -v python3 >/dev/null"]
        checks += ["test -s " + shlex.quote(config.potcar_root.rstrip("/") + "/" + label + "/POTCAR") for label in labels]
        _command(transport, " && ".join(checks))
        _command(transport, "mkdir -p " + shlex.quote(remote))
        previous = state["remote_root"] + "/" + state["stages"][index-1]["folder"] if index else None
        (inputs / "submit.sh").write_text(_script(config, stage, previous))
        files = [inputs / name for name in ("POSCAR", "INCAR", "KPOINTS", "submit.sh")]
        files.append(Path(__file__).with_name("dispatch.py"))
        if hasattr(transport, "upload_many"):
            transport.upload_many(files, remote)
        else:
            for file in files:
                transport.upload(file, remote + "/" + file.name)
        stage["staged"] = True
        _save(root, state, f"Uploaded {stage['name']} inputs")
    # Remote program owns the durable intent/receipt under a file lock. Replaying
    # this call after a lost SSH response returns/reconciles the same job.
    result = _command(transport, f"python3 {shlex.quote(remote + '/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])}", timeout=100)
    receipt = json.loads(result.splitlines()[-1])
    if not str(receipt["job_id"]).isdigit():
        raise ValueError("Invalid Slurm job identity")
    stage["job_id"] = str(receipt["job_id"])
    stage["status"] = state["status"] = "queued"
    state["last_error"] = None
    state["remote_failures"] = 0
    stage["submitted_at"] = utc_now()
    return _save(root, state, f"Slurm job {stage['job_id']} acknowledged")


def _poll(transport, stage):
    job = stage["job_id"]
    if not str(job).isdigit():
        raise ValueError("Invalid persisted job identity")
    live = _command(transport, f"squeue -h -j {job} -o '%T'")
    if live:
        return live.splitlines()[0].strip().split()[0]
    accounting = _command(transport, f"sacct -n -X -P -j {job} --format=JobIDRaw,State,ExitCode")
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job:
            state = fields[1].split()[0].rstrip("+")
            if state == "COMPLETED" and fields[2] != "0:0":
                return "FAILED"
            return state
    return "UNKNOWN"  # Scheduler/accounting lag is never treated as job death.


def _collect(root, state, transport, stage):
    remote = state["remote_root"] + "/" + stage["folder"]
    output = root / stage["folder"] / "outputs"
    output.mkdir(exist_ok=True)
    script = "from pathlib import Path;import json;root=Path(" + repr(remote) + ");print(json.dumps([n for n in " + repr(list(FILES)) + " if (root/n).is_file()]))"
    names = json.loads(_command(transport, "python3 -c " + shlex.quote(script)))
    if any(name not in FILES for name in names):
        raise ValueError("Unexpected remote artifact")
    if hasattr(transport, "download_many"):
        manifest = transport.download_many(remote, output, names)
    else:
        manifest = {name: transport.download(remote + "/" + name, output / name) for name in names}
    _write(output / "artifact_manifest.json", manifest)
    metadata_file = root / stage["folder"] / "inputs" / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if stage["name"] == "bands" and state["current_stage"]:
            previous_result = state["stages"][state["current_stage"] - 1].get("result") or {}
            metadata["scf_fermi_energy_ev"] = previous_result.get("fermi_energy_ev")
        _write(output / "metadata.json", metadata)
    expected = root / stage["folder"] / "inputs" / "POSCAR"
    result = analyze_outputs(output, stage["name"], expected)
    checks = {}
    hashes_file = output / "input_hashes.sha256"
    if hashes_file.exists():
        recorded = {line.split()[-1]: line.split()[0] for line in hashes_file.read_text().splitlines() if len(line.split()) >= 2}
        for name in ("INCAR", "KPOINTS", "POSCAR"):
            digest = stage["metadata"]["input_sha256"][name]
            checks[name] = recorded.get(name) == digest
    result["input_identity_verified"] = len(checks) == 3 and all(checks.values())
    result["scheduler_state"] = stage.get("scheduler_state")
    if not result["input_identity_verified"]:
        result.update(success=False, reason="Input checksums are missing or do not match the approved local inputs")
    if stage.get("scheduler_state") != "COMPLETED":
        result.update(success=False, reason=f"Slurm ended in {stage.get('scheduler_state')}; any structure is partial")
    _write(output / "result.json", result)
    stage["result"] = result
    stage["status"] = "succeeded" if result.get("success") else "needs_attention"
    if result.get("success") and state["current_stage"] + 1 < len(state["stages"]):
        state["current_stage"] += 1
        state["status"] = "planned"
    else:
        state["status"] = stage["status"]
    return _save(root, state, f"Collected {stage['name']}: {result.get('reason', stage['status'])}")


def advance(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL:
            return state
        config = ClusterConfig.load(root / "config.json")
        owned = transport is None
        transport = transport or SSHTransport(config)
        stage = state["stages"][state["current_stage"]]
        try:
            _check_config(root, state)
            if stage["status"] in {"planned", "submitting"} and not state["cancel_requested"]:
                return _submit(root, state, config, transport)
            scheduler = _poll(transport, stage)
            stage["scheduler_state"] = scheduler
            state["last_error"] = None
            if state["cancel_requested"] and scheduler not in SLURM_TERMINAL:
                cancellation = transport.run("scancel " + stage["job_id"])
                if cancellation.returncode:
                    # Jobs can leave the live queue between poll and scancel.
                    # Keep the request and reconcile accounting next tick.
                    state["last_error"] = cancellation.stderr or cancellation.stdout
                    state["cancel_failures"] = state.get("cancel_failures", 0) + 1
                    if state["cancel_failures"] >= MAX_REMOTE_FAILURES:
                        state["status"] = "needs_attention"
                        return _save(root, state, "Cancellation not acknowledged; original job identity preserved")
                else:
                    state["cancel_failures"] = 0
            if scheduler == "CANCELLED":
                stage["status"] = state["status"] = "cancelled"
                state["remote_failures"] = 0
                return _save(root, state, "Slurm confirmed cancellation")
            if scheduler in SLURM_TERMINAL:
                stage["status"] = state["status"] = "collecting"
                _save(root, state)
                result = _collect(root, state, transport, stage)
                result["remote_failures"] = 0
                return _save(root, result)
            if scheduler == "UNKNOWN":
                state["last_error"] = "Waiting for Slurm/accounting to report this job; no resubmission performed"
            else:
                stage["status"] = state["status"] = "queued" if scheduler == "PENDING" else "running"
            state["remote_failures"] = 0
            return _save(root, state)
        except TransportError as exc:
            state["last_error"] = str(exc)
            state["remote_failures"] = state.get("remote_failures", 0) + 1
            # A lost reply is not evidence that dispatch failed. Keep identity and
            # stage so the next tick can reconnect/reconcile the same request.
            if state["remote_failures"] >= MAX_REMOTE_FAILURES or "outcome uncertain" in str(exc) or "did not acknowledge" in str(exc):
                state["status"] = "needs_attention"
            return _save(root, state, "Remote operation unavailable; identity preserved")
        except (ValueError, OSError, KeyError) as exc:
            state["status"] = "needs_attention"
            state["last_error"] = str(exc)
            return _save(root, state, "Stopped for inspection")
        finally:
            if owned and hasattr(transport, "close"):
                transport.close()


def resume(run_dir, transport=None):
    """Reconnect to the same calculation; never change inputs or repeat sbatch.

    A completed, unconverged calculation can be collected again, but running a
    corrected calculation requires preparing and reviewing a separate run.
    """
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] not in {"needs_attention", "failed"}:
            return state
        _check_config(root, state)
        stage = state["stages"][state["current_stage"]]
        owned = transport is None
        try:
            if not stage.get("job_id") and stage.get("staged"):
                config = ClusterConfig.load(root / "config.json")
                transport = transport or SSHTransport(config)
                remote = state["remote_root"] + "/" + stage["folder"]
                script = remote + "/reconcile.py"
                transport.upload(Path(__file__).with_name("dispatch.py"), script)
                reply = _command(transport, f"python3 {shlex.quote(script)} {shlex.quote(remote)} {shlex.quote(stage['job_name'])} --reconcile-only", timeout=100)
                receipt = json.loads(reply.splitlines()[-1])
                if not str(receipt["job_id"]).isdigit():
                    raise ValueError("Invalid reconciled job identity")
                stage["job_id"] = str(receipt["job_id"])
            if stage.get("job_id"):
                stage["status"] = state["status"] = "queued"
            elif state.get("cancel_requested"):
                stage["status"] = state["status"] = "cancelled"
            else:
                # staged=False means dispatch has never been called.
                stage["status"] = state["status"] = "planned"
            state["last_error"] = None
            state["remote_failures"] = 0
            state["cancel_failures"] = 0
            return _save(root, state, "Resumed existing run; calculation inputs unchanged")
        except (TransportError, ValueError, OSError, KeyError) as exc:
            state["last_error"] = str(exc)
            return _save(root, state, "Reconciliation unavailable; no new job submitted")
        finally:
            if owned and transport is not None:
                transport.close()


def cancel(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in {"succeeded", "cancelled"}:
            return state
        state["cancel_requested"] = True
        stage = state["stages"][state["current_stage"]]
        if not stage.get("job_id") and not stage.get("staged"):
            stage["status"] = state["status"] = "cancelled"
        elif not stage.get("job_id"):
            state["status"] = "needs_attention"
            state["last_error"] = "Submission acknowledgement missing. Reconcile remote job before cancellation; do not start a new run."
        elif state["status"] in {"needs_attention", "failed"}:
            stage["status"] = state["status"] = "queued"
            state["remote_failures"] = state["cancel_failures"] = 0
        _save(root, state, "Cancellation requested")
    if stage.get("job_id"):
        return advance(root, transport)
    return state


def watch(run_dir, interval=20):
    root = Path(run_dir).expanduser().resolve()
    with (root / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return read_state(root)
        _write(root / "worker.json", {"pid": os.getpid(), "started_at": utc_now()})
        transport = SSHTransport(ClusterConfig.load(root / "config.json"))
        try:
            while True:
                state = advance(root, transport)
                if state["status"] in TERMINAL:
                    bundle_run(root)
                    return state
                time.sleep(interval)
        finally:
            transport.close()


def start_worker(run_dir):
    root = Path(run_dir).expanduser().resolve()
    # Worker lock, rather than an unverified PID file, owns singleton execution.
    with (root / "worker.log").open("ab") as stream:
        child = subprocess.Popen([sys.executable, "-m", "vasp_slurm_agent.cli", "watch", str(root)], stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
    return child.pid


def bundle_run(run_dir):
    root = Path(run_dir).expanduser().resolve()
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "run.json", "run.json")
        for stage in read_state(root)["stages"]:
            base = root / stage["folder"]
            for folder in (base / "inputs", base / "outputs"):
                if folder.exists():
                    for file in sorted(folder.iterdir()):
                        if file.is_file() and file.name not in {"POTCAR", "CHGCAR", "WAVECAR"}:
                            archive.write(file, str(file.relative_to(root)))
    os.replace(target.with_suffix(".tmp"), target)
    return target
