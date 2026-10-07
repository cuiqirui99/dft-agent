"""Remote submission transaction, uploaded as plain Python (stdlib only).

Once intent exists, an uncertain submission is reconciled, never blindly retried.
Uses one lock per stage and records a durable receipt even if the client disconnects.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def save(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def dispatch(root: Path, name: str, reconcile_only: bool = False):
    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = root / "job.json"
        intent = root / "submit_intent.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value.get("job_name") != name or not str(value.get("job_id", "")).isdigit():
                raise RuntimeError("Stored submission receipt does not match this run; no job action performed")
            return value
        if intent.exists():
            original = json.loads(intent.read_text())
            if original.get("job_name") != name:
                raise RuntimeError("Stored submission intent does not match this run; no job action performed")
            jobs = set()
            commands = [
                ["squeue", "-h", "--name", name, "-o", "%i|%j"],
                ["sacct", "-n", "-X", "-P", "--starttime", original["date"], "--name", name, "--format=JobIDRaw,JobName%100"],
            ]
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError("Cannot reconcile submission: Slurm query failed")
                for line in result.stdout.splitlines():
                    fields = line.strip().split("|")
                    if len(fields) >= 2 and fields[1].strip() == name and fields[0].isdigit():
                        jobs.add(fields[0])
            if len(jobs) != 1:
                raise RuntimeError("Submission outcome uncertain; no automatic resubmission. Inspect queue/accounting using job name " + name)
            value = {"job_id": jobs.pop(), "job_name": name, "reconciled": True}
            save(receipt, value)
            return value
        if reconcile_only:
            raise RuntimeError("No recorded submission exists; reconciliation cannot submit a new job")
        save(intent, {"job_name": name, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "created_at": datetime.now(timezone.utc).isoformat()})
        result = subprocess.run(["sbatch", "--parsable", "--job-name=" + name, "submit.sh"], cwd=root, capture_output=True, text=True, timeout=60)
        job = result.stdout.strip().split(";")[0]
        if result.returncode or not job.isdigit():
            save(root / "submission_error.json", {"returncode": result.returncode, "stderr": result.stderr, "stdout": result.stdout})
            raise RuntimeError("Slurm did not acknowledge a job; inspect submission_error.json before any new run")
        value = {"job_id": job, "job_name": name, "reconciled": False}
        save(receipt, value)
        return value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit once, or reconcile an existing Slurm job")
    parser.add_argument("root", type=Path)
    parser.add_argument("name")
    parser.add_argument("--reconcile-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(dispatch(args.root, args.name, reconcile_only=args.reconcile_only)))
