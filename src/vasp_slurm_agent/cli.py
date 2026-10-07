from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import sys

from .config import ClusterConfig

DEFAULT_CONFIG = Path.home() / ".config" / "vasp-slurm-agent" / "cluster.json"


def doctor(config, transport=None):
    from .transport import SSHTransport
    owned = transport is None
    remote = transport or SSHTransport(config)
    checks = {}
    try:
        for executable in ("python3", "sbatch", "squeue", "sacct", "scancel"):
            result = remote.run("command -v " + executable)
            checks[executable] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
        for name, path in (("potcar_root", config.potcar_root), ("remote_parent", str(Path(config.remote_root).parent))):
            result = remote.run("test -d " + shlex.quote(path))
            checks[name] = {"ok": result.returncode == 0, "detail": path}
        # A user-maintained command/setup is trusted configuration, not LLM code.
        words = shlex.split(config.vasp_command)
        solver = next((w for w in words if "vasp" in Path(w).name.lower()), "")
        if solver:
            command = "\n".join(config.setup_commands + ["command -v " + shlex.quote(solver)])
            result = remote.run(command)
            checks["vasp"] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
        else:
            checks["vasp"] = {"ok": False, "detail": "Could not find VASP in the launch command. If you use a wrapper script, check that it starts VASP correctly."}
        return {"ok": all(value["ok"] for value in checks.values()), "checks": checks}
    finally:
        if owned:
            remote.close()


def main():
    parser = argparse.ArgumentParser(description="Run VASP calculations on your Slurm cluster.")
    commands = parser.add_subparsers(dest="command", required=True)
    ui = commands.add_parser("ui", help="Open the app in your browser")
    ui.add_argument("--port", type=int, default=8501)
    init = commands.add_parser("init", help="Set up a cluster connection")
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    probe = commands.add_parser("doctor", help="Check your cluster connection and VASP setup")
    probe.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    probe.add_argument("--password", action="store_true", help="Ask for your SSH password")
    prep = commands.add_parser("prepare", help="Prepare calculation files without submitting a job")
    prep.add_argument("structure", type=Path)
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--task", choices=("relax", "scf", "bands", "dos"), default="relax")
    prep.add_argument("--parameters", default="{}", help="Calculation settings as a JSON object")
    command_help = {
        "watch": "Submit a prepared calculation and follow its progress",
        "resume": "Reconnect to an existing calculation and keep monitoring",
        "status": "Show the saved status without contacting the cluster",
        "cancel": "Request cancellation of a calculation",
        "bundle": "Save the available inputs and results as a ZIP file",
    }
    for name, help_text in command_help.items():
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("run_dir", type=Path)
        if name in {"watch", "resume", "cancel"}:
            sub.add_argument("--password", action="store_true", help="Ask for your SSH password")
        if name == "watch":
            sub.add_argument("--interval", type=float, default=20)
    args = parser.parse_args()
    try:
        if getattr(args, "password", False):
            os.environ["DFT_AGENT_SSH_PASSWORD"] = getpass.getpass("SSH password: ")
        if args.command == "ui":
            if not 1024 <= args.port <= 65535:
                parser.error("Choose a local port between 1024 and 65535")
            from streamlit.web import cli as streamlit_cli
            sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), "--server.address=127.0.0.1", f"--server.port={args.port}", "--server.headless=true", "--browser.gatherUsageStats=false"]
            return streamlit_cli.main()
        if args.command == "init":
            if args.config.expanduser().exists():
                raise FileExistsError("This configuration file already exists. Edit it or choose a different path with --config.")
            cfg = ClusterConfig(host=input("SSH host: ").strip(), user=input("SSH user: ").strip(), remote_root=input("Remote run folder (absolute path): ").strip(), vasp_command=input("VASP command [srun vasp_std]: ").strip() or "srun vasp_std", potcar_root=input("POTCAR folder (absolute path): ").strip(), partition=input("Slurm partition: ").strip())
            print(cfg.save(args.config))
            return 0
        if args.command == "doctor":
            result = doctor(ClusterConfig.load(args.config))
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        from .workflow import prepare_run, read_state, watch, resume, cancel, bundle_run, TERMINAL
        if args.command == "prepare":
            result = prepare_run(args.structure, args.run_dir, ClusterConfig.load(args.config), args.task, json.loads(args.parameters))
        elif args.command == "watch":
            if args.interval < 1:
                parser.error("Choose a polling interval of at least one second.")
            result = watch(args.run_dir, args.interval)
        elif args.command == "status":
            result = read_state(args.run_dir)
        elif args.command == "resume":
            result = resume(args.run_dir)
            if result["status"] not in TERMINAL:
                result = watch(args.run_dir)
        elif args.command == "cancel":
            result = cancel(args.run_dir)
        else:
            print(bundle_run(args.run_dir))
            return 0
        print(json.dumps(result, indent=2))
        return 1 if result.get("status") in {"failed", "needs_attention"} else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
