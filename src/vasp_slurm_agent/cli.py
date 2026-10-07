from __future__ import annotations

import argparse
import getpass
import hashlib
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
        # Launch commands come from the user's configuration.
        words = shlex.split(config.vasp_command)
        solver = next((w for w in words if "vasp" in Path(w).name.lower()), "")
        if solver:
            command = "\n".join(config.setup_commands + ["command -v " + shlex.quote(solver)])
            result = remote.run(command)
            checks["vasp"] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
        else:
            checks["vasp"] = {"ok": False, "detail": "VASP not found in launch command. Check any wrapper script."}
        if config.vasp_ncl_command:
            words = shlex.split(config.vasp_ncl_command)
            solver = next((w for w in words if "vasp" in Path(w).name.lower()), "")
            if solver:
                result = remote.run("\n".join(config.setup_commands + ["command -v " + shlex.quote(solver)]))
                checks["vasp_ncl"] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
            else:
                checks["vasp_ncl"] = {"ok": False, "detail": "Check the noncollinear launch command."}
        return {"ok": all(value["ok"] for value in checks.values()), "checks": checks}
    finally:
        if owned:
            remote.close()


def main():
    parser = argparse.ArgumentParser(description="Run VASP on Slurm.")
    commands = parser.add_subparsers(dest="command", required=True)
    ui = commands.add_parser("ui", help="Open app")
    ui.add_argument("--port", type=int, default=8501)
    init = commands.add_parser("init", help="Set up cluster")
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    probe = commands.add_parser("doctor", help="Check cluster and VASP")
    probe.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    probe.add_argument("--password", action="store_true", help="Ask for your SSH password")
    plan = commands.add_parser("plan", help="Plan from a goal")
    plan.add_argument("structure", type=Path)
    plan.add_argument("goal")
    plan.add_argument("--provider", choices=("responses", "chat_completions", "codex"), default="responses")
    plan.add_argument("--model", default=os.environ.get("DFT_AGENT_MODEL", ""))
    plan.add_argument("--base-url", default=os.environ.get("DFT_AGENT_BASE_URL", ""))
    plan.add_argument("--output", type=Path, default=Path("plan.json"))
    prep = commands.add_parser("prepare", help="Prepare inputs locally")
    prep.add_argument("structure", type=Path)
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source = prep.add_mutually_exclusive_group()
    source.add_argument("--task", choices=("relax", "scf", "bands", "dos"))
    source.add_argument("--plan", type=Path)
    prep.add_argument("--magnetic-states", nargs="+", choices=("NM", "FM", "AFM"))
    prep.add_argument("--parameters", default="{}", help="Calculation settings as a JSON object")
    command_help = {
        "watch": "Submit and monitor",
        "resume": "Reconnect and monitor",
        "status": "Show saved status",
        "cancel": "Cancel job",
        "bundle": "Export results ZIP",
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
        if args.command == "plan":
            from .agent import ModelSettings, draft_plan
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = draft_plan(args.goal, args.structure, settings)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        from .workflow import prepare_run, prepare_plan, read_state, watch, resume, cancel, bundle_run, TERMINAL
        if args.command == "prepare":
            config = ClusterConfig.load(args.config)
            if args.plan:
                if args.parameters != "{}" or args.magnetic_states:
                    raise ValueError("Edit the plan before changing its settings.")
                proposal = json.loads(args.plan.read_text())
                if proposal.get("status") != "ready":
                    raise ValueError("Resolve the plan's questions before preparing inputs.")
                if proposal.get("source_sha256") != hashlib.sha256(args.structure.read_bytes()).hexdigest():
                    raise ValueError("The structure differs from the plan. Create a new plan.")
                result = prepare_plan(args.structure, args.run_dir, config, proposal["tasks"], proposal["parameters"],
                                      proposal.get("magnetic_states") or None, proposal.get("initial_moment", 3.0))
                (args.run_dir / "proposal.json").write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n")
            elif args.magnetic_states:
                result = prepare_plan(args.structure, args.run_dir, config, [args.task or "scf"],
                                      json.loads(args.parameters), args.magnetic_states)
            else:
                result = prepare_run(args.structure, args.run_dir, config, args.task or "relax", json.loads(args.parameters))
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
