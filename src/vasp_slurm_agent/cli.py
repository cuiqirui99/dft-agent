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
    plan.add_argument("--previous", type=Path, help="Revise a saved calculation plan")
    plan.add_argument("--provider", choices=("responses", "chat_completions", "codex"), default="responses")
    plan.add_argument("--model", default=os.environ.get("DFT_AGENT_MODEL", ""))
    plan.add_argument("--base-url", default=os.environ.get("DFT_AGENT_BASE_URL", ""))
    plan.add_argument("--output", type=Path, default=Path("plan.json"))
    plan.add_argument("--runs", type=Path, help="Past runs for method guidance")
    structure_plan = commands.add_parser("structure-plan", help="Plan structure edits")
    structure_plan.add_argument("structure", type=Path)
    structure_plan.add_argument("goal")
    structure_plan.add_argument("--previous", type=Path, help="Revise a saved structure plan")
    structure_plan.add_argument("--provider", choices=("responses", "chat_completions", "codex"), default="responses")
    structure_plan.add_argument("--model", default=os.environ.get("DFT_AGENT_MODEL", ""))
    structure_plan.add_argument("--base-url", default=os.environ.get("DFT_AGENT_BASE_URL", ""))
    structure_plan.add_argument("--output", type=Path, default=Path("structure-plan.json"))
    structure_prep = commands.add_parser("prepare-structure", help="Apply reviewed structure edits")
    structure_prep.add_argument("structure", type=Path)
    structure_prep.add_argument("output_dir", type=Path)
    structure_prep.add_argument("--plan", type=Path, required=True)
    convert = commands.add_parser("convert", help="Convert CIF or POSCAR")
    convert.add_argument("structure", type=Path)
    convert.add_argument("output_dir", type=Path)
    convert.add_argument("--format", choices=("cif", "poscar", "both"), default="both")
    explain = commands.add_parser("explain", help="Explain saved results")
    explain.add_argument("run_dir", type=Path)
    explain.add_argument("--question", default="Explain the results.")
    explain.add_argument("--provider", choices=("responses", "chat_completions", "codex"), default="responses")
    explain.add_argument("--model", default=os.environ.get("DFT_AGENT_MODEL", ""))
    explain.add_argument("--base-url", default=os.environ.get("DFT_AGENT_BASE_URL", ""))
    repair = commands.add_parser("repair", help="Plan a bounded repair")
    repair.add_argument("run_dir", type=Path)
    repair.add_argument("--provider", choices=("responses", "chat_completions", "codex"), default="responses")
    repair.add_argument("--model", default=os.environ.get("DFT_AGENT_MODEL", ""))
    repair.add_argument("--base-url", default=os.environ.get("DFT_AGENT_BASE_URL", ""))
    repair.add_argument("--output", type=Path, default=Path("repair.json"))
    repair_prep = commands.add_parser("prepare-repair", help="Prepare reviewed repair inputs")
    repair_prep.add_argument("run_dir", type=Path)
    repair_prep.add_argument("new_run_dir", type=Path)
    repair_prep.add_argument("--plan", type=Path, required=True)
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
            goal, history = args.goal, []
            if args.previous:
                previous = json.loads(args.previous.read_text())
                if previous.get("source_sha256") != hashlib.sha256(args.structure.read_bytes()).hexdigest():
                    raise ValueError("The source changed. Start a new calculation plan.")
                if not isinstance(previous.get("goal"), str) or not isinstance(previous.get("dialogue"), list):
                    raise ValueError("This plan has no conversation. Start a new calculation plan.")
                goal = previous["goal"]
                history = previous["dialogue"] + [{"role": "user", "content": args.goal}]
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = draft_plan(goal, args.structure, settings, history=history, runs_root=args.runs)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        if args.command == "structure-plan":
            from .agent import ModelSettings
            from .structure_agent import draft_structure
            goal, history = args.goal, []
            if args.previous:
                previous = json.loads(args.previous.read_text())
                if previous.get("source_sha256") != hashlib.sha256(args.structure.read_bytes()).hexdigest():
                    raise ValueError("The source changed. Start a new structure plan.")
                goal = previous["goal"]
                history = previous["dialogue"] + [{"role": "user", "content": args.goal}]
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = draft_structure(goal, args.structure, settings, history=history)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        if args.command in {"prepare-structure", "convert"}:
            from .structures import apply_structure_plan
            proposal = json.loads(args.plan.read_text()) if args.command == "prepare-structure" else {
                "schema_version": 1, "status": "ready", "operations": [], "output_format": args.format,
                "source_sha256": hashlib.sha256(args.structure.read_bytes()).hexdigest()}
            result = apply_structure_plan(args.structure, args.output_dir, proposal)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "explain":
            from .agent import ModelSettings
            from .explanation import explain_run
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = explain_run(args.run_dir, settings, question=args.question)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "repair":
            from .agent import ModelSettings
            from .recovery import draft_repair
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = draft_repair(args.run_dir, settings)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        if args.command == "prepare-repair":
            from .recovery import prepare_repair
            result = prepare_repair(args.run_dir, args.new_run_dir, json.loads(args.plan.read_text()))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
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
