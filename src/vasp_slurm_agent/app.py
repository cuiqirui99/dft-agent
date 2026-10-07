"""DFT Agent app."""

from __future__ import annotations

from datetime import datetime
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from pymatgen.core import Structure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from vasp_slurm_agent import __version__
from vasp_slurm_agent.cli import doctor
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.workflow import (
    SLURM_TERMINAL,
    bundle_run,
    cancel,
    prepare_run,
    prepare_plan,
    read_state,
    resume,
    start_worker,
)


TERMINAL = {"succeeded", "needs_attention", "failed", "cancelled"}
STATUS_LABELS = {
    "planned": "Awaiting review",
    "submitting": "Submitting",
    "queued": "Queued",
    "running": "Running",
    "collecting": "Collecting results",
    "succeeded": "Completed",
    "needs_attention": "Needs attention",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def _with_password(function, *args):
    """Pass the session password to an action and its worker."""
    previous = os.environ.get("DFT_AGENT_SSH_PASSWORD")
    password = st.session_state.get("ssh_password", "")
    if password:
        os.environ["DFT_AGENT_SSH_PASSWORD"] = password
    try:
        return function(*args)
    finally:
        if previous is None:
            os.environ.pop("DFT_AGENT_SSH_PASSWORD", None)
        else:
            os.environ["DFT_AGENT_SSH_PASSWORD"] = previous


def _load_config(path: Path) -> ClusterConfig | None:
    if not path.is_file():
        return None
    try:
        return ClusterConfig.load(path)
    except Exception as exc:
        st.error(f"Cannot read cluster settings: {exc}")
        return None


def _config_editor(path: Path, config: ClusterConfig | None) -> None:
    st.subheader("Cluster setup")
    st.caption("Settings are saved on this computer.")

    def value(name: str, default):
        return getattr(config, name, default)

    with st.form("cluster_config"):
        left, right = st.columns(2)
        with left:
            host = st.text_input("SSH host", value=value("host", ""))
            user = st.text_input("SSH user", value=value("user", ""))
            port = st.number_input("SSH port", 1, 65535, int(value("port", 22)))
            remote_root = st.text_input("Remote run folder (absolute path)", value=value("remote_root", ""))
            potcar_root = st.text_input("Remote POTCAR folder", value=value("potcar_root", ""))
            vasp_command = st.text_input("VASP command", value=value("vasp_command", "srun vasp_std"))
            ncl_command = st.text_input("SOC / noncollinear command", value=value("vasp_ncl_command", ""),
                                        placeholder="srun vasp_ncl")
        with right:
            partition = st.text_input("Slurm partition", value=value("partition", ""))
            account = st.text_input("Slurm account (optional)", value=value("account", ""))
            tasks = st.number_input("MPI tasks", 1, 4096, int(value("tasks", 8)),
                                    help="Start with 8 tasks for small cells. More cores may not help.")
            walltime = st.text_input("Time limit per stage", value=value("walltime", "00:30:00"))
            timeout = st.number_input("SSH connection timeout (s)", 1, 120, int(value("connect_timeout", 15)))
            setup = st.text_area("Environment setup commands (one per line)", value="\n".join(value("setup_commands", [])),
                                 placeholder="module load your-vasp-module")
        symbols = st.text_area("POTCAR element mapping (JSON)", value=json.dumps(value("potcar_symbols", {}), ensure_ascii=False),
                               help='Use your POTCAR folder names, e.g. {"Ti": "Ti_pv"}.')
        save = st.form_submit_button("Save cluster settings", type="primary")
    if save:
        try:
            mapping = json.loads(symbols)
            if not isinstance(mapping, dict):
                raise ValueError("POTCAR mapping must be a JSON object.")
            candidate = ClusterConfig(
                host=host.strip(), user=user.strip(), remote_root=remote_root.strip(),
                vasp_command=vasp_command.strip(), potcar_root=potcar_root.strip(),
                partition=partition.strip(), port=int(port), tasks=int(tasks), walltime=walltime.strip(),
                account=account.strip(), setup_commands=[line for line in setup.splitlines() if line.strip()],
                connect_timeout=int(timeout), potcar_symbols=mapping,
                vasp_ncl_command=ncl_command.strip(),
            )
            candidate.save(path)
        except Exception as exc:
            st.error(f"Cannot save settings: {exc}")
        else:
            st.success(f"Saved to {path}")
            st.rerun()
    if st.button("Check cluster connection", disabled=config is None):
        try:
            with st.spinner("Checking SSH, Slurm and VASP. No jobs submitted."):
                report = _with_password(doctor, config)
            if report.get("ok"):
                st.success("Cluster checks passed.")
            else:
                st.warning("Some checks failed. See below.")
            st.json(report)
        except Exception as exc:
            st.error(f"Cluster checks failed: {exc}")


def _structure_plot(structure: Structure) -> None:
    fig = plt.figure(figsize=(6, 4))
    try:
        axis = fig.add_subplot(111, projection="3d")
        corners = np.asarray(list(product((0, 1), repeat=3)))
        cartesian = structure.lattice.get_cartesian_coords(corners)
        for i, first in enumerate(corners):
            for j in range(i + 1, len(corners)):
                if np.abs(first - corners[j]).sum() == 1:
                    edge = cartesian[[i, j]]
                    axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#8794a3", linewidth=0.8)
        species = sorted({site.species_string for site in structure})
        for name in species:
            positions = np.asarray([site.coords for site in structure if site.species_string == name])
            axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=75, label=name)
        axis.set(xlabel="x / Å", ylabel="y / Å", zlabel="z / Å")
        axis.set_box_aspect(np.maximum(np.ptp(cartesian, axis=0), 1e-6))
        axis.legend(loc="upper left")
        st.pyplot(fig)
        st.caption("Markers do not show atomic radii.")
    finally:
        plt.close(fig)


def _preview(upload) -> Structure | None:
    suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
    try:
        with tempfile.TemporaryDirectory(prefix="vasp-agent-preview-") as directory:
            source = Path(directory) / f"structure{suffix}"
            source.write_bytes(upload.getvalue())
            structure = Structure.from_file(source)
        cols = st.columns(3)
        cols[0].metric("Formula", structure.composition.reduced_formula)
        cols[1].metric("Atoms", len(structure))
        cols[2].metric("Cell volume (Å³)", f"{structure.volume:.3f}")
        st.caption("a, b, c (Å): " + ", ".join(f"{v:.4f}" for v in structure.lattice.abc))
        _structure_plot(structure)
        with st.expander("Elements and fractional coordinates"):
            st.dataframe([
                {"Element": str(site.species_string), "x": site.a, "y": site.b, "z": site.c}
                for site in structure
            ], hide_index=True)
        return structure
    except Exception as exc:
        st.error(f"Cannot read structure: {exc}")
        return None


def _stage_results(run_dir: Path, stage: dict, scf_fermi_energy: float | None = None) -> None:
    result = stage.get("result")
    if not result:
        return
    label = f"{stage['label']} · " if stage.get("label") else ""
    st.subheader(f"{label}{stage['name']} · Results")
    if not result.get("success"):
        st.warning(result.get("reason", "Result checks failed."))
        if result.get("recovery_hint"):
            st.info(result["recovery_hint"])
        return
    if stage["name"] in {"relax", "scf"}:
        metrics = st.columns(3)
        energy = result.get("final_energy_ev")
        force = result.get("final_max_force_ev_angstrom")
        if energy is not None:
            metrics[0].metric("Final total energy (eV)", f"{energy:.6f}")
        if force is not None:
            metrics[1].metric("Maximum atomic force (eV/Å)", f"{force:.6f}")
        per_atom = result.get("final_energy_ev_per_atom")
        if per_atom is not None:
            metrics[2].metric("Energy per atom (eV)", f"{per_atom:.6f}")
    elif stage["name"] in {"bands", "dos"}:
        mode = (stage.get("metadata") or {}).get("spectral_charge_mode", "fixed")
        if mode == "self_consistent":
            fermi = result.get("energy_reference_ev", result.get("fermi_energy_ev"))
            if fermi is not None:
                st.metric("Fermi energy (eV)", f"{fermi:.6f}")
        else:
            if result.get("energy_reference_source") == "preceding_scf":
                scf_fermi_energy = result.get("energy_reference_ev", scf_fermi_energy)
            if scf_fermi_energy is not None:
                st.metric("SCF Fermi energy (eV)", f"{scf_fermi_energy:.6f}")
        st.caption("Self-consistent spectrum." if mode == "self_consistent" else "Fixed-charge spectrum.")
    if result.get("magnetization"):
        with st.expander("Magnetic moments"):
            st.json(result["magnetization"])
    output = run_dir / stage["folder"] / "outputs"
    final_structure = output / "final_structure.cif"
    if final_structure.is_file():
        try:
            structure = Structure.from_file(final_structure)
            with st.expander("Final structure", expanded=True):
                st.write(f"{structure.composition.reduced_formula} · {len(structure)} atoms · "
                         f"Cell volume {structure.volume:.3f} Å³")
                _structure_plot(structure)
            st.download_button("Download structure (.cif)", final_structure.read_bytes(),
                               file_name=f"{run_dir.name}-{stage['name']}.cif",
                               mime="chemical/x-cif", key=f"structure_{run_dir.name}_{stage['folder']}")
        except Exception as exc:
            st.error(f"Cannot display structure: {exc}")


def _model_editor():
    from vasp_slurm_agent.agent import ModelSettings

    with st.expander("Model"):
        provider = st.selectbox("Provider", ["responses", "chat_completions", "codex"],
                               format_func=lambda value: {"responses": "OpenAI", "chat_completions": "Compatible API", "codex": "Codex CLI"}[value])
        model = st.text_input("Model name", value=os.environ.get("DFT_AGENT_MODEL", ""),
                              help="Leave blank for the Codex CLI default.")
        base_url = st.text_input("API URL (optional)", value=os.environ.get("DFT_AGENT_BASE_URL", ""))
        key = st.text_input("API key (optional)", type="password", key="model_api_key",
                            help="Kept in memory. Environment keys also work.")
        st.button("Clear API key", on_click=lambda: st.session_state.update(model_api_key=""))
    return ModelSettings(provider=provider, model=model.strip(), base_url=base_url.strip() or None, api_key=key or None)


def _prepare_from_plan(upload, runs_root, config, plan):
    if plan.get("source_sha256") != hashlib.sha256(upload.getvalue()).hexdigest():
        raise ValueError("The structure changed. Create a new plan.")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = runs_root / run_id
    suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
    with tempfile.TemporaryDirectory(prefix="dft-agent-input-") as directory:
        source = Path(directory) / f"structure{suffix}"
        source.write_bytes(upload.getvalue())
        prepare_plan(source, run_dir, config, tasks=plan["tasks"], parameters=plan["parameters"],
                     magnetic_states=plan.get("magnetic_states") or None, initial_moment=plan.get("initial_moment", 3.0))
    (run_dir / "proposal.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    st.session_state["active_run"] = str(run_dir)
    st.success("Inputs ready. Review below, then submit.")


def _agent_plan(upload, structure, config, runs_root, settings):
    from vasp_slurm_agent.agent import _redact, _secrets, draft_plan

    goal = st.text_area("Goal", placeholder="Relax this structure, then calculate its bands with SOC.", max_chars=4000)
    stamp = hashlib.sha256((upload.getvalue() if upload else b"") + goal.encode()).hexdigest()
    if st.session_state.get("agent_stamp") != stamp:
        for key in ("agent_proposal", "agent_history", "agent_clear_revision", "plan_revision"):
            st.session_state.pop(key, None)
        st.session_state["agent_stamp"] = stamp
    saved = st.session_state.get("agent_proposal")
    if st.session_state.pop("agent_clear_revision", False):
        st.session_state["plan_revision"] = ""
    revision = st.text_input("Change the plan", key="plan_revision") if saved or st.session_state.get("agent_history") else ""
    if st.button("Plan", type="primary", disabled=structure is None or not goal.strip()):
        history = list(st.session_state.get("agent_history", []))
        pending = {"role": "user", "content": revision.strip()}
        if pending["content"] and (not history or history[-1] != pending):
            history.append(pending)
        st.session_state["agent_history"] = history
        st.session_state.pop("agent_proposal", None)
        saved = None
        try:
            suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
            with tempfile.TemporaryDirectory(prefix="dft-agent-plan-") as directory:
                source = Path(directory) / f"structure{suffix}"
                source.write_bytes(upload.getvalue())
                with st.spinner("Planning…"):
                    plan = draft_plan(goal.strip(), source, settings, history=history[-12:])
            prefix = [{**item, "content": _redact(item["content"], _secrets(settings))} for item in history[:-12]]
            history = prefix + plan.get("dialogue", history[-12:] + [{"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}])
            saved = {"stamp": stamp, "plan": plan, "goal": plan.get("goal", goal.strip())}
            st.session_state["agent_proposal"] = saved
            st.session_state["agent_history"] = history
            st.session_state["agent_clear_revision"] = True
            revision = ""
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if not saved:
        return
    plan = saved["plan"]
    st.write(plan["summary"])
    for question in plan.get("questions", []):
        st.info(question)
    if plan.get("status") == "unsupported":
        st.warning("This plan cannot run yet.")
    with st.expander("Plan details", expanded=plan.get("status") == "ready"):
        st.json({key: value for key, value in plan.items() if key not in {"provenance", "dialogue", "goal"}})
    if revision.strip():
        st.info("Update the plan before preparing inputs.")
    if st.button("Prepare inputs", disabled=plan.get("status") != "ready" or config is None or bool(revision.strip())):
        try:
            _prepare_from_plan(upload, runs_root, config, {**plan, "goal": saved["goal"],
                              "dialogue": st.session_state.get("agent_history", [])})
        except Exception as exc:
            st.error(f"Cannot prepare inputs: {exc}")
    if config is None:
        st.info("Save Cluster setup before preparing inputs.")


def _new_run(config: ClusterConfig | None, runs_root: Path, model_settings=None) -> None:
    st.subheader("New calculation")
    st.write("Prepare inputs locally. Review before submitting.")
    upload = st.file_uploader("Upload a structure", help="CIF or POSCAR; no extension needed for POSCAR. POTCAR stays on the cluster.")
    structure = _preview(upload) if upload is not None else None
    mode = st.radio("Mode", ["Agent", "Manual"], horizontal=True)
    if mode == "Agent":
        _agent_plan(upload, structure, config, runs_root, model_settings)
        return
    with st.form("prepare_run"):
        task = st.selectbox("Calculation", ["relax", "scf", "bands", "dos"],
                            format_func=lambda key: {"relax": "Structure relaxation", "scf": "Self-consistent calculation (SCF)",
                                                     "bands": "Band structure", "dos": "Density of states (DOS)"}[key])
        st.caption("Starting values. Check convergence for your material.")
        left, right = st.columns(2)
        with left:
            encut = st.number_input("ENCUT / eV", 100.0, 2000.0, 520.0, 10.0)
            ediff = st.number_input("EDIFF / eV", min_value=1e-10, max_value=1e-2, value=1e-5,
                                    format="%.1e")
            ismear = st.selectbox("ISMEAR", [-5, -1, 0, 1, 2], index=2)
            sigma = st.number_input("SIGMA / eV", 0.001, 1.0, 0.05, 0.01, format="%.3f")
        with right:
            k_grid = st.text_input("Gamma-centered k-point grid", "4 4 4", help="Three positive integers, e.g. 4 4 4.")
            nsw = st.number_input("Maximum relaxation steps (NSW)", 1, 1000, 100)
            ediffg = st.number_input("Force convergence threshold (eV/Å)", 0.0001, 1.0, 0.03, 0.005, format="%.4f")
            cell_relax = st.checkbox("Relax the cell as well as atomic positions", value=False)
        with st.expander("Method"):
            functional = st.selectbox("Functional", ["PBE", "HSE06", "PBE0"])
            spin = st.selectbox("Spin", ["none", "collinear", "noncollinear"])
            soc = st.checkbox("SOC")
            moments = st.text_area("Site moments (JSON)", value="null", help="One value per input site, or three components for noncollinear spins.")
            axis = st.text_input("Spin axis", "0 0 1")
            hubbard = st.text_area("Hubbard U (JSON)", value="{}", help='Example: {"Ni": {"l": 2, "u": 5, "j": 0}}')
            comparisons = st.multiselect("Compare magnetic states", ["NM", "FM", "AFM"], help="SCF seeds on the same cell.")
        prepared = st.form_submit_button("Prepare inputs", type="primary",
                                         disabled=structure is None or config is None)
    if config is None:
        st.info("Save Cluster setup before preparing inputs.")
    if prepared:
        try:
            grid = [int(part) for part in k_grid.replace(",", " ").split()]
            if len(grid) != 3 or any(part < 1 for part in grid):
                raise ValueError("Use three positive integers for the k-point grid.")
            parameters = dict(encut=encut, ediff=ediff, ediffg=-ediffg, nsw=int(nsw),
                              kpoint_grid=grid, ismear=ismear, sigma=sigma, cell_relax=cell_relax,
                              functional=functional, spin=spin, soc=soc, magmom=json.loads(moments),
                              saxis=[float(v) for v in axis.split()], hubbard_u=json.loads(hubbard))
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = runs_root / run_id
            suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
            with tempfile.TemporaryDirectory(prefix="vasp-agent-input-") as directory:
                source = Path(directory) / f"structure{suffix}"
                source.write_bytes(upload.getvalue())
                prepare_plan(source, run_dir, config, tasks=[task], parameters=parameters, magnetic_states=comparisons or None)
            st.session_state["active_run"] = str(run_dir)
            st.success("Inputs ready. Review below, then submit.")
        except Exception as exc:
            st.error(f"Cannot prepare inputs: {exc}")


def _result_dialogue(run_dir, settings, context):
    from vasp_slurm_agent.explanation import explain_run, load_run_context

    if context.get("goal"):
        st.subheader("Saved goal")
        st.write(context["goal"])
    st.subheader("Outcome")
    st.write(context.get("outcome") or f"Run status: {STATUS_LABELS.get(context['status'], context['status'])}.")
    record_path = run_dir / "explanations.json"
    record = {}
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text())
            if not isinstance(record, dict):
                raise ValueError("Invalid dialogue record.")
        except (OSError, ValueError) as exc:
            st.warning(f"Cannot read saved explanations: {exc}")
            record = {}
    if record and record.get("context_sha256") != context["context_sha256"]:
        st.info("Results changed. Ask again for an updated explanation.")
        record = {}
    question = st.text_input("Ask about this run", key=f"result_question_{run_dir}", max_chars=4000)
    actions = st.columns(2)
    explain = actions[0].button("Explain results", key=f"explain_{run_dir}")
    ask = actions[1].button("Ask", disabled=not question.strip(), key=f"ask_{run_dir}")
    if explain or ask:
        prompt = question.strip() if ask else "Explain the results."
        history = record.get("history", [])
        try:
            with st.spinner("Reading results…"):
                response = explain_run(run_dir, settings, question=prompt, history=history[-12:])
            if response["context_sha256"] != context["context_sha256"] or response["context_sha256"] != load_run_context(run_dir)["context_sha256"]:
                record = {}
                raise ValueError("Results changed. Ask again for an updated explanation.")
            exchange = {"question": prompt, **response}
            reply = {key: response[key] for key in ("answer", "evidence", "limits", "next_steps")}
            record = {"context_sha256": response["context_sha256"],
                      "history": history + [{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)}],
                      "exchanges": record.get("exchanges", []) + [exchange]}
            with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, prefix=".explanations-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            try:
                os.replace(temporary, record_path)
            finally:
                temporary.unlink(missing_ok=True)
        except Exception as exc:
            st.error(f"Cannot explain results: {exc}")
    for exchange in record.get("exchanges", []):
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant"):
            st.write(exchange["answer"])
            with st.expander("Evidence"):
                for fact_id in exchange.get("evidence", []):
                    fact = context["facts"].get(fact_id)
                    if fact:
                        unit = f" {fact['unit']}" if fact.get("unit") else ""
                        st.write(f"{fact_id} · {fact['label']}: {fact['value']}{unit}")
                        st.caption(fact["source"])
            for limit in exchange.get("limits", []):
                st.caption(limit)
            for step in exchange.get("next_steps", []):
                st.write(step)


@st.fragment(run_every="5s")
def _run_panel(run_dir: Path, model_settings=None) -> None:
    from vasp_slurm_agent.explanation import load_run_context

    try:
        state = read_state(run_dir)
        context = load_run_context(run_dir)
    except Exception as exc:
        st.error(f"Cannot read run: {exc}")
        return
    status = context["status"]
    stages = []
    for index, original in enumerate(state.get("stages", []), 1):
        stage = dict(original)
        if (stage.get("result") or {}).get("success") and not context["facts"].get(f"stage_{index}.accepted", {}).get("value"):
            message = context["facts"].get(f"stage_{index}.availability", {}).get("value", "Saved results could not be verified.")
            stage = {**stage, "status": "needs_attention", "result": {"success": False, "reason": message}}
        stages.append(stage)
    awaiting_submission = status == "planned" and not any(stage.get("job_id") for stage in stages)
    st.subheader(f"{state.get('formula', '')} · {state.get('task', '')}")
    st.caption(str(run_dir))
    status_label = "Preparing next stage" if status == "planned" and not awaiting_submission else STATUS_LABELS.get(status, status)
    st.write(f"Status: **{status_label}**")
    if state.get("comparison") and all((stage.get("result") or {}).get("success") for stage in stages):
        with st.expander("Magnetic comparison", expanded=True):
            st.json(state["comparison"])
    if state.get("last_error"):
        st.warning(str(state["last_error"]))
    st.dataframe([
        {"Stage": stage.get("label", stage.get("name", "")), "Status": STATUS_LABELS.get(stage.get("status"), stage.get("status", "")),
         "Slurm job": stage.get("job_id") or "—"}
        for stage in stages
    ], hide_index=True)
    with st.expander("Review settings", expanded=awaiting_submission):
        st.json(state.get("parameters", {}))
        try:
            run_config = ClusterConfig.load(run_dir / "config.json")
            st.json(run_config.to_dict())
        except Exception as exc:
            st.error(f"Cannot read run settings: {exc}")
    with st.expander("Stage details and run history"):
        st.json({"stages": stages, "history": state.get("history", [])})
    with st.expander("Generated INCAR and KPOINTS"):
        for stage in stages:
            folder = stage.get("folder")
            if not folder:
                continue
            st.write(stage.get("name", folder))
            for name in ("INCAR", "KPOINTS"):
                source = run_dir / folder / "inputs" / name
                if source.is_file():
                    st.caption(name)
                    st.code(source.read_text(), language="text")
    scf_fermi_energy = None
    for stage in stages:
        if stage.get("name") == "scf" and (stage.get("result") or {}).get("success"):
            scf_fermi_energy = stage["result"].get("fermi_energy_ev")
        _stage_results(run_dir, stage, scf_fermi_energy)
        folder = stage.get("folder")
        if folder and (stage.get("result") or {}).get("success"):
            for plot in sorted((run_dir / folder / "outputs").glob("*.png")):
                st.image(str(plot), caption=f"{stage.get('name', '')} · {plot.name}")

    actions = st.columns(3)
    if awaiting_submission:
        reviewed = st.checkbox("Structure, settings and resources reviewed.", key=f"review_{run_dir.name}")
        if actions[0].button("Submit calculation", type="primary", disabled=not reviewed, key=f"submit_{run_dir.name}"):
            try:
                pid = _with_password(start_worker, run_dir)
                st.success(f"Monitoring started (PID {pid}). You can close this page.")
            except Exception as exc:
                st.error(f"Cannot start monitoring: {exc}")
    elif status not in TERMINAL:
        if actions[0].button("Resume monitoring", key=f"watch_{run_dir.name}",
                             help="Resume monitoring existing jobs after a restart or interruption."):
            try:
                st.success(f"Monitoring started (PID {_with_password(start_worker, run_dir)}).")
            except Exception as exc:
                st.error(f"Cannot resume monitoring: {exc}")
    elif status in {"needs_attention", "failed"}:
        if actions[0].button("Reconnect", key=f"reconnect_{run_dir.name}",
                             help="Collect output from the original job. Settings and convergence checks stay unchanged."):
            try:
                restored = _with_password(resume, run_dir)
                if restored.get("status") not in TERMINAL:
                    st.success(f"Monitoring started (PID {_with_password(start_worker, run_dir)}).")
                else:
                    st.warning(restored.get("last_error") or "Check the run history.")
            except Exception as exc:
                st.error(f"Cannot reconnect: {exc}")
    current_stage = stages[state.get("current_stage", 0)] if stages else {}
    has_job = bool(current_stage.get("job_id"))
    can_cancel = status not in {"succeeded", "cancelled"} and (
        (has_job and current_stage.get("scheduler_state") not in SLURM_TERMINAL)
        or (not has_job and status not in TERMINAL)
    )
    if can_cancel:
        if actions[1].button("Cancel calculation", key=f"cancel_{run_dir.name}"):
            try:
                _with_password(cancel, run_dir)
                st.rerun()
            except Exception as exc:
                st.error(f"Cancellation unconfirmed: {exc}")
    bundle_key = f"bundle_{run_dir}"
    if actions[2].button("Prepare download", key=f"bundle_btn_{run_dir.name}"):
        try:
            st.session_state[bundle_key] = str(bundle_run(run_dir))
        except Exception as exc:
            st.error(f"Cannot prepare download: {exc}")
    if bundle_key in st.session_state:
        archive = Path(st.session_state[bundle_key])
        if archive.is_file():
            st.download_button("Download results (.zip)", archive.read_bytes(), file_name=archive.name,
                               mime="application/zip", key=f"download_{run_dir.name}")
    if status == "succeeded":
        st.info("Complete. Results are ready to download.")
    elif status == "needs_attention":
        st.info("Check the error before reconnecting. Settings are unchanged.")
    _result_dialogue(run_dir, model_settings, context)


def main() -> None:
    st.set_page_config(page_title="DFT Agent", page_icon="⚛", layout="wide")
    st.title("DFT Agent")
    st.caption(f"Version {__version__}")
    with st.sidebar:
        st.header("Local settings")
        config_path = Path(st.text_input("Configuration file", value=os.environ.get(
            "VASP_AGENT_CONFIG", str(Path.home() / ".config/vasp-slurm-agent/cluster.json")))).expanduser()
        runs_root = Path(st.text_input("Run folder", value=str(Path.home() / "vasp-slurm-agent-runs"))).expanduser()
        st.text_input("SSH password (optional)", type="password", key="ssh_password",
                      help="SSH keys are preferred. Passwords stay in memory and are never saved.")
        st.button("Clear password", on_click=lambda: st.session_state.update(ssh_password=""))
        st.caption("Requires a Slurm account and VASP license.")
        model_settings = _model_editor()
    config = _load_config(config_path)
    task_tab, history_tab, config_tab = st.tabs(["New calculation", "Runs", "Cluster setup"])
    with config_tab:
        _config_editor(config_path, config)
    with task_tab:
        _new_run(config, runs_root, model_settings)
    with history_tab:
        available = sorted(runs_root.glob("*/run.json"), reverse=True) if runs_root.is_dir() else []
        selected = st.selectbox("Select a run", [str(path.parent) for path in available], index=None,
                                placeholder="Choose a run")
        if st.button("View run", disabled=selected is None):
            st.session_state["active_run"] = selected
        manual = st.text_input("Or enter a run folder")
        if st.button("Open run", disabled=not manual.strip()):
            st.session_state["active_run"] = str(Path(manual).expanduser())
    if st.session_state.get("active_run"):
        st.divider()
        _run_panel(Path(st.session_state["active_run"]), model_settings)


if __name__ == "__main__":
    main()
