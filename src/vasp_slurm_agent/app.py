"""Local, single-user Streamlit workbench for reviewed VASP/Slurm runs."""

from __future__ import annotations

from datetime import datetime
from itertools import product
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

from vasp_slurm_agent.cli import doctor
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.workflow import (
    SLURM_TERMINAL,
    bundle_run,
    cancel,
    prepare_run,
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
    """Provide a session-only password to one local action and its worker."""
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
        st.error(f"Could not read the cluster configuration: {exc}")
        return None


def _config_editor(path: Path, config: ClusterConfig | None) -> None:
    st.subheader("Cluster setup")
    st.caption("Settings are saved locally. If you need a password, enter it in the sidebar; it stays in memory for this session.")

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
        with right:
            partition = st.text_input("Slurm partition", value=value("partition", ""))
            account = st.text_input("Slurm account (optional)", value=value("account", ""))
            tasks = st.number_input("MPI tasks", 1, 4096, int(value("tasks", 8)),
                                    help="For small cells, start with 8 MPI tasks. More cores may not be faster. Adjust for your system and cluster.")
            walltime = st.text_input("Time limit per stage", value=value("walltime", "00:30:00"))
            timeout = st.number_input("SSH connection timeout (s)", 1, 120, int(value("connect_timeout", 15)))
            setup = st.text_area("Environment setup commands (one per line)", value="\n".join(value("setup_commands", [])),
                                 placeholder="module load your-vasp-module")
        symbols = st.text_area("POTCAR element mapping (JSON)", value=json.dumps(value("potcar_symbols", {}), ensure_ascii=False),
                               help='For example, {"Ti": "Ti_pv"}. Use the names in your licensed pseudopotential folder.')
        save = st.form_submit_button("Save cluster settings", type="primary")
    if save:
        try:
            mapping = json.loads(symbols)
            if not isinstance(mapping, dict):
                raise ValueError("POTCAR element mapping must be a JSON object.")
            candidate = ClusterConfig(
                host=host.strip(), user=user.strip(), remote_root=remote_root.strip(),
                vasp_command=vasp_command.strip(), potcar_root=potcar_root.strip(),
                partition=partition.strip(), port=int(port), tasks=int(tasks), walltime=walltime.strip(),
                account=account.strip(), setup_commands=[line for line in setup.splitlines() if line.strip()],
                connect_timeout=int(timeout), potcar_symbols=mapping,
            )
            candidate.save(path)
        except Exception as exc:
            st.error(f"Could not save the settings: {exc}")
        else:
            st.success(f"Saved to {path}")
            st.rerun()
    if st.button("Check cluster connection", disabled=config is None):
        try:
            with st.spinner("Checking SSH, Slurm and VASP using the saved settings. No job will be submitted."):
                report = _with_password(doctor, config)
            if report.get("ok"):
                st.success("Connection checks passed. You're ready to prepare a calculation.")
            else:
                st.warning("Some checks failed. See the results below.")
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
        st.caption("Unit cell and atomic positions. Marker sizes do not represent atomic radii.")
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
        st.caption("Lattice lengths a, b, c (Å): " + ", ".join(f"{v:.4f}" for v in structure.lattice.abc))
        _structure_plot(structure)
        with st.expander("Elements and fractional coordinates"):
            st.dataframe([
                {"Element": str(site.species_string), "x": site.a, "y": site.b, "z": site.c}
                for site in structure
            ], hide_index=True)
        return structure
    except Exception as exc:
        st.error(f"Could not read the structure: {exc}")
        return None


def _stage_results(run_dir: Path, stage: dict, scf_fermi_energy: float | None = None) -> None:
    result = stage.get("result")
    if not result:
        return
    st.subheader(f"{stage['name']} · Results")
    if not result.get("success"):
        st.warning(result.get("reason", "Results have not passed the checks."))
        if result.get("recovery_hint"):
            st.info(result["recovery_hint"])
        return
    if stage["name"] in {"relax", "scf"}:
        metrics = st.columns(2)
        energy = result.get("final_energy_ev")
        force = result.get("final_max_force_ev_angstrom")
        if energy is not None:
            metrics[0].metric("Final total energy (eV)", f"{energy:.6f}")
        if force is not None:
            metrics[1].metric("Maximum atomic force (eV/Å)", f"{force:.6f}")
    elif stage["name"] in {"bands", "dos"}:
        if result.get("energy_reference_source") == "preceding_scf":
            scf_fermi_energy = result.get("energy_reference_ev", scf_fermi_energy)
        if scf_fermi_energy is not None:
            st.metric("SCF Fermi energy (eV)", f"{scf_fermi_energy:.6f}")
        st.caption("Fixed-charge spectrum. See the preceding SCF stage for ground-state energy and atomic forces; its Fermi energy is shown here for reference.")
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
            st.error(f"Could not display the final structure: {exc}")


def _new_run(config: ClusterConfig | None, runs_root: Path) -> None:
    st.subheader("New calculation")
    st.write("Choose a structure and a calculation. You can review all inputs before submitting to your cluster.")
    st.caption("Nonmagnetic PBE (ISPIN=1). Band structure and DOS runs start with SCF and use the structure as uploaded.")
    upload = st.file_uploader("Upload a structure", help="Choose a CIF or POSCAR file. POSCAR files do not need an extension. POTCAR stays on your cluster.")
    structure = _preview(upload) if upload is not None else None
    with st.form("prepare_run"):
        task = st.selectbox("Calculation", ["relax", "scf", "bands", "dos"],
                            format_func=lambda key: {"relax": "Structure relaxation", "scf": "Self-consistent calculation (SCF)",
                                                     "bands": "Band structure", "dos": "Density of states (DOS)"}[key])
        st.caption("Start with these settings and adjust them for your material. Check convergence before comparing properties.")
        left, right = st.columns(2)
        with left:
            encut = st.number_input("ENCUT / eV", 100.0, 2000.0, 520.0, 10.0)
            ediff = st.number_input("EDIFF / eV", min_value=1e-10, max_value=1e-2, value=1e-5,
                                    format="%.1e")
            ismear = st.selectbox("ISMEAR", [-5, -1, 0, 1, 2], index=2)
            sigma = st.number_input("SIGMA / eV", 0.001, 1.0, 0.05, 0.01, format="%.3f")
        with right:
            k_grid = st.text_input("Gamma-centered k-point grid", "4 4 4", help="Three positive integers, such as 4 4 4.")
            nsw = st.number_input("Maximum relaxation steps (NSW)", 1, 1000, 100)
            ediffg = st.number_input("Force convergence threshold (eV/Å)", 0.0001, 1.0, 0.03, 0.005, format="%.4f")
            cell_relax = st.checkbox("Relax the cell as well as atomic positions", value=False)
        prepared = st.form_submit_button("Prepare inputs", type="primary",
                                         disabled=structure is None or config is None)
    if config is None:
        st.info("Save your settings in Cluster setup before preparing a calculation.")
    if prepared:
        try:
            grid = [int(part) for part in k_grid.replace(",", " ").split()]
            if len(grid) != 3 or any(part < 1 for part in grid):
                raise ValueError("The k-point grid needs three positive integers.")
            parameters = dict(encut=encut, ediff=ediff, ediffg=-ediffg, nsw=int(nsw),
                              kpoint_grid=grid, ismear=ismear, sigma=sigma, cell_relax=cell_relax)
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = runs_root / run_id
            suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
            with tempfile.TemporaryDirectory(prefix="vasp-agent-input-") as directory:
                source = Path(directory) / f"structure{suffix}"
                source.write_bytes(upload.getvalue())
                prepare_run(source, run_dir, config, task=task, parameters=parameters)
            st.session_state["active_run"] = str(run_dir)
            st.success("Inputs are ready. Review the settings below, then choose Submit calculation.")
        except Exception as exc:
            st.error(f"Could not prepare the inputs: {exc}")


@st.fragment(run_every="5s")
def _run_panel(run_dir: Path) -> None:
    try:
        state = read_state(run_dir)
    except Exception as exc:
        st.error(f"Could not read the run: {exc}")
        return
    status = state.get("status", "unknown")
    stages = state.get("stages", [])
    awaiting_submission = status == "planned" and not any(stage.get("job_id") for stage in stages)
    st.subheader(f"{state.get('formula', '')} · {state.get('task', '')}")
    st.caption(str(run_dir))
    status_label = "Preparing the next stage" if status == "planned" and not awaiting_submission else STATUS_LABELS.get(status, status)
    st.write(f"Status: **{status_label}**")
    if state.get("last_error"):
        st.warning(str(state["last_error"]))
    st.dataframe([
        {"Stage": stage.get("name", ""), "Status": STATUS_LABELS.get(stage.get("status"), stage.get("status", "")),
         "Slurm job": stage.get("job_id") or "—"}
        for stage in stages
    ], hide_index=True)
    with st.expander("Review settings", expanded=awaiting_submission):
        st.json(state.get("parameters", {}))
        try:
            run_config = ClusterConfig.load(run_dir / "config.json")
            st.json(run_config.to_dict())
        except Exception as exc:
            st.error(f"Could not read the run settings: {exc}")
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
        if folder:
            for plot in sorted((run_dir / folder / "outputs").glob("*.png")):
                st.image(str(plot), caption=f"{stage.get('name', '')} · {plot.name}")

    actions = st.columns(3)
    if awaiting_submission:
        reviewed = st.checkbox("I've reviewed the structure, settings and cluster resources.", key=f"review_{run_dir.name}")
        if actions[0].button("Submit calculation", type="primary", disabled=not reviewed, key=f"submit_{run_dir.name}"):
            try:
                pid = _with_password(start_worker, run_dir)
                st.success(f"Monitoring started (PID {pid}). You can close this page and check progress later in Runs.")
            except Exception as exc:
                st.error(f"Could not start monitoring: {exc}")
    elif status not in TERMINAL:
        if actions[0].button("Resume monitoring", key=f"watch_{run_dir.name}",
                             help="Resume after a local restart or an interrupted monitor. The saved state keeps track of submitted jobs."):
            try:
                st.success(f"Monitoring started (PID {_with_password(start_worker, run_dir)}).")
            except Exception as exc:
                st.error(f"Could not resume monitoring: {exc}")
    elif status in {"needs_attention", "failed"}:
        if actions[0].button("Reconnect", key=f"reconnect_{run_dir.name}",
                             help="Reconnect to the original job and collect its output with the same calculation settings. Results still need to pass convergence checks."):
            try:
                restored = _with_password(resume, run_dir)
                if restored.get("status") not in TERMINAL:
                    st.success(f"Monitoring started (PID {_with_password(start_worker, run_dir)}).")
                else:
                    st.warning(restored.get("last_error") or "This run still needs attention. Check the run history.")
            except Exception as exc:
                st.error(f"Could not reconnect: {exc}")
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
                st.error(f"Cancellation was not confirmed: {exc}")
    bundle_key = f"bundle_{run_dir}"
    if actions[2].button("Prepare download", key=f"bundle_btn_{run_dir.name}"):
        try:
            st.session_state[bundle_key] = str(bundle_run(run_dir))
        except Exception as exc:
            st.error(f"Could not prepare the download: {exc}")
    if bundle_key in st.session_state:
        archive = Path(st.session_state[bundle_key])
        if archive.is_file():
            st.download_button("Download results (.zip)", archive.read_bytes(), file_name=archive.name,
                               mime="application/zip", key=f"download_{run_dir.name}")
    if status == "succeeded":
        st.info("Calculation complete. Your structure and results are ready to download.")
    elif status == "needs_attention":
        st.info("Check the error above before reconnecting or preparing a new calculation. Your settings have not been changed.")


def main() -> None:
    st.set_page_config(page_title="VASP Slurm Agent", page_icon="⚛", layout="wide")
    st.title("VASP Slurm Agent")
    st.caption("VASP on your Slurm cluster · Version 0.1.1")
    with st.sidebar:
        st.header("Local settings")
        config_path = Path(st.text_input("Configuration file", value=os.environ.get(
            "VASP_AGENT_CONFIG", str(Path.home() / ".config/vasp-slurm-agent/cluster.json")))).expanduser()
        runs_root = Path(st.text_input("Run folder", value=str(Path.home() / "vasp-slurm-agent-runs"))).expanduser()
        st.text_input("SSH password (optional)", type="password", key="ssh_password",
                      help="SSH keys are preferred. A password stays in session memory and is used only for cluster actions you start.")
        st.button("Clear password", on_click=lambda: st.session_state.update(ssh_password=""))
        st.caption("Connect with your own Slurm account and licensed VASP installation.")
    config = _load_config(config_path)
    task_tab, history_tab, config_tab = st.tabs(["New calculation", "Runs", "Cluster setup"])
    with config_tab:
        _config_editor(config_path, config)
    with task_tab:
        _new_run(config, runs_root)
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
        _run_panel(Path(st.session_state["active_run"]))


if __name__ == "__main__":
    main()
