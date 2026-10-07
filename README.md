# VASP Slurm Agent

A local app for running VASP calculations on a Slurm cluster. Prepare inputs, review them before submission, follow the job and download the results. Start with structure optimization, or use the SCF, band structure and density of states workflows. No LLM account is needed.

**Version: `0.1.1`.** This update makes the interface and documentation fully English. It is intended for individual researchers who already have VASP and Slurm access. The nonmagnetic PBE workflows are unchanged; see the [v0.1.0 validation report](docs/validation.md) for the original cluster tests, recovery checks and numerical limits.

## Install and open

You need macOS or Linux, Python 3.11 or later, an SSH client, a Slurm account, and access to a licensed VASP installation and suitable POTCAR files on your cluster. Native Windows is not supported.

The remote login and compute environments need Python 3.8 or later, Bash and `sha256sum`; Slurm must provide `sbatch`, `squeue`, `sacct` and `scancel`. Set any required module-loading commands in the cluster configuration.

Clone the repository and install it in a virtual environment:

```bash
git clone https://github.com/cuiqirui99/vasp-slurm-agent.git
cd vasp-slurm-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
vasp-agent ui
```

The app opens on `127.0.0.1` and is designed for one user on their own machine. Keep it local; it has no multi-user login. Follow the [quickstart](docs/quickstart.md) for setup and your first calculation.

For command-line use, follow the [Si relaxation, Al SCF and MgO DOS examples](docs/examples.md).

1. Open **Cluster setup** and enter your SSH, VASP and Slurm settings.
2. Open **New calculation**, upload a CIF or POSCAR and inspect the structure.
3. Choose a **Calculation** and click **Prepare inputs**. This only writes local files.
4. Check **Review settings**, tick the confirmation box and click **Submit calculation**.
5. Follow the job in **Runs**, then download the structure or result bundle.

The repository and both distribution formats include small CIF examples. To extract silicon from an installed wheel without the source checkout:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
Path("Si.cif").write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
PY
```

Upload this file to try preparing a calculation. The examples are starting structures. Completed relaxation and SCF stages show the final energy, maximum atomic force and final structure. Bands and DOS show plots, data and the preceding SCF Fermi energy when available.

Closing the browser leaves remote jobs running. Monitoring needs the local computer to stay on; after a restart, select the run and click **Resume monitoring**. Three consecutive connection or collection failures pause the run as `needs_attention`. Check the error, then click **Reconnect** to find the same job or collect its outputs again. This keeps the original calculation settings.

## Configuration and credentials

The sidebar's **Configuration file** defaults to `~/.config/vasp-slurm-agent/cluster.json`, and **Run folder** defaults to `~/vasp-slurm-agent-runs`. You can change both paths. `VASP_AGENT_CONFIG` sets the initial configuration path. Each prepared run saves a copy of its settings.

Use SSH keys or an SSH agent where possible, and verify the cluster's host key before connecting. For password login, use **SSH password (optional)** in the sidebar and **Clear password** when finished. CLI commands such as `vasp-agent watch <run_dir> --password` prompt privately; `DFT_AGENT_SSH_PASSWORD` is also supported. Keep passwords out of configuration files and shared results. See the [quickstart](docs/quickstart.md#2-connect-to-your-cluster) for password handling and MFA limitations.

**Cluster setup** includes an environment check. It connects only when requested and does not submit a job.

New configurations start with 8 MPI tasks for the small examples. Adjust this to your structure and cluster allocation rules. The v0.1.0 [resource adjustment record](validation/resource-adjustments.v1.json) documents a carbon calculation that failed with 64 tasks and completed with 8.

VASP and POTCAR data stay on your cluster. Set `potcar_symbols` to select the potentials you intend to use. This project is available under the [MIT License](LICENSE); it does not distribute VASP or POTCAR files. Their separate terms are covered in [NOTICE.md](NOTICE.md).

## CLI

The CLI provides `vasp-agent init`, `doctor`, `prepare`, `watch`, `resume`, `status`, `cancel` and `bundle`. Use `vasp-agent <command> --help` for arguments. `prepare` writes local inputs; `watch` submits a prepared calculation and follows its stages. Review the inputs before running `watch`. Use `status` to read saved progress or `resume` to reconnect and continue monitoring.

## Scientific scope

The four workflows use nonmagnetic PBE (`ISPIN=1`) for ordered periodic structures. `relax` and `scf` each run one stage. `bands` and `dos` run SCF first, then calculate the requested spectrum on the same structure. To optimize the structure first, run `relax` separately and use its result.

Default settings are a starting point. Check the cutoff, k mesh, smearing and force tolerance for the material and property you want to study. Magnetism, DFT+U, hybrid functionals, spin–orbit coupling, defects, phonons, NEB and molecular dynamics are outside these workflows. Failed or unconverged calculations stop for inspection; the app does not change calculation settings automatically.

## Development and validation

```bash
python -m pip install ".[dev]"
python -m pytest -q
python -m build
```

CI runs offline core and Streamlit AppTest checks on Python 3.11–3.13 on Linux, then installs a wheel outside the source checkout and prepares an included example. It does not contact a cluster. Development dependencies require Streamlit 1.65 or later for AppTest's file-upload support.

The v0.1.0 cluster tests followed a [fixed protocol](validation/protocol.v1.json). The [campaign record](validation/results/campaign.json) and [execution records](validation/results/) include completed runs and failed attempts. [Independent numerical checks](validation/reference/README.md) are recorded separately from software tests. This package runs on its own; the [legacy scope note](docs/legacy-scope.md) explains its relationship to the earlier `AI_AGNET` platform.

Use [CITATION.cff](CITATION.cff) to cite the software, and record the version and calculation settings used for your results.
