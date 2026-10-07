# Quickstart

This guide covers VASP Slurm Agent `0.1.1`. It is for researchers who already have
access to VASP, a POTCAR library and a Slurm cluster. The app runs locally and
needs no LLM account. Start with a small structure optimization to check your
setup, then use SCF, bands or DOS as needed.

## 1. Install and open

On macOS or Linux, install Python 3.11 or later and an SSH client. Then run:

```bash
git clone https://github.com/cuiqirui99/vasp-slurm-agent.git
cd vasp-slurm-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
vasp-agent ui
```

The app listens on `127.0.0.1` for local use. It has no multi-user login and
should not be exposed as a public service. Native Windows is not supported.

Your cluster's login and compute environments need Python 3.8 or later, Bash
and `sha256sum`. Slurm must provide `sbatch`, `squeue`, `sacct` and `scancel`.

The repository's `examples/` directory contains small CIF structures. They are
also included in the installed package. Without a source checkout, extract
the Si example with:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
Path("Si.cif").write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
PY
```

These files are starting structures, ready to use for input preparation.

## 2. Connect to your cluster

First check that you can log in over SSH from a terminal and verify the
cluster's host key. Open **Cluster setup** and enter:

| Setting | What to enter |
|---|---|
| SSH host, user and port | The details from your working SSH connection |
| Remote run directory | An absolute path where you can write calculation files |
| VASP command | The launch command used on your cluster, such as `srun vasp_std` |
| POTCAR directory and element mapping | Your licensed potential library and the potential to use for each element |
| Slurm partition and account | The queue and billing account you use |
| MPI tasks and walltime | Resources and time limit for each calculation stage |
| Environment commands | Any required module or environment setup, one command per line |

New configurations use 8 MPI tasks as a starting point for small examples.
Choose resources that suit your structure and your cluster's allocation rules.
Save the configuration, then run the environment check. This checks access
and required programs without submitting a calculation.

The sidebar's **Configuration file** defaults to
`~/.config/vasp-slurm-agent/cluster.json`, and **Run folder** defaults to
`~/vasp-slurm-agent-runs`. Both can be changed. Each prepared calculation saves
its own configuration, so later edits to the global settings do not alter it.

SSH keys or an SSH agent are the easiest way to connect. If your cluster needs
a password, enter it under **SSH password (optional)**. It stays in session
memory and is passed to operations you start; it is not written to the
configuration. **Clear password** removes it from the session. An already
running monitor keeps its copy until that process exits.

For the CLI, `vasp-agent watch <run_dir> --password` and
`vasp-agent doctor --password` prompt privately. The environment variable
`DFT_AGENT_SSH_PASSWORD` is also supported. Keep passwords out of configuration
files, command history and shared results. If your site requires interactive
MFA, follow its connection procedure; the app does not automate MFA.

## 3. Prepare and submit a calculation

Open **New calculation** and use **Upload a structure** to select a CIF or
POSCAR. Check the composition, cell and atomic positions in the preview.
Choose a **Calculation**:

| Task | What it does |
|---|---|
| `relax` | Optimizes atomic positions, with an option to relax the cell too |
| `scf` | Calculates the self-consistent electronic state of the supplied structure |
| `bands` | Runs SCF, then calculates bands along the displayed path |
| `dos` | Runs SCF, then calculates the density of states on the chosen k mesh |

All four use nonmagnetic PBE (`ISPIN=1`) for ordered periodic structures with
fully occupied sites. Bands and DOS use the supplied structure. If it needs
optimization, run `relax` first and use the resulting structure.

Review the cutoff, k mesh, electronic convergence threshold and smearing.
For relaxation, also choose the force threshold, step limit and whether to
relax the cell. The interface takes a positive force threshold and writes it
as negative `EDIFFG` in INCAR. The defaults help you get started; check their
suitability for any property you plan to study.

Click **Prepare inputs** to write the files locally. Under **Review settings**,
check the stages, calculation settings, account and resources. Tick
**I've reviewed the structure, settings and cluster resources.**, then click
**Submit calculation** to start the job and background monitor.

## 4. Follow a run

Open **Runs** to see saved progress and Slurm job IDs. The display refreshes
every five seconds. `queued`, `running` and `collecting` describe the current
stage. `succeeded` means the task's output checks passed. For `needs_attention`
or `failed`, read the error and stage results before deciding what to do next.

Closing the browser leaves remote jobs running. If your computer or monitor
restarts, select the original run and click **Resume monitoring**. It uses the
saved job identity, so there is no need to create the calculation again.

After three consecutive connection or collection failures, monitoring pauses
as `needs_attention`. Check the error and click **Reconnect** to find the same
job or try collecting its outputs again. Reconnecting keeps the calculation
settings; it cannot fix a calculation that did not converge.

**Cancel calculation** requests cancellation of the current run. Wait for
confirmation in the saved status and cluster queue. If needed, check with
your cluster's `squeue` or `sacct` command.

## 5. View and download results

Successful relaxation and SCF stages show the final energy, maximum atomic
force and final structure. Use **Download structure (.cif)** to save the
structure separately. Bands and DOS show plots, data and the preceding SCF
Fermi energy when available. If a stage fails its checks, the app shows the
reason instead of presenting its structure as a successful result.

Click **Prepare download**, then **Download results (.zip)** to save the
current results. The ZIP contains prepared inputs, retained solver outputs,
plots, numerical data and `run.json`. It can include an incomplete run, so
check its status. POTCAR is excluded; CHGCAR and WAVECAR stay on the cluster
when present. Review account names and paths in the metadata before sharing.

## 6. Use the CLI

The CLI provides `init`, `doctor`, `prepare`, `watch`, `resume`, `status`,
`cancel` and `bundle`. To see the arguments:

```bash
vasp-agent --help
vasp-agent prepare --help
vasp-agent watch --help
```

`prepare` writes local inputs. Review them before running `watch`, which can
submit the calculation. `status` reads saved progress; `resume` reconnects
and continues monitoring. Follow the [Si, Al and MgO examples](examples.md)
for complete commands and result locations.

## Scope and validation

Magnetism, DFT+U, spin–orbit coupling, hybrid functionals, defects, phonons,
NEB and molecular dynamics are outside the four workflows. The app stops on
failed or unconverged calculations without changing their settings.

The [v0.1.0 validation report](validation.md) describes the completed cluster
tests, including limitations and failed attempts. Version 0.1.1 updates the
interface and documentation while keeping those calculation workflows unchanged.
Code and documentation use the [MIT License](../LICENSE). See
[NOTICE.md](../NOTICE.md) for VASP and dependency terms and
[CITATION.cff](../CITATION.cff) for citation details.
