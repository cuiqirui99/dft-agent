# Quickstart

## 1. Install

Use macOS or Linux with Python 3.11+ and SSH:

```bash
git clone https://github.com/cuiqirui99/vasp-slurm-agent.git
cd vasp-slurm-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
vasp-agent ui
```

Keep the app local on `127.0.0.1`. The cluster needs licensed VASP and POTCAR
files, Python 3.8+, Bash, `sha256sum`, and Slurm's `sbatch`, `squeue`, `sacct`
and `scancel` commands.

## 2. Connect to your cluster

Check SSH access and the host key first. In **Cluster setup**, enter:

- SSH host, user and port.
- A writable remote run directory.
- VASP command, POTCAR directory and element mapping.
- Slurm partition, account, MPI tasks and walltime.
- Required environment commands, one per line.

Start with 8 MPI tasks for small examples, adjusting to your cluster's rules.
Save and run the environment check; it submits no jobs.

Use SSH keys or **SSH password (optional)**. Passwords stay in memory;
**Clear password** clears the session copy, while running monitors keep theirs
until exit. Add `--password` to CLI `watch`, `resume`, `doctor` or `cancel`. Keep
credentials out of saved files. Interactive MFA requires your site's setup.

Set **Configuration file** and **Run folder** in the sidebar. Each run saves
its own configuration.

## 3. Prepare and submit

In **New calculation**, upload a CIF or POSCAR. Try `examples/Si.cif`.
To extract it from an installed package:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
Path("Si.cif").write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
PY
```

Choose `relax`, `scf`, `bands` or `dos`. Bands and DOS run SCF first, using the
supplied structure; optimize it separately if needed.

Check the structure, cutoff, k mesh, smearing and convergence thresholds.
For relaxation, also set the step limit and whether to relax the cell.
The positive force threshold becomes negative `EDIFFG` in INCAR.

Click **Prepare inputs**, review the settings, tick the confirmation box,
then click **Submit calculation**. Preparation alone submits nothing.

## 4. Monitor and download

Open **Runs** to follow progress. Closing the browser leaves jobs running.
Keep your computer awake to monitor and run later stages.
After a local restart, select the original run and use **Resume monitoring**.
Three consecutive connection or collection failures pause monitoring as `needs_attention`;
read the error, then use **Reconnect**. Both actions keep the same job and settings.

Use **Cancel calculation** to request cancellation and check the resulting
status. For results, use **Download structure (.cif)** or **Prepare download**
followed by **Download results (.zip)**. Check the saved status before using
an incomplete run's files.

The ZIP includes inputs, retained outputs, plots and `run.json`. POTCAR is
excluded; CHGCAR and WAVECAR stay remote. Check account names and paths before sharing.

## CLI

```bash
vasp-agent --help
vasp-agent prepare --help
vasp-agent watch --help
```

See the [CLI examples](examples.md). `prepare` is local; `watch` submits.
`status` reads saved progress, and `resume` reconnects.

## Scope

Nonmagnetic PBE for ordered, fully occupied periodic structures. Defaults need
checking for your material. Magnetism, DFT+U, SOC, hybrids, defects, phonons,
NEB and molecular dynamics are outside these workflows. Failed or unconverged
calculations stop without automatic parameter changes.
