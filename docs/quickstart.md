# Quickstart

## 1. Install

Use macOS or Linux with Python 3.11+ and SSH:

```bash
git clone https://github.com/cuiqirui99/dft-agent.git
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[agent]'
dft-agent ui
```

Keep the app local on `127.0.0.1`. The cluster needs licensed VASP and POTCAR
files, Python 3.8+, Bash, `sha256sum`, and Slurm's `sbatch`, `squeue`, `sacct`
and `scancel` commands.

## 2. Connect a model

DFT Agent includes no model tokens. Use your own API key or Codex CLI login.
Follow [Model setup](models.md) for account, billing and key instructions.
An API key is your credential; tokens measure model usage.

In the sidebar, open **Model** and choose OpenAI, Claude, Qwen, Grok, GLM or
DeepSeek. Enter your provider's API key and model name. Leave **API URL** blank
for the preset endpoint; Qwen needs the URL for your Model Studio region.
[Model setup](models.md) lists the account links and URLs, including BigModel
for GLM in China. For Codex, run `codex login` first and choose **Codex CLI**.
Other services can use **Compatible API** if they support strict JSON-schema
output through Chat Completions.

Test before connecting a cluster: open **New calculation**, select
**Example → Si**, choose **Agent**, and enter “Relax this structure with PBE.”
Click **Plan**. A plan or a follow-up question confirms the connection.
**Model usage** shows reported token counts. This uses your model account but
submits no calculation. To work without a model, choose **Manual**.

## 3. Connect to your cluster

Check SSH access and the host key first. In **Cluster setup**, enter:

- SSH host, user and port.
- A writable remote run directory.
- VASP command, POTCAR directory and element mapping.
- SOC / noncollinear command, using `vasp_ncl`, if needed.
- Slurm partition, account, nodes, total MPI tasks and walltime.
- **Extra Slurm options** for memory, QoS or other resources, one per line.
- Required environment commands, one per line.

Start with 8 MPI tasks for small examples, adjusting to your cluster's rules.
Save and run the environment check; it submits no jobs.
For example, use `--mem=16G` or `--qos=normal` in **Extra Slurm options**.
Keep `#SBATCH` directives out of environment commands. In a configuration file,
use `"nodes": 2` and `"extra_sbatch": ["--mem=16G", "--ntasks-per-node=8"]`
with `"tasks": 16`. Your VASP launch command must support the requested resources.

Use SSH keys or **SSH password (optional)**. Passwords stay in memory;
**Clear password** clears the session copy, while running monitors keep theirs
until exit. Add `--password` to CLI `watch`, `resume`, `doctor` or `cancel`. Keep
credentials out of saved files. Interactive MFA requires your site's setup.

For an existing authenticated SSH master, set **SSH control socket (optional)**
to its absolute socket path. The app reuses it and leaves it open when finished.
If that session expires, authenticate again before reconnecting.

Set **Configuration file** and **Run folder** in the sidebar. Each run saves
its own configuration.

## 4. Prepare and submit

In **New calculation**, upload a CIF or POSCAR, or choose **Example**.
All [17 examples](structures.md) are included in the installed package.
Select a material to preview it; **Download input** saves its structure for
command-line use.

In **Agent** mode, describe the structure edits and calculations together in
**Goal**, then click **Plan**. Review the edited structure, stage methods and
any comparison points. Use **Change the plan** to answer questions or revise
the whole task. Edits are reapplied to the original upload on each revision.
[Workflows](workflows.md).

In **Edit structure**, describe a change and click **Plan structure**. Answer any
questions in **Follow-up**, then **Preview structure**. Check the cell and site
order before clicking **Use structure**. Each revision starts from the original
input. Applying an edit clears the calculation plan and site moments.

For format conversion, choose **Convert to** and click **Convert**. Download the
CIF or POSCAR, or use it for a calculation. **Plan structure** uses the model;
applying an edit and converting formats run locally. Neither needs a cluster
connection. Calculations use the edited POSCAR to preserve site order and the
Cartesian frame.

Later stages use the accepted relaxed structure. Each stage can have its own
method. Planning sends the goal, structure summary and dialogue to your provider.

Or choose **Manual** for `relax`, `scf`, `bands` or `dos` without a model.
The **Method** section sets spin, SOC, U/J and the functional.
PBE bands and DOS run SCF first. Hybrid spectra are self-consistent.

Check the structure, cutoff, k mesh, smearing and convergence thresholds.
The k mesh is automatic by default; an explicit grid overrides it.
Choose **Electronic type** when known, or leave it at **auto** for Gaussian smearing.
For relaxation, also set the step limit and whether to relax the cell.
The cell stays fixed unless **Relax the cell** is selected.
The positive force threshold becomes negative `EDIFFG` in INCAR.

Click **Prepare inputs**, review the settings, tick the confirmation box,
then click **Submit calculation**. Preparation alone submits nothing.

## 5. Monitor and download

Open **Runs** to follow progress. Closing the browser leaves jobs running.
Keep your computer awake to monitor and run later stages.
After a local restart, select the original run and use **Resume monitoring**.
Three consecutive connection or collection failures pause monitoring as `needs_attention`;
read the error, then use **Reconnect**. Both actions keep the same job and settings.

Use **Cancel calculation** to request cancellation and check the resulting
status. Use **Explain results** to review findings, then ask a follow-up question.
Explanations use verified saved outputs and never submit calculations.
For a batch, review its comparison table and open individual runs for details.
Use **Continue** to plan a new calculation from an accepted stage. Review and
prepare it before submission; the original run stays intact.
For files, use **Download structure (.cif)** or **Prepare download**
followed by **Download results (.zip)**. Check the saved status before using
an incomplete run's files.

The ZIP includes inputs, retained outputs, plots, plans and saved explanations. POTCAR is
excluded; CHGCAR and WAVECAR stay remote. Check account names and paths before sharing.

## CLI

```bash
dft-agent --help
dft-agent prepare --help
dft-agent watch --help
```

See the [CLI examples](examples.md). `prepare` is local; `watch` submits.
`status` reads saved progress, and `resume` reconnects.

## Updates

For a cloned install, activate its environment and run:

```bash
git pull --ff-only
python -m pip install '.[agent]'
```

This updates the checked-out branch. New features reach `main` after review;
a new release then provides versioned installation files on
[GitHub](https://github.com/cuiqirui99/dft-agent/releases).
`main` is the maintained version; temporary development branches are removed
after merging. Release tags identify earlier versions without extra branches.
If you installed a release wheel, download the new wheel and install it in the
same environment with `python -m pip install --upgrade 'PATH_TO_WHEEL[agent]'`.
Updating does not restart saved calculations.

The app is available to researchers with their own cluster and VASP access.
Model features also need a model account. Provider support includes request
format and local output checks; an account's available models and limits vary.

## Scope

Ordered, fully occupied periodic structures, with PBE, magnetism, SOC,
Dudarev DFT+U, HSE06 and PBE0. See [Methods](methods.md) for settings and limits.
The structure editor can prepare substitutions, vacancies and slabs from an input
crystal. Phonons, NEB and molecular dynamics are not implemented.
Failed or unconverged calculations stop without automatic parameter changes.
