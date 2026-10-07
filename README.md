# VASP Slurm Agent

A local, single-user workbench for preparing, submitting and following VASP calculations on an existing Slurm cluster. It provides four tasks: structure relaxation, self-consistent calculations, band structure and density of states. No LLM account is required.

**Status: `0.1.0a1`, development preview.** The initial scope is nonmagnetic PBE for ordered periodic structures, with practical starting parameters. The [live acceptance record](validation/results/campaign.json) tracks completed workflows, solver versions, convergence checks and artifact hashes; its completion fields describe the current evidence. This is a standalone project and does not depend on the older `AI_AGNET` workspace. Code and documentation are available under the [MIT License](LICENSE); external solver terms are described in [NOTICE.md](NOTICE.md).

## Install and open

Requires macOS or Linux, Python 3.11 or later, an SSH client, a Slurm account, and access to a licensed VASP installation and appropriate POTCAR files on the remote system. Native Windows is not supported.

The remote login and compute environments need Python 3.8 or later, Bash and `sha256sum`; Slurm must provide `sbatch`, `squeue`, `sacct` and `scancel`. Set any required module-loading commands in the cluster configuration.

Clone the repository, enter its directory and create a local environment. Authenticate with GitHub first if your repository access requires it.

```bash
git clone https://github.com/cuiqirui99/vasp-slurm-agent.git
cd vasp-slurm-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
vasp-agent ui
```

The UI binds to `127.0.0.1`. Keep it local: it is designed for one trusted user and is not an authenticated multi-user web service. See the [Chinese quickstart](docs/quickstart.zh-CN.md) for the complete workflow.

1. Configure the SSH host/user, remote run directory, POTCAR directory, VASP command and Slurm resources.
2. Upload a CIF or POSCAR and inspect the composition, lattice and atomic positions.
3. Select `relax`, `scf`, `bands` or `dos`; set numerical parameters and generate the local inputs.
4. Review the stages, settings and account, then explicitly submit. Preparation alone does not submit a job.
5. Follow persisted stage/job status. Restart monitoring after a local interruption, cancel a run, or download its result bundle.

The repository and both distribution formats include small CIF examples. To extract silicon from an installed wheel without the source checkout:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
Path("Si.cif").write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
PY
```

Upload that file to try local input preparation. The example structures are starting inputs, not converged reference results. Accepted results show the final energy, maximum atomic force and final structure, with a separate CIF download as well as the full result ZIP.

Closing the browser does not deliberately cancel remote jobs. The background monitor requires the local machine to remain available; after a restart, reopen the run and resume monitoring. Three consecutive remote-operation failures pause monitoring as `needs_attention`. Inspect the error, then use **重新连接 / 回收** to reconnect to the same job or collect its outputs again. Reconnection does not change scientific parameters.

## Configuration and credentials

The UI defaults to `~/.config/vasp-slurm-agent/cluster.json` and saves runs under `~/vasp-slurm-agent-runs`; both paths are editable. `VASP_AGENT_CONFIG` changes the UI's initial configuration path. Each prepared run retains its own configuration snapshot.

Use SSH keys or an SSH agent where possible. For password-only sites, enter the password in the UI sidebar; it stays in session memory and is passed to an explicitly started operation or worker. Clear it with the sidebar button when no longer needed. CLI users can use `vasp-agent watch <run_dir> --password` or `vasp-agent doctor --password` for a private prompt. The transport also accepts `DFT_AGENT_SSH_PASSWORD`; never put a password or private key in the configuration or a shared result bundle. Establish and verify the host's SSH identity before using the agent. Sites requiring interactive MFA may require a separately established session; this preview does not claim general MFA integration.

The configuration page includes an explicit SSH/Slurm/VASP environment check. It connects only when clicked and does not submit a job.

New configurations start with 8 MPI tasks, suitable for trying the small examples. Adjust this to your structure and cluster allocation rules. The initial carbon test failed during diagonalization with 64 tasks and completed with unchanged physics inputs at 8 tasks; the [resource adjustment record](validation/resource-adjustments.v1.json) preserves that attempt.

The VASP executable and POTCAR data remain on your cluster. Configure `potcar_symbols` for the intended potential choices. The software does not provide a VASP license, distribute VASP binaries, or grant permission to redistribute potential files. Review [NOTICE.md](NOTICE.md) before sharing this project or its artifacts.

## CLI

The same local workflow is available through `vasp-agent init`, `doctor`, `prepare`, `watch`, `resume`, `status`, `cancel`, and `bundle`. Run `vasp-agent <command> --help` for the supported arguments. `prepare` generates inputs locally; `watch` can submit a prepared run and follow its stages, so run it only after reviewing the prepared inputs. `resume` reconnects a paused run and continues monitoring; `status` is read-only.

## Scientific scope

`relax` and `scf` each run one stage. `bands` and `dos` run an SCF stage followed by the requested stage on the supplied structure; they do not implicitly relax it. All four recipes set `ISPIN=1`. Review ENCUT, k mesh, smearing and force tolerance for your material; property calculations intended for research need their own convergence checks.

Magnetism, DFT+U, hybrid functionals, spin–orbit coupling, defects, phonons, NEB and molecular dynamics are outside these recipes. Failed or unconverged runs stop for inspection instead of silently changing scientific parameters.

## Development and acceptance

```bash
python -m pip install ".[dev]"
python -m pytest -q
python -m build
```

CI runs offline core and Streamlit AppTest checks on Python 3.11–3.13 on Linux, then installs a built wheel outside the source checkout and prepares an included example without contacting a cluster. The [first hosted run for commit `4f320d7`](https://github.com/cuiqirui99/vasp-slurm-agent/actions/runs/37621743051) passed all three versions. Development dependencies require Streamlit 1.65 or later for AppTest's file-upload support.

Live acceptance follows the [frozen protocol](validation/protocol.v1.json). The [campaign record](validation/results/campaign.json) and [execution records](validation/results/) retain completed and incomplete outcomes. [Independent numerical checks](validation/reference/README.md) are recorded separately from software tests. The [legacy scope note](docs/legacy-scope.md) explains how this package relates to the earlier platform.

Software citation details are provided in [CITATION.cff](CITATION.cff). Report the installed version and the settings and evidence associated with your calculation when citing a result.
