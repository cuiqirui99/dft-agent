# Three CLI examples

Start in the repository directory with `vasp-agent` installed and a reviewed
`cluster.json` for your licensed VASP installation and Slurm account. Create that
configuration with `vasp-agent init --config cluster.json` if needed, then set
your cluster's environment commands, POTCAR mapping and resource limits. See the
[installation guide](../README.md#install-and-open) or [Chinese quickstart](quickstart.zh-CN.md).

These examples use nonmagnetic PBE. The default cutoff, k mesh, smearing and
convergence thresholds are starting settings; choose and check them for your
material. Recorded live acceptance is tracked in the [campaign record](../validation/results/campaign.json).

## Prepare locally

`prepare` writes local inputs and a configuration snapshot. It does not connect
to the cluster or submit a job. Use a new run directory for each new calculation.

**Si: structure and cell relaxation**

```bash
vasp-agent prepare examples/Si.cif runs/si-relax --config cluster.json \
  --task relax --parameters '{"cell_relax": true}'
```

This explicitly allows the cell to relax. Omit the parameter override to keep
the supplied cell fixed while optimizing atomic positions.

**Al: static self-consistent calculation**

```bash
vasp-agent prepare examples/Al.cif runs/al-scf --config cluster.json --task scf
```

SCF uses the supplied structure without optimizing it. Review the k mesh and
smearing settings for this metallic example before submission.

**MgO: SCF followed by density of states**

```bash
vasp-agent prepare examples/MgO.cif runs/mgo-dos --config cluster.json --task dos
```

This prepares `01_scf` and `02_dos`; the DOS stage uses the preceding SCF charge
density. It does not insert a structure optimization step.

## Review, then submit

For your chosen example, inspect each stage's `inputs/INCAR`, `inputs/KPOINTS`
and `inputs/POSCAR`, plus the run's `config.json` for the cluster and resources.
If settings need changing, prepare and review a new run; do not edit a prepared
run's frozen inputs in place.

After that review, run the matching command below. **`watch` submits a prepared
run and follows its stages; it is not a read-only status command.**

```bash
vasp-agent watch runs/si-relax
vasp-agent watch runs/al-scf
vasp-agent watch runs/mgo-dos
```

Choose the line for the calculation you intend to run. Password-only sites can
add `--password` for a private prompt; keep credentials out of configuration files.

## Check, reconnect and export

For example, after starting the Si calculation:

```bash
vasp-agent status runs/si-relax
vasp-agent resume runs/si-relax
vasp-agent bundle runs/si-relax
```

`status` only reads saved state. Use `resume` when monitoring was interrupted or
the run paused: it reconnects to the same run without changing its scientific
parameters. `needs_attention` requires inspection; reconnecting does not make an
unconverged calculation successful. Substitute `runs/al-scf` or `runs/mgo-dos` for
the other examples.

| Example | Main result files after accepted completion |
| --- | --- |
| Si relaxation | `runs/si-relax/01_relax/outputs/final_structure.cif`, `relax_energy.csv`, `relax_energy.png`, `result.json` |
| Al SCF | `runs/al-scf/01_scf/outputs/result.json` and `final_structure.cif` |
| MgO DOS | `runs/mgo-dos/02_dos/outputs/dos.csv`, `dos.png`, `result.json`; preceding SCF results are under `01_scf/outputs/` |

The run's `run.json` retains stage status and Slurm job IDs. `bundle` writes
`results.zip` in that run directory, containing prepared inputs, retained solver
outputs, plots, numerical results and `run.json`. POTCAR is not distributed;
CHGCAR and WAVECAR, when present, remain remote and are excluded from the ZIP. A bundle can also
capture an incomplete run, so check its saved status before interpreting results.
