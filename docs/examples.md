# Three CLI examples

Start in the repository directory with `vasp-agent` installed and a
`cluster.json` configured for your VASP installation and Slurm account. If you
need to create one, run `vasp-agent init --config cluster.json`, then set your
cluster's environment commands, POTCAR choices and resources. See the
[installation guide](../README.md#install-and-open) or [quickstart](quickstart.md).

These examples use nonmagnetic PBE. Start with the defaults, then check the
cutoff, k mesh, smearing and convergence thresholds for your material and the
property you want to calculate. Results from the v0.1.0 cluster tests are in the
[validation report](validation.md).

## Prepare locally

`prepare` writes inputs and a copy of the configuration locally. It does not
connect to the cluster or submit a job. Use a new directory for each calculation.

**Si: structure and cell relaxation**

```bash
vasp-agent prepare examples/Si.cif runs/si-relax --config cluster.json \
  --task relax --parameters '{"cell_relax": true}'
```

This allows the cell to relax along with the atomic positions. Omit
`--parameters` to keep the cell fixed.

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

This prepares `01_scf` and `02_dos`. The DOS stage uses the SCF charge density
and the same input structure. Run a separate relaxation first if needed.

## Review, then submit

For your chosen example, inspect each stage's `inputs/INCAR`, `inputs/KPOINTS`
and `inputs/POSCAR`, plus the run's `config.json` for the cluster and resources.
If settings need changing, prepare and review a new run so its saved settings
and inputs stay consistent.

After reviewing the inputs, choose the matching command below.
**`watch` submits the calculation and follows its stages.**

```bash
vasp-agent watch runs/si-relax
vasp-agent watch runs/al-scf
vasp-agent watch runs/mgo-dos
```

For password login, add `--password` to get a private prompt.

## Check, reconnect and export

For example, after starting the Si calculation:

```bash
vasp-agent status runs/si-relax
vasp-agent resume runs/si-relax
vasp-agent bundle runs/si-relax
```

`status` reads saved progress. Use `resume` after an interruption or a paused
connection: it reconnects to the same run without changing the calculation
settings. If the state is `needs_attention`, read the error first; reconnecting
will not resolve a convergence problem. Use `runs/al-scf` or `runs/mgo-dos` for
the other examples.

| Example | Main result files after successful completion |
| --- | --- |
| Si relaxation | `runs/si-relax/01_relax/outputs/final_structure.cif`, `relax_energy.csv`, `relax_energy.png`, `result.json` |
| Al SCF | `runs/al-scf/01_scf/outputs/result.json` and `final_structure.cif` |
| MgO DOS | `runs/mgo-dos/02_dos/outputs/dos.csv`, `dos.png`, `result.json`; preceding SCF results are under `01_scf/outputs/` |

The run's `run.json` records stage status and Slurm job IDs. `bundle` writes
`results.zip` in that run directory, containing prepared inputs, retained solver
outputs, plots, numerical data and `run.json`. POTCAR is excluded. CHGCAR and
WAVECAR stay on the cluster when present and are not included in the ZIP. You
can also download an incomplete run, so check its saved status before using
the results.
