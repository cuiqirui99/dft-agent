# CLI examples

Follow the [quickstart](quickstart.md), then work in the repository directory.
Create `cluster.json` with `vasp-agent init --config cluster.json` and review
its VASP, POTCAR and Slurm settings. The PBE defaults are starting points.

## Prepare locally

Use a new directory for each run. `prepare` submits nothing.

**Si: relax atoms and cell**

```bash
vasp-agent prepare examples/Si.cif runs/si-relax --config cluster.json \
  --task relax --parameters '{"cell_relax": true}'
```

Omit `--parameters` to keep the cell fixed.

**Al: SCF**

```bash
vasp-agent prepare examples/Al.cif runs/al-scf --config cluster.json --task scf
```

**MgO: SCF, then DOS**

```bash
vasp-agent prepare examples/MgO.cif runs/mgo-dos --config cluster.json --task dos
```

SCF and DOS use the supplied structure without relaxing it.

## Submit

Review `config.json` and each stage's `inputs/INCAR`, `KPOINTS` and `POSCAR`.
To change settings, prepare a new run. Choose the matching command:

```bash
vasp-agent watch runs/si-relax
vasp-agent watch runs/al-scf
vasp-agent watch runs/mgo-dos
```

**`watch` submits the calculation.** Add `--password` for a private prompt.

## Status, recovery and download

```bash
vasp-agent status runs/si-relax
vasp-agent resume runs/si-relax
vasp-agent bundle runs/si-relax
```

`status` reads saved progress. `resume` reconnects to the same run; inspect
`needs_attention` errors first. Substitute the Al or MgO directory as needed.

| Run | Results |
|---|---|
| Si | `runs/si-relax/01_relax/outputs/`: `final_structure.cif`, `relax_energy.csv`, `relax_energy.png`, `result.json` |
| Al | `runs/al-scf/01_scf/outputs/`: `result.json`, `final_structure.cif` |
| MgO | `runs/mgo-dos/02_dos/outputs/`: `dos.csv`, `dos.png`, `result.json`; SCF results in `01_scf/outputs/` |

`bundle` writes `results.zip` with inputs, retained outputs, plots and `run.json`.
Check the saved status before using results. POTCAR is excluded; CHGCAR and
WAVECAR stay remote.
