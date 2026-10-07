# Independent numerical reference

These scripts do not import the application or pymatgen. They independently
author plain-text VASP inputs and use ASE's OUTCAR and XML readers plus Python's
XML parser. Install ASE in the validation environment (`pip install 'ase>=3.23,<4'`).
They do not submit jobs, download files, or read/copy a licensed POTCAR.

## Same-model SCF

Run Si, Al and MgO with the unchanged frozen protocol. Use each application's
**input POSCAR**, not its relaxed CONTCAR, as the reference structure. This keeps
cell orientation, site order, and force-vector coordinates identical while the
reference INCAR/KPOINTS are independently authored.

```sh
python validation/reference/prepare_reference.py \
  --material Si --source-poscar /private/agent-scf/inputs/POSCAR \
  --destination /private/reference-Si/inputs
```

Repeat for `Al` and `MgO`. The operator submits these files through the same
approved VASP runtime, resolving the same POTCAR on the remote machine. Preserve
the original inputs, complete OUTCAR/XML/CONTCAR, `potcar_hash.sha256`,
`execution.json` and the SHA-256/size `artifact_manifest.json`. Do not regenerate
the reference inputs through the application before submission.

```sh
python validation/reference/compare_reference.py compare \
  --agent /private/agent-scf/outputs --reference /private/reference-Si/outputs \
  --output /private/reference-Si/comparison.json
```

The comparator checks both complete independent parses, solver termination and
electronic convergence, every retained artifact receipt, identical input
structure/model/mesh/POTCAR identity, then applies the existing protocol's
**1 meV/atom energy** and **0.01 eV/angstrom maximum force-vector difference**.
An agreement pass establishes software/model consistency, not physical accuracy.

Use `--variant higher-cutoff`, `denser-mesh`, or `combined` to prepare the frozen
520 eV and/or 6x6x6 sensitivity cases, and `compare --mode sensitivity` to record
their differences. Sensitivity reports are measurements without a convergence
pass claim or an experimental comparison. Preserve the original baseline.

## Nontrivial ionic relaxation

The initial Si cell test succeeded with three cell steps, but all ionic forces
were zero by symmetry. The separate
`internal-relaxation-protocol.v1.json` therefore freezes Si and MgO cases with a
0.10 angstrom displacement of the last atom. Generate their POSCARs with
`prepare_reference.py --task displaced-relax`. Feed that POSCAR to the application
with **cell_relax=false** and the frozen baseline parameters. This exercises real
ionic motion rather than only a symmetric cell-volume optimization.

```sh
python validation/reference/compare_reference.py audit \
  --run /private/displaced-relax/outputs --require-internal-relaxation \
  --output /private/displaced-relax/independent-audit.json
```

The additional frozen gates require initial force above 0.03 eV/angstrom, at
least two ionic steps, actual internal coordinate motion above 0.01 angstrom,
decreased energy, final force at most 0.03 eV/angstrom and an unchanged cell.

## Recorded evidence

`si-relax-independent-review.v1.json` audits the actual first Si run using full
OUTCAR and XML independently of the application. All 15 retained artifact
hashes/sizes passed; energies and forces from the two ASE readers and the
application agree. Its three energies are -10.65673788, -10.66111132 and
-10.66523726 eV, and its volume changes from 40.0478695 to 41.1400102 angstrom³.
The atomic fractional coordinates do not move; this establishes cell relaxation
only. No independent SCF reference or displaced-atom run is claimed by this file.
`si-scf-independent-review.v1.json` separately audits the application's actual Si
SCF output at -10.65673788 eV; this is an independent parser check of that same
run, not a separately executed reference calculation.
The summary contains hashes and observables; private cluster credentials and
licensed pseudopotential bytes remain outside this repository.

## Input population and follow-up execution

The complete input population is **eight idealized baseline structures plus two
displaced structures** (Si and MgO). Three manual baseline references and three
combined cutoff/mesh sensitivity calculations reuse Si, Al and MgO; these six
extra solver runs do not count as six new materials or independent experiments.
Together with the two displaced relaxations, the follow-up adds eight serial
solver jobs after the primary 8-material/4-workflow campaign.

The private follow-up runner keeps cluster configuration and job identities
outside the repository. It uses the same campaign lock, refuses to start before
all primary workflows succeed, resumes persisted run identities, and stops on
unaccepted results without resubmitting or weakening the frozen tolerances.
Only whitelisted scientific metrics and input counts enter a public summary.
The combined 520 eV/6x6x6 comparison is a limited sensitivity measurement;
it does not isolate each numerical parameter or establish comprehensive
physical convergence.
