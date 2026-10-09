# Methods

Use these settings in a model plan, the **Method** form, or CLI `--parameters`.
Defaults are starting values, not convergence studies.

| Method | Settings |
|---|---|
| PBE | `"functional": "PBE"` (default) |
| Collinear spin | `"spin": "collinear"`, one `magmom` value per uploaded site |
| Noncollinear spin | `"spin": "noncollinear"`, one three-component `magmom` per site |
| SOC | `"soc": true`, `vasp_ncl_command` in the cluster config |
| DFT+U | `"hubbard_u": {"Ni": {"l": 2, "u": 5, "j": 0}}` |
| Hybrid | `"functional": "HSE06"` or `"PBE0"` |

Moments are in μB, in uploaded site order. Vectors use the SAXIS spinor basis;
`saxis` defaults to `[0, 0, 1]`. SOC with moments requires noncollinear spin.
SOC with `spin: none` uses a zero-moment seed; it does not constrain the final
magnetization. Noncollinear calculations also require `vasp_ncl`.
[VASP SOC guide](https://vasp.at/wiki/LSORBIT).

DFT+U uses Dudarev's `LDAUTYPE=2`, with U and J in eV. Supply both explicitly.
The Ni example above is illustrative. Element settings follow the actual
POSCAR/POTCAR order. [VASP DFT+U guide](https://vasp.at/wiki/LDAU).

HSE06 uses 25% exact exchange and screening of 0.2 Å⁻¹; PBE0 is unscreened.
Hybrid bands combine a weighted mesh and a zero-weight path. They are
self-consistent and do not use `ICHARG=11`. A minimum of ten electronic steps
is enforced. Bands use Davidson iteration and reject equivalent k points whose
energies disagree by more than 0.05 eV. This is a consistency check; convergence
along the full path still needs checking.
[VASP hybrid bands guide](https://vasp.at/wiki/Band-structure_calculation_using_hybrid_functionals).

## Magnetic comparisons

Compare NM, FM and AFM with SCF on the same structure and cell. AFM alternates
initial moments; a shared supercell is made if needed. The default seed is
3 μB per magnetic site and can be changed in a plan.

Only successful, compatible runs enter the energy-per-atom ranking. Seeds can
converge to the same state. Inspect final moments; this is not a ground-state
search over all magnetic orders. PAW-sphere moments are not the full-cell moment.

## Plans

Plans can chain relaxation, SCF, bands and DOS. PBE spectra reuse an SCF charge
density; hybrid spectra solve self-consistently. Magnetic comparisons currently
support SCF only. The same method settings apply throughout a chain.

The structure editor supports substitutions, vacancies and slabs. Defect formation
energies, phonons, NEB and MD remain outside this release.
