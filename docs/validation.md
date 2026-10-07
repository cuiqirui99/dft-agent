# v0.1.0 validation

The first release prioritizes a usable path from an input structure to a completed
VASP calculation and downloadable structure. Acceptance was completed on
2026-10-07 with VASP 6.2.1 and PBE.54 potentials on one Slurm cluster. These are
workflow and implementation checks; they do not establish universal material
accuracy or compatibility with every cluster.

## Completed checks

| Check | Observed result | Evidence |
|---|---|---|
| Eight materials, four tasks each | 32/32 workflows accepted; 48 successful solver stages | [Campaign](../validation/results/campaign.json) |
| Independent output parsing and receipts | All 48 stages checked; 720 file receipts matched actual hashes and sizes | [Evidence audit](../validation/reference/evidence-audit.v1.json) |
| Independently authored SCF references | Si, Al and MgO passed the frozen energy and force comparison gates | [Reference summary](../validation/reference/campaign-summary.v1.json) |
| Nontrivial ionic relaxation | Displaced Si and MgO moved internally, lowered energy and reached the frozen force threshold with the cell fixed | [Reference summary](../validation/reference/campaign-summary.v1.json) |
| Complete transfer | A 4 MiB SSH round trip retained identical bytes | [Transport](../validation/results/transport.json) |
| Monitor interruption | Restart resumed the same existing solver job | [Restart](../validation/results/restart.json) |
| Lost submission reply, reconnect and cancellation | A real scheduler job retained one identity across process restarts and bounded injected offline errors; cancellation was confirmed | [Recovery](../validation/results/recovery.json) |
| Timeout and missing output | Actual Slurm TIMEOUT and a COMPLETED job without VASP output were both rejected, with no automatic resubmission | [Failure cases](../validation/results/failures.json) |
| Isolated wheel installation | The release wheel completed a real displaced Si relaxation, recovered its outputs and produced a result bundle | [Installed wheel](../validation/results/installed-wheel.json) |

The eight baseline structures are Si, C, Ge, Al, Cu, MgO, NaCl and SiC.
Each ran `relax`, `scf`, `bands` and `dos`; bands and DOS include a preceding
SCF stage. The separate reference campaign adds three manual baseline SCFs,
three sensitivity SCFs and two displaced relaxations, rather than eight new
materials. See the [frozen protocol](../validation/protocol.v1.json) and
[reference procedure](../validation/reference/README.md).

The release wheel's SHA-256 is
`dca668ad66b5cb9efd104a51e07dc89fd33b8bd4077ebfb679a1bc0599c048be`.
It was built from code commit `2f7026832b69f7b6a49e44615832328405a2d7ea`;
the release tag adds final documentation and evidence without changing its
Python modules or bundled example structures. It was installed outside the
source checkout with dependency checks passing. Its real Si optimization used
eight ionic steps and reduced the maximum force from 0.79756 to
0.00946 eV/angstrom; composition, fixed cell, coordinates and input identity
were checked. This is an automated installation exercise, not an independent
human usability study.

Offline core, UI and reference checks recorded **138 passed, zero failed**.
The [CI workflow](../.github/workflows/ci.yml) runs these checks on Python
3.11, 3.12 and 3.13 and installs a wheel outside the checkout. Real-cluster
acceptance is recorded separately above and is not run by public CI.

## Precision and retained failures

The frozen main campaign used 400 eV, a 4x4x4 mesh, EDIFF=1e-5 and a
0.03 eV/angstrom force threshold. New application inputs default to 520 eV;
defaults remain practical starting settings, not a guarantee of convergence
for a particular property. The manual baseline references matched the agent
energies and forces at the reported output precision, testing the preparation
and parsing path under the same physical model.

Changing **both** cutoff to 520 eV and mesh to 6x6x6 produced these absolute
total-energy differences from the frozen baseline:

| Material | Difference (eV/atom) |
|---|---:|
| Si | 0.082133 |
| Al | 0.361898 |
| MgO | 0.003059 |

These differences are sensitivity measurements, not convergence passes.
In particular, the coarse Al baseline is unsuitable for small energy
comparisons without further convergence work. The combined change does not
separate cutoff and mesh effects; zero forces in these symmetric SCF examples
do not establish force accuracy for distorted structures.

Initial C relaxation and Cu band calculations failed during diagonalization
with 64 MPI tasks. The preserved attempts were followed by explicit resource
adjustments to eight tasks, with unchanged declared physical inputs. VASP's
automatic band allocation also changed, so this is not a single-cause diagnosis.
The replacement runs passed. See the [C adjustment](../validation/resource-adjustments.v1.json),
[Cu adjustment](../validation/resource-adjustments.v2.json) and
[actual outcomes](../validation/resource-adjustment-outcomes.v1.json).
This history is part of the acceptance record; the 32/32 count is not a claim
that every first attempt succeeded or that the application repaired them automatically.

The supported release scope remains local single-user, nonmagnetic PBE bulk
workflows. Detailed property convergence, magnetism, SOC and other advanced
methods require separate work. Earlier platform test failures are described in
the [legacy scope note](legacy-scope.md), not counted as passing this package's tests.
