# Scientific guidance

Plans use local VASP notes, structure hints and relevant saved runs. Open
**Scientific guidance** to see method choices, checks and sources. The report
is saved with the plan. Past runs supply advice, never automatic parameter changes.

History comes from the selected **Run folder**. Results are rechecked before use;
changed or rejected outputs cannot supply successful examples. Magnetic moments
and spin axes require compatible methods and site geometry.

For a failed run, open **Repair**, then **Plan repair**. Review the proposed change,
click **Prepare repair**, and submit the new run after checking its inputs.
The original run stays intact. A repair chain allows at most two attempts.

Repairs can increase electronic or ionic iteration limits, or the Slurm time limit.
They keep the method, structure, k-points and convergence thresholds unchanged.
Unknown failures, uncertain job status, magnetic comparisons and failed PBE spectra
that need the original charge density require manual review. Reconnect first if
the original job has not been collected.

CLI:

```bash
dft-agent plan examples/Si.cif "Relax, then SCF" --provider codex --runs ./runs
dft-agent repair ./runs/failed --provider codex --output repair.json
dft-agent prepare-repair ./runs/failed ./runs/repaired --plan repair.json
dft-agent watch ./runs/repaired
```

This adapts the earlier project's document guidance, scientific reports, scoped
experience and repair planning. It keeps the current executor and its result checks.
It does not import the old private corpus, run unrestricted model tools, or search
the literature automatically. See [sources and method limits](harness-sources.md).

Validation: 355 offline tests passed. Real model checks covered HSE06, SOC and
missing U/J values. On Slurm, a deliberately unconverged Si SCF was rejected;
the reviewed NELM increase produced a converged run with unchanged structure,
k-points and thresholds. Ionic and time-limit repairs were tested offline.
See the [validation record](../validation/results/harness.json).
