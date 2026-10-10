# VASPKIT

DFT Agent can run an installed copy of VASPKIT on your cluster and download
its tables with the calculation results.

1. Install [VASPKIT](https://vaspkit.com/installation.html) on the cluster, or
   ask your administrator for its location.
2. Enter `vaspkit` or its full path in **Cluster setup → VASPKIT executable**.
   Add a module command under **Environment setup commands** if needed.
3. Finish a bands or DOS calculation. Open **Runs → Data and files** and click
   **Run VASPKIT**.
4. Use **Download results** to save the tables, original VASP files, and logs.

| Calculation | Export |
| --- | --- |
| Bands | Band energies and high-symmetry labels |
| Projected bands | Element and orbital weights |
| DOS | Total and integrated DOS |
| Projected DOS | Element and orbital contributions |

Projected exports require `LORBIT = 10` or `11` and `PROCAR`. New bands and DOS
calculations include these settings. Older calculations without projections
must be rerun to obtain them.

VASPKIT runs in separate working folders. Original calculation files and your
VASPKIT settings are kept unchanged. For explicit band paths, each original
segment is processed separately and joined at its original k-point distances.
Energies and orbital weights are not interpolated. Each export includes the
tool version, input hashes, commands, logs, and k-point mapping.

The adapter currently supports nonmagnetic and collinear calculations.
SOC, noncollinear, and hybrid-band exports use DFT Agent's built-in analysis.
Missing inputs or failed processing are reported without marking the
calculation itself as failed.

From the command line:

```bash
dft-agent postprocess runs/my-run 03_bands --executable /path/to/vaspkit
dft-agent postprocess runs/my-run 04_dos --executable /path/to/vaspkit
```

VASPKIT is distributed separately under its own
[usage agreement](https://vaspkit.com/installation.html#usage-agreement).
When using its results in a publication, cite
[Wang et al., Computer Physics Communications 267, 108033 (2021)](https://doi.org/10.1016/j.cpc.2021.108033).
See the [VASPKIT tutorials](https://vaspkit.com/tutorials.html) for the underlying
band and DOS formats.
