# Changelog

## 0.1.1 — 2026-10-07

- The interface is now entirely in English, with shorter labels and clearer instructions throughout.
- Rewrote the quickstart and examples in plain English, and made CLI help and error messages more useful.
- Calculation settings, job handling and numerical checks are unchanged. The original v0.1.0 cluster results remain available in the [validation report](docs/validation.md).

## 0.1.0 — 2026-10-07

- Standalone local VASP/Slurm workflows for nonmagnetic PBE relaxation, SCF, bands and DOS.
- Local input preparation and explicit submission, persisted job identity, bounded connection retries and SHA-256-verified transfers.
- Paused runs can reconnect to the same job and repeat collection without changing scientific parameters; potentially active jobs remain cancellable when monitoring pauses.
- Streamlit structure preview, final CIF download and result bundles containing retained inputs, outputs, figures and run records.
- Energy and force metrics are displayed for relaxation and SCF; bands and DOS display the available preceding SCF Fermi reference with their spectral results.
- CLI examples, bundled CIF inputs, offline and UI tests, and CI checks for wheel installation outside the source checkout.
- MIT license and software citation metadata.

The [validation report](docs/validation.md) links the completed 32-workflow campaign, independent numerical checks, live failure recovery and real installed-wheel calculation. It documents numerical sensitivity and retained failed attempts.
