# Changelog

## 0.1.0 — Unreleased

- Standalone local VASP/Slurm workflows for nonmagnetic PBE relaxation, SCF, bands and DOS.
- Local input preparation and explicit submission, persisted job identity, bounded connection retries and SHA-256-verified transfers.
- Paused runs can reconnect to the same job and repeat collection without changing scientific parameters; potentially active jobs remain cancellable when monitoring pauses.
- Streamlit structure preview, final CIF download and result bundles containing retained inputs, outputs, figures and run records.
- Energy and force metrics are displayed for relaxation and SCF; bands and DOS display the available preceding SCF Fermi reference with their spectral results.
- CLI examples, bundled CIF inputs, offline and UI tests, and CI checks for wheel installation outside the source checkout.
- MIT license and software citation metadata.

Live workflow outcomes and artifact receipts are tracked in the [campaign record](validation/results/campaign.json); [independent numerical checks](validation/reference/README.md) are recorded separately.
