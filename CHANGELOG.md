# Changelog

## 0.1.0a1 — Unreleased

- New standalone scope: local single-user VASP workflows on Slurm for relaxation, SCF, bands and DOS.
- Local preparation and explicit submission, persisted task status, background monitoring, cancellation and result export.
- Streamlit interface, Python package entry point, offline checks and wheel-build CI configuration.
- Accepted-stage energy/force display, final-structure preview and CIF download.
- CIF examples included in source and wheel distributions; AppTest covers import, explicit submission review and result display without remote jobs.

- Si relaxation completed on BSCC with convergence and file-integrity checks; its scoped record is in `validation/results/Si-relax.json`.
- Paused runs can reconnect to their existing job identity and repeat collection without changing scientific parameters.
