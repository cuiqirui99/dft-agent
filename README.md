# DFT Agent

Version `0.1.3`.

Run VASP on a Slurm cluster from a local app. Optimize structures, run SCF, calculate bands or DOS, and download the results. No LLM account is needed.

Requires macOS or Linux, Python 3.11+, SSH, and access to licensed VASP and POTCAR files on your cluster.

## Install and open

```bash
git clone https://github.com/cuiqirui99/dft-agent.git
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
dft-agent ui
```

## Use

1. Enter your SSH, VASP and Slurm settings in **Cluster setup**.
2. Open **New calculation** and upload a CIF or POSCAR. Try `examples/Si.cif`.
3. Choose a calculation and click **Prepare inputs**. This only writes local files.
4. Review the settings, tick the confirmation box and click **Submit calculation**.
5. Follow progress in **Runs** and download the structure or results ZIP.

After an interruption, select the original run and use **Resume monitoring**. For `needs_attention`, read the error before using **Reconnect**.

Supports nonmagnetic PBE for ordered periodic structures. Bands and DOS include SCF but do not optimize the structure first. Default parameters are starting settings; check them for your material. Keep the app local: it has no multi-user login.

[Quickstart](docs/quickstart.md) · [CLI examples](docs/examples.md) · [v0.1.0 validation](docs/validation.md) · [MIT license](LICENSE) · [Notices](NOTICE.md) · [Citation](CITATION.cff)
