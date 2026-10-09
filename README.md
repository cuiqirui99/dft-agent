# DFT Agent

Version `0.2.1`.

The DFT agent automates first-principles calculations on HPC clusters, requiring only a simple natural language description of the desired calculation.

Describe a calculation, review the plan, and run VASP on your Slurm cluster. DFT Agent handles relaxation, SCF, bands and DOS with magnetism, SOC, DFT+U, HSE06 or PBE0.

Requires macOS or Linux, Python 3.11+, SSH, and access to licensed VASP and POTCAR files on your cluster.

## Install and open

```bash
git clone https://github.com/cuiqirui99/dft-agent.git
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[agent]'
dft-agent ui
```

## Use

1. Enter your SSH, VASP and Slurm settings in **Cluster setup**.
2. Open **New calculation** and upload a CIF or POSCAR. Try `examples/Si.cif`.
3. Select a provider in **Model**, enter your **Goal**, and click **Plan**. Or use **Manual** without a model.
4. Review the plan and click **Prepare inputs**.
5. Check the inputs, tick the confirmation box and click **Submit calculation**.
6. Follow progress in **Runs**, explain the results, and download the structure.

After an interruption, select the original run and use **Resume monitoring**. For `needs_attention`, read the error before using **Reconnect**.

Plans include scientific guidance and relevant past runs. For a failed calculation,
use **Repair** to review a proposed fix and prepare a new run. [Details](docs/harness.md).

Try: “Relax this structure, then calculate its bands with SOC.” Plans can be revised before submission. Missing settings, such as U and J, trigger a question.

Supports OpenAI, compatible APIs and a logged-in Codex CLI. Planning shares your goal and structure; explanations share verified results and dialogue. Manual mode needs no model account; install with `pip install .`.

Use ordered periodic structures. Check the starting parameters for your material. Magnetic comparisons test NM, FM and one AFM seed; they do not search all magnetic orders. Keep the app local.

[Quickstart](docs/quickstart.md) · [Methods](docs/methods.md) · [CLI examples](docs/examples.md) · [Validation](docs/validation.md) · [MIT license](LICENSE) · [Notices](NOTICE.md) · [Citation](CITATION.cff)
