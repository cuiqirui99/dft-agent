# DFT Agent

![DFT Agent](docs/assets/dft-agent-logo.png)

Version `0.3.3`.

[Qirui Cui](https://www.kth.se/profile/qiruic?l=en) · KTH Royal Institute of Technology · [ORCID](https://orcid.org/0009-0005-6165-3237)

The DFT agent automates first-principles calculations on HPC clusters, requiring only a simple natural language description of the desired calculation.

Describe a calculation, review the plan, and run VASP on your Slurm cluster. DFT Agent handles relaxation, SCF, bands and DOS with magnetism, SOC, DFT+U, HSE06 or PBE0.

Requires macOS or Linux, Python 3.11+, SSH, and access to licensed VASP and POTCAR files on your cluster.
Bring your own cluster and model account. DFT Agent includes no compute time or model tokens.

## Install and open

```bash
git clone https://github.com/cuiqirui99/dft-agent.git
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[agent]'
dft-agent ui
```

## Connect a model

Open **Model** in the sidebar. Choose your provider:

| Provider | What to enter | Usage |
| --- | --- | --- |
| OpenAI | Your API key and model name; leave API URL blank | Your OpenAI API account |
| Claude, Grok, GLM or DeepSeek | Your provider's API key and model name | Your provider's API account |
| Qwen | Your key, model name and regional API URL | Your Model Studio account |
| Codex CLI | Run `codex login` first; model name can be blank | Your CLI account's plan or API billing |
| Compatible API | Your provider's key, model name and base URL | Your provider's account |

An **API key** gives the app access to your account. **Tokens** measure how much
the model reads and generates. You do not paste tokens into the app.
Consumer chat subscriptions do not automatically include API credits.

**[Model setup: get a key, connect and check usage](docs/models.md).**

Try **New calculation → Example → Si**, choose **Agent**, enter
“Relax this structure with PBE,” then click **Plan**. This uses your model account
but needs no cluster and submits no calculation. You should see a plan or a
question, followed by **Model usage**.

## Use

1. Enter your SSH, VASP and Slurm settings in **Cluster setup**.
2. Open **New calculation**. Upload a CIF or POSCAR, or choose **Example**.
3. Select a provider in **Model**, enter your **Goal**, and click **Plan**. Or use **Manual** without a model.
4. Review the plan and click **Prepare inputs**.
5. Check the inputs, tick the confirmation box and click **Submit calculation**.
6. Follow progress in **Runs**, explain the results, and download the structure.

Use **Edit structure** to change geometry or convert CIF/POSCAR before planning.
[Browse the 17 examples](docs/structures.md).

After an interruption, select the original run and use **Resume monitoring**. For `needs_attention`, read the error before using **Reconnect**.

Plans include scientific guidance and relevant past runs. For a failed calculation,
use **Repair** to review a proposed fix and prepare a new run. [Details](docs/harness.md).

**Memory** includes checked guidance and calculation cases. Search their evidence
or import selected local records. Rejected and unverified lessons stay out of
recommendations. [Memory](docs/memory.md).

Try: “Relax this structure, then calculate its bands with SOC.” Plans can be revised before submission. Missing settings, such as U and J, trigger a question.

Supports OpenAI, Claude, Qwen, Grok, GLM, DeepSeek, compatible APIs and a logged-in Codex CLI. Planning shares your goal and structure; explanations share verified results and dialogue. Manual mode needs no model account; install with `pip install .`.

Model calls use compact context and record reported token counts. [Model use](docs/token-use.md).

Use ordered periodic structures. Check the starting parameters for your material. Magnetic comparisons test NM, FM and one AFM seed; they do not search all magnetic orders. Keep the app local.

## Citation

Cui, Q. (2026). *DFT Agent* (v0.3.3). [GitHub](https://github.com/cuiqirui99/dft-agent/releases/tag/v0.3.3).

[Archived v0.3.0](https://doi.org/10.5281/zenodo.23260917).

[Quickstart](docs/quickstart.md) · [Methods](docs/methods.md) · [CLI examples](docs/examples.md) · [Validation](docs/validation-0.3.1.md) · [MIT license](LICENSE) · [Notices](NOTICE.md) · [Citation](CITATION.cff)
