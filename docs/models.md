# Connect a model

Anyone can install DFT Agent. To run calculations, you need your own Slurm
account, licensed VASP and POTCAR files. To use the agent, you also need one of
the model accounts below. The download does not include model credits or cluster
access. [Install and open the app](quickstart.md#1-install), then choose a route.

An **API key** is a credential for your model account. **Tokens** measure the
text a model reads and generates. You enter a key, not a number of tokens.

| Route | What you need |
| --- | --- |
| OpenAI | An OpenAI API account, billing and API key |
| Codex CLI | Codex installed and signed in on the computer running DFT Agent |
| Compatible API | A provider with the required API, plus its model name, URL and key |
| Manual | No model account; choose the calculation settings yourself |

## OpenAI

1. Sign in to the [OpenAI Platform](https://platform.openai.com/).
2. Set up [API billing](https://platform.openai.com/settings/organization/billing/overview).
   A ChatGPT subscription does not pay for direct API requests.
3. Create an [API key](https://platform.openai.com/api-keys). Keep it private.
4. In the app's **Model** panel, enter:

   | Field | Value |
   | --- | --- |
   | Provider | OpenAI |
   | Model name | An available model ID, for example `gpt-6.1-sol` |
   | API URL | Leave blank |
   | API key | Your key |

The model must support Responses and structured JSON output. Check that your
account can use the model you choose; the example is not an access guarantee.
See the [API quickstart](https://developers.openai.com/api/docs/quickstart) and
[model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

## Codex CLI

1. [Install Codex CLI](https://developers.openai.com/codex/cli) on the computer
   running DFT Agent.
2. In a terminal, run:

   ```bash
   codex login
   codex login status
   ```

3. Choose **Codex CLI** in **Model**. Leave **Model name** blank to use your
   configured model or the CLI default. API URL and API key are not used.

Signing in to ChatGPT in a browser does not sign the CLI in. Codex can use
eligible ChatGPT subscription access or an API key; API-key use is billed by
usage. See [Codex authentication](https://learn.chatgpt.com/docs/auth) for current
account options and limits. No Codex installation is needed on the cluster.

## Compatible API

Choose **Compatible API** and enter the provider's exact **Model name**, **API
URL** and **API key**. Use its API base URL, not its chat website or a full
`/chat/completions` endpoint.

The service must support Chat Completions with strict JSON-schema output and
the request options used by DFT Agent. An OpenAI-compatible label alone does
not guarantee this. Check the provider's documentation and billing dashboard.

## Try one plan

1. Open **New calculation**, choose **Example**, then **Si**.
2. Select **Agent** and enter `Relax this structure with PBE` in **Goal**.
3. Click **Plan**. A plan or a clarification question confirms a model response.
4. To change it, use **Change the plan** and click **Plan** again.

This test uses model tokens but needs no cluster connection and submits no VASP
job. To run the calculation, complete **Cluster setup**, review **Prepare
inputs**, then confirm **Submit calculation**. Follow the [quickstart](quickstart.md).

## Tokens and cost

Model calls happen when you click **Plan**, **Plan structure**, **Explain
results**, **Ask** or **Plan repair**, including follow-up requests. Input tokens
include your goal, structure, conversation and relevant guidance. Output tokens
cover the response, including reasoning where the provider reports it.

Preparing inputs, applying a structure edit, checking jobs, refreshing the page
and downloading results do not call a model. **Manual** calculations need no
model unless you choose a model action such as **Explain results**.

The app shows token counts after a response when available. Check your
[OpenAI usage](https://platform.openai.com/usage), Codex account limits or your
provider's dashboard for charges and remaining access. DFT Agent does not sell
tokens or set a spending cap. [Usage details](token-use.md).

API costs depend on the model's input and output rates and the tokens used.
Check [current prices](https://developers.openai.com/api/docs/pricing) before
starting. Begin with the Si example and review usage before larger requests.

## Keys and data

The key entered in the app stays in memory and is not saved in calculation
files. **Clear API key** clears that field. Environment keys still work:
`DFT_AGENT_API_KEY` takes precedence over `OPENAI_API_KEY` when the field is
empty. `DFT_AGENT_MODEL` and `DFT_AGENT_BASE_URL` can also fill the model settings.
Keep keys on the computer running the app; they are not needed on the cluster.

Model requests include the structure and task context needed for the action.
Result explanations use extracted results rather than entire output files.
Use a provider suitable for the data you send.

## If it fails

For the general message “The model request failed,” check the items below.

| Message or symptom | Check |
| --- | --- |
| Missing or invalid key | Paste a valid key for the selected provider. |
| Model not found or access denied | Use an exact model ID available to your account. |
| Quota or billing error | Check credits, billing and account limits with the provider. |
| Rate limit | Wait before requesting another plan. |
| JSON schema or unsupported parameter error | The model or endpoint may not support the required API. |
| Codex not found or signed out | Install or update the CLI, then run `codex login status` in the app's terminal environment. |

Do not post API keys when reporting an error.
