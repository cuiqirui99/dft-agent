# Install

Version `0.4.2` is a local desktop preview. Its installers and PyPI package
have not yet been published. Use a supplied preview installer or a source
checkout for now; after publication, download the installer from the
[corresponding GitHub Release](https://github.com/cuiqirui99/dft-agent/releases).

## Desktop app

- **Apple Silicon Mac:** open the `.dmg`, copy **DFT Agent** to Applications,
  then open it.
- **Windows x64:** run the `.exe` installer, then open **DFT Agent** from the
  Start menu. If Microsoft WebView2 is missing, installation needs an internet
  connection to install it. [Windows connection details](windows.md).

Python, scientific libraries and model SDKs are included. You do not need
Python, pip or a terminal to use the desktop app. Windows connects natively;
WSL is optional. Signing, notarization and successful installation on every
supported platform are not implied by a local build. See [desktop build and
acceptance notes](desktop.md) for the checks that accompany each package.

## First use

1. Open **Runs → Load sample runs** to view completed results and plots offline.
2. In **New calculation**, choose an example structure and **Manual**, then
   **Prepare inputs**. This needs neither a cluster nor a model account.
3. To submit a calculation, save your own **Cluster setup**: SSH access to a
   Linux Slurm cluster with Python 3, licensed VASP and POTCAR files.
4. For **Plan**, **Explain results** or model-assisted edits, configure your
   own model account under **Model**. Network access and the provider's billing
   apply. Bundled SDKs include no API keys or credits; Codex CLI is optional
   and installed separately. [Model setup](models.md).

The interactive structure viewer downloads 3Dmol.js. For offline use, turn
it off under **Preferences** to use the static plot. Automatic update checks
can also be turned off there.

## Your first cluster setup

You do not need to create `cluster.json` yourself. Leave **Configuration
file** at its default, fill **Cluster setup**, then click **Save cluster
settings**. Ask your institution's cluster support for a working VASP job
example and the following details:

| App field | Where to get the value |
| --- | --- |
| SSH host, user, port | The login hostname and your cluster login account; usually port 22. These are separate from a model API account. |
| Remote run folder | Your writable scratch or project directory on the cluster, as an absolute Linux path. This is not a folder on your laptop. |
| Remote POTCAR folder | The licensed potential directory supplied by your VASP administrator, containing element folders such as `Si/POTCAR`. |
| VASP command and environment setup | Copy the launch command and module/setup lines from your site's working VASP job. `srun vasp_std` is only a starting value. |
| SOC / noncollinear command | The site's launch command for `vasp_ncl`, required when using SOC or noncollinear spins. |
| Slurm partition, account, nodes, MPI tasks, time | Use the resources and project allocation in that job example or your cluster portal. The Slurm account is the project charged for the job, not the SSH username. |

After entering SSH details, **Detect from cluster** can suggest the remaining
values. Enter an SSH password in the sidebar if your site uses passwords;
otherwise use its configured SSH key/agent. Review the suggestions, save,
then click **Check cluster connection**. Neither action submits a calculation.
Leave optional mappings and extra options unchanged unless your site requires
them. A preset does not grant cluster access or a VASP licence.

## Python installation from source

For Linux users and developers, use Python 3.11 or newer. On macOS/Linux:

```bash
git clone https://github.com/cuiqirui99/dft-agent.git
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[agent]'
dft-agent ui
```

These commands install the checked-out source. For an unpublished preview,
use the supplied preview checkout; the public repository may still contain
an earlier version. The Python interface opens in a local browser at
`http://127.0.0.1:8501`; `--no-browser` skips opening it and `--port` selects
another port. OpenSSH is required for macOS/Linux cluster connections.

`[agent]` adds model SDKs and keychain support; omitting it still allows
manual preparation and calculation. `[dev]` adds test and build tools.
PyPI, uv and pipx installation by package name will be documented once the
package is published; they are not installation paths for this preview.

## Files and updates

| What | Location |
| --- | --- |
| Cluster settings | `~/.dft-agent/cluster.json` |
| Preferences and desktop log | `~/.dft-agent/` |
| Runs | `~/dft-agent-runs/` |

On Windows, `~` means your user folder, normally `%USERPROFILE%`.
Older settings under `~/.config/vasp-slurm-agent/` are copied on first start;
an existing `~/vasp-slurm-agent-runs/` folder continues to be used.
`DFT_AGENT_HOME`, `DFT_AGENT_CONFIG` and `DFT_AGENT_RUNS` override these paths.

Close the app before installing a newer version. Keep settings and run
folders to retain your work. Each published version gets its own tag and
Release; earlier tags, releases and assets remain available and unchanged.

To remove the desktop app, move it to the Trash on macOS or use Installed
apps on Windows. Keep the folders above unless you also want to remove your
settings and results. Remembered keys remain in the system keychain; untick
**Remember on this computer** for each provider to remove them before uninstalling.
