"""Find Slurm partitions, accounts, VASP modules and POTCAR folders on a cluster.

One SSH session runs a read-only shell script. Its output is parsed into facts
and turned into suggested settings that the user reviews before saving.
"""

from __future__ import annotations

from importlib.resources import files
import json
import re

from .config import ClusterConfig

MARK = "__DFT_AGENT_SECTION__"
PROBE = r"""
section() { printf '\n%s %s\n' "__DFT_AGENT_SECTION__" "$1"; }
section home
printf '%s\n' "$HOME"
section user
id -un 2>/dev/null
section slurm
for n in sbatch squeue sacct scancel sinfo; do command -v "$n" >/dev/null 2>&1 && printf '%s\n' "$n"; done
section partitions
sinfo -h -o '%P|%a|%l|%D' 2>/dev/null | head -n 50
section accounts
sacctmgr -n -P show assoc user="$(id -un)" format=account 2>/dev/null | sort -u | head -n 20
section modules
{ module -t avail 2>&1; module -t spider vasp 2>&1; } | grep -i vasp | grep -v -i -E 'error|not found|command' | sed 's/:$//' | sort -u | head -n 30
section executables
for n in vasp_std vasp_ncl vasp_gam; do p=$(command -v "$n" 2>/dev/null) && printf '%s=%s\n' "$n" "$p"; done
section potcar
{
  for d in "$VASP_PP_PATH" "$POTCAR_DIR" "$POTCAR_PATH" "$HOME"/potcar* "$HOME"/POTCAR* "$HOME"/*/potpaw* "$HOME"/*/*potpaw* "$HOME"/*/POTCAR* /opt/*/potpaw* /opt/*/*/potpaw* /sw/*/potpaw* /software/*/potpaw* /software/*/*/potpaw* /apps/*/potpaw* /apps/*/*/potpaw* /public*/soft*/*/potpaw* /public*/soft*/*/*/potpaw*; do
    [ -n "$d" ] && [ -d "$d" ] && printf '%s\n' "$d"
  done
  if command -v timeout >/dev/null 2>&1; then timeout 20 find "$HOME" -maxdepth 3 -type d \( -iname 'potpaw*' -o -iname 'potcar*' \) 2>/dev/null | head -n 20
  else find "$HOME" -maxdepth 3 -type d \( -iname 'potpaw*' -o -iname 'potcar*' \) 2>/dev/null | head -n 20; fi
} | awk '!seen[$0]++' | while IFS= read -r d; do
  if [ -f "$d/Si/POTCAR" ] || [ -f "$d/H/POTCAR" ] || [ -f "$d/Fe/POTCAR" ] || [ -f "$d/O/POTCAR" ]; then printf '%s\n' "$d"; fi
done | head -n 20
section scratch
u=$(id -un)
for d in "$SCRATCH" "$WORK" "$WORKDIR" "$PDC_TMP" "/scratch/$u" "$HOME/scratch" "/cfs/klemming/scratch/$(printf '%.1s' "$u")/$u" "$HOME"; do
  [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ] && printf '%s\n' "$d"
done 2>/dev/null | awk '!seen[$0]++' | head -n 10
"""
SECTIONS = ("home", "user", "slurm", "partitions", "accounts", "modules", "executables", "potcar", "scratch")
_SAFE_TEXT = re.compile(r"[^\x20-\x7e]")
_NAME = re.compile(r"[\w.+/:@-]+")


class DiscoveryError(RuntimeError):
    """The cluster could not be probed."""


def probe_config(host: str, user: str, port: int = 22, *, connect_timeout: int = 15, ssh_control_path: str = "",
                 setup_commands: list[str] | None = None) -> ClusterConfig:
    """A provisional configuration that only carries connection settings."""
    return ClusterConfig(host=host.strip(), user=user.strip(), remote_root="/tmp", vasp_command="vasp_std",
                         potcar_root="/tmp", partition="probe", port=int(port), connect_timeout=int(connect_timeout),
                         ssh_control_path=ssh_control_path or "", setup_commands=list(setup_commands or []))


def _clean(line: str, limit: int = 300) -> str:
    return _SAFE_TEXT.sub("", line).strip()[:limit]


def parse_probe(output: str) -> dict:
    """Turn the probe script output into plain facts. Unknown sections are ignored."""
    sections: dict[str, list[str]] = {name: [] for name in SECTIONS}
    current = None
    for raw in output.splitlines():
        line = _clean(raw)
        if line.startswith(MARK):
            name = line[len(MARK):].strip()
            current = name if name in sections else None
            continue
        if current and line:
            sections[current].append(line)
    partitions = []
    for line in sections["partitions"]:
        fields = line.split("|")
        name = fields[0]
        if not _NAME.fullmatch(name.rstrip("*")):
            continue
        partitions.append({"name": name.rstrip("*"), "default": name.endswith("*"),
                           "state": fields[1] if len(fields) > 1 else "", "timelimit": fields[2] if len(fields) > 2 else "",
                           "nodes": fields[3] if len(fields) > 3 else ""})
    accounts = [line for line in sections["accounts"] if re.fullmatch(r"[\w.-]+", line)]
    modules = [line for line in sections["modules"] if _NAME.fullmatch(line)]
    executables = {}
    for line in sections["executables"]:
        name, _, path = line.partition("=")
        if name in {"vasp_std", "vasp_ncl", "vasp_gam"} and path.startswith("/"):
            executables[name] = path
    potcar = [line for line in sections["potcar"] if line.startswith("/")]
    scratch = [line for line in sections["scratch"] if line.startswith("/")]
    home = next((line for line in sections["home"] if line.startswith("/")), "")
    return {"home": home, "user": sections["user"][0] if sections["user"] else "",
            "slurm": sections["slurm"], "partitions": partitions, "accounts": accounts, "modules": modules,
            "executables": executables, "potcar_roots": potcar, "scratch_dirs": scratch}


def suggest(facts: dict) -> dict:
    """Suggested settings and notes. Every value still needs the user's review."""
    suggested: dict = {}
    notes: list[str] = []
    partitions = facts.get("partitions") or []
    if partitions:
        default = next((item for item in partitions if item["default"]), partitions[0])
        suggested["partition"] = default["name"]
        if default.get("timelimit") and default["timelimit"] not in {"infinite", "UNLIMITED"}:
            notes.append(f"Partition {default['name']} allows up to {default['timelimit']} per job.")
    else:
        notes.append("No Slurm partitions were listed. Check that Slurm commands are on the login node's PATH.")
    if facts.get("accounts"):
        suggested["account"] = facts["accounts"][0]
        if len(facts["accounts"]) > 1:
            notes.append("Several Slurm accounts are available: " + ", ".join(facts["accounts"][:6]) + ".")
    potcar = facts.get("potcar_roots") or []
    if potcar:
        ranked = sorted(potcar, key=lambda path: (("pbe" not in path.lower()), ("potpaw" not in path.lower()), len(path)))
        suggested["potcar_root"] = ranked[0]
        if len(ranked) > 1:
            notes.append("Other POTCAR folders: " + ", ".join(ranked[1:4]) + ".")
    else:
        notes.append("No POTCAR folder was found. Enter the path to your licensed PAW potentials.")
    scratch = facts.get("scratch_dirs") or []
    if scratch:
        suggested["remote_root"] = scratch[0].rstrip("/") + "/dft-agent-runs"
    executables = facts.get("executables") or {}
    modules = facts.get("modules") or []
    if "vasp_std" in executables:
        suggested["vasp_command"] = "srun vasp_std"
        suggested["vasp_ncl_command"] = "srun vasp_ncl" if "vasp_ncl" in executables else ""
    elif modules:
        suggested["setup_commands"] = [f"module load {modules[0]}"]
        suggested["vasp_command"] = "srun vasp_std"
        suggested["vasp_ncl_command"] = "srun vasp_ncl"
        if len(modules) > 1:
            notes.append("Other VASP modules: " + ", ".join(modules[1:6]) + ".")
    else:
        notes.append("VASP was not found on the PATH or as a module. Use the full path to vasp_std in the VASP command.")
    missing = {"sbatch", "squeue", "sacct", "scancel"} - set(facts.get("slurm") or [])
    if missing:
        notes.append("Missing Slurm commands: " + ", ".join(sorted(missing)) + ".")
    return {"suggested": suggested, "notes": notes}


def merge_suggestions(current: dict, suggested: dict) -> dict:
    """Keep the user's launcher and prerequisites when filling detected values."""
    merged = {**current, **suggested}
    for name in ("vasp_command", "vasp_ncl_command"):
        if current.get(name):
            merged[name] = current[name]
    setup = list(current.get("setup_commands") or [])
    for command in suggested.get("setup_commands") or []:
        if command not in setup:
            setup.append(command)
    if setup:
        merged["setup_commands"] = setup
    return merged


def probe_cluster(config: ClusterConfig, transport=None, timeout: int = 90) -> dict:
    """Run the read-only probe over SSH and return facts, suggestions and notes."""
    from .transport import SSHTransport, TransportError

    owned = transport is None
    remote = transport or SSHTransport(config)
    try:
        script = "\n".join([*config.setup_commands, PROBE])
        try:
            result = remote.run(script, timeout=timeout)
        except TransportError as exc:
            raise DiscoveryError(str(exc)) from exc
        output = result.stdout or ""
        if MARK not in output:
            detail = (result.stderr or output or "no output").strip()[-500:]
            raise DiscoveryError(f"The cluster did not answer the probe: {detail}")
        facts = parse_probe(output)
        report = suggest(facts)
        if config.setup_commands and "setup_commands" in report["suggested"]:
            report["suggested"]["setup_commands"] = merge_suggestions(
                {"setup_commands": config.setup_commands}, report["suggested"])["setup_commands"]
        return {"facts": facts, **report}
    finally:
        if owned and hasattr(remote, "close"):
            remote.close()


def load_presets() -> list[dict]:
    """Bundled starting points for known clusters. Values are hints, not facts."""
    folder = files("vasp_slurm_agent").joinpath("clusters")
    presets = []
    for item in sorted(folder.iterdir(), key=lambda entry: entry.name):
        if item.name.endswith(".json"):
            data = json.loads(item.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("id") and data.get("name"):
                presets.append(data)
    presets.sort(key=lambda item: (item["id"] != "generic", item["name"]))
    return presets


PRESET_FIELDS = ("host", "port", "partition", "account", "vasp_command", "vasp_ncl_command", "setup_commands",
                 "potcar_root", "remote_root", "tasks", "nodes", "walltime", "extra_sbatch")


def preset_values(preset: dict) -> dict:
    """Configuration fields carried by a preset, with blanks left out."""
    values = {}
    for key in PRESET_FIELDS:
        value = preset.get(key)
        if value in (None, "", []):
            continue
        values[key] = list(value) if isinstance(value, list) else value
    return values
