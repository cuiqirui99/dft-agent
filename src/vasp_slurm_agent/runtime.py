"""Commands for source installs and desktop bundles."""

import sys


def cli_command(*arguments):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--cli", *map(str, arguments)]
    return [sys.executable, "-m", "vasp_slurm_agent.cli", *map(str, arguments)]
