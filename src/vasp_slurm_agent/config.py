from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ClusterConfig:
    host: str
    user: str
    remote_root: str
    vasp_command: str
    potcar_root: str
    partition: str
    port: int = 22
    tasks: int = 8
    walltime: str = "00:30:00"
    account: str = ""
    setup_commands: list[str] = field(default_factory=list)
    connect_timeout: int = 15
    potcar_symbols: dict[str, str] = field(default_factory=dict)
    vasp_ncl_command: str = ""
    ssh_control_path: str = ""

    def __post_init__(self):
        if not isinstance(self.ssh_control_path, str) or any(c in self.ssh_control_path for c in "\n\r\x00"):
            raise ValueError("Use an absolute SSH control socket path without line breaks.")
        if self.ssh_control_path:
            path = Path(self.ssh_control_path).expanduser()
            if not path.is_absolute():
                raise ValueError("Use an absolute SSH control socket path.")
            object.__setattr__(self, "ssh_control_path", str(path))
        for key in ("host", "user"):
            value = getattr(self, key)
            if not value or value.startswith("-") or re.search(r"[\s\x00-\x1f]", value):
                raise ValueError(f"Enter an SSH {key} without spaces or line breaks; it cannot start with '-'.")
        for key in ("remote_root", "potcar_root"):
            value = getattr(self, key)
            if not value.startswith("/") or any(c in value for c in "\n\r\x00"):
                raise ValueError(f"Use an absolute path for {key}, starting with '/' and without line breaks.")
        for key in ("partition", "account"):
            value = getattr(self, key)
            if (key == "partition" and not value) or (value and not re.fullmatch(r"[\w.-]+", value)):
                raise ValueError(f"Enter a valid Slurm {key} using letters, numbers, dots, dashes or underscores.")
        if not 1 <= self.port <= 65535 or not 1 <= self.tasks <= 4096:
            raise ValueError("Use an SSH port from 1 to 65535 and an MPI task count from 1 to 4096.")
        if not 1 <= self.connect_timeout <= 120:
            raise ValueError("Set the SSH connection timeout to between 1 and 120 seconds.")
        if not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", self.walltime):
            raise ValueError("Enter the time limit as HH:MM:SS or D-HH:MM:SS, for example 00:30:00.")
        if not self.vasp_command.strip() or any(c in self.vasp_command for c in "\n\r\x00"):
            raise ValueError("Enter the VASP launch command on a single line.")
        if not isinstance(self.vasp_ncl_command, str) or any(c in self.vasp_ncl_command for c in "\n\r\x00"):
            raise ValueError("Enter vasp_ncl_command on a single line.")
        if not isinstance(self.setup_commands, list) or any(not isinstance(c, str) for c in self.setup_commands):
            raise ValueError("Provide setup_commands as a list of shell commands for your cluster environment.")
        for symbol, label in self.potcar_symbols.items():
            if not re.fullmatch(r"[A-Z][a-z]?", symbol) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", label):
                raise ValueError('Map each element to a POTCAR folder name, for example {"Ti": "Ti_pv"}.')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "ClusterConfig":
        return cls(**json.loads(Path(path).expanduser().read_text()))

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        target.chmod(0o600)
        return target
