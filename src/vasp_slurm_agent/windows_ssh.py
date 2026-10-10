"""Native Windows SSH and SFTP; the shared transport still verifies every file."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

from .transport import Result, TransportError

_WINDOWS = sys.platform == "win32"


def _read_host_keys(paramiko, path: Path):
    """Read our UTF-8 file independently of the Windows ANSI code page."""
    keys = paramiko.HostKeys()
    if not path.exists():
        return keys
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry = paramiko.hostkeys.HostKeyEntry.from_line(line, number)
        if entry is not None:
            for hostname in entry.hostnames:
                # Retain the first key for duplicate host/algorithm entries.
                existing = keys.lookup(hostname)
                if existing is None or entry.key.get_name() not in existing:
                    keys.add(hostname, entry.key.get_name(), entry.key)
    return keys


@contextmanager
def _known_hosts_lock(path: Path):
    """Serialize first-use host-key writes across app and monitoring processes."""
    with path.with_name(path.name + ".dft-agent.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + 10
        while True:
            stream.seek(0)
            try:
                if _WINDOWS:
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # The backend is also exercised by local integration tests.
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TransportError("The SSH known_hosts file is busy. Try again.") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if _WINDOWS:
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _accept_new_policy(paramiko, path: Path):
    class AcceptNew(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            path.parent.mkdir(parents=True, exist_ok=True)
            with _known_hosts_lock(path):
                known = _read_host_keys(paramiko, path)
                previous = known.lookup(hostname)
                if previous:
                    expected = previous.get(key.get_name()) or next(iter(previous.values()))
                    if expected != key:
                        raise paramiko.BadHostKeyException(hostname, key, expected)
                else:
                    # Preserve comments and existing entries, with atomic replacement.
                    original = path.read_text(encoding="utf-8") if path.exists() else ""
                    entry = f"{hostname} {key.get_name()} {key.get_base64()}\n"
                    descriptor, name = tempfile.mkstemp(prefix=".known-hosts-", dir=path.parent)
                    try:
                        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                            stream.write(original + ("\n" if original and not original.endswith("\n") else "") + entry)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(name, path)
                    finally:
                        Path(name).unlink(missing_ok=True)
                client.get_host_keys().add(hostname, key.get_name(), key)

    return AcceptNew()


class WindowsSSH:
    def __init__(self, config, password):
        try:
            import paramiko
        except ImportError:
            raise TransportError("The Windows SSH component is missing. Reinstall DFT Agent.") from None
        self._paramiko = paramiko
        self.config = config
        self._password = password
        self._client = None
        self._closed = False

    def _connect(self):
        if self._closed:
            raise TransportError("SSH transport has been closed.")
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client
            self._client.close()
            self._client = None
        client = self._paramiko.SSHClient()
        known_hosts = Path.home() / ".ssh" / "known_hosts"
        try:
            if known_hosts.exists():
                known = _read_host_keys(self._paramiko, known_hosts)
                for hostname, entries in known.items():
                    for kind, key in entries.items():
                        client.get_host_keys().add(hostname, kind, key)
            client.set_missing_host_key_policy(_accept_new_policy(self._paramiko, known_hosts))
            timeout = max(1, int(getattr(self.config, "connect_timeout", 15)))
            client.connect(
                hostname=str(self.config.host).removeprefix("[").removesuffix("]"),
                port=int(self.config.port), username=str(self.config.user),
                password=self._password or None, allow_agent=True, look_for_keys=True,
                timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
                channel_timeout=timeout,
            )
            client.get_transport().set_keepalive(30)
        except self._paramiko.BadHostKeyException:
            client.close()
            raise TransportError("SSH host key has changed. Verify it with your cluster administrator before updating known_hosts.") from None
        except self._paramiko.AuthenticationException:
            client.close()
            raise TransportError("SSH authentication failed. Check your username, password or SSH key.") from None
        except Exception as exc:
            client.close()
            raise TransportError(f"Cannot connect with SSH: {exc}") from None
        self._client = client
        return client

    def run(self, command: str, timeout: float) -> Result:
        channel = None
        stdout, stderr = bytearray(), bytearray()
        try:
            client = self._connect()
            deadline = time.monotonic() + timeout
            channel = client.get_transport().open_session(timeout=timeout)
            channel.settimeout(timeout)
            channel.exec_command(command)
            channel.shutdown_write()
            while True:
                # Drain both streams before requesting status: either can fill the SSH window.
                if channel.recv_ready():
                    stdout.extend(channel.recv(65536))
                if channel.recv_stderr_ready():
                    stderr.extend(channel.recv_stderr(65536))
                if (channel.exit_status_ready() and (channel.closed or channel.eof_received)
                        and not channel.recv_ready() and not channel.recv_stderr_ready()):
                    status = channel.recv_exit_status()
                    return Result(status if status >= 0 else 255,
                                  stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"))
                if time.monotonic() >= deadline:
                    raise socket.timeout()
                time.sleep(0.01)
        except (socket.timeout, TimeoutError):
            return Result(124, stdout.decode("utf-8", errors="replace"),
                          stderr.decode("utf-8", errors="replace") + "\nSSH operation timed out.")
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(f"SSH command failed: {exc}") from None
        finally:
            if channel is not None:
                channel.close()

    def transfer(self, local: str, remote: str, *, upload: bool):
        sftp = None
        try:
            sftp = self._connect().open_sftp()
            deadline = time.monotonic() + 600
            sftp.get_channel().settimeout(600)

            def progress(_transferred, _total):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise socket.timeout("SFTP transfer timed out")
                sftp.get_channel().settimeout(remaining)

            if upload:
                sftp.put(local, remote, callback=progress)
            else:
                # Avoid a prefetch thread outliving its channel on timeout/close.
                sftp.get(remote, local, callback=progress, prefetch=False)
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(f"SFTP transfer failed: {exc}") from None
        finally:
            if sftp is not None:
                sftp.close()

    def close(self):
        self._closed = True
        self._password = ""
        if self._client is not None:
            self._client.close()
            self._client = None
