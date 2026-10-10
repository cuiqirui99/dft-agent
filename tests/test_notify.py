"""Completion notices use the desktop and an optional webhook, and never raise."""

import json
import pytest

from vasp_slurm_agent import notify, workflow


class Response:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_desktop_and_webhook_channels(monkeypatch):
    commands = []
    requests = []
    monkeypatch.setattr(notify.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(notify, "_run", lambda argv, **kw: commands.append(argv) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(notify, "_open", lambda request, timeout=10: requests.append(request) or Response())
    settings = {"notifications": True, "notify_webhook": "https://hooks.example/abc"}
    assert notify.notify("DFT Agent", "Si scf completed.", run="/runs/si", settings=settings) == ["desktop", "webhook"]
    assert commands[0][:2] == ["osascript", "-e"] and "Si scf completed." in commands[0][2]
    body = json.loads(requests[0].data.decode())
    assert body["text"] == "DFT Agent: Si scf completed." and body["run"] == "/runs/si"
    assert notify.notify("x", "y", settings={"notifications": False, "notify_webhook": "https://hooks.example/abc"}) == []
    assert notify.notify("x", "y", settings={"notifications": True, "notify_webhook": "ftp://nope"}) == ["desktop"]


def test_failures_are_swallowed(monkeypatch):
    monkeypatch.setattr(notify.platform, "system", lambda: "Linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/usr/bin/" + name)

    def boom(*args, **kwargs):
        raise OSError("no display")

    monkeypatch.setattr(notify, "_run", boom)
    monkeypatch.setattr(notify, "_open", boom)
    assert notify.notify("x", "y", settings={"notifications": True, "notify_webhook": "https://hooks.example"}) == []
    assert notify.notify_run({"status": "needs_attention", "formula": "Fe", "task": "scf"}) == []


def test_workflow_announces_terminal_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(notify, "notify", lambda title, message, run="", settings=None: seen.append((title, message, run)) or ["desktop"])
    workflow._announce("/runs/x", {"status": "succeeded", "formula": "Si", "task": "scf"})
    assert seen == [("DFT Agent", "Si scf completed.", "/runs/x")]

    def broken(state, run=""):
        raise RuntimeError("boom")

    monkeypatch.setattr(notify, "notify_run", broken)
    workflow._announce("/runs/x", {"status": "failed"})


@pytest.mark.parametrize("host", ["discord.com", "discordapp.com", "canary.discord.com", "ptb.discord.com"])
def test_discord_webhooks_send_discord_message_payload(monkeypatch, host):
    requests = []
    monkeypatch.setattr(notify, "_open", lambda request, timeout=10: requests.append(request) or Response())
    assert notify._webhook(f"https://{host}/api/webhooks/123/test-token", "DFT Agent", "Si scf completed.", "/runs/si")
    assert json.loads(requests[0].data) == {"content": "DFT Agent: Si scf completed.", "allowed_mentions": {"parse": []}}


def test_discord_failure_does_not_report_success(monkeypatch):
    class Rejected(Response):
        status = 400

    monkeypatch.setattr(notify, "_open", lambda *args, **kwargs: Rejected())
    assert not notify._webhook("https://discord.com/api/webhooks/123/test-token", "DFT Agent", "completed", "")
