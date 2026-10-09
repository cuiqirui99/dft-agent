import sys

import pytest
from streamlit.testing.v1 import AppTest


def field(app, label):
    return next(item for item in app.text_input if item.label == label)


def model_editor():
    return AppTest.from_string("""
import streamlit as st
from vasp_slurm_agent.app import _model_editor
st.session_state['settings'] = _model_editor()
""").run()


@pytest.mark.parametrize("provider", ["anthropic", "qwen", "grok", "glm", "deepseek", "codex"])
def test_switching_provider_does_not_carry_openai_settings(monkeypatch, provider):
    monkeypatch.setenv("DFT_AGENT_MODEL", "openai-test-model")
    monkeypatch.setenv("DFT_AGENT_BASE_URL", "https://openai.example.invalid/v1")
    app = model_editor()
    field(app, "API key").set_value("openai-test-key").run()
    assert app.session_state["settings"].api_key == "openai-test-key"
    app.selectbox[0].select(provider).run()
    assert not app.exception
    settings = app.session_state["settings"]
    assert settings.provider == provider
    assert settings.api_key is None
    assert settings.base_url is None
    assert settings.model == ""


def test_qwen_settings_and_clear_key_stay_in_selected_provider():
    app = model_editor()
    app.selectbox[0].select("qwen").run()
    field(app, "Model name").set_value("qwen-plus").run()
    field(app, "API URL").set_value("https://workspace.example.invalid/compatible-mode/v1").run()
    field(app, "API key").set_value("qwen-test-key").run()
    settings = app.session_state["settings"]
    assert settings.provider == "qwen"
    assert settings.model == "qwen-plus"
    assert settings.base_url == "https://workspace.example.invalid/compatible-mode/v1"
    assert settings.api_key == "qwen-test-key"
    next(button for button in app.button if button.label == "Clear API key").click().run()
    assert not app.exception
    settings = app.session_state["settings"]
    assert settings.api_key is None
    assert settings.model == "qwen-plus"
    assert settings.base_url == "https://workspace.example.invalid/compatible-mode/v1"


@pytest.mark.parametrize("provider", ["anthropic", "qwen", "grok", "glm", "deepseek"])
def test_cli_accepts_named_providers_without_generic_environment_url(tmp_path, monkeypatch, provider):
    from vasp_slurm_agent import agent, cli, explanation
    from vasp_slurm_agent.providers import PROVIDERS

    monkeypatch.setenv("DFT_AGENT_BASE_URL", "https://openai.example.invalid/v1")
    calls = []

    def explain(run_dir, settings, *, question):
        calls.append(agent._settings(settings))
        return {"answer": "No results yet."}

    monkeypatch.setattr(explanation, "explain_run", explain)
    arguments = ["dft-agent", "explain", str(tmp_path), "--provider", provider,
                 "--model", PROVIDERS[provider]["model_example"]]
    if provider == "qwen":
        arguments.extend(["--base-url", "https://workspace.example.invalid/compatible-mode/v1"])
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0
    assert calls[0].provider == provider
    assert calls[0].base_url != "https://openai.example.invalid/v1"
