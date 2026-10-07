"""Planning boundary tests: real structure parsing, stubbed model providers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import agent
from vasp_slurm_agent.agent import AgentError, ModelSettings, draft_plan


@pytest.fixture(autouse=True)
def isolated_codex_config(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))


@pytest.fixture
def structure(tmp_path):
    path = tmp_path / "private-upload.cif"
    Structure(Lattice.cubic(5), ["Fe", "O", "Fe"],
              [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]).to(filename=path)
    return path


def response(**changes):
    result = {
        "status": "ready", "summary": "Optimize the structure.", "tasks": ["relax"],
        "parameters": {**dict.fromkeys(agent.DEFAULTS), "spin": "none", "magmom": None,
                       "soc": False, "saxis": [0, 0, 1], "functional": "PBE",
                       "hubbard_u": {"Fe": None, "O": None}},
        "magnetic_states": [], "initial_moment": None, "questions": [], "notes": [],
        "intent": {"requested_tasks": ["relax"], "forbidden_tasks": [], "spin": None,
                   "soc": None, "functional": None, "hubbard_u": None, "unsupported": []},
    }
    result.update(changes)
    return result


def stub(monkeypatch, value):
    monkeypatch.setattr(agent, "_request_plan", lambda *args: json.dumps(value))


def test_multistep_preserves_original_sites_and_hides_paths(monkeypatch, structure):
    plan = response(tasks=["relax", "scf", "bands", "dos"])
    plan["intent"]["requested_tasks"] = plan["tasks"]
    def request(payload, schema, settings):
        data = json.loads(payload)
        original = Structure.from_file(structure)
        assert [site["element"] for site in data["structure"]["sites"]] == [site.specie.symbol for site in original]
        assert [site["index"] for site in data["structure"]["sites"]] == list(range(3))
        assert str(structure) not in payload and structure.name not in payload
        assert data["history"] == [{"role": "user", "content": "Then add DOS."}]
        return json.dumps(plan)
    monkeypatch.setattr(agent, "_request_plan", request)
    result = draft_plan("Optimize, then SCF and bands.", structure, ModelSettings("codex"),
                        [{"role": "user", "content": "Then add DOS."}])
    assert result["status"] == "ready" and result["tasks"] == plan["tasks"]
    assert result["source_sha256"] == hashlib.sha256(structure.read_bytes()).hexdigest()
    assert "encut" not in result["parameters"]
    assert result["provenance"] == {"provider": "codex", "model": "Codex CLI default"}


@pytest.mark.parametrize("bad", ["not JSON", "```json\n{}\n```", '{"status":"ready"}', '[1]', 'NaN'])
def test_malformed_response_is_rejected(monkeypatch, structure, bad):
    monkeypatch.setattr(agent, "_request_plan", lambda *args: bad)
    with pytest.raises(AgentError, match="invalid plan"):
        draft_plan("Relax", structure, ModelSettings("codex"))


def test_missing_u_j_and_moments_require_input(monkeypatch, structure):
    plan = response()
    plan["parameters"].update(spin="collinear", hubbard_u={"Fe": {"l": 2, "u": None, "j": None}, "O": None})
    plan["intent"].update(spin="collinear", hubbard_u=True)
    stub(monkeypatch, plan)
    result = draft_plan("Relax using spin polarized +U.", structure, ModelSettings("codex"))
    assert result["status"] == "needs_input"
    assert any("U and J" in question for question in result["questions"])
    assert any("moment" in question for question in result["questions"])
    assert result["parameters"]["hubbard_u"]["Fe"]["u"] is None


def test_missing_u_question_is_not_repeated(monkeypatch, structure):
    question = "Which elements use +U? Supply l, U and J for each."
    plan = response(questions=[question])
    plan["intent"]["hubbard_u"] = True
    stub(monkeypatch, plan)
    result = draft_plan("Use +U but ask for its parameters", structure, ModelSettings("codex"))
    assert result["status"] == "needs_input"
    assert result["questions"] == [question]


def test_followup_supplies_explicit_u_without_dropping_exclusions(monkeypatch, structure):
    goal = "Use +U; ask me for its parameters."
    history = [{"role": "assistant", "content": "Supply l, U and J for each element."},
               {"role": "user", "content": "SCF only, no relaxation or SOC. Fe l=2 U=5 J=0; use collinear moments [3,-3,0]."}]
    plan = response(tasks=["scf"])
    plan["parameters"].update(spin="collinear", magmom=[3, -3, 0], hubbard_u={"Fe": {"l": 2, "u": 5, "j": 0}, "O": None})
    plan["intent"].update(requested_tasks=["scf"], forbidden_tasks=["relax"], spin="collinear", soc=False, hubbard_u=True)
    def request(payload, *args):
        data = json.loads(payload)
        assert data["goal"] == goal and data["history"] == history
        return json.dumps(plan)
    monkeypatch.setattr(agent, "_request_plan", request)
    result = draft_plan(goal, structure, ModelSettings("codex"), history)
    assert result["status"] == "ready" and result["tasks"] == ["scf"]
    assert result["parameters"]["soc"] is False
    assert result["parameters"]["hubbard_u"]["Fe"] == {"l": 2, "u": 5, "j": 0}


def test_explicit_u_soc_and_negative_relax_preserved(monkeypatch, structure):
    plan = response(tasks=["scf", "bands"])
    plan["parameters"].update(spin="noncollinear", soc=True, magmom=[[0, 0, 3], [0, 0, 0], [0, 0, -3]],
                              hubbard_u={"Fe": {"l": 2, "u": 4.5, "j": 0.5}, "O": None}, mesh=[6, 6, 6])
    plan["intent"].update(requested_tasks=["scf", "bands"], forbidden_tasks=["relax"],
                          soc=True, hubbard_u=True, spin="noncollinear")
    stub(monkeypatch, plan)
    result = draft_plan("Do not relax. Compute SCF and bands with SOC and Fe U=4.5 J=0.5 l=2, axis z and the supplied moments.", structure, ModelSettings("codex"))
    assert result["status"] == "ready"
    assert result["parameters"]["soc"] is True
    assert result["parameters"]["hubbard_u"] == {"Fe": {"l": 2, "u": 4.5, "j": 0.5}}
    assert result["parameters"]["magmom"] == plan["parameters"]["magmom"]
    assert result["tasks"] == ["scf", "bands"]


def test_silent_method_downgrade_is_not_ready(monkeypatch, structure):
    plan = response()
    plan["intent"].update(soc=True, hubbard_u=True, forbidden_tasks=["relax"], functional="HSE06")
    stub(monkeypatch, plan)
    result = draft_plan("No relaxation. HSE06 +U and SOC.", structure, ModelSettings("codex"))
    assert result["status"] == "needs_input"
    assert len(result["questions"]) >= 4


@pytest.mark.parametrize("moments", [[3, -3], [[0, 0, 3]] * 3])
def test_bad_collinear_site_mapping_cannot_be_prepared(monkeypatch, structure, moments):
    plan = response()
    plan["parameters"].update(spin="collinear", magmom=moments)
    stub(monkeypatch, plan)
    result = draft_plan("Spin polarized relaxation", structure, ModelSettings("codex"))
    assert result["status"] == "needs_input"


def test_soc_missing_axis_asks_and_nonmagnetic_soc_has_zero_vectors(monkeypatch, structure):
    plan = response()
    plan["parameters"].update(soc=True, saxis=None)
    plan["intent"].update(soc=True, spin="none")
    stub(monkeypatch, plan)
    result = draft_plan("Nonmagnetic SOC", structure, ModelSettings("codex"))
    assert result["status"] == "needs_input"
    assert result["parameters"]["spin"] == "noncollinear"
    assert result["parameters"]["magmom"] == [[0.0, 0.0, 0.0]] * 3
    assert any("SAXIS" in question for question in result["questions"])


def test_comparison_does_not_drop_downstream_spectra(monkeypatch, structure):
    plan = response(tasks=["scf", "bands"], magnetic_states=["NM", "FM", "AFM"])
    plan["intent"]["requested_tasks"] = ["scf", "bands"]
    stub(monkeypatch, plan)
    result = draft_plan("Compare magnetic states and then bands", structure, ModelSettings("codex"))
    assert result["status"] == "needs_input"
    assert result["tasks"] == ["scf", "bands"]
    assert result["initial_moment"] == 3
    assert any("initial guess" in note for note in result["notes"])


def test_supported_comparison_ready(monkeypatch, structure):
    plan = response(tasks=["scf"], magnetic_states=["NM", "FM", "AFM"])
    plan["intent"]["requested_tasks"] = ["scf"]
    stub(monkeypatch, plan)
    assert draft_plan("Compare NM FM AFM", structure, ModelSettings("codex"))["status"] == "ready"


@pytest.mark.parametrize("bad_task", [True, False])
def test_unsupported_physics_is_explicit(monkeypatch, structure, bad_task):
    plan = response()
    if bad_task:
        plan["tasks"] = ["phonons"]
    else:
        plan["intent"]["unsupported"] = ["phonons"]
    stub(monkeypatch, plan)
    result = draft_plan("Calculate phonons", structure, ModelSettings("codex"))
    assert result["status"] == "unsupported"
    assert any("phonons" in note for note in result["notes"])


def test_no_structure_needs_input(monkeypatch):
    plan = response()
    plan["parameters"]["hubbard_u"] = {}
    stub(monkeypatch, plan)
    result = draft_plan("Optimize", None, ModelSettings("codex"))
    assert result["status"] == "needs_input" and result["source_sha256"] is None


def test_source_hash_binds_the_structure_seen_by_model(monkeypatch, structure):
    expected = hashlib.sha256(structure.read_bytes()).hexdigest()
    def request(*args):
        structure.write_text("changed after the model request")
        return json.dumps(response())
    monkeypatch.setattr(agent, "_request_plan", request)
    result = draft_plan("Relax", structure, ModelSettings("codex"))
    assert result["source_sha256"] == expected
    assert result["source_sha256"] != hashlib.sha256(structure.read_bytes()).hexdigest()


def test_hybrid_request_preserves_functional(monkeypatch, structure):
    plan = response(tasks=["bands"])
    plan["parameters"]["functional"] = "HSE06"
    plan["intent"].update(requested_tasks=["bands"], functional="HSE06")
    stub(monkeypatch, plan)
    result = draft_plan("Calculate HSE06 bands", structure, ModelSettings("codex"))
    assert result["status"] == "ready" and result["parameters"]["functional"] == "HSE06"


def test_schema_rejects_single_state_mislabelled_as_comparison(monkeypatch, structure):
    stub(monkeypatch, response(magnetic_states=["NM"]))
    with pytest.raises(AgentError, match="invalid plan"):
        draft_plan("Nonmagnetic bands", structure, ModelSettings("codex"))


def test_schema_keeps_static_nsw_handling_in_runner(monkeypatch, structure):
    plan = response(tasks=["scf"])
    plan["parameters"]["nsw"] = 0
    plan["intent"]["requested_tasks"] = ["scf"]
    stub(monkeypatch, plan)
    with pytest.raises(AgentError, match="invalid plan"):
        draft_plan("SCF", structure, ModelSettings("codex"))


@pytest.mark.parametrize("key,value", [("encut", True), ("mesh", [0, 4, 4]), ("ediff", -1), ("sigma", float("inf"))])
def test_invalid_numeric_settings_rejected(monkeypatch, structure, key, value):
    plan = response()
    plan["parameters"][key] = value
    stub(monkeypatch, plan)
    with pytest.raises(AgentError):
        draft_plan("Relax", structure, ModelSettings("codex"))


@pytest.mark.parametrize("key", ["test-api-key-private", 'test-escaped-"key-private'])
def test_secrets_redacted_from_request_and_never_returned(monkeypatch, structure, key):
    password = "test-ssh-password-private"
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", password)
    settings = ModelSettings("codex", api_key=key)
    assert key not in repr(settings)
    def request(payload, *args):
        assert key not in payload and password not in payload
        assert key not in json.loads(payload)["goal"]
        return json.dumps(response(summary=key))
    monkeypatch.setattr(agent, "_request_plan", request)
    with pytest.raises(AgentError) as caught:
        draft_plan(f"Relax {key} {password}", structure, settings)
    assert key not in str(caught.value) and password not in str(caught.value)


def test_sdk_uses_strict_structured_output_and_hides_provider_error(monkeypatch, structure):
    import sys
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "private-key"
            self.responses = SimpleNamespace(create=self.create)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            assert kwargs["store"] is False
            assert kwargs["text"]["format"]["strict"] is True
            assert "private-key" not in kwargs["input"]
            raise RuntimeError("private-key server failure with credentials")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    with pytest.raises(AgentError, match="model request failed") as caught:
        draft_plan("Relax", structure, ModelSettings("responses", "test-model", api_key="private-key"))
    assert "private-key" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_chat_provider_schema_and_env_key(monkeypatch, structure):
    import sys
    monkeypatch.setenv("DFT_AGENT_API_KEY", "env-key-private")
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "env-key-private"
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            assert kwargs["response_format"]["json_schema"]["strict"] is True
            assert kwargs["messages"][0]["role"] == "system"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response()), refusal=None))])
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    result = draft_plan("Relax", structure, ModelSettings("chat_completions", "test-model"))
    assert result["status"] == "ready"


def test_codex_isolated_readonly_without_tools_or_password(monkeypatch, structure):
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/codex")
    monkeypatch.setenv("DFT_AGENT_SSH_PASSWORD", "ssh-private")
    monkeypatch.setenv("OPENAI_API_KEY", "api-private")
    def run(command, **kwargs):
        assert "-m" not in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "--ephemeral" in command and "--ignore-user-config" in command
        assert "features.shell_tool=false" in command and 'web_search="disabled"' in command
        assert "project_doc_max_bytes=0" in command
        assert kwargs["cwd"] != str(structure.parent)
        assert "DFT_AGENT_SSH_PASSWORD" not in kwargs["env"]
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "Relax" in kwargs["input"] and "Relax" not in command
        Path(command[command.index("-o") + 1]).write_text(json.dumps(response()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(agent.subprocess, "run", run)
    assert draft_plan("Relax", structure, ModelSettings("codex"))["status"] == "ready"


def test_codex_error_does_not_echo_stderr(monkeypatch, structure):
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/codex")
    monkeypatch.setattr(agent.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="secret-value"))
    with pytest.raises(AgentError, match="Codex could not") as caught:
        draft_plan("Relax", structure, ModelSettings("codex"))
    assert "secret-value" not in str(caught.value)


def test_codex_reuses_only_configured_model(monkeypatch, structure, tmp_path):
    import os
    config_home = Path(os.environ["CODEX_HOME"])
    config_home.mkdir()
    (config_home / "config.toml").write_text('model = "configured-model"\nsandbox_mode = "danger-full-access"\nmodel_instructions_file = "/private/instructions"\n')
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/codex")
    def run(command, **kwargs):
        assert command[command.index("-m") + 1] == "configured-model"
        assert "--ignore-user-config" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "danger-full-access" not in command
        assert "/private/instructions" not in command
        Path(command[command.index("-o") + 1]).write_text(json.dumps(response()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(agent.subprocess, "run", run)
    result = draft_plan("Relax", structure, ModelSettings("codex"))
    assert result["provenance"]["model"] == "configured-model"


def test_explicit_codex_model_overrides_local_default(monkeypatch):
    monkeypatch.setattr(agent, "_codex_default_model", lambda: "user-default")
    assert agent._settings(ModelSettings("codex", "explicit-model")).model == "explicit-model"


def test_codex_uses_app_cli_instead_of_stale_path_install(monkeypatch, structure):
    monkeypatch.setenv("CODEX_CLI_PATH", "/app/current/codex")
    def which(value):
        assert value == "/app/current/codex"
        return value
    monkeypatch.setattr(agent.shutil, "which", which)
    def run(command, **kwargs):
        assert command[0] == "/app/current/codex"
        Path(command[command.index("-o") + 1]).write_text(json.dumps(response()))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(agent.subprocess, "run", run)
    assert draft_plan("Relax", structure, ModelSettings("codex"))["status"] == "ready"


def test_old_codex_has_safe_actionable_error(monkeypatch, structure):
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/codex")
    monkeypatch.setattr(agent.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stderr="private-data: The model requires a newer version of Codex."))
    with pytest.raises(AgentError, match="too old") as caught:
        draft_plan("Relax", structure, ModelSettings("codex"))
    assert "private-data" not in str(caught.value)
