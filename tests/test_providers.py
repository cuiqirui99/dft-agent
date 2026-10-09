"""Exercise vendor request contracts through their real SDK serializers."""

from __future__ import annotations

import json

import pytest
from pymatgen.core import Lattice, Structure

from vasp_slurm_agent import agent
from vasp_slurm_agent.agent import AgentError, ModelSettings, draft_plan
from vasp_slurm_agent.model_usage import normalize_usage


VENDORS = {
    "anthropic": ("claude-sonnet-4-6", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "qwen": ("qwen-plus", "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "grok": ("grok-4.7", "https://api.x.ai/v1", "XAI_API_KEY"),
    "glm": ("glm-5.3", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    "deepseek": ("deepseek-flash", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


@pytest.fixture(autouse=True)
def isolated_keys(monkeypatch, tmp_path):
    for key in {value[2] for value in VENDORS.values()} | {
        "ZHIPUAI_API_KEY", "DFT_AGENT_API_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DFT_AGENT_MODEL", "DFT_AGENT_BASE_URL", "DFT_AGENT_MAX_OUTPUT_TOKENS",
    }:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))


@pytest.fixture
def structure(tmp_path):
    path = tmp_path / "Si.cif"
    Structure(Lattice.cubic(5.43), ["Si", "Si"],
              [[0, 0, 0], [0.25, 0.25, 0.25]]).to(filename=path)
    return path


def plan_json():
    return json.dumps({
        "status": "ready", "summary": "Relax Si.", "tasks": ["relax"],
        "parameters": {**dict.fromkeys(agent.DEFAULTS), "spin": "none", "soc": False,
                       "saxis": [0, 0, 1], "magmom": None, "functional": "PBE",
                       "hubbard_u": {"Si": None}},
        "magnetic_states": [], "initial_moment": None, "questions": [], "notes": [],
        "intent": {"requested_tasks": ["relax"], "forbidden_tasks": [], "spin": None,
                   "soc": None, "functional": None, "hubbard_u": None, "unsupported": []},
    })


def vendor_settings(provider, *, api_key=None, base_url=None):
    model, endpoint, _ = VENDORS[provider]
    return ModelSettings(provider, model, base_url or (endpoint if provider == "qwen" else None), api_key)


def claude_wire(value, schema):
    nullable = agent._claude_nullable(schema)
    if nullable is not None:
        return [] if value is None else [claude_wire(value, nullable)]
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                agent._check_schema(value, option)
            except AgentError:
                continue
            return claude_wire(value, option)
        return value
    if isinstance(value, dict) and schema.get("type") == "object":
        return {key: claude_wire(item, schema["properties"].get(key, {})) for key, item in value.items()}
    if isinstance(value, list) and schema.get("type") == "array":
        return [claude_wire(item, schema["items"]) for item in value]
    return value


def mock_sdk(monkeypatch, provider, *, content=None, stop=None, status=200, schema=None,
             error_type="rate_limit_error", error_message="private-server-key", failure=None,
             blocks=None, raw_claude=False):
    """Keep SDK validation/serialization and replace only HTTP transport."""
    if provider == "anthropic":
        import anthropic as sdk
        from anthropic import _base_client
        constructor = "Anthropic"
    else:
        import openai as sdk
        from openai import _base_client
        constructor = "OpenAI"
    http = getattr(_base_client, "httpx2", None) or _base_client.httpx
    original = getattr(sdk, constructor)
    requests = []
    content = plan_json() if content is None else content

    def handle(request):
        requests.append(request)
        if failure:
            raise getattr(http, failure)("private-server-key", request=request)
        if status != 200:
            return http.Response(status, json={"error": {
                "type": error_type, "message": error_message}}, request=request)
        if provider == "anthropic":
            encoded = content
            if not raw_claude:
                try:
                    encoded = json.dumps(claude_wire(json.loads(content), schema or agent._schema(["Si"])))
                except (ValueError, TypeError):
                    pass
            response = {"id": "msg_test", "type": "message", "role": "assistant",
                        "model": VENDORS[provider][0], "content": blocks or [{"type": "text", "text": encoded}],
                        "stop_reason": stop or "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 100, "output_tokens": 40,
                                  "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30}}
        elif provider == "responses":
            response = {"id": "resp_test", "object": "response", "created_at": 1,
                        "model": "test-model", "status": "completed" if stop == "refusal" else stop or "completed",
                        "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                                    "content": [{"type": "refusal", "refusal": "private refusal text"}] if stop == "refusal"
                                    else [{"type": "output_text", "text": content, "annotations": []}]}],
                        "usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}}
        else:
            message = {"role": "assistant", "content": content}
            if stop == "refusal":
                message["refusal"] = "I cannot do that."
            response = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                        "model": VENDORS.get(provider, ("test-model",))[0],
                        "choices": [{"index": 0, "message": message,
                                     "finish_reason": "stop" if stop == "refusal" else stop or "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                                  "prompt_tokens_details": {"cached_tokens": 20}}}
        return http.Response(200, json=response, request=request)

    monkeypatch.setattr(sdk, constructor, lambda **kwargs: original(
        **kwargs, http_client=http.Client(transport=http.MockTransport(handle))))
    return requests


@pytest.mark.parametrize("stop,content,match", [
    ("refusal", None, "declined"), ("incomplete", None, "incomplete"),
    ("failed", None, "complete response"), (None, "{}", "invalid structured data"),
])
def test_responses_rejects_unaccepted_output_through_sdk(monkeypatch, structure, stop, content, match):
    requests = mock_sdk(monkeypatch, "responses", stop=stop, content=content)
    with pytest.raises(AgentError, match=match) as caught:
        draft_plan("Relax Si.", structure, ModelSettings("responses", "test-model", api_key="private-key"))
    assert len(requests) == 1 and caught.value.model_usage["output_tokens"] == 40
    assert "private refusal text" not in str(caught.value)


@pytest.mark.parametrize("provider", VENDORS)
def test_vendor_request_and_validated_plan(monkeypatch, structure, provider):
    model, endpoint, key_name = VENDORS[provider]
    monkeypatch.setenv(key_name, "vendor-private-key")
    requests = mock_sdk(monkeypatch, provider)
    result = draft_plan("Relax Si.", structure, vendor_settings(provider))
    assert result["status"] == "ready"
    assert result["provenance"] == {"provider": provider, "model": model}
    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert body["model"] == model
    assert "vendor-private-key" not in request.content.decode()
    assert request.url == endpoint + ("/v1/messages" if provider == "anthropic" else "/chat/completions")
    if provider == "anthropic":
        assert request.headers["x-api-key"] == "vendor-private-key"
        assert body["max_tokens"] == 8192
        assert "JSON" in body["system"] and "'Nullable value' use [] for null" in body["system"]
        assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["format"]["schema"]["additionalProperties"] is False
        assert not {"tools", "response_format", "store"} & body.keys()
    else:
        assert request.headers["authorization"] == "Bearer vendor-private-key"
        assert "store" not in body
        assert [item["role"] for item in body["messages"]] == ["system", "user"]
        if provider == "grok":
            assert body["max_completion_tokens"] == 8192 and "max_tokens" not in body
            assert body["response_format"]["json_schema"]["strict"] is True
            assert body["response_format"]["type"] == "json_schema"
        else:
            assert body["max_tokens"] == 8192 and "max_completion_tokens" not in body
            assert body["response_format"] == {"type": "json_object"}
            assert '"properties"' in body["messages"][0]["content"]
            assert "JSON" in body["messages"][0]["content"]
        if provider == "qwen":
            assert body["enable_thinking"] is False
        elif provider == "deepseek":
            assert body["thinking"] == {"type": "disabled"}
        elif provider == "glm":
            assert "thinking" not in body


@pytest.mark.parametrize("provider", VENDORS)
def test_vendor_does_not_borrow_other_account_keys(monkeypatch, structure, provider):
    monkeypatch.setenv("DFT_AGENT_API_KEY", "generic-private")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-private")
    requests = mock_sdk(monkeypatch, provider)
    with pytest.raises(AgentError, match="key"):
        draft_plan("Relax Si.", structure, vendor_settings(provider))
    assert requests == []


@pytest.mark.parametrize("provider", VENDORS)
def test_explicit_key_wins_and_all_environment_keys_are_redacted(monkeypatch, structure, provider):
    all_keys = {value[2] for value in VENDORS.values()} | {"ZHIPUAI_API_KEY", "DFT_AGENT_API_KEY", "OPENAI_API_KEY"}
    secrets = []
    for index, key in enumerate(sorted(all_keys)):
        value = f"private-credential-{index}-end"
        secrets.append(value)
        monkeypatch.setenv(key, value)
    requests = mock_sdk(monkeypatch, provider)
    draft_plan("Relax Si. " + " ".join(secrets), structure,
               vendor_settings(provider, api_key="explicit-private-key"))
    assert len(requests) == 1
    header = "x-api-key" if provider == "anthropic" else "authorization"
    assert requests[0].headers[header].endswith("explicit-private-key")
    for secret in secrets + ["explicit-private-key"]:
        assert secret not in requests[0].content.decode()


def test_glm_china_endpoint_and_key(monkeypatch, structure):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "china-private-key")
    requests = mock_sdk(monkeypatch, "glm")
    draft_plan("Relax Si.", structure, vendor_settings("glm", base_url="https://open.bigmodel.cn/api/paas/v4"))
    assert requests[0].url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer china-private-key"


@pytest.mark.parametrize("provider", VENDORS)
def test_named_vendor_requires_its_own_model(monkeypatch, provider):
    monkeypatch.setenv("DFT_AGENT_MODEL", "wrong-account-model")
    endpoint = VENDORS[provider][1] if provider == "qwen" else None
    with pytest.raises(AgentError, match="model"):
        agent._settings(ModelSettings(provider, base_url=endpoint))


def test_qwen_requires_real_workspace_url(monkeypatch):
    monkeypatch.setenv("DFT_AGENT_BASE_URL", "https://unrelated.example/v1")
    with pytest.raises(AgentError, match="URL|url|endpoint"):
        agent._settings(ModelSettings("qwen", "qwen-plus"))


@pytest.mark.parametrize("provider", ["anthropic", "qwen", "grok", "glm", "deepseek"])
@pytest.mark.parametrize("content", ["not JSON", '{"status":"ready"}'])
def test_vendor_output_still_passes_local_plan_validation(monkeypatch, structure, provider, content):
    requests = mock_sdk(monkeypatch, provider, content=content)
    with pytest.raises(AgentError, match="invalid") as caught:
        draft_plan("Relax Si.", structure, vendor_settings(provider, api_key="private-key"))
    assert len(requests) == 1
    assert caught.value.model_usage["provider"] == provider
    assert caught.value.model_usage["output_tokens"] == 40


@pytest.mark.parametrize("provider,stop", [
    ("anthropic", "max_tokens"), ("anthropic", "refusal"), ("anthropic", "tool_use"),
    ("qwen", "length"), ("grok", "refusal"), ("glm", "content_filter"), ("deepseek", "tool_calls"),
])
def test_incomplete_or_refused_response_cannot_become_plan(monkeypatch, structure, provider, stop):
    requests = mock_sdk(monkeypatch, provider, stop=stop)
    with pytest.raises(AgentError) as caught:
        draft_plan("Relax Si.", structure, vendor_settings(provider, api_key="private-key"))
    assert len(requests) == 1
    assert caught.value.model_usage["provider"] == provider
    assert caught.value.model_usage["output_tokens"] == 40


@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
def test_sdk_does_not_retry_or_echo_provider_errors(monkeypatch, structure, provider):
    requests = mock_sdk(monkeypatch, provider, status=429)
    with pytest.raises(AgentError) as caught:
        draft_plan("Relax Si.", structure, vendor_settings(provider, api_key="private-server-key"))
    assert len(requests) == 1
    assert "private-server-key" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("cache_read,cache_created,expected", [(20, 30, 150), (None, None, 100), (-1, 30, None), (True, 30, None)])
def test_anthropic_input_usage_includes_cache_once(cache_read, cache_created, expected):
    usage = normalize_usage({"input_tokens": 100, "output_tokens": 40,
                             "cache_read_input_tokens": cache_read,
                             "cache_creation_input_tokens": cache_created}, "anthropic")
    assert usage["input_tokens"] == expected
    assert usage["output_tokens"] == 40
    assert usage["total_tokens"] == (expected + 40 if expected is not None else None)


@pytest.mark.parametrize("provider", ["qwen", "glm", "deepseek"])
def test_named_chat_usage_maps_provider_fields(provider):
    usage = normalize_usage({"prompt_tokens": 100, "completion_tokens": 40,
                             "prompt_tokens_details": {"cached_tokens": 20},
                             "completion_tokens_details": {"reasoning_tokens": 10}}, provider)
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 40
    assert usage["cached_input_tokens"] == 20 and usage["reasoning_tokens"] == 10
    assert usage["total_tokens"] == 140


def test_grok_usage_includes_separate_reasoning_tokens():
    # Values from the xAI Chat Completions reference response.
    usage = normalize_usage({"prompt_tokens": 32, "completion_tokens": 9,
                             "total_tokens": 135,
                             "prompt_tokens_details": {"cached_tokens": 6},
                             "completion_tokens_details": {"reasoning_tokens": 94}}, "grok")
    assert usage["input_tokens"] == 32
    assert usage["output_tokens"] == 103
    assert usage["reasoning_tokens"] == 94
    assert usage["cached_input_tokens"] == 6
    assert usage["total_tokens"] == 135


def test_claude_does_not_forward_ambient_auth_token(monkeypatch, structure):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-login-token")
    requests = mock_sdk(monkeypatch, "anthropic")
    draft_plan("Relax Si.", structure, vendor_settings("anthropic", api_key="explicit-api-key"))
    assert len(requests) == 1
    assert requests[0].headers["x-api-key"] == "explicit-api-key"
    assert "authorization" not in requests[0].headers
    assert "unrelated-login-token" not in requests[0].content.decode()


def union_count(value):
    if isinstance(value, dict):
        return int("anyOf" in value) + sum(union_count(item) for item in value.values())
    return sum(union_count(item) for item in value) if isinstance(value, list) else 0


def test_claude_native_schemas_fit_limits_without_changing_local_constraints():
    from copy import deepcopy
    from anthropic import transform_schema
    from vasp_slurm_agent import structure_agent, task_agent
    for schema in (agent._schema(["Ni", "O"]), structure_agent._schema(), task_agent._schema()):
        original = deepcopy(schema)
        transformed = transform_schema(agent._claude_wire_schema(schema))
        assert union_count(transformed) <= 16
        assert schema == original
        assert transformed["additionalProperties"] is False


def test_claude_native_structure_plan(monkeypatch, structure):
    from vasp_slurm_agent import structure_agent
    reply = {"status": "ready", "summary": "Make a supercell.", "operations": [
        {"type": "supercell", "matrix": [2, 1, 1]}], "output_format": "both", "questions": [], "notes": []}
    requests = mock_sdk(monkeypatch, "anthropic", content=json.dumps(reply), schema=structure_agent._schema())
    result = structure_agent.draft_structure("Make a 2x1x1 supercell.", structure,
        vendor_settings("anthropic", api_key="private-key"))
    assert result["status"] == "ready" and result["preview"]["output"]["number_of_sites"] == 4
    assert len(requests) == 1


def test_claude_native_task_with_variants(monkeypatch, structure):
    from copy import deepcopy
    from vasp_slurm_agent import task_agent
    legacy = json.loads(plan_json())
    parameters = legacy["parameters"]
    parameters["hubbard_u"] = []
    stages = [{"task": "scf", "parameters": parameters, "requirements": {
        "spin": None, "soc": None, "functional": None, "hubbard_u": None}}]
    reply = {"status": "ready", "summary": "Compare two strains.", "operations": [], "stages": stages,
             "variants": [{"label": "zero", "strain": None, "stages": deepcopy(stages)},
                          {"label": "one_percent", "strain": [0.01, 0, 0], "stages": deepcopy(stages)}],
             "questions": [], "notes": [], "intent": {"requested_tasks": ["scf"], "forbidden_tasks": [], "unsupported": []}}
    requests = mock_sdk(monkeypatch, "anthropic", content=json.dumps(reply), schema=task_agent._schema())
    result = task_agent.draft_task("Compare unstrained SCF and 1% a-axis strain.", structure,
        vendor_settings("anthropic", api_key="private-key"))
    assert result["status"] == "ready" and len(result["variants"]) == 2
    assert len(requests) == 1
    assert union_count(json.loads(requests[0].content)["output_config"]["format"]["schema"]) <= 16


def test_claude_explanation_schema_through_sdk(monkeypatch):
    schema = agent._object({"answer": {"type": "string"}, **{
        field: {"type": "array", "items": {"type": "string"}}
        for field in ("evidence", "limits", "next_steps")}})
    reply = {"answer": "The SCF calculation passed its checks.", "evidence": ["stage_1.accepted"],
             "limits": [], "next_steps": []}
    requests = mock_sdk(monkeypatch, "anthropic", content=json.dumps(reply), schema=schema)
    result = agent._request_structured('{"goal":"Explain results"}', schema,
        vendor_settings("anthropic", api_key="private-key"), instructions="Explain verified facts.", name="dft_explanation")
    assert json.loads(result) == reply and len(requests) == 1


def test_claude_repair_schema_through_sdk(monkeypatch):
    schema = agent._object({"action": {"type": "string", "enum": ["increase_nelm", "none"]},
        "diagnosis": {"type": "string"}, "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}})
    reply = {"action": "increase_nelm", "diagnosis": "The electronic step limit was reached.", "questions": []}
    requests = mock_sdk(monkeypatch, "anthropic", content=json.dumps(reply), schema=schema)
    result = agent._request_structured('{"goal":"Review failed SCF"}', schema,
        vendor_settings("anthropic", api_key="private-key"), instructions="Review a bounded repair.", name="dft_repair")
    assert json.loads(result) == reply and len(requests) == 1


@pytest.mark.parametrize("content,accepted", [
    ("```json\n" + plan_json() + "\n```", True),
    ("Here is the plan:\n```json\n" + plan_json() + "\n```", False),
    ("```json\n" + plan_json() + "\n```\nExtra instructions", False),
    ('```json\n{"status":"ready"}\n```', False),
])
def test_claude_only_accepts_one_complete_fence_with_valid_original_schema(monkeypatch, structure, content, accepted):
    mock_sdk(monkeypatch, "anthropic", content=content, raw_claude=True)
    settings = vendor_settings("anthropic", api_key="private-key")
    if accepted:
        assert draft_plan("Relax Si.", structure, settings)["status"] == "ready"
    else:
        with pytest.raises(AgentError, match="invalid structured data"):
            draft_plan("Relax Si.", structure, settings)


def test_claude_local_constraints_reject_values_omitted_by_api_schema(monkeypatch):
    schema = agent._object({"value": agent._nullable({"type": "number", "minimum": 0}),
                           "axis": agent._nullable({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3})})
    requests = mock_sdk(monkeypatch, "anthropic", content='{"value":-1,"axis":[1,2]}', schema=schema)
    with pytest.raises(AgentError, match="invalid structured data") as caught:
        agent._request_structured('{}', schema, vendor_settings("anthropic", api_key="private-key"), instructions="JSON")
    assert len(requests) == 1 and caught.value.model_usage["output_tokens"] == 40


@pytest.mark.parametrize("provider", [*VENDORS, "responses", "chat_completions"])
@pytest.mark.parametrize("status,fragment", [
    (400, "request format"), (401, "API key"), (402, "credit or spending"),
    (403, "denied access"), (404, "model or endpoint"), (413, "request size"),
    (429, "rate or usage limit"), (500, "temporarily unavailable"), (504, "timed out"), (529, "temporarily unavailable"),
])
def test_provider_http_errors_are_specific_private_and_not_retried(monkeypatch, provider, status, fragment):
    requests = mock_sdk(monkeypatch, provider, status=status)
    settings = (vendor_settings(provider, api_key="private-server-key") if provider in VENDORS else
                ModelSettings(provider, "test-model", api_key="private-server-key"))
    with pytest.raises(AgentError, match=fragment) as caught:
        agent._request_structured('{}', agent._object({"answer": {"type": "string"}}),
                                  agent._settings(settings), instructions="JSON")
    assert len(requests) == 1
    assert "private-server-key" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert caught.value.model_usage["input_tokens"] is None


@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "responses"])
@pytest.mark.parametrize("status,error_type,error_message", [
    (429, "insufficient_quota", "private-server-key"),
    (400, "invalid_request_error", "The credit balance is too low: private-server-key"),
])
def test_quota_errors_explain_billing_without_echoing_body(monkeypatch, provider, status, error_type, error_message):
    requests = mock_sdk(monkeypatch, provider, status=status, error_type=error_type, error_message=error_message)
    settings = (vendor_settings(provider, api_key="private-server-key") if provider in VENDORS else
                ModelSettings(provider, "test-model", api_key="private-server-key"))
    with pytest.raises(AgentError, match="credit or spending") as caught:
        agent._request_structured('{}', agent._object({"answer": {"type": "string"}}), settings, instructions="JSON")
    assert len(requests) == 1 and "private-server-key" not in str(caught.value)


@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "responses"])
@pytest.mark.parametrize("failure,fragment", [("ReadTimeout", "timed out"), ("ConnectError", "Cannot connect")])
def test_sdk_network_errors_keep_unknown_usage(monkeypatch, provider, failure, fragment):
    requests = mock_sdk(monkeypatch, provider, failure=failure)
    settings = (vendor_settings(provider, api_key="private-server-key") if provider in VENDORS else
                ModelSettings(provider, "test-model", api_key="private-server-key"))
    with pytest.raises(AgentError, match=fragment) as caught:
        agent._request_structured('{}', agent._object({"answer": {"type": "string"}}), settings, instructions="JSON")
    assert len(requests) == 1 and "private-server-key" not in str(caught.value)
    assert caught.value.model_usage["input_tokens"] is None


def test_claude_null_values_remain_unanswered_not_invented(monkeypatch, structure):
    reply = json.loads(plan_json())
    reply["parameters"]["hubbard_u"] = {"Si": {"l": 1, "u": 2, "j": None}}
    reply["intent"]["hubbard_u"] = True
    mock_sdk(monkeypatch, "anthropic", content=json.dumps(reply))
    result = draft_plan("SCF with the supplied +U settings; ask for J.", structure,
                        vendor_settings("anthropic", api_key="private-key"))
    assert result["status"] == "needs_input"
    assert result["parameters"]["hubbard_u"]["Si"]["j"] is None


@pytest.mark.parametrize("blocks", [
    [{"type": "thinking", "thinking": "private thought", "signature": "signature"}],
    [{"type": "tool_use", "id": "tool_test", "name": "do_not_run", "input": {}}],
    [{"type": "text", "text": plan_json()}, {"type": "text", "text": '{}'}],
])
def test_claude_unexpected_content_does_not_become_a_plan(monkeypatch, structure, blocks):
    mock_sdk(monkeypatch, "anthropic", blocks=blocks)
    with pytest.raises(AgentError, match="invalid structured data") as caught:
        draft_plan("Relax Si.", structure, vendor_settings("anthropic", api_key="private-key"))
    assert caught.value.model_usage["output_tokens"] == 40


def test_claude_thinking_plus_native_json_keeps_reported_usage(monkeypatch, structure):
    encoded = json.dumps(claude_wire(json.loads(plan_json()), agent._schema(["Si"])))
    requests = mock_sdk(monkeypatch, "anthropic", blocks=[
        {"type": "thinking", "thinking": "private thought", "signature": "signature"},
        {"type": "text", "text": encoded}])
    result = draft_plan("Relax Si.", structure, vendor_settings("anthropic", api_key="private-key"))
    assert result["status"] == "ready"
    assert result["model_usage"]["output_tokens"] == 40
    assert "private thought" not in json.dumps(result)
    sent = json.loads(requests[0].content)["output_config"]["format"]["schema"]
    assert result["model_usage"]["schema_characters"] == len(json.dumps(sent, separators=(",", ":")))
