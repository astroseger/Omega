"""Exercise the ChatGPT subscription protocol without credentials or a network."""

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    settings = {}
    config = types.ModuleType("config")
    config.config_get_by_key = lambda key, default=None: settings.get(key, default)
    auth = types.ModuleType("chatgpt_auth")

    class ChatGPTAuthError(RuntimeError):
        pass

    auth.ChatGPTAuthError = ChatGPTAuthError
    auth.get_access_token = lambda auth_file=None: "test-access-token"
    auth.list_models = lambda auth_file=None: [
        {"slug": "available-first", "display_name": "First", "visibility": "list"},
        {"slug": "available-second", "display_name": "Second", "visibility": "list"},
    ]
    openai = types.ModuleType("openai")

    class APIStatusError(Exception):
        def __init__(self, status_code, body=None):
            super().__init__("upstream request failed")
            self.status_code = status_code
            self.body = body
            self.code = (body or {}).get("error", {}).get("code")
            self.request_id = "req_test"
            self.response = NS(headers={})

    openai.APIStatusError = APIStatusError
    openai.AuthenticationError = APIStatusError
    openai.RateLimitError = APIStatusError
    openai.OpenAI = lambda **kwargs: pytest.fail("Unexpected model connection")
    monkeypatch.setitem(sys.modules, "config", config)
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "chatgpt_auth", auth)
    monkeypatch.setitem(sys.modules, "src.chatgpt_auth", auth)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    providers = load("providers", "src/providers.py")
    llm = load("lib_llm_ext", "providers/lib_llm_ext.py")
    load("omega_openai", "providers/omega_openai.py")
    chatgpt = load("omega_chatgpt", "providers/omega_chatgpt.py")
    return NS(P=providers, llm=llm, chatgpt=chatgpt, auth=auth,
              openai=openai, settings=settings)


def request(modules):
    p = modules.P
    send = (p.LLMTool().with_name("send").with_description("Send a reply")
            .add_parameter(p.LLMToolParameter().with_name("content")))
    return p.llmRequest(p.llmRequestMessage("system", "Agent instructions"),
                        [p.llmRequestMessage("user", "Hello")],
                        6000, "medium", [send])


def function_call(**overrides):
    values = dict(type="function_call", id="fc_item", call_id="call_execution",
                  namespace="omega", name="send", arguments='{"content":"Hello"}',
                  status="completed")
    values.update(overrides)
    return NS(**values)


def completed(*output):
    return NS(type="response.completed", response=NS(
        id="resp_test", status="completed", output=list(output),
        usage=None, incomplete_details=None, error=None))


class Stream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event


def connected(modules, monkeypatch, *streams):
    pending = list(streams)
    calls = []
    constructed = []
    client = NS(api_key=None, close=lambda: None)

    def create(**kwargs):
        calls.append(kwargs)
        next_stream = pending.pop(0)
        if isinstance(next_stream, Exception):
            raise next_stream
        return next_stream

    client.responses = NS(create=create)

    def make_client(**kwargs):
        constructed.append(kwargs)
        client.api_key = kwargs["api_key"]
        return client

    monkeypatch.setattr(modules.openai, "OpenAI", make_client)
    provider = modules.chatgpt.ChatGPTProviderImpl("test-model")
    return provider, client, calls, constructed


def test_request_uses_subscription_fields_and_namespaced_tools(modules):
    provider = modules.chatgpt.ChatGPTProviderImpl("gpt-5.5")
    raw = provider.convert_request(request(modules))
    assert raw["store"] is False
    assert raw["stream"] is True
    assert raw["tool_choice"] == "required"
    assert "max_output_tokens" not in raw
    assert "prompt_cache_retention" not in raw
    assert "previous_response_id" not in raw
    assert raw["input"][0] == {"role": "developer", "content": "Agent instructions"}
    namespace, = raw["tools"]
    assert namespace["type"] == "namespace"
    assert namespace["name"] == "omega"
    tool, = namespace["tools"]
    assert tool["name"] == "send"
    assert tool["type"] == "function"
    assert tool["strict"] is True
    assert tool["parameters"]["required"] == ["content"]
    assert tool["parameters"]["additionalProperties"] is False


def test_completed_call_identity_survives_history_round_trip(modules, monkeypatch):
    provider, _, _, _ = connected(modules, monkeypatch, Stream([completed(function_call())]))
    result = provider.chat(request(modules))
    assert not result.is_error()
    call, = result.calls
    assert call.id == "call_execution"
    assert call.name == "send"
    assert call.arguments == {"content": "Hello"}
    p = modules.P
    history = request(modules)
    history.add_message(p.llmToolCallMessage("assistant", [call], ""))
    history.add_message(p.llmToolCallResponseMessage("tool", call.id, "sent"))
    raw = provider.convert_request(history)
    sent_call, sent_result = raw["input"][-2:]
    assert sent_call["call_id"] == sent_result["call_id"] == "call_execution"
    assert sent_call["namespace"] == "omega"
    assert sent_result["type"] == "function_call_output"
    assert sent_result["output"] == "sent"


def test_only_completed_response_delivers_tools(modules, monkeypatch):
    partial = NS(type="response.output_item.done", item=function_call(arguments='{"content":"partial"}'))
    stream = Stream([partial, completed(function_call())])
    provider, _, calls, _ = connected(modules, monkeypatch, stream)
    result = provider.chat(request(modules))
    assert result.calls[0].arguments == {"content": "Hello"}
    assert len(calls) == 1
    assert stream.closed


@pytest.mark.parametrize("terminal", [
    NS(type="response.failed", response=NS(error=NS(code="server_error", message="Failed"))),
    NS(type="response.incomplete", response=NS(incomplete_details=NS(reason="max_output_tokens"))),
    NS(type="error", code="server_error", message="Failed"),
    None,
], ids=["failed", "incomplete", "error-event", "eof"])
def test_partial_tool_calls_are_discarded_on_failure(modules, monkeypatch, terminal):
    events = [NS(type="response.output_item.done", item=function_call())]
    if terminal is not None:
        events.append(terminal)
    stream = Stream(events)
    provider, _, calls, _ = connected(modules, monkeypatch, stream)
    result = provider.chat(request(modules))
    assert result.is_error()
    assert result.calls == []
    assert stream.closed
    assert len(calls) == 1


@pytest.mark.parametrize("changes", [
    {"namespace": "someone_else"},
    {"arguments": "not JSON"},
    {"arguments": "[]"},
    {"call_id": None},
], ids=["foreign-namespace", "invalid-json", "non-object-arguments", "missing-call-id"])
def test_invalid_calls_do_not_reach_tool_execution(modules, monkeypatch, changes):
    provider, _, _, _ = connected(modules, monkeypatch, Stream([completed(function_call(**changes))]))
    result = provider.chat(request(modules))
    assert result.is_error() or (result.calls and all(call.is_error() for call in result.calls))


def test_stream_usage_limit_pauses_later_inference(modules, monkeypatch):
    failure = NS(type="response.failed", response=NS(error=NS(
        code="subscription_sharing_usage_limit_exceeded", message="App limit reached")))
    provider, _, calls, _ = connected(modules, monkeypatch, Stream([failure]))
    first = provider.chat(request(modules))
    second = provider.chat(request(modules))
    assert first.is_error() and second.is_error()
    assert "usage" in first.error.lower()
    assert len(calls) == 1


def test_http_usage_limit_is_not_retried(modules, monkeypatch):
    failure = modules.openai.APIStatusError(429, {
        "error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "App limit reached"}})
    provider, _, calls, _ = connected(modules, monkeypatch, failure)
    assert provider.chat(request(modules)).is_error()
    assert provider.chat(request(modules)).is_error()
    assert len(calls) == 1


def test_unsupported_capability_pauses_identical_requests(modules, monkeypatch):
    failure = NS(type="response.failed", response=NS(error=NS(
        code="subscription_sharing_unsupported_capability")))
    provider, _, calls, _ = connected(modules, monkeypatch, Stream([failure]))
    assert provider.chat(request(modules)).is_error()
    assert provider.chat(request(modules)).is_error()
    assert len(calls) == 1


def test_usage_service_failure_backs_off_across_agent_loops(modules, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(modules.chatgpt.time, "monotonic", lambda: now[0])
    failure = NS(type="response.failed", response=NS(error=NS(
        code="subscription_sharing_usage_unavailable")))
    provider, _, calls, _ = connected(modules, monkeypatch,
        Stream([failure]), Stream([failure]), Stream([completed(function_call())]))
    assert provider.chat(request(modules)).is_error()
    now[0] += 1
    assert provider.chat(request(modules)).is_error()
    assert len(calls) == 1
    now[0] = 105
    assert provider.chat(request(modules)).is_error()
    now[0] = 110
    assert provider.chat(request(modules)).is_error()
    assert len(calls) == 2
    now[0] = 115
    assert not provider.chat(request(modules)).is_error()
    assert len(calls) == 3


def test_interrupted_stream_closes_and_discards_partial_tools(modules, monkeypatch):
    stream = Stream([NS(type="response.output_item.done", item=function_call()),
                     OSError("request included private-bearer-token")])
    provider, _, calls, _ = connected(modules, monkeypatch, stream)
    result = provider.chat(request(modules))
    assert result.is_error()
    assert result.calls == []
    assert "private-bearer-token" not in result.error
    assert stream.closed
    assert len(calls) == 1


def test_http_error_keeps_diagnostics_without_private_body(modules, monkeypatch, caplog):
    failure = modules.openai.APIStatusError(403, {
        "error": {"code": "subscription_sharing_route_not_supported",
                  "message": "private-bearer-token"}})
    provider, _, _, _ = connected(modules, monkeypatch, failure)
    result = provider.chat(request(modules))
    assert result.is_error()
    assert "403" in result.error
    assert "req_test" in result.error
    assert "subscription_sharing_route_not_supported" in result.error
    assert "private-bearer-token" not in result.error + caplog.text


def test_each_request_uses_current_bearer_without_api_key_fallback(modules, monkeypatch):
    tokens = iter(["first-access-token", "refreshed-access-token"])
    monkeypatch.setattr(modules.auth, "get_access_token", lambda *a, **kw: next(tokens))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    provider, client, calls, constructed = connected(
        modules, monkeypatch, Stream([completed(function_call())]), Stream([completed(function_call())]))
    assert not provider.chat(request(modules)).is_error()
    assert client.api_key == "first-access-token"
    assert not provider.chat(request(modules)).is_error()
    assert client.api_key == "refreshed-access-token"
    assert len(calls) == 2
    assert constructed[0]["base_url"] == "https://api.openai.com/v1"
    assert constructed[0]["max_retries"] == 0


def test_authentication_failure_never_uses_api_key(modules, monkeypatch):
    def fail(*args, **kwargs):
        raise modules.auth.ChatGPTAuthError("Sign in with scripts/chatgpt-auth login")

    monkeypatch.setattr(modules.auth, "get_access_token", fail)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    provider = modules.chatgpt.ChatGPTProviderImpl("test-model")
    result = provider.chat(request(modules))
    assert result.is_error()
    assert "login" in result.error


def test_gateway_mode_is_rejected_before_loading_credentials(modules, monkeypatch):
    modules.settings["GATEWAY_URL"] = "http://proxy"
    monkeypatch.setattr(modules.auth, "get_access_token", lambda *a, **kw: pytest.fail("Read credentials"))
    with pytest.raises((RuntimeError, ValueError), match="(?i)(gateway|docker|standalone)"):
        modules.chatgpt.ChatGPTProvider().start()


def test_default_model_comes_from_account_catalog(modules):
    provider = modules.chatgpt.ChatGPTProvider()
    provider.start()
    assert provider.delegate._model_name == "available-first"


def test_explicit_model_overrides_subscription_default(modules):
    modules.settings.update(model="available-second", chatgpt_model="available-first")
    provider = modules.chatgpt.ChatGPTProvider()
    provider.start()
    assert provider.delegate._model_name == "available-second"


def test_unavailable_explicit_model_is_rejected(modules):
    modules.settings["model"] = "not-available"
    with pytest.raises(modules.auth.ChatGPTAuthError, match="(?i)(model|listed)"):
        modules.chatgpt.ChatGPTProvider().start()


def test_empty_catalog_is_reported(modules, monkeypatch):
    monkeypatch.setattr(modules.auth, "list_models", lambda *a, **kw: [])
    with pytest.raises(modules.auth.ChatGPTAuthError, match="(?i)no.*models"):
        modules.chatgpt.ChatGPTProvider().start()


def test_stopping_before_first_request_is_safe(modules):
    modules.chatgpt.ChatGPTProvider().stop()
    modules.chatgpt.ChatGPTProviderImpl("test-model").stop()


def test_plugin_registers_distinct_provider(modules):
    modules.chatgpt.loadOmegaPlugin()
    assert isinstance(modules.P._llmProviderRegistry["ChatGPT"], modules.chatgpt.ChatGPTProvider)
