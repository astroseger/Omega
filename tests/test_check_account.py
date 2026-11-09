"""Check the one-request account probe without credentials or network access."""

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace as NS

import pytest

import chatgpt_auth as auth
import check_account as check


ROOT = Path(__file__).resolve().parents[1]
NONCE = "unique-account-check-nonce"


@pytest.fixture
def provider(monkeypatch, tmp_path):
    # The probe adds checkout paths for its worker; restore them after each test.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    settings = {}
    config = types.ModuleType("config")
    config.config_get_by_key = lambda key, default=None: settings.get(key, default)

    def init_config(arguments):
        settings.clear()
        settings.update(argument.split("=", 1) for argument in arguments)

    config.init_config = init_config
    monkeypatch.setitem(sys.modules, "config", config)
    spec = importlib.util.spec_from_file_location("providers", ROOT / "src/providers.py")
    models = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "providers", models)
    spec.loader.exec_module(models)
    state = NS(events=[], requests=[], settings=settings, models=models,
               start_error=None, stop_error=None, auth_file=tmp_path / "chatgpt.json")
    state.response = models.LLMResponse().add_tool_call(
        models.LLMToolCall().with_name("account_check").with_id("call_probe")
        .with_arguments({"nonce": NONCE}))

    class FakeProvider:
        def __init__(self):
            self.delegate = None

        def start(self):
            state.events.append("start")
            state.selected_auth_file = os.environ.get("OMEGA_CHATGPT_AUTH_FILE")
            if state.start_error:
                raise state.start_error
            self.delegate = NS(_model_name=settings.get("model") or "available-model")

        def chat(self, request):
            state.events.append("chat")
            state.requests.append(request)
            return state.response

        def stop(self):
            state.events.append("stop")
            if state.stop_error:
                raise state.stop_error

    module = types.ModuleType("omega_chatgpt")
    module.ChatGPTProvider = FakeProvider
    monkeypatch.setitem(sys.modules, "omega_chatgpt", module)
    monkeypatch.setattr(check.secrets, "token_hex", lambda *args: NONCE)
    monkeypatch.setenv("OMEGA_CHATGPT_AUTH_FILE", str(tmp_path / "unused-default.json"))
    return state


def test_probe_uses_selected_account_model_and_only_one_inert_request(provider):
    assert check._check_once(provider.auth_file, model="chosen-model") == "chosen-model"
    assert provider.selected_auth_file == str(provider.auth_file)
    assert provider.events == ["start", "chat", "stop"]
    request, = provider.requests
    tool, = request.tools
    assert tool.name == "account_check"
    assert [parameter.name for parameter in tool.parameters] == ["nonce"]
    assert NONCE in " ".join(message.content for message in request.messages)


@pytest.mark.parametrize("defect", [
    "response-error", "call-error", "no-calls", "multiple-calls", "wrong-tool",
    "wrong-nonce", "extra-argument", "missing-id", "synthetic-timeout",
])
def test_probe_rejects_false_success_and_always_closes(provider, defect):
    response = provider.response
    call = response.calls[0]
    if defect == "response-error":
        response.error = "Usage unavailable"
    elif defect == "call-error":
        call.error = "Invalid tool result"
    elif defect == "no-calls":
        response.calls.clear()
    elif defect == "multiple-calls":
        response.calls.append(call)
    elif defect == "wrong-tool":
        call.name = "exec"
    elif defect == "wrong-nonce":
        call.arguments["nonce"] = "not-the-request-nonce"
    elif defect == "extra-argument":
        call.arguments["command"] = "unexpected"
    elif defect == "missing-id":
        call.id = None
    else:
        # Omega converts an SDK timeout into a send tool, without response.error.
        call.name = "send"
        call.arguments = {"content": "The provider timed out"}

    with pytest.raises(check.AccountCheckError):
        check._check_once(provider.auth_file)
    assert provider.events == ["start", "chat", "stop"]
    assert len(provider.requests) == 1


def test_start_failure_closes_provider_without_inference(provider):
    provider.start_error = auth.ChatGPTAuthError("Sign in again")
    with pytest.raises(auth.ChatGPTAuthError, match="Sign in again"):
        check._check_once(provider.auth_file)
    assert provider.events == ["start", "stop"]
    assert not provider.requests


def test_cleanup_failure_preserves_original_failure(provider):
    provider.response.error = "Usage unavailable"
    provider.stop_error = RuntimeError("private-refresh-token")
    with pytest.raises(check.AccountCheckError, match="Usage unavailable"):
        check._check_once(provider.auth_file)
    assert provider.events[-1] == "stop"


def test_cleanup_failure_cannot_report_success(provider, capsys):
    provider.stop_error = RuntimeError("private-refresh-token")
    assert check._worker(provider.auth_file) == 1
    output = capsys.readouterr()
    assert "Account OK" not in output.out
    assert "private-refresh-token" not in output.out + output.err


def test_worker_prints_success_only_after_valid_response(provider, capsys):
    assert check._worker(provider.auth_file, model="chosen-model") == 0
    output = capsys.readouterr()
    assert "Account OK" in output.out
    assert "chosen-model" in output.out
    assert not output.err


@pytest.mark.parametrize("failure", [
    RuntimeError("private-refresh-token"),
    ValueError("private-refresh-token"),
    OSError("private-refresh-token"),
])
def test_worker_never_prints_unexpected_exception_payloads(provider, capsys, failure):
    provider.start_error = failure
    assert check._worker(provider.auth_file) == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err
    assert "private-refresh-token" not in output.err


def test_worker_preserves_safe_authentication_diagnostic(provider, capsys):
    provider.start_error = auth.ChatGPTAuthError("ChatGPT is not signed in")
    assert check._worker(provider.auth_file) == 1
    assert "not signed in" in capsys.readouterr().err


def test_worker_reports_missing_dependencies_without_exception_payload(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("private-refresh-token", name="openai")

    monkeypatch.setattr(check, "_check_once", missing)
    assert check._worker("unused.json") == 1
    output = capsys.readouterr()
    assert "openai" in output.err
    assert "requirements.txt" in output.err
    assert "private-refresh-token" not in output.out + output.err


def test_selected_file_model_and_deadline_reach_worker(monkeypatch, tmp_path):
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return NS(returncode=0)

    monkeypatch.setattr(check.subprocess, "run", run)
    credential = tmp_path / "chosen account" / "chatgpt.json"
    assert check.check_account(auth_file=credential, model="chosen-model", timeout=9) == 0
    assert str(credential) in captured["command"]
    assert "chosen-model" in captured["command"]
    assert captured["timeout"] == 9
    assert captured["command"][0] == sys.executable


def test_environment_selects_worker_credentials(monkeypatch, tmp_path):
    credential = tmp_path / "from-env.json"
    commands = []
    monkeypatch.setenv("OMEGA_CHATGPT_AUTH_FILE", str(credential))

    def run(command, **kwargs):
        commands.append(command)
        return NS(returncode=0)

    monkeypatch.setattr(check.subprocess, "run", run)
    assert check.check_account() == 0
    assert str(credential) in commands[0]


def test_selected_symlink_is_not_resolved_around_auth_ownership_checks(monkeypatch, tmp_path):
    target = tmp_path / "target.json"
    link = tmp_path / "selected.json"
    link.symlink_to(target)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return NS(returncode=1)

    monkeypatch.setattr(check.subprocess, "run", run)
    assert check.check_account(auth_file=link) == 1
    assert str(link) in commands[0]
    assert str(target) not in commands[0]


def test_timeout_returns_distinct_status_without_echoing_worker_output(monkeypatch, capsys):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="private-refresh-token")

    monkeypatch.setattr(check.subprocess, "run", run)
    assert check.check_account(timeout=1) == 124
    output = capsys.readouterr()
    assert not output.out
    assert "timed out" in output.err.lower()
    assert "private-refresh-token" not in output.err


def test_deadline_kills_and_reaps_real_worker(monkeypatch, tmp_path, capsys):
    pid_file = tmp_path / "pid"
    worker = tmp_path / "sleeping_worker.py"
    worker.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n")
    monkeypatch.setattr(check, "__file__", str(worker))
    assert check.check_account(auth_file=tmp_path / "unused.json", timeout=1) == 124
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "timed out" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("timeout", [0, -1, 1.5, float("nan"), True])
def test_invalid_timeout_never_starts_worker(monkeypatch, timeout):
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected worker"))
    assert check.check_account(timeout=timeout) == 2


def test_auth_cli_passes_account_model_and_timeout(monkeypatch, tmp_path):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return 124

    monkeypatch.setattr(check, "check_account", run)
    credential = str(tmp_path / "account.json")
    assert auth.main(["--auth-file", credential, "check", "--model", "chosen-model", "--timeout", "7"]) == 124
    assert captured == {"auth_file": credential, "model": "chosen-model", "timeout": 7}


def test_auth_help_and_status_keep_provider_dependencies_lazy(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"check_account", "omega_chatgpt", "openai"}:
            pytest.fail(f"Unexpected heavy import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as stopped:
        auth.main(["--help"])
    assert stopped.value.code == 0
    assert auth.main(["--auth-file", str(tmp_path / "missing.json"), "status"]) == 1
