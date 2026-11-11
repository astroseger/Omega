"""Check a ChatGPT account with one request through Omega's real provider.

The public command runs the probe in a subprocess so its deadline also covers
imports, token refresh, model discovery, and streamed inference. The worker
does not start the agent loop or execute any returned tools.
"""

import argparse
import os
from pathlib import Path
import secrets
import subprocess
import sys

from chatgpt_auth import ChatGPTAuthError, credential_path


DEFAULT_TIMEOUT = 120


class AccountCheckError(RuntimeError):
    """An account-check diagnostic safe to print without model output or tokens."""


def _check_once(auth_file, model=None):
    """Run inside the disposable worker; return the model used on success."""
    root = Path(__file__).resolve().parents[1]
    # src/providers.py must take precedence over the providers directory.
    for location in (root, root / "providers", root / "src"):
        sys.path.insert(0, str(location))

    # These imports deliberately stay out of login, status, and CLI help.
    from config import config_get_by_key, init_config
    from omega_chatgpt import ChatGPTProvider
    from providers import LLMMessage, LLMRequest, LLMTool, LLMToolParameter

    os.environ["OMEGA_CHATGPT_AUTH_FILE"] = str(auth_file)
    init_config([f"model={model}"] if model is not None else [])
    if config_get_by_key("GATEWAY_URL"):
        raise AccountCheckError("The account check requires direct ChatGPT access; GATEWAY_URL is unsupported.")

    nonce = secrets.token_hex(16)
    tool = (LLMTool().with_name("account_check")
            .with_description("Return the supplied nonce to verify this connection. This tool has no side effects.")
            .add_parameter(LLMToolParameter().with_name("nonce")))
    request = (LLMRequest()
               .add_message(LLMMessage().with_role("system").with_content(
                   "This is a connection check. Call account_check exactly once with the user's nonce."))
               .add_message(LLMMessage().with_role("user").with_content(f"Nonce: {nonce}"))
               .add_tool(tool))

    provider = ChatGPTProvider()
    failed = True
    try:
        provider.start()
        response = provider.chat(request)
        if response.is_error():
            # The ChatGPT provider already sanitizes server/authentication errors.
            raise AccountCheckError(response.error)
        if len(response.calls) != 1:
            raise AccountCheckError("Expected exactly one account_check response.")
        call = response.calls[0]
        if (call.is_error() or not call.id or call.name != "account_check"
                or call.arguments != {"nonce": nonce}):
            # In particular, a synthetic 'send' timeout notice is not success.
            raise AccountCheckError("The model did not return the expected account_check response.")
        selected_model = provider.delegate._model_name
        failed = False
        return selected_model
    finally:
        try:
            provider.stop()
        except Exception:
            if not failed:
                raise AccountCheckError("Could not close the ChatGPT connection after the check.") from None


def _worker(auth_file, model=None):
    try:
        selected_model = _check_once(auth_file, model)
    except (AccountCheckError, ChatGPTAuthError) as error:
        print(f"Account check failed: {error}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as error:
        print(f"Account check failed: missing Python dependency {error.name!r}; "
              "install Omega's requirements.txt.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Account check cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        # Exception messages may contain HTTP request data or credential bodies.
        print(f"Account check failed ({type(error).__name__}).", file=sys.stderr)
        return 1
    print(f"Account OK: {auth_file} (model: {selected_model})")
    return 0


def check_account(*, auth_file=None, model=None, timeout=DEFAULT_TIMEOUT):
    """Return 0 on success, 1 on failure, 2 on invalid timeout, or 124 on timeout."""
    if type(timeout) is not int or timeout <= 0:
        print("Account check timeout must be a positive integer in seconds.", file=sys.stderr)
        return 2
    # Do not resolve symlinks: the auth helper must enforce its own O_NOFOLLOW
    # checks on the credential and lock files.
    path = credential_path(auth_file).absolute()
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", "--auth-file", str(path)]
    if model is not None:
        command.extend(["--model", model])
    try:
        # run() kills and reaps the worker on timeout/exception, including when
        # this command is itself run inside Docker. No detached work remains.
        result = subprocess.run(command, stdin=subprocess.DEVNULL, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"Account check timed out after {timeout} seconds; the worker was stopped. "
              "The request may already have consumed plan usage.", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        print("Account check cancelled.", file=sys.stderr)
        return 130
    except OSError as error:
        print(f"Could not start the account check ({type(error).__name__}).", file=sys.stderr)
        return 1
    if result.returncode in (0, 1, 130):
        return result.returncode
    print(f"Account check worker exited unexpectedly (status {result.returncode}).", file=sys.stderr)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a ChatGPT account with one Omega inference request.")
    parser.add_argument("--auth-file", help="Credential file (default: OMEGA_CHATGPT_AUTH_FILE or ~/.omega_auth/chatgpt.json)")
    parser.add_argument("--model", help="Account model to check (default: Omega configuration or first available model)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Overall deadline in seconds (default: 120)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(credential_path(args.auth_file), args.model)
    return check_account(auth_file=args.auth_file, model=args.model, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
