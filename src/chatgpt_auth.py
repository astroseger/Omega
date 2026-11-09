"""Local OAuth credentials for OpenAI's open-source ChatGPT plan usage flow.

One account/workspace registration is stored per file. Credentials are never
read from Codex or treated as Platform API keys. See the official contract:
https://developers.openai.com/siwc/token-sharing-open-source/sign-in
"""

import argparse
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import webbrowser


ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
TOKEN_FIELDS = ("access_token", "refresh_token", "id_token", "expires_at", "scopes")
TERMINAL_REFRESH_ERRORS = {
    "invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused",
}


class ChatGPTAuthError(RuntimeError):
    """A safe, user-facing authentication error; never includes token payloads."""


class _HTTPFailure(ChatGPTAuthError):
    def __init__(self, status, code=None):
        self.status = status
        self.code = code
        # Only recognized protocol codes are exposed, not arbitrary server text.
        suffix = f" ({code})" if code else ""
        super().__init__(f"OpenAI authentication request failed: HTTP {status}{suffix}.")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ChatGPTAuthError("OpenAI authentication request unexpectedly redirected.")


class _CallbackServer(HTTPServer):
    # Browser preconnects can accept a TCP connection without sending a request.
    # Bound each accepted connection as well as the listener's accept timeout.
    connection_timeout = 5

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.connection_timeout)
        return connection, address


def _request_json(url, *, form=None, bearer=None, empty_ok=False):
    headers = {"Accept": "application/json"}
    data = None
    if form is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urlencode(form).encode("utf-8")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    try:
        with build_opener(_NoRedirect()).open(
            Request(url, data=data, headers=headers), timeout=30
        ) as response:
            raw = response.read(2 * 1024 * 1024)
    except HTTPError as error:
        code = None
        try:
            body = json.loads(error.read(65536))
            value = body.get("error") if isinstance(body, dict) else None
            if isinstance(value, dict):
                value = value.get("code")
            if isinstance(value, str) and value in TERMINAL_REFRESH_ERRORS | {"invalid_client"}:
                code = value
        except (ValueError, OSError):
            pass
        raise _HTTPFailure(error.code, code) from None
    except (URLError, OSError, TimeoutError):
        raise ChatGPTAuthError("Could not reach OpenAI. Check the connection and try again.") from None
    if not raw and empty_ok:
        return {}
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ChatGPTAuthError("OpenAI returned an invalid JSON response.") from None
    if not isinstance(result, dict):
        raise ChatGPTAuthError("OpenAI returned an unexpected response format.")
    return result


def credential_path(auth_file=None):
    return Path(auth_file or os.environ.get("OMEGA_CHATGPT_AUTH_FILE")
                or "~/.config/omega/chatgpt.json").expanduser()


@contextmanager
def _locked(path):
    try:
        import fcntl
    except ImportError:
        raise ChatGPTAuthError("ChatGPT local credential locking requires a Unix system.") from None
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path) + ".lock", flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ChatGPTAuthError("The ChatGPT credential lock must be owned by the current user.")
        os.fchmod(fd, 0o600)
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ChatGPTAuthError("Another process is signing in or refreshing ChatGPT credentials; try again later.")
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)


def _read(path):
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return {}
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ChatGPTAuthError("ChatGPT credentials must be a regular file owned by you with mode 0600.")
        with os.fdopen(fd, "r", encoding="utf-8") as source:
            fd = None
            result = json.load(source)
    except (ValueError, UnicodeError):
        raise ChatGPTAuthError("The ChatGPT credential file is invalid; restore it or use a new --auth-file.") from None
    finally:
        if fd is not None:
            os.close(fd)
    if not isinstance(result, dict):
        raise ChatGPTAuthError("The ChatGPT credential file has an invalid format.")
    for key in ("client_id", "subject", "ext_agent_host_id", "access_token", "refresh_token", "id_token"):
        if key in result and not isinstance(result[key], str):
            raise ChatGPTAuthError("The ChatGPT credential file has an invalid format.")
    scopes = result.get("scopes", [])
    if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
        raise ChatGPTAuthError("The ChatGPT credential file has invalid scopes.")
    try:
        if "expires_at" in result and not math.isfinite(float(result["expires_at"])):
            raise ValueError()
    except (ValueError, TypeError):
        raise ChatGPTAuthError("The ChatGPT credential file has an invalid expiry.") from None
    return result


def _write(path, record):
    fd, temporary = tempfile.mkstemp(prefix=".chatgpt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(record, destination, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _clear_tokens(record):
    return {key: value for key, value in record.items() if key not in TOKEN_FIELDS}


def _discovery_endpoint(field):
    discovery = _request_json(DISCOVERY_URL)
    if discovery.get("issuer") != ISSUER:
        raise ChatGPTAuthError("OpenAI discovery returned an unexpected issuer.")
    endpoint = discovery.get(field, "")
    if not isinstance(endpoint, str):
        raise ChatGPTAuthError("OpenAI discovery returned an unexpected endpoint.")
    parts = urlsplit(endpoint)
    if parts.scheme != "https" or parts.netloc != "auth.openai.com" or parts.fragment:
        raise ChatGPTAuthError("OpenAI discovery returned an unexpected endpoint.")
    return endpoint


def _validate_identity(token, client_id, nonce):
    try:
        import jwt
    except ImportError:
        raise ChatGPTAuthError("Install PyJWT[crypto]>=2.8,<3 to sign in with ChatGPT.") from None
    jwks = _request_json(_discovery_endpoint("jwks_uri"))
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm not in {"RS256", "ES256"}:
            raise ValueError("Unsupported signing algorithm")
        keys = jwks.get("keys", [])
        if not isinstance(keys, list):
            raise ValueError("Invalid signing keys")
        candidates = [key for key in keys
                      if isinstance(key, dict) and key.get("kid") == header.get("kid")
                      and key.get("use", "sig") == "sig"
                      and key.get("alg", algorithm) == algorithm]
        if len(candidates) != 1:
            raise ValueError("Unknown signing key")
        public_key = jwt.PyJWK.from_dict(candidates[0], algorithm=algorithm).key
        identity = jwt.decode(token, public_key, algorithms=[algorithm],
                              issuer=ISSUER, audience=client_id, leeway=5,
                              options={"require": ["iss", "aud", "exp", "iat", "sub", "nonce"]})
        if not isinstance(identity["sub"], str) or not identity["sub"]:
            raise ValueError("Invalid subject")
        if not isinstance(identity["nonce"], str) or not secrets.compare_digest(identity["nonce"], nonce):
            raise ValueError("Nonce mismatch")
    except (jwt.PyJWTError, ValueError, TypeError, KeyError):
        raise ChatGPTAuthError("ChatGPT identity verification failed; sign in again.") from None
    return identity


def _token_record(record, tokens, *, initial=False):
    access = tokens.get("access_token")
    token_type = tokens.get("token_type", "")
    if (not isinstance(access, str) or not access or not isinstance(token_type, str)
            or token_type.lower() != "bearer"):
        raise ChatGPTAuthError("OpenAI did not return a valid bearer token.")
    try:
        expires = float(tokens["expires_in"])
        if not math.isfinite(expires) or expires <= 0:
            raise ValueError()
    except (KeyError, ValueError, TypeError):
        raise ChatGPTAuthError("OpenAI did not return a valid token expiry.") from None
    scope = tokens.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ChatGPTAuthError("OpenAI did not return valid granted scopes.")
    scopes = scope.split() if scope is not None else ([] if initial else record.get("scopes", []))
    refresh = tokens.get("refresh_token", record.get("refresh_token"))
    if not isinstance(refresh, str) or not refresh:
        raise ChatGPTAuthError("OpenAI did not return a refresh token; sign in again.")
    updated = dict(record, access_token=access, refresh_token=refresh,
                   scopes=scopes, expires_at=time.time() + expires)
    # The original validated ID token is retained as identity metadata. Refresh
    # doesn't need a new ID token, and unvalidated replacement claims aren't used.
    return updated


def _require_plan(record):
    if PLAN_SCOPE not in record.get("scopes", []):
        raise ChatGPTAuthError("ChatGPT plan usage was not authorized. Run chatgpt-auth login --consent to enable it.")


def get_access_token(auth_file=None):
    """Return a current token, serializing refresh-token rotation across processes."""
    path = credential_path(auth_file)
    with _locked(path):
        record = _read(path)
        if not record.get("access_token") or not record.get("client_id") or not record.get("subject"):
            raise ChatGPTAuthError("ChatGPT is not signed in. Run scripts/chatgpt-auth login first.")
        _require_plan(record)
        if float(record.get("expires_at", 0)) > time.time() + 60:
            return record["access_token"]
        try:
            tokens = _request_json(TOKEN_URL, form={
                "grant_type": "refresh_token", "client_id": record["client_id"],
                "refresh_token": record.get("refresh_token", ""), "resource": RESOURCE,
            })
        except _HTTPFailure as error:
            if error.code in TERMINAL_REFRESH_ERRORS:
                _write(path, _clear_tokens(record))
                raise ChatGPTAuthError("ChatGPT sign-in expired or was revoked. Run scripts/chatgpt-auth login again.") from None
            raise
        updated = _token_record(record, tokens)
        _write(path, updated)
        _require_plan(updated)
        return updated["access_token"]


def list_models(auth_file=None):
    """List visible account-specific model choices without making an inference request."""
    data = _request_json(RESOURCE + "/models", bearer=get_access_token(auth_file))
    models = data.get("models")
    if not isinstance(models, list):
        raise ChatGPTAuthError("OpenAI returned an unexpected ChatGPT model catalog.")
    return [model for model in models if isinstance(model, dict)
            and model.get("visibility") == "list" and isinstance(model.get("slug"), str)]


def _authorization(record, redirect_uri, consent=False):
    verifier = secrets.token_urlsafe(48)
    pending = {"state": secrets.token_urlsafe(32), "nonce": secrets.token_urlsafe(32),
               "verifier": verifier, "redirect_uri": redirect_uri,
               "client_id": record.get("client_id") or "dynamic_agent_client"}
    params = {"client_id": pending["client_id"], "ext_agent_host_id": record["ext_agent_host_id"],
              "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES,
              "resource": RESOURCE, "state": pending["state"], "nonce": pending["nonce"],
              "code_challenge_method": "S256",
              "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()}
    if pending["client_id"] == "dynamic_agent_client":
        params["agent_name_hint"] = "Omega"
    if consent:
        params["prompt"] = "consent"
    # No ID-token hint in the URL: --no-browser prints it for manual navigation.
    return pending, AUTHORIZE_URL + "?" + urlencode(params)


def _complete_login(path, record, pending, callback):
    if (not isinstance(callback.get("state"), str)
            or not secrets.compare_digest(callback["state"].encode(), pending["state"].encode())):
        raise ChatGPTAuthError("ChatGPT callback state did not match.")
    if callback.get("error"):
        raise ChatGPTAuthError("ChatGPT sign-in was declined or failed. No new credentials were saved.")
    client_id = callback.get("client_id") or pending["client_id"]
    if client_id == "dynamic_agent_client" or not client_id or not callback.get("code"):
        raise ChatGPTAuthError("ChatGPT registration did not return a client ID and authorization code.")
    if pending["client_id"] != "dynamic_agent_client" and client_id != pending["client_id"]:
        raise ChatGPTAuthError("ChatGPT callback changed the selected account's client ID.")
    registration = dict(record, client_id=client_id)
    # Retain the issued registration even if the one-time code expires. The
    # next sign-in must reuse it instead of registering a duplicate app.
    if pending["client_id"] == "dynamic_agent_client":
        _write(path, registration)
    tokens = _request_json(TOKEN_URL, form={
        "grant_type": "authorization_code", "client_id": client_id,
        "code": callback["code"], "code_verifier": pending["verifier"],
        "redirect_uri": pending["redirect_uri"], "resource": RESOURCE,
    })
    identity = _validate_identity(tokens.get("id_token"), client_id, pending["nonce"])
    if record.get("subject") and identity["sub"] != record["subject"]:
        raise ChatGPTAuthError("ChatGPT account changed. Use a separate --auth-file for another account.")
    registration.update(subject=identity["sub"], issuer=ISSUER,
                        email=identity.get("email", ""), id_token=tokens["id_token"])
    updated = _token_record(registration, tokens, initial=True)
    _write(path, updated)
    _require_plan(updated)
    return updated


def login(auth_file=None, *, no_browser=False, port=0, consent=False):
    path = credential_path(auth_file)
    with _locked(path):
        record = _read(path)
        if not record.get("ext_agent_host_id"):
            record["ext_agent_host_id"] = "urn:uuid:" + str(uuid.uuid4())
            _write(path, record)
        result = {}
        pending = {}

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Request paths contain authorization codes.

            def do_GET(self):
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query, keep_blank_values=True)
                valid = (parsed.path == "/auth/callback"
                         and all(len(values) == 1 for values in query.values())
                         and secrets.compare_digest(query.get("state", [""])[0].encode(), pending["state"].encode()))
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                if valid:
                    result.update({key: values[0] for key, values in query.items()})
                self.wfile.write(b"Return to the terminal to finish sign-in." if valid else b"Invalid sign-in callback.")

        with _CallbackServer(("127.0.0.1", port), Callback) as server:
            server.timeout = 1
            redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
            pending, url = _authorization(record, redirect_uri, consent)
            print("Continue with ChatGPT. Authorize Omega to use your ChatGPT plan.")
            if no_browser or not webbrowser.open(url):
                print("Open this URL in a browser on this computer:\n" + url)
            deadline = time.monotonic() + 300
            while not result and time.monotonic() < deadline:
                server.handle_request()
            if not result:
                raise ChatGPTAuthError("ChatGPT sign-in timed out; run login again.")
        return _complete_login(path, record, pending, result)


def logout(auth_file=None):
    """Clear local tokens, attempting remote refresh-session revocation first."""
    path = credential_path(auth_file)
    with _locked(path):
        record = _read(path)
        confirmed = not bool(record.get("refresh_token"))
        if record.get("refresh_token"):
            for attempt in range(3):
                try:
                    _request_json(_discovery_endpoint("revocation_endpoint"), form={
                        "token": record["refresh_token"], "token_type_hint": "refresh_token",
                        "client_id": record["client_id"],
                    }, empty_ok=True)
                    confirmed = True
                    break
                except ChatGPTAuthError as error:
                    if isinstance(error, _HTTPFailure) and error.status < 500:
                        break
                    if attempt < 2:
                        time.sleep(0.5 * 2 ** attempt)
        _write(path, _clear_tokens(record))
        return confirmed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage Omega's local Sign in with ChatGPT connection.")
    parser.add_argument("--auth-file", help="Credential file (default: OMEGA_CHATGPT_AUTH_FILE or ~/.config/omega/chatgpt.json)")
    commands = parser.add_subparsers(dest="command", required=True)
    sign_in = commands.add_parser("login", help="Continue with ChatGPT in your browser")
    sign_in.add_argument("--no-browser", action="store_true", help="Print the sign-in URL instead of opening a browser")
    sign_in.add_argument("--port", type=int, default=0, help="Loopback callback port (default: choose a free port)")
    sign_in.add_argument("--consent", action="store_true", help="Ask to enable ChatGPT plan usage after previously declining it")
    commands.add_parser("status", help="Show local connection status without refreshing tokens")
    commands.add_parser("models", help="List models available to the connected ChatGPT account")
    commands.add_parser("logout", help="Revoke the session and clear local credentials")
    args = parser.parse_args(argv)
    try:
        if args.command == "login":
            record = login(args.auth_file, no_browser=args.no_browser, port=args.port, consent=args.consent)
            print(f"Signed in as {record.get('email') or record['subject']}. Using your ChatGPT plan.")
        elif args.command == "models":
            for model in list_models(args.auth_file):
                print(f"{model['slug']}\t{model.get('display_name', model['slug'])}")
        elif args.command == "logout":
            confirmed = logout(args.auth_file)
            print("Signed out locally." + (" Remote session revoked." if confirmed else
                  " Remote revocation was not confirmed; disconnect Omega in ChatGPT Settings > Security and login."))
        else:
            path = credential_path(args.auth_file)
            with _locked(path):
                record = _read(path)
            if not record.get("access_token"):
                print("Not signed in. Run scripts/chatgpt-auth login.")
                return 1
            print(f"Account: {record.get('email') or record.get('subject', 'unknown')}")
            print("ChatGPT plan usage: " + ("enabled" if PLAN_SCOPE in record.get("scopes", []) else "not authorized"))
            print("Token: " + ("current" if float(record.get("expires_at", 0)) > time.time() else "expired; will refresh on use"))
        return 0
    except (ChatGPTAuthError, OSError, ValueError) as error:
        # OSError/ValueError here contain local paths or parsing diagnostics, not
        # HTTP response bodies. Protocol errors are sanitized at the boundary.
        print(f"ChatGPT: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ChatGPT sign-in cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
