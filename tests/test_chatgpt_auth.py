import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import chatgpt_auth as auth


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(auth, "_request_json", lambda *a, **kw: pytest.fail("Unexpected HTTP request"))


@pytest.fixture
def credentials(tmp_path):
    path = tmp_path / "chatgpt.json"
    record = {
        "client_id": "oaiapp_example", "ext_agent_host_id": "urn:uuid:host",
        "subject": "user-one", "email": "user@example.com", "issuer": auth.ISSUER,
        "access_token": "private-access", "refresh_token": "private-refresh",
        "id_token": "private-id", "scopes": auth.SCOPES.split(),
        "expires_at": time.time() + 3600,
    }
    auth._write(path, record)
    return path, record


@pytest.fixture
def signing(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    public.update(kid="test-key", alg="RS256", use="sig")

    def request(url, **kwargs):
        if url == auth.DISCOVERY_URL:
            return {"issuer": auth.ISSUER, "jwks_uri": auth.ISSUER + "/.well-known/jwks.json"}
        if url == auth.ISSUER + "/.well-known/jwks.json":
            return {"keys": [public]}
        pytest.fail("Unexpected HTTP request")

    monkeypatch.setattr(auth, "_request_json", request)

    def token(**changes):
        claims = {"iss": auth.ISSUER, "aud": "oaiapp_example", "sub": "user-one",
                  "nonce": "nonce", "iat": int(time.time()), "exp": int(time.time()) + 3600}
        claims.update(changes)
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})

    return token


def test_authorization_uses_pkce_and_dynamic_registration():
    pending, url = auth._authorization({"ext_agent_host_id": "urn:uuid:host"}, "http://127.0.0.1:1234/auth/callback")
    query = parse_qs(urlsplit(url).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["agent_name_hint"] == ["Omega"]
    assert query["resource"] == [auth.RESOURCE]
    assert query["scope"] == [auth.SCOPES]
    expected = base64.urlsafe_b64encode(hashlib.sha256(pending["verifier"].encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == [expected]
    assert query["code_challenge_method"] == ["S256"]
    assert "code_verifier" not in query
    assert "id_token_hint" not in query
    second, _ = auth._authorization({"ext_agent_host_id": "urn:uuid:host"}, pending["redirect_uri"])
    assert all(second[key] != pending[key] for key in ("nonce", "state", "verifier"))


def test_reauthorization_keeps_client_id_and_host_without_exposing_id_token(credentials):
    _, record = credentials
    _, url = auth._authorization(record, "http://127.0.0.1:1234/auth/callback", consent=True)
    query = parse_qs(urlsplit(url).query)
    assert query["client_id"] == [record["client_id"]]
    assert query["ext_agent_host_id"] == [record["ext_agent_host_id"]]
    assert query["prompt"] == ["consent"]
    assert "agent_name_hint" not in query
    assert "private-id" not in url


def test_identity_verifies_signature_and_claims(signing):
    identity = auth._validate_identity(signing(), "oaiapp_example", "nonce")
    assert identity["sub"] == "user-one"


@pytest.mark.parametrize("changes", [
    {"iss": "https://attacker.example"}, {"aud": "wrong-client"},
    {"nonce": "wrong-nonce"}, {"exp": 1}, {"sub": ""},
])
def test_identity_rejects_invalid_claims(signing, changes):
    with pytest.raises(auth.ChatGPTAuthError, match="verification failed"):
        auth._validate_identity(signing(**changes), "oaiapp_example", "nonce")


def test_identity_rejects_forged_signature(signing):
    token = signing()
    pieces = token.split(".")
    pieces[2] = "A" * len(pieces[2])
    with pytest.raises(auth.ChatGPTAuthError, match="verification failed"):
        auth._validate_identity(".".join(pieces), "oaiapp_example", "nonce")


def test_discovery_rejects_untrusted_key_endpoint(monkeypatch):
    monkeypatch.setattr(auth, "_request_json", lambda *_a, **_kw: {
        "issuer": auth.ISSUER, "jwks_uri": "https://attacker.example/jwks"})
    with pytest.raises(auth.ChatGPTAuthError, match="unexpected endpoint"):
        auth._validate_identity("not-a-token", "oaiapp_example", "nonce")


def pending(client_id="dynamic_agent_client"):
    return {"state": "expected-state", "nonce": "nonce", "verifier": "verifier",
            "redirect_uri": "http://127.0.0.1:1234/auth/callback", "client_id": client_id}


def test_new_login_exchanges_issued_id_and_stores_protected_credentials(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    record = {"ext_agent_host_id": "urn:uuid:host"}
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return {"access_token": "access", "refresh_token": "refresh", "id_token": "identity",
                "expires_in": 3600, "token_type": "Bearer", "scope": auth.SCOPES}

    monkeypatch.setattr(auth, "_request_json", request)
    monkeypatch.setattr(auth, "_validate_identity", lambda *args: {"sub": "user-one", "email": "u@example.com"})
    result = auth._complete_login(path, record, pending(), {
        "state": "expected-state", "code": "authorization-code", "client_id": "oaiapp_example"})
    form = calls[0][1]["form"]
    assert form["client_id"] == "oaiapp_example"
    assert form["code_verifier"] == "verifier"
    assert form["redirect_uri"] == pending()["redirect_uri"]
    assert "client_secret" not in form
    assert result["subject"] == "user-one"
    assert result["scopes"] == auth.SCOPES.split()
    assert path.stat().st_mode & 0o777 == 0o600
    assert auth._read(path) == result


@pytest.mark.parametrize("callback", [
    {"state": "wrong", "code": "secret", "client_id": "oaiapp_example"},
    {"state": "неверно", "code": "secret", "client_id": "oaiapp_example"},
    {"state": "expected-state", "code": "secret"},
    {"state": "expected-state", "error": "access_denied"},
])
def test_bad_callback_never_exchanges_code(tmp_path, callback):
    with pytest.raises(auth.ChatGPTAuthError):
        auth._complete_login(tmp_path / "auth.json", {"ext_agent_host_id": "host"}, pending(), callback)


def test_reauthorization_rejects_changed_client_without_overwriting(credentials):
    path, record = credentials
    with pytest.raises(auth.ChatGPTAuthError, match="changed.*client ID"):
        auth._complete_login(path, record, pending(record["client_id"]), {
            "state": "expected-state", "code": "code", "client_id": "oaiapp_different"})
    assert auth._read(path) == record


def test_reauthorization_rejects_changed_identity_without_overwriting(credentials, monkeypatch):
    path, record = credentials
    monkeypatch.setattr(auth, "_request_json", lambda *a, **kw: {"id_token": "identity"})
    monkeypatch.setattr(auth, "_validate_identity", lambda *a: {"sub": "another-user"})
    with pytest.raises(auth.ChatGPTAuthError, match="account changed"):
        auth._complete_login(path, record, pending(record["client_id"]), {"state": "expected-state", "code": "code"})
    assert auth._read(path) == record


def test_identity_only_grant_is_saved_but_cannot_be_used(credentials, monkeypatch):
    path, record = credentials
    monkeypatch.setattr(auth, "_request_json", lambda *a, **kw: {
        "id_token": "identity", "access_token": "access", "refresh_token": "refresh",
        "expires_in": 3600, "token_type": "Bearer", "scope": "openid profile email"})
    monkeypatch.setattr(auth, "_validate_identity", lambda *a: {"sub": "user-one"})
    with pytest.raises(auth.ChatGPTAuthError, match="not authorized"):
        auth._complete_login(path, record, pending(record["client_id"]), {"state": "expected-state", "code": "code"})
    with pytest.raises(auth.ChatGPTAuthError, match="not authorized"):
        auth.get_access_token(path)


def test_valid_token_needs_no_request(credentials):
    path, _ = credentials
    assert auth.get_access_token(path) == "private-access"


def test_refresh_rotates_token_atomically_and_preserves_registration(credentials, monkeypatch):
    path, record = credentials
    record["expires_at"] = 0
    auth._write(path, record)
    calls = []

    def request(url, **kwargs):
        calls.append(kwargs["form"])
        return {"access_token": "new-access", "refresh_token": "new-refresh",
                "token_type": "Bearer", "expires_in": 3600}

    monkeypatch.setattr(auth, "_request_json", request)
    assert auth.get_access_token(path) == "new-access"
    assert calls == [{"grant_type": "refresh_token", "client_id": "oaiapp_example",
                      "refresh_token": "private-refresh", "resource": auth.RESOURCE}]
    updated = auth._read(path)
    assert updated["refresh_token"] == "new-refresh"
    assert updated["subject"] == record["subject"]
    assert updated["scopes"] == record["scopes"]
    assert updated["expires_at"] > time.time() + 3500


def test_concurrent_refresh_uses_rotating_token_once(credentials, monkeypatch):
    path, record = credentials
    record["expires_at"] = 0
    auth._write(path, record)
    calls = []
    start = threading.Barrier(2)

    def request(*a, **kw):
        calls.append(kw)
        time.sleep(0.1)
        return {"access_token": "new-access", "refresh_token": "new-refresh",
                "token_type": "Bearer", "expires_in": 3600}

    def obtain():
        start.wait(timeout=2)
        return auth.get_access_token(path)

    monkeypatch.setattr(auth, "_request_json", request)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(obtain) for _ in range(2)]
        assert [job.result(timeout=3) for job in jobs] == ["new-access", "new-access"]
    assert len(calls) == 1


@pytest.mark.parametrize("code", sorted(auth.TERMINAL_REFRESH_ERRORS))
def test_terminal_refresh_error_clears_tokens_but_keeps_registration(credentials, monkeypatch, code):
    path, record = credentials
    record["expires_at"] = 0
    auth._write(path, record)

    def fail(*a, **kw):
        raise auth._HTTPFailure(400, code)

    monkeypatch.setattr(auth, "_request_json", fail)
    with pytest.raises(auth.ChatGPTAuthError, match="expired or was revoked"):
        auth.get_access_token(path)
    assert auth._read(path) == auth._clear_tokens(record)


def test_transient_refresh_failure_keeps_credentials(credentials, monkeypatch):
    path, record = credentials
    record["expires_at"] = 0
    auth._write(path, record)

    def fail(*a, **kw):
        raise auth._HTTPFailure(503)

    monkeypatch.setattr(auth, "_request_json", fail)
    with pytest.raises(auth.ChatGPTAuthError, match="503"):
        auth.get_access_token(path)
    assert auth._read(path) == record


def test_logout_revokes_then_clears_tokens(credentials, monkeypatch):
    path, record = credentials
    calls = []
    monkeypatch.setattr(auth, "_discovery_endpoint", lambda _: auth.ISSUER + "/revoke")
    monkeypatch.setattr(auth, "_request_json", lambda *a, **kw: calls.append((a, kw)) or {})
    assert auth.logout(path) is True
    assert calls[0][1]["form"] == {"token": "private-refresh", "token_type_hint": "refresh_token", "client_id": "oaiapp_example"}
    assert auth._read(path) == auth._clear_tokens(record)


def test_logout_without_remote_confirmation_still_clears_local_tokens(credentials, monkeypatch):
    path, record = credentials
    monkeypatch.setattr(auth, "_discovery_endpoint", lambda _: auth.ISSUER + "/revoke")
    monkeypatch.setattr(auth.time, "sleep", lambda _: None)

    def fail(*a, **kw):
        raise auth.ChatGPTAuthError("Offline")

    monkeypatch.setattr(auth, "_request_json", fail)
    assert auth.logout(path) is False
    assert auth._read(path) == auth._clear_tokens(record)


def test_model_catalog_is_account_scoped_and_filters_hidden_models(credentials, monkeypatch):
    path, _ = credentials
    calls = []

    def request(*a, **kw):
        calls.append((a, kw))
        return {"models": [{"slug": "visible", "display_name": "Visible", "visibility": "list"},
                           {"slug": "hidden", "visibility": "hide"}]}

    monkeypatch.setattr(auth, "_request_json", request)
    assert [m["slug"] for m in auth.list_models(path)] == ["visible"]
    assert calls == [((auth.RESOURCE + "/models",), {"bearer": "private-access"})]


def test_insecure_credential_permissions_are_rejected(credentials):
    path, _ = credentials
    path.chmod(0o644)
    with pytest.raises(auth.ChatGPTAuthError, match="0600"):
        auth.get_access_token(path)


@pytest.mark.parametrize("changes", [{"expires_at": []}, {"expires_at": float("inf")}, {"scopes": "bad"}, {"access_token": []}])
def test_malformed_credential_record_is_rejected(credentials, changes):
    path, record = credentials
    auth._write(path, dict(record, **changes))
    with pytest.raises(auth.ChatGPTAuthError, match="invalid"):
        auth.get_access_token(path)


def test_idle_browser_preconnect_does_not_block_callback_listener(monkeypatch):
    monkeypatch.setattr(auth._CallbackServer, "connection_timeout", 0.1)

    class QuietHandler(auth.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

    with auth._CallbackServer(("127.0.0.1", 0), QuietHandler) as server:
        server.timeout = 1
        with socket.create_connection(server.server_address, timeout=1):
            with ThreadPoolExecutor(max_workers=1) as executor:
                # A connected browser socket sends no HTTP bytes. Processing it
                # must return, letting login enforce its own overall deadline.
                executor.submit(server.handle_request).result(timeout=2)


def test_status_never_prints_tokens(credentials, capsys):
    path, _ = credentials
    assert auth.main(["--auth-file", str(path), "status"]) == 0
    output = capsys.readouterr()
    assert "user@example.com" in output.out
    assert "private" not in output.out + output.err


def test_environment_selects_auth_file(credentials, monkeypatch):
    path, _ = credentials
    monkeypatch.setenv("OMEGA_CHATGPT_AUTH_FILE", str(path))
    assert auth.get_access_token() == "private-access"


def test_http_errors_never_echo_token_bodies(monkeypatch):
    # Restore the real transport after the autouse network guard, then replace
    # its opener with a local fake which raises a response containing a secret.
    import importlib.util
    spec = importlib.util.spec_from_file_location("auth_http_test", auth.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Opener:
        def open(self, *a, **kw):
            body = json.dumps({"error": "invalid_grant", "error_description": "private-refresh"}).encode()
            raise HTTPError(auth.TOKEN_URL, 400, "bad", {}, io.BytesIO(body))

    monkeypatch.setattr(module, "build_opener", lambda *_: Opener())
    with pytest.raises(module.ChatGPTAuthError) as error:
        module._request_json(auth.TOKEN_URL, form={"refresh_token": "private-refresh"})
    assert "invalid_grant" in str(error.value)
    assert "private-refresh" not in str(error.value)
