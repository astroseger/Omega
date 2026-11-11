"""Local ChatGPT plan access using the official Sign in with ChatGPT flow.

Protocol: https://developers.openai.com/siwc/token-sharing-open-source
Credentials belong to this Omega instance, independently of API-key providers.
"""

import json
import re
import time

import openai
import chatgpt_auth
import lib_llm_ext as llm
from config import config_get_by_key
from omega_openai import OpenAIProviderImpl
from providers import LLMProvider, LLMResponse, LLMToolCall, registerLLMProvider


USAGE_URL = "https://chatgpt.com/settings/usage"
NAMESPACE = "omega"


def _diagnostic(value):
    """Keep error identifiers, never arbitrary server text or credentials."""
    value = str(value or "unknown")
    return value if re.fullmatch(r"[a-zA-Z0-9_.-]{1,120}", value) else "unknown"


class ChatGPTProvider(LLMProvider):
    def __init__(self):
        self.delegate = None

    def start(self):
        if config_get_by_key("GATEWAY_URL"):
            raise RuntimeError(
                "ChatGPT currently supports standalone local Omega only; "
                "the Docker API-key gateway does not support ChatGPT OAuth."
            )
        models = chatgpt_auth.list_models()
        model = config_get_by_key("model") or config_get_by_key("chatgpt_model", "")
        slugs = [m["slug"] for m in models]
        if not slugs:
            raise chatgpt_auth.ChatGPTAuthError("No ChatGPT models are available for this account.")
        if model and model not in slugs:
            raise chatgpt_auth.ChatGPTAuthError(
                "The selected model is not listed for this ChatGPT account. "
                "Run scripts/chatgpt-auth models and select a listed slug."
            )
        self.delegate = ChatGPTProviderImpl(model or slugs[0])
        llm.logger.info("ChatGPT provider model=%s; usage counts against your ChatGPT plan. "
                        "Manage limits at %s. maxOutputToken is not supported on this route.",
                        self.delegate._model_name, USAGE_URL)

    def chat(self, request):
        return self.delegate.chat(request)

    def stop(self):
        if self.delegate is not None:
            self.delegate.stop()
            self.delegate = None


class ChatGPTProviderImpl(OpenAIProviderImpl):
    def __init__(self, model):
        super().__init__("ChatGPT", "", model, "https://api.openai.com/v1")
        self._paused_error = None
        self._retry_at = 0
        self._transient_failures = 0
        self._last_error = None

    def _ensure_client(self):
        if config_get_by_key("GATEWAY_URL"):
            raise chatgpt_auth.ChatGPTAuthError(
                "ChatGPT requires standalone local Omega; the API-key gateway is unsupported."
            )
        # Re-read on each request so refreshes and local logout take effect.
        token = chatgpt_auth.get_access_token()
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=token, base_url=self._base_url,
                max_retries=0, timeout=llm.CHAT_REQUEST_TIMEOUT_SECONDS,
            )
        else:
            self._client.api_key = token

    def convert_message(self, message):
        result = super().convert_message(message)
        for item in result:
            if item.get("role") == "system":
                item["role"] = "developer"
        return result

    def convert_tool_call(self, call):
        return {**super().convert_tool_call(call), "namespace": NAMESPACE}

    def convert_request(self, request):
        # Build the supported subset explicitly: the API-key implementation
        # includes output limits and cache retention that this route rejects.
        result = {
            "model": self._model_name,
            "input": [item for msg in request.messages for item in self.convert_message(msg)],
            "store": False,
            "stream": True,
        }
        if request.reasoning_mode:
            result["reasoning"] = {"effort": request.reasoning_mode}
        if request.tools:
            result["tools"] = [{
                "type": "namespace", "name": NAMESPACE,
                "description": "Omega agent tools.",
                "tools": [self.convert_tool(tool) for tool in request.tools],
            }]
            result["tool_choice"] = "required"
        return result

    def convert_response(self, raw):
        response = LLMResponse()
        for item in raw.output or []:
            if item.type != "function_call":
                continue
            if getattr(item, "namespace", None) not in (None, NAMESPACE):
                return LLMResponse().with_error("ChatGPT returned an unknown tool namespace.")
            # Response item IDs (fc_...) are not function call IDs (call_...).
            call_id = getattr(item, "call_id", None)
            if not isinstance(call_id, str) or not call_id:
                return LLMResponse().with_error("ChatGPT returned a tool call without a call_id.")
            try:
                arguments = json.loads(item.arguments)
            except (ValueError, TypeError):
                return LLMResponse().with_error("ChatGPT returned invalid tool arguments.")
            if not isinstance(arguments, dict):
                return LLMResponse().with_error("ChatGPT tool arguments must be a JSON object.")
            response.add_tool_call(LLMToolCall().with_name(item.name)
                                   .with_id(call_id).with_arguments(arguments))
        if not response.calls:
            return response.with_error("ChatGPT completed without an Omega tool call.")
        return response

    def _failure(self, code, status=None, request_id=None):
        code = _diagnostic(code)
        error = f"ChatGPT request failed ({code})"
        if status is not None:
            error += f"; HTTP {_diagnostic(status)}"
        if request_id:
            error += f"; request ID {_diagnostic(request_id)}"
        if code == "subscription_sharing_usage_limit_exceeded":
            error += f". Requests paused. Check {USAGE_URL}, then restart Omega when usage is available."
            self._paused_error = error
        elif code == "subscription_sharing_user_not_eligible":
            error += ". ChatGPT plan usage is unavailable for this account or workspace."
            self._paused_error = error
        elif status == 401:
            error += ". Check the account with scripts/chatgpt-auth status; sign in again if disconnected."
            self._paused_error = error + " Restart Omega after reconnecting."
        elif code == "subscription_sharing_unsupported_capability":
            error += ". The selected model or request capability is unsupported on the subscription route."
            self._paused_error = error + " Restart Omega after correcting the configuration."
        elif status in (400, 403) or code in (
            "subscription_sharing_route_not_supported", "subscription_sharing_invalid_user",
            "chatpass_v2_scope_not_authorized", "chatpass_v2_invalid_authorization_context",
        ):
            self._paused_error = error + " Requests paused; check the connection and configuration before restarting Omega."
        elif status in (429, 503) or code in (
            "subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable",
        ):
            # Omega's agent loop calls chat again after errors. Apply bounded
            # backoff across those calls, not just within one SDK invocation.
            self._transient_failures = min(self._transient_failures + 1, 5)
            delay = min(60, 5 * 2 ** (self._transient_failures - 1))
            self._retry_at = time.monotonic() + delay
            error += f". ChatGPT requests will wait {delay} seconds before retrying."
        error = self._paused_error or error
        self._last_error = error
        return LLMResponse().with_error(error)

    def chat(self, request):
        if self._paused_error:
            return LLMResponse().with_error(self._paused_error)
        if time.monotonic() < self._retry_at:
            return LLMResponse().with_error(self._last_error)
        self._start_of_turn(request)
        try:
            self._ensure_client()
            # Do not retry partially consumed streams: they may have consumed
            # subscription usage. Never execute tools from intermediate events.
            with self._client.responses.create(**self.convert_request(request)) as stream:
                for event in stream:
                    if event.type == "response.completed":
                        self._answered()
                        self._transient_failures = 0
                        self._retry_at = 0
                        llm._log_responses_completion(self._name, self._model_name, event.response)
                        return self.convert_response(event.response)
                    if event.type == "response.failed":
                        return self._failure(getattr(getattr(event.response, "error", None), "code", None))
                    if event.type == "response.incomplete":
                        reason = getattr(getattr(event.response, "incomplete_details", None), "reason", None)
                        return LLMResponse().with_error(
                            f"ChatGPT response incomplete ({_diagnostic(reason)}); no tools were executed.")
                    if event.type == "error":
                        return self._failure(getattr(event, "code", None))
            return LLMResponse().with_error("ChatGPT stream ended before completion; no tools were executed.")
        except chatgpt_auth.ChatGPTAuthError as error:
            return LLMResponse().with_error(str(error))
        except Exception as error:
            if llm._is_timeout_error(error):
                return self._timeout_reply()
            # The exception string/body can contain request data. Return only
            # structured identifiers, without logging bearer tokens or bodies.
            return self._failure(getattr(error, "code", type(error).__name__),
                                 getattr(error, "status_code", None),
                                 getattr(error, "request_id", None))

    def stop(self):
        if self._client is not None:
            self._client.close()
            self._client = None


def loadOmegaPlugin():
    registerLLMProvider("ChatGPT", ChatGPTProvider())
