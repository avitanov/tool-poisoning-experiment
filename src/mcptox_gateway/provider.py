"""A serial, cache-first chat-completions transport boundary."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mcptox_gateway.budget import Budget
from mcptox_gateway.storage import CallStore, canonical_json


@dataclass(frozen=True)
class ProviderConfig:
    """Qualified transport configuration without a credential value."""

    provider: str
    model: str
    api_key_env: str
    timeout_seconds: float


class OpenRouterHTTPError(ValueError):
    """Preserve rate-limit evidence without persisting credentials or response headers."""

    def __init__(self, error: HTTPError) -> None:
        super().__init__(f"OpenRouter HTTP {error.code}; no automatic retry")
        self.details = {
            "http_status": error.code,
            "retry_after": error.headers.get("Retry-After"),
            "rate_limit": error.headers.get("X-RateLimit-Limit"),
            "rate_remaining": error.headers.get("X-RateLimit-Remaining"),
            "rate_reset": error.headers.get("X-RateLimit-Reset"),
        }


class ChatCompletionsAdapter:
    """Dispatch cached, non-streaming chat requests through an injected sender.

    The sender is deliberately injected so tests do not import a provider SDK or
    make network calls. It receives the body and a credential only after every
    cache, reservation, and claim check succeeds.
    """

    def __init__(
        self,
        store: CallStore,
        budget: Budget,
        sender: Callable[[dict[str, Any], str, float], dict[str, Any]],
    ) -> None:
        self.store = store
        self.budget = budget
        self.sender = sender

    def collect(
        self,
        record: dict[str, Any],
        reservation_id: str,
        config: ProviderConfig,
    ) -> tuple[str, bytes, bool]:
        """Load a completed response or perform one guarded transport attempt."""

        body = canonical_json(record["payload"]).encode("utf-8")
        request_id = self.store.plan(record, body)
        try:
            return request_id, self.store.response(request_id), True
        except ValueError:
            pass
        if not config.provider or not config.model or config.timeout_seconds <= 0:
            raise ValueError("Provider configuration is not qualified")
        key = os.environ.get(config.api_key_env)
        if not key:
            raise ValueError("Credential environment variable is missing")
        attempt = self.store.claim(request_id, reservation_id)
        self.store.dispatched(request_id, attempt)
        try:
            response = self.sender(record["payload"], key, config.timeout_seconds)
        except Exception as error:
            self.budget.unresolved(
                reservation_id,
                {
                    "transport_error": type(error).__name__,
                    "transport_details": getattr(error, "details", None),
                },
            )
            raise
        raw = canonical_json(response).encode("utf-8")
        metadata = {
            "returned_model": response.get("model"),
            "returned_provider": response.get("provider"),
            "requested_gateway": config.provider,
            "provider_request_id": response.get("id"),
            "usage": response.get("usage"),
        }
        self.store.save_response(request_id, attempt, raw, metadata)
        # Billing reconciliation remains explicit: unknown billing cannot be guessed.
        return request_id, raw, False


def extract_content(response: bytes) -> str:
    """Extract one chat-completions text response without repairing malformed data."""

    try:
        payload = json.loads(response.decode("utf-8"))
        choices = payload["choices"]
        content = choices[0]["message"]["content"]
    except (UnicodeDecodeError, TypeError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise ValueError("Provider response lacks one text completion") from error
    if not isinstance(content, str):
        raise ValueError("Provider completion content is not text")
    return content


def openrouter_sender(
    payload: dict[str, Any], api_key: str, timeout_seconds: float
) -> dict[str, Any]:
    """Send one non-streaming OpenRouter chat completion.

    This sender is intentionally narrow: callers must supply a fixed ``:free``
    model and the adapter has already persisted the exact request and claimed a
    ledger reservation. Authorization headers are never returned or persisted.

    Raises:
        ValueError: If OpenRouter returns an invalid or non-success response.
    """

    if not isinstance(payload.get("model"), str) or not payload["model"].endswith(":free"):
        raise ValueError("This sender requires an explicit free model")
    routing = payload.get("provider", {})
    if (
        len(routing.get("only", [])) != 1
        or routing.get("allow_fallbacks") is not False
        or routing.get("max_price") != {"prompt": 0, "completion": 0, "request": 0}
        or any(key in payload for key in ("tools", "plugins", "models", "route"))
    ):
        raise ValueError("Free requests require one pinned provider, zero prices, and no tools")
    return openrouter_bounded_sender(payload, api_key, timeout_seconds)


def openrouter_bounded_sender(
    payload: dict[str, Any], api_key: str, timeout_seconds: float
) -> dict[str, Any]:
    """Send a fixed-provider, price-capped request after ledger reservation.

    Supports paid qualification as well as free calls. Provider-native tools and
    plugins remain disabled; this function never interprets a proposed action.
    """
    routing = payload.get("provider", {})
    caps = routing.get("max_price", {})
    try:
        prices = [Decimal(str(caps[k])) for k in ("prompt", "completion", "request")]
    except (KeyError, InvalidOperation, TypeError) as error:
        raise ValueError("Explicit finite price caps are required") from error
    if (
        not isinstance(payload.get("model"), str)
        or not payload["model"]
        or len(routing.get("only", [])) != 1
        or not isinstance(routing["only"][0], str)
        or not routing["only"][0]
        or routing.get("allow_fallbacks") is not False
        or routing.get("require_parameters") is not True
        or any(not price.is_finite() or price < 0 for price in prices)
        or prices[2] != 0
        or payload.get("stream") is not False
        or type(payload.get("max_tokens")) is not int
        or payload["max_tokens"] <= 0
        or any(key in payload for key in ("tools", "plugins", "models", "route"))
    ):
        raise ValueError("Require bounded output, price caps, one provider and no tools/fallback")
    body = canonical_json(payload).encode("utf-8")
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except HTTPError as error:
        raise OpenRouterHTTPError(error) from error
    except (URLError, TimeoutError) as error:
        raise ValueError(f"OpenRouter transport failed: {type(error).__name__}") from error
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OpenRouter returned non-JSON content") from error
    if not isinstance(parsed, dict):
        raise ValueError("OpenRouter returned a non-object response")
    return parsed
