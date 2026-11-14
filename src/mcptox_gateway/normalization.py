"""Strict normalization of the benchmark's text-JSON actor interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcptox_gateway.storage import parse_exact_json


PARSER_VERSION = "1"


@dataclass(frozen=True)
class NormalizedProposal:
    """One immutable interpretation of an actor response.

    ``action`` is populated only for a structurally valid, single benchmark
    object. Raw text is intentionally retained by the call store, not copied
    into this evaluation record.
    """

    status: str
    action: dict[str, Any] | None
    detail: str | None = None


def normalize_actor_output(text: str, *, truncated: bool = False) -> NormalizedProposal:
    """Parse one exact JSON object without selecting from malformed batches.

    Args:
        text: The exact text returned by the actor provider.
        truncated: Whether provider metadata reports a truncated generation.

    Returns:
        A valid action, no-action/refusal, or an explicit non-executable state.
    """

    if truncated:
        return NormalizedProposal("truncation", None, "provider reported truncation")
    if not isinstance(text, str) or not text:
        return NormalizedProposal("invalid_format", None, "empty or non-text response")
    # The frozen primary protocol deliberately accepts neither fences nor prose.
    try:
        parsed = parse_exact_json(text)
    except (TypeError, ValueError):
        return NormalizedProposal("invalid_format", None, "response is not one strict JSON object")
    if not isinstance(parsed, dict):
        return NormalizedProposal("invalid_format", None, "response must be an object")
    if set(parsed) == {"refusal"} and isinstance(parsed["refusal"], str):
        return NormalizedProposal("refusal", None, parsed["refusal"])
    if set(parsed) == {"no_action"} and parsed["no_action"] is True:
        return NormalizedProposal("no_action", None)
    if set(parsed) != {"tool", "arguments"}:
        return NormalizedProposal("invalid_format", None, "expected exactly tool and arguments")
    if not isinstance(parsed["tool"], str) or not parsed["tool"]:
        return NormalizedProposal("invalid_format", None, "tool must be a nonempty string")
    if not isinstance(parsed["arguments"], dict):
        return NormalizedProposal("invalid_format", None, "arguments must be an object")
    return NormalizedProposal("valid_action", {"tool": parsed["tool"], "arguments": parsed["arguments"]})
