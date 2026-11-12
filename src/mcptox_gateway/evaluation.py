"""Evaluation-only annotation access, isolated from model request construction."""

from __future__ import annotations

from typing import Any


EVALUATION_SCHEMA_VERSION = 1


def assert_no_evaluation_labels(payload: object) -> None:
    """Reject an outbound model payload that contains answer-key-only fields.

    Args:
        payload: JSON-compatible prospective model message payload.

    Raises:
        ValueError: If a recursively nested evaluation-only key is found.
    """

    forbidden = {
        "authorized_alternatives",
        "success_criteria",
        "attacker_goal",
        "adjudication_status",
    }
    if isinstance(payload, dict):
        overlap = forbidden & set(payload)
        if overlap:
            raise ValueError(f"Model payload contains evaluation-only keys: {sorted(overlap)}")
        for value in payload.values():
            assert_no_evaluation_labels(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_no_evaluation_labels(value)


def annotation_stub(item_id: str, kind: str, request: str) -> dict[str, Any]:
    """Create a review-required answer-key record without inferring permissions.

    The explicit unsupported status prevents these fields from being mistaken for
    completed human annotations or model-visible authorization instructions.
    """

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "item_id": item_id,
        "kind": kind,
        "legitimate_request": request,
        "authorized_alternatives": [],
        "argument_boundaries": [],
        "synthetic_state": None,
        "success_criteria": [],
        "attacker_goal": None,
        "adjudication_status": "unsupported_pending_independent_review",
    }
