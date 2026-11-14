"""Evaluation-only annotation access, isolated from model request construction."""

from __future__ import annotations


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
