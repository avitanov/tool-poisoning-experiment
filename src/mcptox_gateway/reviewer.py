"""Construct and validate the matched B/C reviewer requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcptox_gateway.evaluation import assert_no_evaluation_labels
from mcptox_gateway.storage import parse_exact_json

REVIEWER_PROMPT_VERSION = "semantics-availability-v3"


@dataclass(frozen=True)
class ReviewerVerdict:
    """A locally validated reviewer decision."""

    decision: str
    reason: str


def _evidence_message(report: dict[str, Any]) -> str:
    """Serialize attacker-controlled descriptions as literal JSON data."""

    return json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _selected_semantics_status(
    clean_semantics: list[dict[str, Any]], proposal: dict[str, Any]
) -> dict[str, str | None]:
    """Describe supplied semantic context, without making an authorization decision."""
    matches = [item for item in clean_semantics if item.get("tool_name") == proposal.get("tool")]
    if not matches:
        return {"status": "unavailable", "reason": "not_provided"}
    if len(matches) != 1:
        return {"status": "unavailable", "reason": "ambiguous_tool_identity"}
    description = matches[0].get("description")
    if not isinstance(description, str) or not description.strip():
        return {"status": "unavailable", "reason": "description_missing_or_empty"}
    return {"status": "available", "reason": None}


def build_reviewer_messages(
    arm: str,
    *,
    user_request: str,
    clean_semantics: list[dict[str, Any]],
    proposal: dict[str, Any],
    integrity_report: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build stateless B/C messages while isolating C-only evidence.

    Descriptions never enter a trusted instruction message. B has no integrity
    package at all; C receives one distinct literal-data evidence message.
    """

    if arm not in {"B", "C"}:
        raise ValueError("Reviewer arm must be B or C")
    common: dict[str, Any] = {
        "user_request": user_request,
        "clean_tool_semantics": clean_semantics,
        "selected_tool_semantics": _selected_semantics_status(clean_semantics, proposal),
        "proposal": proposal,
    }
    assert_no_evaluation_labels(common)
    base_path, evidence_path = prompt_paths()
    base = base_path.read_text(encoding="utf-8").strip()
    messages = [
        {"role": "system", "content": base},
        {"role": "user", "content": json.dumps(common, ensure_ascii=False, sort_keys=True)},
    ]
    if arm == "C":
        if integrity_report is None:
            raise ValueError("C requires the complete integrity report")
        messages.append(
            {
                "role": "user",
                "content": (
                    evidence_path.read_text(encoding="utf-8").strip()
                    + "\n"
                    + _evidence_message(integrity_report)
                ),
            }
        )
    return messages


def parse_reviewer_verdict(text: str) -> ReviewerVerdict:
    """Validate the common strict reviewer output schema."""

    try:
        value = parse_exact_json(text)
    except (TypeError, ValueError) as error:
        raise ValueError("Reviewer verdict is not strict JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"decision", "reason"}
        or value.get("decision") not in {"allow", "block", "uncertain"}
        or not isinstance(value.get("reason"), str)
        or not value["reason"].strip()
    ):
        raise ValueError("Reviewer verdict has an invalid schema")
    return ReviewerVerdict(value["decision"], value["reason"])


def prompt_paths() -> tuple[Path, Path]:
    """Return the version-controlled reviewer prompt paths for audit displays."""

    root = Path(__file__).resolve().parents[2]
    return root / "prompts" / "reviewer_base.txt", root / "prompts" / "reviewer_hash_evidence.txt"
