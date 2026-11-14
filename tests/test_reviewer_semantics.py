"""Regression checks for missing reviewer context, distinct from model errors."""

import copy
import json
from typing import Any

import pytest

from mcptox_gateway.normalization import normalize_actor_output
from mcptox_gateway.reviewer import build_reviewer_messages, parse_reviewer_verdict
from mcptox_gateway.runner import run_branches


@pytest.mark.parametrize(
    ("semantics", "status", "reason"),
    [
        ([], "unavailable", "not_provided"),
        ([{"tool_name": "other", "description": "Other tool"}], "unavailable", "not_provided"),
        ([{"tool_name": "write_file"}], "unavailable", "description_missing_or_empty"),
        (
            [{"tool_name": "write_file", "description": None}],
            "unavailable",
            "description_missing_or_empty",
        ),
        (
            [{"tool_name": "write_file", "description": "  "}],
            "unavailable",
            "description_missing_or_empty",
        ),
        (
            [{"tool_name": "write_file", "description": "Write text"}] * 2,
            "unavailable",
            "ambiguous_tool_identity",
        ),
        ([{"tool_name": "write_file", "description": " Write text\n"}], "available", None),
    ],
)
def test_shared_context_marks_missing_semantics_without_integrity_leakage(
    semantics: list[dict[str, Any]], status: str, reason: str | None
) -> None:
    """B/C see the same availability; descriptions and inputs stay untouched."""
    proposal = {"tool": "write_file", "arguments": {"path": "/tmp/a", "content": "Hello"}}
    original = copy.deepcopy(semantics)
    shared = dict(
        user_request="Write Hello to /tmp/a", clean_semantics=semantics, proposal=proposal
    )
    b = build_reviewer_messages("B", **shared)
    c = build_reviewer_messages("C", **shared, integrity_report={"entries": []})
    assert b == c[:2]
    context = json.loads(b[1]["content"])
    assert context["selected_tool_semantics"] == {"status": status, "reason": reason}
    assert context["clean_tool_semantics"] == original == semantics
    assert set(context) == {
        "user_request",
        "clean_tool_semantics",
        "selected_tool_semantics",
        "proposal",
    }


def test_uncertain_and_missing_response_are_distinct_and_never_execute() -> None:
    """Uncertainty is a model verdict; a missing response remains an error."""
    proposal = normalize_actor_output(
        '{"tool":"write_file","arguments":{"path":"/tmp/a","content":"Hello"}}'
    )
    uncertain = parse_reviewer_verdict('{"decision":"uncertain","reason":"Missing semantics"}')
    branches = run_branches(
        proposal, initial_state={"files": {}}, verdict_b=uncertain, verdict_c=None
    )
    assert branches["A"].effect.status == "executed"
    assert branches["B"].reviewer_status == "uncertain"
    assert branches["C"].reviewer_status == "review_error"
    assert branches["B"].effect is branches["C"].effect is None
    # Preserve a model's real block, even when its reasoning is mistaken.
    assert (
        parse_reviewer_verdict('{"decision":"block","reason":"No reference"}').decision == "block"
    )


@pytest.mark.parametrize("text", ["null", "", '{"decision":null,"reason":"missing"}'])
def test_invalid_verdict_is_not_rewritten_as_uncertain(text: str) -> None:
    with pytest.raises(ValueError):
        parse_reviewer_verdict(text)
