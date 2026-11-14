"""Construct a paired actor/reviewer/judge case without executing tools."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from mcptox_gateway.actor import build_actor_messages
from mcptox_gateway.dataset import actor_visible_inventory
from mcptox_gateway.judge import (
    build_judge_messages,
    judge_request_record,
    judge_response_format,
    parse_judge_verdict,
)
from mcptox_gateway.metadata import compare_descriptions
from mcptox_gateway.normalization import PARSER_VERSION, normalize_actor_output
from mcptox_gateway.provider import extract_content
from mcptox_gateway.reviewer import (
    REVIEWER_PROMPT_VERSION,
    build_reviewer_messages,
    parse_reviewer_verdict,
)
from mcptox_gateway.runner import judged_branches
from mcptox_gateway.storage import request_identity


def _complete_content(raw: bytes) -> str:
    """Reject truncated outputs before treating a verdict as a measurement."""
    try:
        response = json.loads(raw)
        if response["choices"][0].get("finish_reason") == "length":
            raise ValueError("Model output was truncated")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid provider response") from exc
    return extract_content(raw)


def _payload(
    route: dict[str, Any], messages: list[dict[str, str]], *, judge: bool = False
) -> dict[str, Any]:
    """Build an exact, fixed-provider transport body without provider tools."""
    if set(route["generation"]) - {"max_tokens", "temperature", "reasoning"}:
        raise ValueError("Unsupported generation parameters")
    body = {
        "model": route["model"],
        "messages": messages,
        **route["generation"],
        "stream": False,
        "usage": {"include": True},
        "provider": {
            "only": [route["provider_slug"]],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": route["max_price"],
        },
    }
    if judge:
        body["response_format"] = judge_response_format()
    return body


def run_judged_case(
    *,
    case: dict[str, Any],
    clean_semantics: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    registry_version: str,
    clean_semantics_version: str,
    attack_goal: str | None,
    routes: dict[str, dict[str, Any]],
    review_order: tuple[str, str],
    send: Callable[[str, dict[str, Any]], bytes],
) -> dict[str, Any]:
    """Build one proposal and independent reviews, then judge it once.

    The injected sender must enforce frozen cost bounds, reserve all four roles
    before actor dispatch, persist raw responses, and replay completed calls.
    This function cannot itself authorize a research run or make network calls.
    The caller must supply independently reviewed case context and references.
    """
    if review_order not in (("B", "C"), ("C", "B")):
        raise ValueError("Review order must contain B and C exactly once")
    if set(routes) != {"actor", "reviewer", "judge"}:
        raise ValueError("Explicit actor, reviewer and judge routes are required")
    case_id = case["case_id"]
    versions = {"experiment_mode": "judged-v1", "clean_semantics": clean_semantics_version}
    actor_payload = _payload(routes["actor"], build_actor_messages(case["system"], case["query"]))
    actor_record = {
        "role": "actor",
        "case_id": case_id,
        "payload": actor_payload,
        "provider": "openrouter",
        "model": routes["actor"]["model"],
        "generation": routes["actor"]["generation"],
        "repetition_id": "0",
        "prompt_version": "light-warning-v1",
        "reference_versions": {"experiment_mode": "judged-v1"},
    }
    actor_raw = send("actor", actor_record)
    response = json.loads(actor_raw)
    if len(extract_content(actor_raw).encode("utf-8")) > 8192:
        raise ValueError("Actor response exceeds the frozen proposal byte bound")
    proposal = normalize_actor_output(
        extract_content(actor_raw),
        truncated=response["choices"][0].get("finish_reason") == "length",
    )
    ids = {"actor": request_identity(actor_record)}
    if proposal.action is None:
        return {
            "case_id": case_id,
            "proposal_status": proposal.status,
            "request_ids": ids,
            "branches": judged_branches(
                proposal, judge_outcome=None, verdict_b=None, verdict_c=None
            ),
        }
    inventory = actor_visible_inventory(case, proposal.action["tool"])
    report = compare_descriptions(inventory, registry, registry_version)
    visible_report = {
        **report,
        "entries": [
            {
                k: v
                for k, v in entry.items()
                if k not in {"approved_sha256", "observed_sha256", "observed_source_span"}
            }
            for entry in report["entries"]
        ],
    }
    verdicts: dict[str, Any] = {}
    for arm in review_order:
        messages = build_reviewer_messages(
            arm,
            user_request=case["query"],
            clean_semantics=clean_semantics,
            proposal=proposal.action,
            integrity_report=visible_report if arm == "C" else None,
        )
        record = {
            "role": "reviewer",
            "arm": arm,
            "case_id": case_id,
            "payload": _payload(routes["reviewer"], messages),
            "provider": "openrouter",
            "model": routes["reviewer"]["model"],
            "generation": routes["reviewer"]["generation"],
            "repetition_id": "0",
            "prompt_version": REVIEWER_PROMPT_VERSION,
            "reference_versions": versions,
            "actor_response_hash": hashlib.sha256(actor_raw).hexdigest(),
            "proposal": proposal.action,
            "normalization_version": PARSER_VERSION,
            "clean_semantics": clean_semantics,
            "clean_semantics_version": clean_semantics_version,
        }
        if arm == "C":
            record.update(
                integrity_report=visible_report,
                registry_version=registry_version,
                hash_rules_version="1",
                extraction_version="1",
                report_schema_version="1",
                serialization_version="1",
            )
        ids[arm] = request_identity(record)
        raw = send(arm, record)
        try:
            verdicts[arm] = parse_reviewer_verdict(_complete_content(raw))
        except ValueError:
            verdicts[arm] = None
    judge_route = routes["judge"]
    messages = build_judge_messages(
        request=case["query"],
        clean_semantics=clean_semantics,
        proposal=proposal.action,
        attack_goal=attack_goal,
    )
    judge_record = judge_request_record(
        payload=_payload(judge_route, messages, judge=True),
        provider="openrouter",
        model=judge_route["model"],
        generation=judge_route["generation"],
        case_id=case_id,
        repetition_id="0",
        actor_response=actor_raw,
        proposal=proposal.action,
        reference_versions=versions,
    )
    ids["judge"] = request_identity(judge_record)
    raw = send("judge", judge_record)
    try:
        judgment = parse_judge_verdict(_complete_content(raw))
    except ValueError:
        judgment = None
    return {
        "case_id": case_id,
        "proposal_status": proposal.status,
        "proposal": proposal.action,
        "request_ids": ids,
        "integrity_report": report,
        "judgment": judgment,
        "review_verdicts": {
            k: None if v is None else {"decision": v.decision, "reason": v.reason}
            for k, v in verdicts.items()
        },
        "branches": judged_branches(
            proposal,
            judge_outcome=None if judgment is None else judgment["outcome"],
            verdict_b=verdicts.get("B"),
            verdict_c=verdicts.get("C"),
        ),
    }
