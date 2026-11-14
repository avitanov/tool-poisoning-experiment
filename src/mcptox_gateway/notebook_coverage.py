"""Summarize recorded logical model-call results for the notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcptox_gateway.pipelines.common import read_json


def role_call_coverage(actor_dir: Path, evaluation_dir: Path, kind: str) -> list[dict[str, Any]]:
    """Count planned cases and usable recorded results by role.

    One logical request per role and case is counted; HTTP-level retries are not
    counted separately. Missing traces are not assumed to be undispatched.
    Valid uncertain decisions are recorded separately, not treated as failures.
    This does not assess semantic correctness or parse actor response text.
    """
    if kind not in {"attack", "control"}:
        raise ValueError("Case kind must be attack or control")
    manifest_path = actor_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    cases = [case for case in read_json(manifest_path)["cases"] if case["kind"] == kind]
    roles = ("actor", "B", "C", "judge")
    counts = {
        role: {"planned": len(cases), "recorded": 0, "usable": 0, "invalid": 0, "uncertain": 0}
        for role in roles
    }
    for case in cases:
        source_id = case["source_id"]
        actor_path = actor_dir / "traces" / (source_id + ".json")
        if actor_path.is_file():
            actor = read_json(actor_path)
            if (actor.get("request_ids") or {}).get("actor"):
                counts["actor"]["recorded"] += 1
                if actor.get("proposal_status") == "not_parsed":
                    counts["actor"]["usable"] += 1
                else:
                    counts["actor"]["invalid"] += 1
        evaluation_path = evaluation_dir / "traces" / (source_id + ".json")
        if not evaluation_path.is_file():
            continue
        trace = read_json(evaluation_path)
        ids = trace.get("request_ids") or {}
        reviews = trace.get("review_verdicts") or {}
        for role in ("B", "C", "judge"):
            if not ids.get(role):
                continue
            counts[role]["recorded"] += 1
            verdict = trace.get("judgment") if role == "judge" else reviews.get(role)
            allowed = (
                {
                    "attacker_goal",
                    "poisoned_tool_selected",
                    "authorized_task",
                    "neither",
                    "uncertain",
                }
                if role == "judge"
                else {"allow", "block", "uncertain"}
            )
            decision = (
                verdict.get("outcome" if role == "judge" else "decision")
                if isinstance(verdict, dict)
                else None
            )
            if (
                isinstance(verdict, dict)
                and decision in allowed
                and isinstance(verdict.get("reason"), str)
            ):
                counts[role]["usable"] += 1
                if decision == "uncertain":
                    counts[role]["uncertain"] += 1
            else:
                counts[role]["invalid"] += 1
    return [
        {"role": role, **data, "not_recorded": data["planned"] - data["recorded"]}
        for role, data in counts.items()
    ]
