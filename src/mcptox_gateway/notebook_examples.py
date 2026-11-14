"""Read saved judge and reviewer examples for the explanatory notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcptox_gateway.pipelines.common import read_json


def _attack_examples(actor_dir: Path, evaluation_dir: Path) -> list[dict[str, Any]]:
    """Load saved attack evidence without interpreting the actor text."""
    actor_path = actor_dir / "manifest.json"
    traces = evaluation_dir / "traces"
    if not actor_path.is_file() or not traces.is_dir():
        return []
    cases = {item["source_id"]: item for item in read_json(actor_path)["cases"]}
    found = []
    for path in sorted(traces.glob("*.json")):
        trace = read_json(path)
        if trace.get("kind") != "attack":
            continue
        verdicts = trace.get("review_verdicts") or {}
        item = cases[trace["source_id"]]
        found.append(
            {
                "source_id": trace["source_id"],
                "request": item["case"]["query"],
                "source_risk": item["goal"],
                "poisoned_tool": item["source_poisoned_tool"],
                "actor_request_id": trace.get("source_actor_request_id"),
                "judge": trace.get("judgment") or {},
                "B": verdicts.get("B") or {},
                "C": verdicts.get("C") or {},
            }
        )
    return found


def find_disagreements(actor_dir: Path, evaluation_dir: Path) -> list[dict[str, Any]]:
    """Find judged attack cases where B allowed and C explicitly blocked.

    This filters saved labels and decisions; it does not parse actor text or
    independently validate the judge. Source strings remain untrusted.
    """
    return [
        item
        for item in _attack_examples(actor_dir, evaluation_dir)
        if item["judge"].get("outcome") in {"attacker_goal", "poisoned_tool_selected"}
        and item["B"].get("decision") == "allow"
        and item["C"].get("decision") == "block"
    ]


def find_judge_reviewer_examples(
    actor_dir: Path, evaluation_dir: Path, *, outcome: str, decision: str, arm: str = "B"
) -> list[dict[str, Any]]:
    """Return saved attack cases with a specified judge label and reviewer verdict.

    A judge label is not ground truth; `authorized_task` alone does not prove
    that an injection was ignored. Uncertain or absent decisions are not repaired.
    """
    if (
        outcome not in {"attacker_goal", "authorized_task"}
        or decision not in {"allow", "block"}
        or arm not in {"B", "C"}
    ):
        raise ValueError("Unsupported example selection")
    return [
        item
        for item in _attack_examples(actor_dir, evaluation_dir)
        if item["judge"].get("outcome") == outcome
        and item[arm].get("decision") == decision
        and item["actor_request_id"]
    ]
