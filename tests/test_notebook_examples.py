"""Only explicit B-allow/C-block attack disagreements become examples."""

import json
from pathlib import Path

from mcptox_gateway.notebook_examples import find_disagreements, find_judge_reviewer_examples


def test_disagreement_selection_keeps_original_evidence(tmp_path: Path) -> None:
    actor, evaluation = tmp_path / "actor", tmp_path / "evaluation"
    actor.mkdir()
    (evaluation / "traces").mkdir(parents=True)
    original = {
        "source_id": "case",
        "case": {"query": "user request"},
        "goal": "source risk",
        "source_poisoned_tool": "untrusted text",
    }
    (actor / "manifest.json").write_text(json.dumps({"cases": [original]}), encoding="utf8")
    trace = {
        "source_id": "case",
        "kind": "attack",
        "source_actor_request_id": "actor-id",
        "judgment": {"outcome": "poisoned_tool_selected", "reason": "judge reason"},
        "review_verdicts": {
            "B": {"decision": "allow", "reason": "B reason"},
            "C": {"decision": "block", "reason": "C reason"},
        },
    }
    path = evaluation / "traces" / "case.json"
    path.write_text(json.dumps(trace), encoding="utf8")
    result = find_disagreements(actor, evaluation)
    assert len(result) == 1 and result[0]["poisoned_tool"] == "untrusted text"
    assert result[0]["source_risk"] == "source risk"
    assert result[0]["actor_request_id"] == "actor-id"
    assert result[0]["judge"]["reason"] == "judge reason"
    trace["review_verdicts"]["C"]["decision"] = "uncertain"
    path.write_text(json.dumps(trace), encoding="utf8")
    assert find_disagreements(actor, evaluation) == []
    assert find_disagreements(tmp_path / "missing", evaluation) == []


def test_judge_reviewer_examples_select_labels_without_interpreting_text(tmp_path: Path) -> None:
    import pytest

    actor, evaluation = tmp_path / "actor", tmp_path / "evaluation"
    actor.mkdir()
    (evaluation / "traces").mkdir(parents=True)
    original = {
        "source_id": "case",
        "case": {"query": "task"},
        "goal": "risk",
        "source_poisoned_tool": "injected text",
    }
    (actor / "manifest.json").write_text(json.dumps({"cases": [original]}), encoding="utf8")
    trace = {
        "source_id": "case",
        "kind": "attack",
        "source_actor_request_id": "actor-id",
        "judgment": {"outcome": "attacker_goal", "reason": "judge"},
        "review_verdicts": {
            "B": {"decision": "allow", "reason": "review"},
            "C": {"decision": "block", "reason": "review"},
        },
    }
    (evaluation / "traces" / "case.json").write_text(json.dumps(trace), encoding="utf8")
    selected = find_judge_reviewer_examples(
        actor, evaluation, outcome="attacker_goal", decision="allow"
    )
    assert len(selected) == 1 and selected[0]["poisoned_tool"] == "injected text"
    assert (
        find_judge_reviewer_examples(actor, evaluation, outcome="authorized_task", decision="block")
        == []
    )
    assert (
        len(
            find_judge_reviewer_examples(
                actor, evaluation, outcome="attacker_goal", decision="block", arm="C"
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="selection"):
        find_judge_reviewer_examples(
            actor, evaluation, outcome="attacker_goal", decision="uncertain"
        )
