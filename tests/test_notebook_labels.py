"""Judge label and ablation tables account for every planned case once."""

import json
from pathlib import Path

from mcptox_gateway.notebook_labels import (
    ablation_label_counts,
    judge_label_counts,
    load_label_traces,
)


def _trace(case: str, label: str | None, b: str | None, c: str | None) -> dict:
    return {
        "source_id": case,
        "kind": "attack",
        "judgment": {"outcome": label, "reason": "saved"} if label else None,
        "request_ids": {"judge": "j"},
        "review_verdicts": {
            "B": {"decision": b} if b else None,
            "C": {"decision": c} if c else None,
        },
    }


def test_labels_are_exclusive_and_reviewer_abstentions_are_withheld() -> None:
    traces = [
        _trace("1", "attacker_goal", "allow", "block"),
        _trace("2", "poisoned_tool_selected", "uncertain", "allow"),
        _trace("3", "authorized_task", "allow", "allow"),
        _trace("4", "neither", "allow", "allow"),
        _trace("5", "uncertain", "allow", "allow"),
        _trace("6", None, "allow", "allow"),
    ]
    labels = judge_label_counts(traces)
    assert [
        labels[name]
        for name in (
            "attacker_goal",
            "poisoned_tool_selected",
            "authorized_task",
            "neither",
            "uncertain",
        )
    ] == [1] * 5
    assert labels["judge_invalid"] == 1 and labels["planned"] == 6
    rows = {row["condition"]: row for row in ablation_label_counts(traces)}
    assert rows["A"]["attacker_goal"] == 1 and rows["A"]["poisoned_tool_selected"] == 1
    assert rows["B"]["withheld_uncertain"] == 1 and rows["C"]["withheld_block"] == 1
    assert rows["A"]["excluded"] == rows["C"]["excluded"] == 2
    for row in rows.values():
        assert (
            sum(
                row[key]
                for key in (
                    "attacker_goal",
                    "poisoned_tool_selected",
                    "authorized_task",
                    "neither",
                    "uncertain",
                    "withheld_block",
                    "withheld_uncertain",
                    "excluded",
                )
            )
            == row["planned"]
        )


def test_retries_replace_only_missing_judgments_without_changing_saved_traces(
    tmp_path: Path,
) -> None:
    actor, evaluation, retry = (tmp_path / name for name in ("actor", "evaluation", "retry"))
    actor.mkdir()
    (evaluation / "traces").mkdir(parents=True)
    (retry / "traces").mkdir(parents=True)
    (actor / "manifest.json").write_text(
        json.dumps({"cases": [{"source_id": "x", "kind": "attack"}]})
    )
    original = _trace("x", None, None, None)
    original["actor_error"] = "provider_no_text"
    path = evaluation / "traces/x.json"
    path.write_text(json.dumps(original))
    (retry / "traces/x.json").write_text(json.dumps(_trace("x", "attacker_goal", "allow", "block")))
    rows = load_label_traces(actor, evaluation, "attack", actor_retry_evaluation=retry)
    assert judge_label_counts(rows)["attacker_goal"] == 1
    assert json.loads(path.read_text()) == original
    assert judge_label_counts(load_label_traces(actor, evaluation, "attack"))["actor_unusable"] == 1


def test_judge_only_count_keeps_case_with_missing_reviewer() -> None:
    traces = [_trace("x", "attacker_goal", "allow", None)]
    assert judge_label_counts(traces)["attacker_goal"] == 1
    rows = ablation_label_counts(traces)
    assert all(row["excluded"] == 1 and row["attacker_goal"] == 0 for row in rows)


def test_judge_retry_reuses_existing_reviews_without_overwriting_trace(tmp_path: Path) -> None:
    actor, evaluation = tmp_path / "actor", tmp_path / "evaluation"
    actor.mkdir()
    (evaluation / "traces").mkdir(parents=True)
    (actor / "manifest.json").write_text(
        json.dumps({"cases": [{"source_id": "x", "kind": "attack"}]})
    )
    original = _trace("x", None, "allow", "block")
    path = evaluation / "traces/x.json"
    path.write_text(json.dumps(original))
    retry = tmp_path / "judge_retry.json"
    retry.write_text(
        json.dumps(
            {"source_id": "x", "judgment": {"outcome": "authorized_task", "reason": "saved retry"}}
        )
    )
    rows = load_label_traces(actor, evaluation, "attack", judge_retry_file=retry)
    assert judge_label_counts(rows)["authorized_task"] == 1
    assert ablation_label_counts(rows)[2]["withheld_block"] == 1
    assert json.loads(path.read_text()) == original
