"""Coverage distinguishes invalid outputs, uncertainty and unrecorded calls."""

import json
from pathlib import Path

import pytest

from mcptox_gateway.notebook_coverage import role_call_coverage


def test_role_coverage_distinguishes_no_text_missing_and_uncertain(tmp_path: Path) -> None:
    actor, evaluation = tmp_path / "actor", tmp_path / "evaluation"
    (actor / "traces").mkdir(parents=True)
    (evaluation / "traces").mkdir(parents=True)
    cases = [{"source_id": key, "kind": "attack"} for key in ("valid", "blank", "missing")]
    (actor / "manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf8")
    for key, status in (("valid", "not_parsed"), ("blank", "provider_no_text")):
        (actor / "traces" / f"{key}.json").write_text(
            json.dumps({"request_ids": {"actor": key}, "proposal_status": status}), encoding="utf8"
        )
    (evaluation / "traces" / "valid.json").write_text(
        json.dumps(
            {
                "request_ids": {"B": "b", "C": "c", "judge": "j"},
                "review_verdicts": {
                    "B": {"decision": "uncertain", "reason": "missing reference"},
                    "C": None,
                },
                "judgment": {"outcome": "authorized_task", "reason": "aligned"},
            }
        ),
        encoding="utf8",
    )
    rows = {row["role"]: row for row in role_call_coverage(actor, evaluation, "attack")}
    assert (
        rows["actor"]["recorded"],
        rows["actor"]["usable"],
        rows["actor"]["invalid"],
        rows["actor"]["not_recorded"],
    ) == (2, 1, 1, 1)
    assert (rows["B"]["usable"], rows["B"]["uncertain"], rows["B"]["not_recorded"]) == (1, 1, 2)
    assert (rows["C"]["recorded"], rows["C"]["invalid"]) == (1, 1)
    assert (rows["judge"]["usable"], rows["judge"]["invalid"]) == (1, 0)
    assert role_call_coverage(tmp_path / "missing", evaluation, "attack") == []
    with pytest.raises(ValueError, match="kind"):
        role_call_coverage(actor, evaluation, "unexpected")
