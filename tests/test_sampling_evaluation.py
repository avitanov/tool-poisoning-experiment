"""Offline tests for frozen sample selection and answer-key isolation."""

from __future__ import annotations

import pytest

from mcptox_gateway.evaluation import assert_no_evaluation_labels
from mcptox_gateway.sampling import select_attack_samples


def _case(family: str, template: str, position: int) -> dict[str, object]:
    return {
        "case_id": f"case-{family}",
        "family_id": family,
        "template": template,
        "wrong_data": 0,
        "poisoned_tool": "ordinary text",
        "source_position": str(position),
    }


def test_selection_is_repeatable_and_keeps_families_disjoint() -> None:
    """Pilot, main, and reserve use each family at most once."""

    cases = [
        _case(f"{template}-{index}", template, index)
        for index in range(50)
        for template in ("Template-1", "Template-2", "Template-3")
    ]
    first = select_attack_samples(cases)
    second = select_attack_samples(cases)
    assert first == second
    pilot = {item["family_id"] for item in first["pilot"]}
    main = {item["family_id"] for item in first["main"]}
    reserve = {item["family_id"] for item in first["reserve"]}
    assert len(first["pilot"]) == 10
    assert len(first["main"]) == 100
    assert not pilot & main and not pilot & reserve and not main & reserve


def test_answer_key_fields_cannot_enter_model_payloads() -> None:
    """A deliberately wrong payload is rejected before transport exists."""

    assert_no_evaluation_labels({"request": "safe"})
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_no_evaluation_labels({"request": "safe", "success_criteria": ["hidden"]})
