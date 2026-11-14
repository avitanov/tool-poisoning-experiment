"""Offline validation of bounded paid transport and qualification output checks."""

import json
from pathlib import Path
from typing import Any

import pytest

from mcptox_gateway.pipelines.qualify_models import inspect_probe
from mcptox_gateway.provider import openrouter_bounded_sender


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": 0},
        {"stream": True},
        {"tools": []},
        {"plugins": []},
        {"provider": {"only": ["a"], "allow_fallbacks": True}},
        {
            "provider": {
                "only": ["a"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "max_price": {"prompt": "NaN", "completion": 1, "request": 0},
            }
        },
    ],
)
def test_paid_sender_rejects_unbounded_or_executable_requests(change: dict[str, Any]) -> None:
    payload = {
        "model": "fixture-model",
        "stream": False,
        "max_tokens": 100,
        "provider": {
            "only": ["a"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": 1, "completion": 2, "request": 0},
        },
    }
    payload.update(change)
    with pytest.raises(ValueError):
        openrouter_bounded_sender(payload, "unused", 1)


def test_qualification_marks_truncated_and_wrong_outputs_failed() -> None:
    from mcptox_gateway.pipelines.free_pilot import make_record

    spec = json.loads((Path(__file__).resolve().parents[1] / "config/free_pilot.json").read_text())
    record = make_record(spec, "qualification", "fixture", [])
    probe = {"record": record, "kind": "missing_reference", "arm": "B"}
    response = {
        "usage": {"cost": 0},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"decision":"block","reason":"missing semantics"}'},
            }
        ],
    }
    assert inspect_probe(probe, json.dumps(response).encode())["passed"] is False
    response["choices"][0]["message"]["content"] = (
        '{"decision":"uncertain","reason":"missing semantics"}'
    )
    assert inspect_probe(probe, json.dumps(response).encode())["passed"] is True
    response["choices"][0]["finish_reason"] = "length"
    assert inspect_probe(probe, json.dumps(response).encode())["passed"] is False
