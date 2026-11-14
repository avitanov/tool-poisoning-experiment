"""Checks for the frozen multi-actor experiment inputs."""

from copy import deepcopy
from pathlib import Path

from mcptox_gateway.pipelines import research_multi_actor as multi
from mcptox_gateway.pipelines.research_pilot import ROUTES


def test_main_manifest_reuses_source_context_and_freezes_each_actor(
    monkeypatch, tmp_path: Path
) -> None:
    case = {
        "source_id": "source-1",
        "kind": "attack",
        "case": {"case_id": "old", "query": "original task", "system": "original tool text"},
        "goal": "source risk",
        "source_poisoned_tool": "poison text",
        "bounds": {"actor": "0", "B": "1", "C": "1", "judge": "1"},
    }
    base = {"cases": [case], "routes": ROUTES, "known_limitations": []}
    original = deepcopy(base)
    monkeypatch.setattr(multi, "verify_manifest", lambda *args: None)
    monkeypatch.setattr(multi, "read_json", lambda path: deepcopy(base))
    monkeypatch.setattr(multi, "_check_endpoint", lambda route, judge: {"model": route["model"]})
    for name in ("gpt", "qwen", "minimax"):
        manifest = multi.make_manifest(tmp_path, name, "main")
        actual = manifest["cases"][0]
        assert actual["case"]["query"] == "original task"
        assert actual["case"]["system"] == "original tool text"
        assert actual["goal"] == "source risk"
        assert actual["source_poisoned_tool"] == "poison text"
        assert manifest["routes"]["actor"]["model"] == multi.ACTORS[name]["model"]
        assert float(actual["bounds"]["actor"]) > 0
        assert actual["bounds"]["judge"] == "0"
    assert base == original


def test_new_main_run_requires_completed_pilot_before_endpoint_checks(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    import pytest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_multi_actor",
            "--model",
            "gpt",
            "--split",
            "main",
            "--run-id",
            "fresh-main-gpt",
            "--collect-missing",
        ],
    )
    monkeypatch.setattr(
        multi, "make_manifest", lambda *args, **kwargs: pytest.fail("pilot gate bypassed")
    )
    with pytest.raises(ValueError, match="completed pilot"):
        multi.main()
