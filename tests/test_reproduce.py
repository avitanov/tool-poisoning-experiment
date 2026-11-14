"""Offline orchestration and pilot-gate checks for fresh reproduction."""

import json
from pathlib import Path

import pytest

from mcptox_gateway.pipelines import reproduce


def test_fresh_runs_pilot_before_main_and_keeps_models_separate(
    monkeypatch, tmp_path: Path
) -> None:
    events = []
    for name in ("download_sources", "ingest", "create_splits", "verify"):
        monkeypatch.setattr(
            reproduce, name, lambda *args, _name=name, **kwargs: events.append(_name)
        )

    def stage(root, module, args):
        events.append((module, args))
        if module.endswith("pilot_raw_comparison"):
            run_id = args[args.index("--run-id") + 1]
            output = root / "results" / run_id
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "attack_text_pass_through": {"complete_triples": 10},
                        "control_text_pass_through": {"complete_triples": 3},
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(reproduce, "_stage", stage)
    reproduce.run_pipeline(tmp_path, mode="fresh", models=["gpt"], experiment_id="check")
    assert events[:4] == ["download_sources", "ingest", "create_splits", "verify"]
    stages = [event for event in events if isinstance(event, tuple)]
    assert len(stages) == 4
    assert stages[0][1][stages[0][1].index("--split") + 1] == "pilot"
    assert stages[2][1][stages[2][1].index("--split") + 1] == "main"
    assert all("--collect-missing" in args for _, args in stages)
    comparison = json.loads((tmp_path / "results/check-comparison.json").read_text())
    assert comparison["actors"]["gpt"]["run_id"] == "check-main-gpt-evaluation"


def test_failed_pilot_stops_before_main(monkeypatch, tmp_path: Path) -> None:
    for name in ("download_sources", "ingest", "create_splits", "verify"):
        monkeypatch.setattr(reproduce, name, lambda *args, **kwargs: None)
    calls = []

    def stage(root, module, args):
        calls.append(args)
        if module.endswith("pilot_raw_comparison"):
            output = root / "results" / args[args.index("--run-id") + 1]
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "attack_text_pass_through": {"complete_triples": 9},
                        "control_text_pass_through": {"complete_triples": 3},
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(reproduce, "_stage", stage)
    with pytest.raises(ValueError, match="pilot has incomplete"):
        reproduce.run_pipeline(tmp_path, mode="fresh", models=["gpt"], experiment_id="test")
    assert len(calls) == 2
    assert not (tmp_path / "results/test-comparison.json").exists()


def test_replay_never_requests_collection(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(reproduce, "_stage", lambda root, module, args: calls.append(args))
    reproduce.run_pipeline(tmp_path, mode="replay", models=["gpt", "nemotron"], experiment_id=None)
    assert len(calls) == 4
    assert all("--collect-missing" not in args for args in calls)
    with pytest.raises(ValueError, match="requires a new experiment identifier"):
        reproduce.run_pipeline(tmp_path, mode="fresh", models=["gpt"], experiment_id=None)
