"""Offline checks for the live engineering pilot and its zero-spend replay."""

import json
import shutil
from pathlib import Path

import pytest

from mcptox_gateway.actor import build_actor_messages
from mcptox_gateway.pipelines import free_pilot
from mcptox_gateway.provider import openrouter_sender
from mcptox_gateway.storage import CallStore

ROOT = Path(__file__).resolve().parents[1]


def test_actor_preserves_source_and_appends_one_light_warning() -> None:
    source = "Original tools\n  exact whitespace\n"
    messages = build_actor_messages(source, "task")
    assert messages[0]["content"].startswith(source + "\n\n")
    assert messages[0]["content"].count("Be careful when using tools") == 1
    assert messages[1] == {"role": "user", "content": "task"}


@pytest.mark.parametrize(
    "change",
    [
        {"model": "paid/model"},
        {"provider": {"only": ["a", "b"], "allow_fallbacks": False}},
        {"tools": []},
        {"plugins": []},
    ],
)
def test_free_sender_rejects_paid_or_unpinned_routes_before_network(change: dict) -> None:
    config = json.loads((ROOT / "config/free_pilot.json").read_text())
    payload = free_pilot.make_record(config, "actor", "case", [])["payload"]
    payload.update(change)
    with pytest.raises(ValueError):
        openrouter_sender(payload, "unused-secret", 1)


def test_pilot_collects_once_then_replays_without_network_or_key(
    tmp_path: Path, monkeypatch
) -> None:
    for folder in ("config", "prompts"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    for filename in (".python-version", "pyproject.toml", "uv.lock", "data/manifest.json"):
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, target)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-secret")
    monkeypatch.setattr(free_pilot.time, "sleep", lambda _: None)
    import io

    endpoint = {
        "data": {"endpoints": [{"tag": "nvidia", "pricing": {"prompt": "0", "completion": "0"}}]}
    }
    monkeypatch.setattr(
        free_pilot, "urlopen", lambda *a, **_kwargs: io.BytesIO(json.dumps(endpoint).encode())
    )
    sent = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        sent.append(payload)
        if payload["messages"][0]["content"].startswith("Return exactly"):
            content = '{"status":"ok"}'
        elif payload["messages"][0]["content"].startswith("Choose one tool"):
            content = (
                '{"tool":"send_message","arguments":'
                '{"recipient":"Alice","content":"Meeting at 10."}}'
            )
        else:
            content = '{"decision":"allow","reason":"fixture"}'
        return {
            "id": f"fixture-{len(sent)}",
            "model": payload["model"],
            "provider": "Nvidia",
            "usage": {"cost": 0},
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr(free_pilot, "openrouter_sender", sender)
    first = free_pilot.run_pilot(tmp_path, collect=True)
    assert len(sent) == 19
    assert first["valid_reviews"] == 12
    store = CallStore(tmp_path / "results")
    with store.connect() as db:
        before = db.execute("SELECT * FROM attempts").fetchall()
    monkeypatch.delenv("OPENROUTER_API_KEY")

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not access network")

    monkeypatch.setattr(free_pilot, "urlopen", forbidden)
    monkeypatch.setattr(free_pilot, "openrouter_sender", forbidden)
    assert free_pilot.run_pilot(tmp_path) == first
    assert free_pilot.run_pilot(tmp_path, collect=False) == first
    with store.connect() as db:
        assert db.execute("SELECT * FROM attempts").fetchall() == before
    traces = list(
        (tmp_path / "results" / "engineering-pilot-2026-11-12-v1" / "traces").glob("*.json")
    )
    assert len(traces) == 6
    assert {json.loads(p.read_text())["review_order"] for p in traces} == {"BC", "CB"}
