"""Exercise actual worker overlap with mocked transport and the real ledger."""

import io
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError

import pytest

from mcptox_gateway.budget import Budget
from mcptox_gateway.concurrent_collection import CollectionJob, ConcurrentCollector
from mcptox_gateway.provider import OpenRouterHTTPError, ProviderConfig
from mcptox_gateway.storage import CallStore, canonical_json, request_identity


def jobs_for(store: CallStore, budget: Budget, count: int) -> list[CollectionJob]:
    jobs = []
    for index in range(count):
        case = f"case-{index}"
        reservation = budget.reserve_case(case, {"actor": "1", "B": "0", "C": "0"})
        record = {
            "role": "actor",
            "case_id": case,
            "model": "fixture-model",
            "provider": "fixture",
            "generation": {},
            "repetition_id": "0",
            "prompt_version": "1",
            "reference_versions": {},
            "payload": {
                "model": "fixture-model",
                "messages": [{"role": "user", "content": case}],
                "provider": {"only": ["fixture"], "allow_fallbacks": False},
            },
        }
        jobs.append(
            CollectionJob(
                record,
                reservation["actor"],
                ProviderConfig("fixture", "fixture-model", "TEST_CONCURRENT_KEY", 2),
                "Fixture",
            )
        )
    return jobs


def response(payload: dict) -> dict:
    return {
        "model": "fixture-model",
        "provider": "Fixture",
        "id": payload["messages"][0]["content"],
        "usage": {"cost": 0.25},
        "choices": [{"message": {"content": "retained"}}],
    }


def test_judge_collection_replays_without_key_or_transport(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1", "judge": "1"})
    record = {
        "role": "judge",
        "case_id": "case",
        "model": "fixture-model",
        "provider": "fixture",
        "generation": {},
        "repetition_id": "0",
        "prompt_version": "1",
        "reference_versions": {"experiment_mode": "judged-v1"},
        "actor_response_hash": "actor-sha",
        "proposal": {"tool": "x", "arguments": {}},
        "normalization_version": "1",
        "judge_schema_version": "1",
        "payload": {
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "case"}],
            "provider": {"only": ["fixture"], "allow_fallbacks": False},
        },
    }
    job = CollectionJob(
        record, ids["judge"], ProviderConfig("fixture", "fixture-model", "TEST_KEY", 2), "Fixture"
    )
    calls = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        calls.append(1)
        return response(payload)

    monkeypatch.setenv("TEST_KEY", "fixture")
    collector = ConcurrentCollector(store, budget, sender, max_workers=1, min_interval_seconds=0)
    first = collector.collect([job], collect_missing=True)
    monkeypatch.delenv("TEST_KEY")
    assert collector.collect([job]) == first
    assert calls == [1]
    assert budget.summary()["settled"] == "0.25"


def test_received_429_is_retried_with_same_request_and_new_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    job = jobs_for(store, budget, 1)[0]
    job = replace(
        job,
        record={**job.record, "provider": "openrouter"},
        config=replace(job.config, provider="openrouter"),
    )
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")
    calls = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        calls.append(payload)
        if len(calls) == 1:
            raise OpenRouterHTTPError(
                HTTPError("https://openrouter.ai", 429, "rate", {"Retry-After": "0"}, io.BytesIO())
            )
        return response(payload)

    collector = ConcurrentCollector(store, budget, sender, max_workers=1, min_interval_seconds=0)
    result = collector.collect([job], collect_missing=True, maximum_new_requests=3)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert request_identity(job.record) in result
    assert budget.summary()["settled"] == "0.25"
    with store.connect() as db:
        assert db.execute("SELECT state FROM attempts ORDER BY number").fetchall() == [
            ("rate_limited",),
            ("completed",),
        ]


def test_timeout_does_not_retry_or_release_charge(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    job = jobs_for(store, budget, 1)[0]
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")
    calls = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        calls.append(1)
        raise TimeoutError("unknown billing")

    collector = ConcurrentCollector(store, budget, sender, max_workers=1, min_interval_seconds=0)
    with pytest.raises(ValueError, match="Collection stopped"):
        collector.collect([job], collect_missing=True, maximum_new_requests=3)
    assert calls == [1]
    assert budget.summary()["unresolved_reservations"] == "1"


def test_rate_limit_retries_stop_at_three_attempts(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    job = jobs_for(store, budget, 1)[0]
    job = replace(
        job,
        record={**job.record, "provider": "openrouter"},
        config=replace(job.config, provider="openrouter"),
    )
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")
    calls = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        calls.append(1)
        raise OpenRouterHTTPError(
            HTTPError("https://openrouter.ai", 429, "rate", {"Retry-After": "0"}, io.BytesIO())
        )

    collector = ConcurrentCollector(store, budget, sender, max_workers=1, min_interval_seconds=0)
    with pytest.raises(ValueError, match="Collection stopped"):
        collector.collect([job], collect_missing=True, maximum_new_requests=5)
    assert len(calls) == 3
    assert budget.summary()["unresolved_reservations"] == "1"


def test_overlap_replay_duplicates_and_exclusive_coordinator(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 4)
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "never-persisted")
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    observed = {"active": 0, "peak": 0, "calls": 0}
    coordinator = threading.get_ident()

    def sender(payload: dict, key: str, timeout: float) -> dict:
        assert threading.get_ident() != coordinator
        with lock:
            observed["active"] += 1
            observed["calls"] += 1
            observed["peak"] = max(observed["peak"], observed["active"])
        # A second store must never recover these live requests as abandoned.
        other = CallStore(tmp_path)
        with pytest.raises(ValueError, match="active"):
            with other.collection_session():
                pass
        with pytest.raises(ValueError, match="coordinator"):
            store.check_recovery()
        barrier.wait(timeout=3)
        with lock:
            observed["active"] -= 1
        return response(payload)

    collector = ConcurrentCollector(store, budget, sender, max_workers=2, min_interval_seconds=0)
    first = collector.collect(jobs + [jobs[0]], collect_missing=True)
    assert observed["calls"] == 4 and observed["peak"] == 2
    assert len(first) == 4 and budget.summary()["settled"] == "1.00"
    monkeypatch.delenv("TEST_CONCURRENT_KEY")
    assert collector.collect(jobs) == first
    assert observed["calls"] == 4
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 4
        assert (
            db.execute("SELECT count(*) FROM reservations WHERE status='unresolved'").fetchone()[0]
            == 0
        )


def test_failure_stops_queue_but_saves_other_inflight_response(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 4)
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")
    barrier = threading.Barrier(2)
    sent = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        case = payload["messages"][0]["content"]
        sent.append(case)
        barrier.wait(timeout=3)
        if case == "case-0":
            raise TimeoutError("unknown billing")
        time.sleep(0.05)
        return response(payload)

    collector = ConcurrentCollector(store, budget, sender, max_workers=2, min_interval_seconds=0)
    with pytest.raises(ValueError, match="in-flight responses retained"):
        collector.collect(jobs, collect_missing=True)
    assert sorted(sent) == ["case-0", "case-1"]
    assert store.response(request_identity(jobs[1].record))
    with store.connect() as db:
        assert db.execute("SELECT status FROM reservations WHERE id='case-0:actor'").fetchone() == (
            "unresolved",
        )
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 2
    with pytest.raises(ValueError, match="reconciliation"):
        collector.collect(jobs, collect_missing=True)
    assert len(sent) == 2


def test_quota_and_free_rate_checks_precede_dispatch(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 2)
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")

    def forbidden(*args):
        raise AssertionError("No dispatch permitted")

    collector = ConcurrentCollector(store, budget, forbidden, min_interval_seconds=0)
    with pytest.raises(ValueError, match="quota"):
        collector.collect(jobs, collect_missing=True, maximum_new_requests=1)
    with pytest.raises(ValueError, match="replay"):
        collector.collect(jobs)
    record = dict(jobs[0].record, model="fixture:free")
    record["payload"] = dict(record["payload"], model="fixture:free")
    free = CollectionJob(
        record,
        jobs[0].reservation_id,
        ProviderConfig("fixture", "fixture:free", "TEST_CONCURRENT_KEY", 2),
        "Fixture",
    )
    with pytest.raises(ValueError, match="3.2"):
        collector.collect([free], collect_missing=True)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0


def test_rate_limit_headers_are_retained_without_credentials() -> None:
    error = HTTPError(
        "https://example.invalid",
        429,
        "limited",
        {"Retry-After": "30", "X-RateLimit-Reset": "1234", "Authorization": "secret"},
        io.BytesIO(b"private response"),
    )
    captured = OpenRouterHTTPError(error)
    assert captured.details["retry_after"] == "30"
    assert captured.details["http_status"] == 429
    assert "secret" not in json.dumps(captured.details)


def test_spacing_is_shared_between_workers(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 3)
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")
    starts = []

    def sender(payload: dict, key: str, timeout: float) -> dict:
        starts.append(time.monotonic())
        return response(payload)

    collector = ConcurrentCollector(store, budget, sender, min_interval_seconds=0.04)
    collector.collect(jobs, collect_missing=True)
    assert all(b - a >= 0.03 for a, b in zip(starts, starts[1:]))


def test_provider_drift_is_saved_and_still_rejected_on_replay(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 1)
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")

    def sender(payload: dict, key: str, timeout: float) -> dict:
        return {**response(payload), "provider": "Unexpected"}

    collector = ConcurrentCollector(store, budget, sender, min_interval_seconds=0)
    with pytest.raises(ValueError, match="stopped"):
        collector.collect(jobs, collect_missing=True)
    assert store.response(request_identity(jobs[0].record))
    monkeypatch.delenv("TEST_CONCURRENT_KEY")
    with pytest.raises(ValueError, match="drift"):
        collector.collect(jobs)


def test_missing_billing_is_saved_without_resending(tmp_path: Path, monkeypatch) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 1)
    monkeypatch.setenv("TEST_CONCURRENT_KEY", "fixture")

    def sender(payload: dict, key: str, timeout: float) -> dict:
        return {**response(payload), "usage": {}}

    collector = ConcurrentCollector(store, budget, sender, min_interval_seconds=0)
    with pytest.raises(ValueError, match="stopped"):
        collector.collect(jobs, collect_missing=True)
    assert store.response(request_identity(jobs[0].record))
    with pytest.raises(ValueError, match="reconciliation"):
        collector.collect(jobs, collect_missing=True)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1


def test_abandoned_batch_recovers_every_inflight_request(tmp_path: Path) -> None:
    store = CallStore(tmp_path)
    budget = Budget(store)
    jobs = jobs_for(store, budget, 2)
    with pytest.raises(RuntimeError):
        with store.collection_session():
            for job in jobs:
                identity = store.plan(job.record, canonical_json(job.record["payload"]).encode())
                attempt = store.claim(identity, job.reservation_id)
                store.dispatched(identity, attempt)
            raise RuntimeError("simulated process interruption")
    reopened = CallStore(tmp_path)
    with reopened.connect() as db:
        assert (
            db.execute("SELECT count(*) FROM reservations WHERE status='unresolved'").fetchone()[0]
            == 2
        )
