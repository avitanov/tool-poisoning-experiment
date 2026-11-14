from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from mcptox_gateway.budget import Budget, maximum_charge
from mcptox_gateway.storage import CallStore, canonical_json, parse_exact_json, request_identity


def actor():
    return {"role": "actor", "payload": {"messages": [{"content": " x Ω "}]}, "provider": "fixture", "model": "mock", "generation": {}, "case_id": "case", "repetition_id": "1", "prompt_version": "p1", "reference_versions": {}}


def reviewer(arm):
    record = {**actor(), "role": "reviewer", "arm": arm, "actor_response_hash": "hash", "proposal": {"tool": "x"}, "normalization_version": "n1", "clean_semantics": ["clean"], "clean_semantics_version": "s1"}
    if arm == "C":
        record.update(integrity_report=[{"approved_description": None, "observed_description": " bad ", "missing_reason": "unknown"}], registry_version="r1", hash_rules_version="h1", extraction_version="e1", report_schema_version="schema1", serialization_version="json1")
    return record


def test_identity_and_strict_json():
    b, c = reviewer("B"), reviewer("C")
    assert request_identity(b) != request_identity(c)
    assert request_identity(c) != request_identity({**c, "registry_version": "r2"})
    assert request_identity(c) != request_identity({**c, "integrity_report": []})
    for record in (b, c):
        assert request_identity(record) != request_identity({**record, "clean_semantics_version": "s2"})
        assert request_identity(record) != request_identity({**record, "repetition_id": "2"})
    assert request_identity(actor()) == request_identity(actor())
    with pytest.raises(ValueError):
        request_identity({**b, "registry_version": "r2"})
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(ValueError):
            parse_exact_json(text)
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


def test_judge_identity_and_complete_bundle(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    record = {**actor(), "role": "judge", "actor_response_hash": "saved-actor",
              "proposal": {"tool": "x", "arguments": {}}, "normalization_version": "1",
              "judge_schema_version": "1"}
    identity = request_identity(record)
    assert identity != request_identity({**record, "judge_schema_version": "2"})
    with pytest.raises(ValueError):
        request_identity({**record, "arm": "B"})
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1", "judge": "1"})
    assert budget.summary()["unresolved_reservations"] == "4"
    request = store.plan(record, canonical_json(record["payload"]).encode())
    assert store.claim(request, ids["judge"]) == 1
    store.dispatched(request, 1)
    store.save_response(request, 1, b'{"outcome":"uncertain"}', {})
    budget.settle(ids["judge"], "0.1", "actual", {})
    store.complete(request)
    assert store.response(identity) == b'{"outcome":"uncertain"}'
    store.close()


def test_judged_actor_requires_judge_reservation(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    record = {**actor(), "reference_versions": {"experiment_mode": "judged-v1"}}
    request = store.plan(record, canonical_json(record["payload"]).encode())
    with pytest.raises(ValueError, match="bundle"):
        store.claim(request, ids["actor"])
    store.close()


def test_recorded_http_rejection_is_closed_at_upper_bound_without_retry(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    ids = budget.reserve_case("case", {"actor": "0.1", "B": "0.1", "C": "0.1"})
    record = actor()
    request = store.plan(record, canonical_json(record["payload"]).encode())
    attempt = store.claim(request, ids["actor"])
    store.dispatched(request, attempt)
    budget.unresolved(ids["actor"], {"transport_details": {"http_status": 404}})
    store.close()
    store.reconcile_rejected_request(request, attempt)
    assert Decimal(budget.summary()["settled"]) == Decimal("0.1")
    with store.connect() as db:
        assert db.execute("SELECT status FROM requests WHERE id=?", (request,)).fetchone() == ("failed",)
        assert db.execute("SELECT charge_status FROM reservations WHERE id=?", (ids["actor"],)).fetchone() == ("upper-bound",)
    with pytest.raises(ValueError):
        store.reconcile_rejected_request(request, attempt)


def test_claim_recovery_and_cache(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    record = actor()
    request = store.plan(record, canonical_json(record["payload"]).encode())
    def claim():
        try:
            return store.claim(request, ids["actor"])
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2)), key=lambda x: x or 0) == [None, 1]
    store.dispatched(request, 1)
    store.save_response(request, 1, b'{"refusal":"no"}', {"provider_request_id": "fixture", "usage": None})
    reopened = CallStore(tmp_path)
    assert reopened.response(request) == b'{"refusal":"no"}'
    budget.settle(ids["actor"], "0.5", "actual", {})
    reopened.complete(request)
    assert reopened.response(request) == b'{"refusal":"no"}'
    with pytest.raises(ValueError):
        reopened.claim(request, ids["B"])
    budget.release(ids["B"])
    budget.release(ids["C"])
    assert Budget(reopened).summary()["settled"] == "0.5"


def test_budget_bounds_and_unknown(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    ids = budget.reserve_case("one", {"actor": "4", "B": "10", "C": "5.50"})
    with pytest.raises(ValueError):
        budget.reserve_attempt("extra", "0.01", case_id="extra", role="actor")
    budget.unresolved(ids["actor"], {"error": "timeout"})
    with pytest.raises(ValueError):
        budget.release(ids["actor"])
    budget.settle(ids["actor"], "4", "upper-bound", {})
    budget.settle(ids["actor"], "4", "upper-bound", {})
    with pytest.raises(ValueError):
        budget.settle(ids["actor"], "3", "actual", {})
    assert budget.summary()["operational_remaining"] == "0.00"
    assert maximum_charge("1", "2", "3", "0.1", reasoning_in_output=False) == "6.1"
    with pytest.raises(ValueError):
        maximum_charge("1", "2", "3", "0", reasoning_in_output=True)


def test_safe_retry_limit_and_ambiguous_dispatch(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    record = actor()
    request = store.plan(record, canonical_json(record["payload"]).encode())
    bundle = budget.reserve_case("case", {"actor": "0.1", "B": "0.1", "C": "0.1"})
    reservation = bundle["actor"]
    for number in range(1, 4):
        assert store.claim(request, reservation) == number
        store.safe_pre_dispatch_failure(request, number, {"reason": "local validation"})
        if number < 3:
            reservation = f"retry-{number}"
            budget.reserve_attempt(reservation, "0.1", case_id="case", role="actor", request_id=request)
    with pytest.raises(ValueError):
        budget.reserve_attempt("fourth", "0.1", case_id="case", role="actor", request_id=request)
    second = {**record, "case_id": "other"}
    other = store.plan(second, canonical_json(second["payload"]).encode())
    other_bundle = budget.reserve_case("other", {"actor": "0.1", "B": "0.1", "C": "0.1"})
    assert store.claim(other, other_bundle["actor"]) == 1
    store.dispatched(other, 1)
    budget.unresolved(other_bundle["actor"], {"reason": "timeout"})
    with pytest.raises(ValueError):
        store.safe_pre_dispatch_failure(other, 1, {})
    store.close()
    with pytest.raises(ValueError):
        store.claim(other, other_bundle["actor"])



def test_completion_requires_reconciliation_and_integrity(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    record = actor()
    request = store.plan(record, canonical_json(record["payload"]).encode())
    store.claim(request, "case:actor")
    store.dispatched(request, 1)
    store.save_response(request, 1, b"invalid output", {})
    with pytest.raises(ValueError):
        store.complete(request)
    budget.settle("case:actor", "0.1", "actual", {})
    store.complete(request)
    artifact = next((tmp_path / "raw" / request).glob("response-*.json"))
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError):
        store.response(request)


def test_reservation_ownership_and_unresolved_blocks_collection(tmp_path: Path):
    store = CallStore(tmp_path, output_root=tmp_path / "run-a")
    budget = Budget(store)
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    request = store.plan(actor(), canonical_json(actor()["payload"]).encode())
    for invalid in ("missing", ids["B"]):
        with pytest.raises(ValueError):
            store.claim(request, invalid)
    with pytest.raises(ValueError):
        store.claim(request, ids["C"])
    store.claim(request, ids["actor"])
    second = {**actor(), "repetition_id": "2"}
    request2 = store.plan(second, canonical_json(second["payload"]).encode())
    with pytest.raises(ValueError):
        store.claim(request2, ids["actor"])
    store.dispatched(request, 1)
    budget.unresolved(ids["actor"], {"timeout": True})
    reopened = Budget(CallStore(tmp_path, output_root=tmp_path / "run-b"))
    assert reopened.summary()["unresolved_reservations"] == "3"
    with pytest.raises(ValueError):
        reopened.reserve_attempt("new", "0.1", case_id="case", role="actor")
    with pytest.raises(ValueError):
        reopened.store.claim(request2, ids["B"])


def test_crash_after_response_fsync_recovers_journal(tmp_path: Path, monkeypatch):
    store = CallStore(tmp_path)
    budget = Budget(store)
    budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    request = store.plan(actor(), canonical_json(actor()["payload"]).encode())
    store.claim(request, "case:actor")
    store.dispatched(request, 1)
    original = store._artifact
    def interrupted(request_id, name, content):
        result = original(request_id, name, content)
        if name == "response":
            raise RuntimeError("power loss after artifact fsync")
        return result
    monkeypatch.setattr(store, "_artifact", interrupted)
    with pytest.raises(RuntimeError):
        store.save_response(request, 1, b"exact response", {"usage": {"tokens": 1}})
    store.close()
    reopened = CallStore(tmp_path)
    assert reopened.response(request) == b"exact response"
    Budget(reopened).settle("case:actor", "0.1", "actual", {"invoice": "fixture"})
    reopened.complete(request)
    with pytest.raises(ValueError):
        reopened.claim(request, "case:actor")


def test_overflow_and_schema_version(tmp_path: Path):
    import sqlite3
    with pytest.raises(ValueError):
        parse_exact_json('{"nested": [1e999]}')
    store = CallStore(tmp_path)
    with store.connect() as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="migration"):
        CallStore(tmp_path)
    other = tmp_path / "old"
    other.mkdir()
    with sqlite3.connect(other / "calls.sqlite") as db:
        db.execute("CREATE TABLE old_data (id TEXT)")
    with pytest.raises(ValueError, match="migration"):
        CallStore(other)



def test_abandoned_dispatch_and_live_collector(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    request = store.plan(actor(), canonical_json(actor()["payload"]).encode())
    store.claim(request, ids["actor"])
    store.dispatched(request, 1)
    live = CallStore(tmp_path)
    with live.connect() as db:
        assert db.execute("SELECT status FROM reservations WHERE id=?", (ids["actor"],)).fetchone() == ("reserved",)
    store.close()  # Simulate process death: OS releases its collector lock.
    recovered = CallStore(tmp_path)
    with recovered.connect() as db:
        assert db.execute("SELECT status FROM reservations WHERE id=?", (ids["actor"],)).fetchone() == ("unresolved",)
    with pytest.raises(ValueError, match="Unresolved"):
        Budget(recovered).reserve_case("other", {"actor": "1", "B": "1", "C": "1"})
    with pytest.raises(ValueError):
        recovered.claim(request, ids["B"])


def test_lower_legacy_limit_preserves_charges_and_rejects_active_reservations(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store)
    budget.reserve_attempt("probe", "0.20", case_id="probe", role="qualification")
    with store.connect() as db:
        db.execute("UPDATE budget_config SET ceiling='15', stop='14.50'")
    with pytest.raises(ValueError, match="reservations"):
        Budget.lower_legacy_limit(store)
    budget.settle("probe", "0.10", "actual", {})
    Budget.lower_legacy_limit(store)
    assert Budget(store, ceiling="10", operational_stop="9.50").summary()["settled"] == "0.10"
    with store.connect() as db:
        assert db.execute("SELECT old_ceiling, new_ceiling FROM budget_amendments").fetchall() == [("15", "10")]
    store.close()


def test_raise_to_twenty_preserves_ledger(tmp_path: Path):
    store = CallStore(tmp_path)
    budget = Budget(store, ceiling="10", operational_stop="9.50")
    budget.reserve_attempt("probe", "0.2", case_id="probe", role="qualification")
    with pytest.raises(ValueError, match="reservations"):
        Budget.raise_to_twenty(store)
    budget.settle("probe", "0.1", "actual", {})
    Budget.raise_to_twenty(store)
    assert Budget(store).summary()["settled"] == "0.1"
    with store.connect() as db:
        assert db.execute("SELECT old_ceiling, new_ceiling FROM budget_amendments").fetchall() == [("10", "20")]
    with pytest.raises(ValueError, match="former"):
        Budget.raise_to_twenty(store)
    store.close()


def test_stop_and_case_bundle_are_mandatory(tmp_path: Path):
    store = CallStore(tmp_path)
    with pytest.raises(ValueError):
        Budget(store, ceiling="20", operational_stop="20")
    with pytest.raises(ValueError):
        Budget(store, ceiling="21", operational_stop="19.50")
    budget = Budget(store)
    request = store.plan(actor(), canonical_json(actor()["payload"]).encode())
    with pytest.raises(ValueError):
        budget.reserve_attempt("solo", "1", case_id="case", role="actor", request_id=request)
    with pytest.raises(ValueError):
        store.claim(request, "solo")
    with pytest.raises(ValueError):
        store.dispatched(request, 1)
    store.close()
    ids = budget.reserve_case("case", {"actor": "1", "B": "1", "C": "1"})
    store.claim(request, ids["actor"])
    store.dispatched(request, 1)
    store.close()
