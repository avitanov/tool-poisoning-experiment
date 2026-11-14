"""Budgeted adapter connecting judged cases to the durable call collector."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

from mcptox_gateway.budget import Budget
from mcptox_gateway.concurrent_collection import CollectionJob, ConcurrentCollector
from mcptox_gateway.provider import ProviderConfig
from mcptox_gateway.storage import CallStore, canonical_json, request_identity


class CaseCollector:
    """Collect or replay one case under a preflighted four-role reservation.

    The caller must freeze and validate the maximum charges and complete context
    before constructing this adapter. It does not grant research readiness.
    """

    def __init__(
        self,
        store: CallStore,
        budget: Budget,
        *,
        case_id: str,
        bounds: dict[str, str],
        routes: dict[str, dict[str, Any]],
        sender: Callable[[dict[str, Any], str, float], dict[str, Any]],
        collect: bool = False,
    ) -> None:
        if not case_id or set(bounds) != {"actor", "B", "C", "judge"}:
            raise ValueError("A judged case requires four explicit cost bounds")
        if set(routes) != {"actor", "reviewer", "judge"}:
            raise ValueError("Actor, reviewer and judge transport routes are required")
        self.store = store
        self.budget = budget
        self.case_id = case_id
        self.bounds = bounds
        self.routes = routes
        self.collect = collect
        self.collector = ConcurrentCollector(
            store, budget, sender, max_workers=1, min_interval_seconds=3.2
        )
        with store.connect() as db:
            rows = db.execute(
                "SELECT role, maximum FROM reservations "
                "WHERE case_id=? AND retry_request_id IS NULL",
                (case_id,),
            ).fetchall()
        if rows:
            if len(rows) != 4 or {r: Decimal(v) for r, v in rows} != {
                r: Decimal(v) for r, v in bounds.items()
            }:
                raise ValueError("Existing case reservations differ from frozen bounds")
        elif collect:
            budget.reserve_case(case_id, bounds)

    def _job(self, role: str, record: dict[str, Any]) -> CollectionJob:
        """Validate a request and bind it to its exact reserved role."""
        if role not in {"actor", "B", "C", "judge"} or record.get("case_id") != self.case_id:
            raise ValueError("Request does not belong to this case")
        expected = "reviewer" if role in {"B", "C"} else role
        if record.get("role") != expected or (expected == "reviewer" and record.get("arm") != role):
            raise ValueError("Request role does not match its reservation")
        route = self.routes["reviewer" if role in {"B", "C"} else role]
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("model") != route["model"]:
            raise ValueError("Request payload differs from the frozen model")
        if payload.get("provider", {}).get("max_price") != route["max_price"]:
            raise ValueError("Request prices differ from the frozen route")
        messages = payload.get("messages")
        max_tokens = payload.get("max_tokens")
        if not isinstance(messages, list) or type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("Request lacks bounded messages or output")
        input_bound = len(canonical_json(payload).encode("utf-8")) + 100 * len(messages)
        if input_bound + max_tokens > route["context_length"]:
            raise ValueError("Request exceeds the frozen context bound")
        prices = route["max_price"]
        maximum = (
            (
                Decimal(input_bound) * Decimal(str(prices["prompt"]))
                + Decimal(max_tokens) * Decimal(str(prices["completion"]))
            )
            * Decimal("1.1")
            / Decimal(1_000_000)
        )
        if maximum > Decimal(self.bounds[role]):
            raise ValueError("Actual request exceeds the frozen role reservation")
        identity = request_identity(record)
        with self.store.connect() as db:
            pending = db.execute(
                "SELECT id FROM reservations WHERE retry_request_id=? "
                "AND case_id=? AND role=? AND status='reserved'",
                (identity, self.case_id, role),
            ).fetchall()
        if len(pending) > 1:
            raise ValueError("Multiple pending retry reservations are ambiguous")
        reservation = pending[0][0] if pending else f"{self.case_id}:{role}"
        return CollectionJob(
            record,
            reservation,
            ProviderConfig(
                "openrouter", route["model"], route["api_key_env"], route["timeout_seconds"]
            ),
            route["returned_provider"],
        )

    def send(self, role: str, record: dict[str, Any]) -> bytes:
        """Replay or collect one correctly owned role under its case reservation."""
        job = self._job(role, record)
        result = self.collector.collect([job], collect_missing=self.collect, maximum_new_requests=3)
        return result[request_identity(record)]

    def send_many(self, jobs: list[tuple[str, dict[str, Any]]]) -> dict[str, bytes]:
        """Collect independent reviews and judgment together, preserving dispatch order."""
        roles = [role for role, _ in jobs]
        if not jobs or len(set(roles)) != len(roles) or "actor" in roles:
            raise ValueError("Batch requires distinct non-actor roles")
        validated = [self._job(role, record) for role, record in jobs]
        self.collector.max_workers = min(3, len(validated))
        responses = self.collector.collect(validated, collect_missing=self.collect,
                                           maximum_new_requests=3 * len(validated))
        return {role: responses[request_identity(record)] for role, record in jobs}

    def release_unused(self) -> None:
        """Release unused role allowances after a case ends without those calls."""
        if not self.collect:
            return
        with self.store.connect() as db:
            unused = db.execute(
                "SELECT r.id FROM reservations r LEFT JOIN attempts a ON a.reservation_id=r.id "
                "WHERE r.case_id=? AND r.status='reserved' AND a.reservation_id IS NULL",
                (self.case_id,),
            ).fetchall()
        for (reservation_id,) in unused:
            self.budget.release(reservation_id)
