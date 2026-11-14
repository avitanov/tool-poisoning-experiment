"""Persistent project-wide reservations and conservative spend accounting."""
from __future__ import annotations

import sqlite3
from decimal import Decimal, InvalidOperation
from typing import Any

from mcptox_gateway.storage import CallStore, _utc, canonical_json


def _money(value: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("Money must be a decimal string")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid money") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Money must be finite and nonnegative")
    return amount


def maximum_charge(input_cost: str, output_cost: str, reasoning_cost: str, fees: str, *, reasoning_in_output: bool) -> str:
    """Sum conservative charge components without charging included reasoning twice."""
    values = [_money(v) for v in (input_cost, output_cost, reasoning_cost, fees)]
    if reasoning_in_output and values[2] != 0:
        raise ValueError("Included reasoning must not also be separately charged")
    return str(sum(values, Decimal(0)))


class Budget:
    """Maintain one immutable project cap and persistent reservation ledger."""
    def __init__(self, store: CallStore, ceiling: str = "20", operational_stop: str = "19.50") -> None:
        self.store = store
        cap, stop = _money(ceiling), _money(operational_stop)
        if cap > Decimal("20") or stop > cap or stop > Decimal("19.50"):
            raise ValueError("Invalid project ceiling or operational stop")
        with store.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS budget_config (singleton INTEGER PRIMARY KEY CHECK(singleton=1), ceiling TEXT NOT NULL, stop TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO budget_config VALUES (1, ?, ?)", (str(cap), str(stop)))
            row = db.execute("SELECT ceiling, stop FROM budget_config WHERE singleton=1").fetchone()
            if tuple(map(Decimal, row)) != (cap, stop):
                raise ValueError("Budget configuration differs from existing ledger")

    @staticmethod
    def lower_legacy_limit(store: CallStore) -> None:
        """Lower the former project limit, preserving every reservation and charge.

        The change is transactional and recorded in the ledger. Unresolved or
        active reservations prevent migration until their billing is reconciled.
        """
        store.check_recovery()
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT ceiling, stop FROM budget_config WHERE singleton=1").fetchone()
            if row is None or tuple(map(Decimal, row)) != (Decimal("15"), Decimal("14.50")):
                raise ValueError("Expected the former $15/$14.50 ledger configuration")
            if db.execute("SELECT 1 FROM reservations WHERE status IN ('reserved', 'unresolved')").fetchone():
                raise ValueError("Active or unresolved reservations prevent migration")
            charges = db.execute("SELECT charge FROM reservations WHERE status='settled'").fetchall()
            total = sum((Decimal(charge[0]) for charge in charges), Decimal(0))
            if total > Decimal("9.50"):
                raise ValueError("Historical charges exceed the new operational stop")
            db.execute("CREATE TABLE IF NOT EXISTS budget_amendments (changed_utc TEXT NOT NULL, old_ceiling TEXT NOT NULL, old_stop TEXT NOT NULL, new_ceiling TEXT NOT NULL, new_stop TEXT NOT NULL, settled_total TEXT NOT NULL)")
            db.execute("INSERT INTO budget_amendments VALUES (?, ?, ?, ?, ?, ?)", (_utc(), row[0], row[1], "10", "9.50", str(total)))
            db.execute("UPDATE budget_config SET ceiling='10', stop='9.50' WHERE singleton=1")
            db.commit()

    @staticmethod
    def raise_to_twenty(store: CallStore) -> None:
        """Record the authorized ceiling increase without resetting the ledger."""
        store.check_recovery()
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT ceiling, stop FROM budget_config WHERE singleton=1").fetchone()
            if row is None or tuple(map(Decimal, row)) != (Decimal("10"), Decimal("9.50")):
                raise ValueError("Expected the former $10/$9.50 ledger configuration")
            if db.execute("SELECT 1 FROM reservations WHERE status IN ('reserved', 'unresolved')").fetchone():
                raise ValueError("Active or unresolved reservations prevent migration")
            charges = db.execute("SELECT charge FROM reservations WHERE status='settled'").fetchall()
            total = sum((Decimal(charge[0]) for charge in charges), Decimal(0))
            db.execute("CREATE TABLE IF NOT EXISTS budget_amendments (changed_utc TEXT NOT NULL, old_ceiling TEXT NOT NULL, old_stop TEXT NOT NULL, new_ceiling TEXT NOT NULL, new_stop TEXT NOT NULL, settled_total TEXT NOT NULL)")
            db.execute("INSERT INTO budget_amendments VALUES (?, ?, ?, ?, ?, ?)", (_utc(), row[0], row[1], "20", "19.50", str(total)))
            db.execute("UPDATE budget_config SET ceiling='20', stop='19.50' WHERE singleton=1")
            db.commit()

    def _totals(self, db: sqlite3.Connection) -> tuple[Decimal, Decimal]:
        rows = db.execute("SELECT maximum, status, charge FROM reservations").fetchall()
        settled = sum((Decimal(row[2]) for row in rows if row[1] == "settled"), Decimal(0))
        reserved = sum((Decimal(row[0]) for row in rows if row[1] in {"reserved", "unresolved"}), Decimal(0))
        return settled, reserved

    def reserve_case(self, case_id: str, amounts: dict[str, str]) -> dict[str, str]:
        """Atomically reserve actor, reviewers and, for judged cases, the judge."""
        if set(amounts) not in ({"actor", "B", "C"}, {"actor", "B", "C", "judge"}) or not case_id:
            raise ValueError("A case requires actor, B and C bounds, optionally with judge")
        self.store.check_recovery()
        parsed = {role: _money(value) for role, value in amounts.items()}
        ids = {role: f"{case_id}:{role}" for role in parsed}
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM reservations WHERE status='unresolved'").fetchone():
                raise ValueError("Unresolved billing blocks new reservations")
            settled, reserved = self._totals(db)
            stop = Decimal(db.execute("SELECT stop FROM budget_config WHERE singleton=1").fetchone()[0])
            if settled + reserved + sum(parsed.values(), Decimal(0)) > stop:
                raise ValueError("Reservation exceeds operational stop")
            for role, amount in parsed.items():
                db.execute("INSERT INTO reservations (id, case_id, role, maximum, status, charge, charge_status, details, created_utc) VALUES (?, ?, ?, ?, 'reserved', NULL, NULL, '{}', ?)", (ids[role], case_id, role, str(amount), _utc()))
            db.commit()
        return ids

    def reserve_attempt(self, reservation_id: str, maximum: str, *, case_id: str, role: str, request_id: str | None = None) -> None:
        """Reserve an additional explicitly identified qualification or safe retry."""
        self.store.check_recovery()
        amount = _money(maximum)
        if not case_id or role not in {"actor", "B", "C", "judge", "qualification"}:
            raise ValueError("Additional reservation requires case and role")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM reservations WHERE status='unresolved'").fetchone():
                raise ValueError("Unresolved billing blocks new reservations")
            settled, reserved = self._totals(db)
            stop = Decimal(db.execute("SELECT stop FROM budget_config WHERE singleton=1").fetchone()[0])
            if role != "qualification":
                roles = {r[0] for r in db.execute("SELECT role FROM reservations WHERE case_id=? AND retry_request_id IS NULL", (case_id,))}
                prior = db.execute("SELECT a.state, a.number, r.case_id, r.role FROM attempts a JOIN reservations r ON r.id=a.reservation_id WHERE a.request_id=? ORDER BY a.number DESC LIMIT 1", (request_id,)).fetchone()
                pending = db.execute("SELECT 1 FROM reservations WHERE retry_request_id=? AND status='reserved'", (request_id,)).fetchone()
                if not ({"actor", "B", "C", "judge"} if role == "judge" else {"actor", "B", "C"}) <= roles or not prior or prior[0] != "safe_failure" or prior[1] >= 3 or prior[2:] != (case_id, role) or pending:
                    raise ValueError("Retry requires a case bundle and a documented safe prior failure")
            if settled + reserved + amount > stop:
                raise ValueError("Reservation exceeds operational stop")
            db.execute("INSERT INTO reservations (id, case_id, role, maximum, status, charge, charge_status, details, created_utc, retry_request_id) VALUES (?, ?, ?, ?, 'reserved', NULL, NULL, '{}', ?, ?)", (reservation_id, case_id, role, str(amount), _utc(), request_id))
            db.commit()

    def release(self, reservation_id: str) -> None:
        """Release an unused reservation only if no attempt has claimed it."""
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT 1 FROM attempts WHERE reservation_id=?", (reservation_id,)).fetchone()
            if used or db.execute("UPDATE reservations SET status='released' WHERE id=? AND status='reserved'", (reservation_id,)).rowcount != 1:
                raise ValueError("Reservation is used or not releasable")
            db.commit()

    def unresolved(self, reservation_id: str, details: dict[str, Any]) -> None:
        """Retain the full allowance for a possibly billed attempt."""
        with self.store.connect() as db:
            cursor = db.execute("UPDATE reservations SET status='unresolved', details=? WHERE id=? AND status='reserved'", (canonical_json(details), reservation_id))
            if cursor.rowcount != 1:
                raise ValueError("Reservation is not active")

    def settle(self, reservation_id: str, charge: str, charge_status: str, details: dict[str, Any]) -> None:
        """Record a charge once, keeping its original billing evidence."""
        amount = _money(charge)
        if charge_status not in {"actual", "estimated", "upper-bound"}:
            raise ValueError("Unknown charge status")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT maximum, status, charge, charge_status, details FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            encoded = canonical_json({"format_version": 1, "evidence_status": "provided" if details else "missing", "evidence": details})
            if row is None:
                raise ValueError("Unknown reservation")
            if row[1] == "settled":
                if (Decimal(row[2]), row[3], row[4]) != (amount, charge_status, encoded):
                    raise ValueError("Conflicting duplicate settlement")
                db.commit()
                return
            if row[1] not in {"reserved", "unresolved"} or amount > Decimal(row[0]):
                raise ValueError("Charge exceeds reservation or reservation inactive")
            db.execute("UPDATE reservations SET status='settled', charge=?, charge_status=?, details=? WHERE id=?", (str(amount), charge_status, encoded, reservation_id))
            db.commit()

    def summary(self) -> dict[str, str]:
        """Return decimal-string historical charges, reservations and allowance."""
        with self.store.connect() as db:
            settled, reserved = self._totals(db)
            cap, stop = map(Decimal, db.execute("SELECT ceiling, stop FROM budget_config WHERE singleton=1").fetchone())
        return {"settled": str(settled), "unresolved_reservations": str(reserved), "ceiling_remaining": str(cap - settled - reserved), "operational_remaining": str(stop - settled - reserved)}

