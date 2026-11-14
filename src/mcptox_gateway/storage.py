"""Canonical request identities and durable, offline call storage."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from threading import get_ident

if os.name == "nt":
    import msvcrt
else:
    import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON values deterministically without normalizing their text.

    Raises:
        ValueError: If values are non-finite or keys are not strings.
    """
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError("Unsupported JSON value")
    validate(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_exact_json(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and nonfinite numbers."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value: str) -> Any:
        raise ValueError(f"Nonfinite JSON number: {value}")
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    canonical_json(result)  # Also reject finite syntax that overflows to infinity.
    return result


def request_identity(record: dict[str, Any]) -> str:
    """Hash an explicit, versioned request record; reject incomplete identities.

    The exact application payload and all supplied versions are part of identity.
    C evidence must not be included in B records.
    """
    common = {"role", "payload", "provider", "model", "generation", "case_id", "repetition_id", "prompt_version", "reference_versions"}
    role = record.get("role")
    required = set(common)
    allowed = set(common) | {"format_version"}
    if role == "reviewer":
        required |= {"arm", "actor_response_hash", "proposal", "normalization_version", "clean_semantics", "clean_semantics_version"}
        allowed |= required
        if record.get("arm") == "C":
            required |= {"integrity_report", "registry_version", "hash_rules_version", "extraction_version", "report_schema_version", "serialization_version"}
            allowed |= required
        elif record.get("arm") != "B":
            raise ValueError("Reviewer arm must be B or C")
    elif role == "judge":
        required |= {"actor_response_hash", "proposal", "normalization_version", "judge_schema_version"}
        allowed |= required
    elif role not in {"actor", "qualification"}:
        raise ValueError("Unknown request role")
    if not required <= record.keys() or record.keys() - allowed:
        raise ValueError("Missing or unexpected identity fields")
    if any(record[key] is None for key in required - {"proposal"}):
        raise ValueError("Identity fields must be explicit")
    versioned = {"format_version": 1, **record}
    if versioned["format_version"] != 1:
        raise ValueError("Unsupported request format")
    return hashlib.sha256(canonical_json(versioned).encode("utf-8")).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class CallStore:
    """Store call lifecycle and immutable content-addressed raw artifacts locally.

    Args:
        root: Project results directory containing the database and raw files.
    """
    def __init__(self, root: Path | str, *, output_root: Path | str | None = None) -> None:
        # root identifies the project, never an individual run/output folder.
        self.root = Path(root).resolve()
        self.output_root = Path(output_root) if output_root is not None else None
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "calls.sqlite"
        self._collector_lock: Any = None
        self._batch_owner: int | None = None
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version != 1 and (version != 0 or tables):
                raise ValueError("Unsupported or unversioned call database; explicit migration required")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, record TEXT NOT NULL,
                status TEXT NOT NULL, request_path TEXT NOT NULL, request_hash TEXT NOT NULL,
                response_path TEXT, response_hash TEXT, response_metadata TEXT);
            CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, case_id TEXT NOT NULL, role TEXT NOT NULL,
                maximum TEXT NOT NULL, status TEXT NOT NULL, charge TEXT, charge_status TEXT,
                details TEXT NOT NULL, created_utc TEXT NOT NULL, format_version INTEGER NOT NULL DEFAULT 1, retry_request_id TEXT);
            CREATE TABLE IF NOT EXISTS attempts (request_id TEXT NOT NULL, number INTEGER NOT NULL,
                state TEXT NOT NULL, reservation_id TEXT NOT NULL UNIQUE, details TEXT NOT NULL, created_utc TEXT NOT NULL,
                format_version INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(request_id, number),
                FOREIGN KEY(request_id) REFERENCES requests(id), FOREIGN KEY(reservation_id) REFERENCES reservations(id));
            PRAGMA user_version=1;
            """)
        if self._acquire_collector():
            try:
                self._recover_abandoned()
            finally:
                self.close()

    def _acquire_collector(self) -> bool:
        if self._collector_lock is not None:
            return True
        lock = open(self.root / ".collector.lock", "a+b")
        try:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            return False
        self._collector_lock = lock
        return True

    def close(self) -> None:
        """Release collector ownership; a dispatched call then requires recovery."""
        if self._batch_owner is not None:
            raise ValueError("Only the batch coordinator may end a collection session")
        if self._collector_lock is not None:
            lock = self._collector_lock
            self._collector_lock = None
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def check_recovery(self) -> None:
        """Stop new work while a collector is live; recover any abandoned dispatch."""
        if self._batch_owner is not None:
            if self._batch_owner != get_ident():
                raise ValueError("Collection state belongs to the coordinator thread")
            return
        if self._collector_lock is not None or not self._acquire_collector():
            raise ValueError("Collector is active; new work must wait")
        try:
            self._recover_abandoned()
        finally:
            self.close()

    @contextmanager
    def collection_session(self) -> Iterator[None]:
        """Hold one process lock while a coordinator manages concurrent network I/O.

        Workers must only perform transport. The calling thread owns all claims,
        persistence and accounting; live batch requests are never crash-recovered.
        """
        if self._collector_lock is not None or not self._acquire_collector():
            raise ValueError("Another collector is active")
        try:
            self._recover_abandoned()
            self._batch_owner = get_ident()
            yield
        finally:
            self._batch_owner = None
            self.close()

    def _recover_abandoned(self) -> None:
        """Recover only while holding the exclusive process-lifetime lock."""
        with self.connect() as db:
            rows = db.execute("SELECT request_id, number, reservation_id FROM attempts WHERE state='dispatched'").fetchall()
        for request_id, number, reservation_id in rows:
            journals = list((self.root / "raw" / request_id).glob(f"journal-{number}-*.json"))
            if journals:
                self.recover_response(request_id, number)
            else:
                with self.connect() as db:
                    db.execute("UPDATE reservations SET status='unresolved', details=? WHERE id=? AND status='reserved'", (canonical_json({"format_version": 1, "reason": "abandoned dispatch; billing unknown"}), reservation_id))

    def connect(self) -> sqlite3.Connection:
        """Open a database connection with durability and foreign keys enabled."""
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _artifact(self, request_id: str, name: str, content: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(content).hexdigest()
        directory = self.root / "raw" / request_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}-{digest}.json"
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return target.relative_to(self.root).as_posix(), digest

    def plan(self, record: dict[str, Any], body: bytes) -> str:
        """Persist an exact application body and its identity before dispatch."""
        request_id = request_identity(record)
        # The application body must match the model-visible identity payload.
        body_text = body.decode("utf-8")
        parsed = parse_exact_json(body_text)
        payload = record["payload"]
        if isinstance(payload, str):
            if body_text != payload:
                raise ValueError("Request body differs from identity payload")
        elif body_text != canonical_json(payload):
            raise ValueError("Object payload requires canonical exact body bytes")
        if isinstance(parsed, dict) and any(key.lower() in {"authorization", "api_key", "headers"} for key in parsed):
            raise ValueError("Authentication and headers must not enter application artifacts")
        path, digest = self._artifact(request_id, "request", body)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO requests VALUES (?, ?, 'planned', ?, ?, NULL, NULL, NULL)",
                       (request_id, canonical_json({"format_version": 1, **record}), path, digest))
            row = db.execute("SELECT request_hash FROM requests WHERE id=?", (request_id,)).fetchone()
            if row[0] != digest:
                raise ValueError("Same identity has a different exact body")
        return request_id

    def claim(self, request_id: str, reservation_id: str) -> int:
        """Atomically claim a planned request; never reclaim ambiguous work."""
        self.check_recovery()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None or row[0] != "planned":
                raise ValueError("Request is missing, active, unresolved, or complete")
            reservation = db.execute("SELECT status, case_id, role, retry_request_id FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            record = json.loads(db.execute("SELECT record FROM requests WHERE id=?", (request_id,)).fetchone()[0])
            expected_role = record.get("arm") if record["role"] == "reviewer" else record["role"]
            if (reservation is None or (reservation[:3] != ("reserved", record["case_id"], expected_role) or reservation[3] not in (None, request_id))
                    or db.execute("SELECT 1 FROM attempts WHERE reservation_id=?", (reservation_id,)).fetchone()):
                raise ValueError("Reservation is missing, used, inactive, or for another case/role")
            if expected_role != "qualification":
                roles = {r[0] for r in db.execute("SELECT role FROM reservations WHERE case_id=? AND retry_request_id IS NULL", (record["case_id"],))}
                required_roles = {"actor", "B", "C"}
                if record["role"] == "judge" or record["reference_versions"].get("experiment_mode") == "judged-v1":
                    required_roles.add("judge")
                if not required_roles <= roles:
                    raise ValueError("Complete case bundle is required")
            if db.execute("SELECT 1 FROM reservations WHERE status='unresolved'").fetchone():
                raise ValueError("Unresolved billing blocks collection")
            number = db.execute("SELECT COUNT(*) FROM attempts WHERE request_id=?", (request_id,)).fetchone()[0] + 1
            if number > 3:
                raise ValueError("Retry limit reached")
            db.execute("INSERT INTO attempts (request_id, number, state, reservation_id, details, created_utc) VALUES (?, ?, 'claimed', ?, ?, ?)", (request_id, number, reservation_id, canonical_json({"format_version": 1}), _utc()))
            db.execute("UPDATE requests SET status='claimed' WHERE id=?", (request_id,))
            db.commit()
            return number

    def safe_pre_dispatch_failure(self, request_id: str, number: int, details: dict[str, Any]) -> None:
        """Record a verified pre-dispatch failure and allow up to two retries.

        Dispatched attempts cannot use this path because billing may be ambiguous.
        Each retry needs a fresh reservation.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state, reservation_id FROM attempts WHERE request_id=? AND number=?", (request_id, number)).fetchone()
            if row is None or row[0] != "claimed":
                raise ValueError("Only verified pre-dispatch failures are safe to retry")
            db.execute("UPDATE reservations SET status='settled', charge='0', charge_status='actual', details=? WHERE id=? AND status='reserved'", (canonical_json(details), row[1]))
            db.execute("UPDATE attempts SET state='safe_failure', details=? WHERE request_id=? AND number=?", (canonical_json(details), request_id, number))
            db.execute("UPDATE requests SET status=? WHERE id=?", ("planned" if number < 3 else "failed", request_id))
            db.commit()

    def retry_rate_limited(self, request_id: str, number: int, details: dict[str, Any]) -> str:
        """Record an explicit HTTP 429 and reserve one bounded retry atomically.

        Only use with a received OpenRouter 429 response and no auxiliary services.
        Network failures and other HTTP responses are not evidence of rejection.
        """
        if details.get("http_status") != 429 or number >= 3:
            raise ValueError("Only up to two explicit rate-limit retries are allowed")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            saved = db.execute("SELECT record FROM requests WHERE id=?", (request_id,)).fetchone()
            if saved is None:
                raise ValueError("Rate-limited request is missing")
            request = json.loads(saved[0])
            if request.get("provider") != "openrouter":
                raise ValueError("Rate-limit retry policy applies only to OpenRouter")
            payload = request.get("payload", {})
            if not isinstance(payload, dict) or any(
                key in payload for key in ("tools", "plugins", "web_search_options")
            ):
                raise ValueError("Auxiliary services make 429 billing uncertain")
            row = db.execute(
                "SELECT a.state, r.id, r.case_id, r.role, r.maximum, r.status "
                "FROM attempts a JOIN reservations r ON r.id=a.reservation_id "
                "WHERE a.request_id=? AND a.number=?", (request_id, number),
            ).fetchone()
            if row is None or row[0] != "dispatched" or row[5] != "reserved":
                raise ValueError("Only a dispatched reserved attempt may be rate-limited")
            retry_id = f"{row[1]}:429-retry-{number}"
            encoded = canonical_json({"format_version": 1, "reason": "openrouter_http_429", **details})
            db.execute("UPDATE reservations SET status='settled', charge='0', charge_status='actual', details=? WHERE id=?", (encoded, row[1]))
            db.execute("UPDATE attempts SET state='rate_limited', details=? WHERE request_id=? AND number=?", (encoded, request_id, number))
            db.execute("UPDATE requests SET status='planned' WHERE id=?", (request_id,))
            db.execute("INSERT INTO reservations (id, case_id, role, maximum, status, charge, charge_status, details, created_utc, retry_request_id) VALUES (?, ?, ?, ?, 'reserved', NULL, NULL, '{}', ?, ?)", (retry_id, row[2], row[3], row[4], _utc(), request_id))
            db.commit()
            return retry_id

    def reconcile_rejected_request(self, request_id: str, number: int) -> None:
        """Close a recorded HTTP rejection at its full reserved upper-bound cost.

        This does not retry the request or infer that the charge was zero.
        """
        self.check_recovery()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT a.state,r.id,r.status,r.maximum,r.details FROM attempts a "
                "JOIN reservations r ON r.id=a.reservation_id "
                "WHERE a.request_id=? AND a.number=?", (request_id, number),
            ).fetchone()
            if row is None or row[0] != "dispatched" or row[2] != "unresolved":
                raise ValueError("Expected an unresolved dispatched request")
            details = json.loads(row[4])
            status = (details.get("transport_details") or {}).get("http_status")
            if not isinstance(status, int) or not 400 <= status < 500 or status == 429:
                raise ValueError("Expected a recorded non-rate-limit HTTP rejection")
            evidence = canonical_json({"format_version": 1, "evidence_status": "provided",
                "evidence": {"reason": "received HTTP rejection, conservatively charged at reserved maximum",
                             "prior_details": details}})
            db.execute("UPDATE reservations SET status='settled',charge=maximum,charge_status='upper-bound',details=? WHERE id=?", (evidence, row[1]))
            db.execute("UPDATE attempts SET state='rejected',details=? WHERE request_id=? AND number=?", (evidence, request_id, number))
            db.execute("UPDATE requests SET status='failed' WHERE id=?", (request_id,))
            db.commit()

    def dispatched(self, request_id: str, number: int) -> None:
        """Mark dispatch before transport; a crash after this is ambiguous."""
        if self._batch_owner is not None:
            self.check_recovery()
        else:
            if self._collector_lock is not None or not self._acquire_collector():
                raise ValueError("Another collector owns dispatch")
            self._recover_abandoned()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM reservations WHERE status='unresolved'").fetchone():
                raise ValueError("Unresolved billing blocks dispatch")
            active = db.execute("SELECT 1 FROM attempts a JOIN reservations r ON r.id=a.reservation_id WHERE a.request_id=? AND a.number=? AND a.state='claimed' AND r.status='reserved'", (request_id, number)).fetchone()
            if not active:
                raise ValueError("Attempt has no active reservation")
            record = json.loads(db.execute("SELECT record FROM requests WHERE id=?", (request_id,)).fetchone()[0])
            if record["role"] != "qualification":
                bundle = dict(db.execute("SELECT role, status FROM reservations WHERE case_id=? AND retry_request_id IS NULL", (record["case_id"],)).fetchall())
                required_roles = {"actor", "B", "C"}
                if record["role"] == "judge" or record["reference_versions"].get("experiment_mode") == "judged-v1":
                    required_roles.add("judge")
                if not required_roles <= bundle.keys():
                    raise ValueError("Complete case bundle is required")
                if record["role"] == "actor" and any(bundle[role] != "reserved" for role in ("B", "C")):
                    raise ValueError("Both reviewer allowances must be reserved before actor dispatch")
                if record["role"] == "actor" and "judge" in required_roles and bundle["judge"] != "reserved":
                    raise ValueError("Judge allowance must be reserved before actor dispatch")
            cursor = db.execute("UPDATE attempts SET state='dispatched' WHERE request_id=? AND number=? AND state='claimed'", (request_id, number))
            if cursor.rowcount != 1:
                raise ValueError("Attempt is not claimed")
            db.execute("UPDATE requests SET status='dispatched' WHERE id=?", (request_id,))
            db.commit()

    def save_response(self, request_id: str, number: int, response: bytes, metadata: dict[str, Any]) -> None:
        """Durably save an exact provider response before completion."""
        with self.connect() as db:
            row = db.execute("SELECT state FROM attempts WHERE request_id=? AND number=?", (request_id, number)).fetchone()
        if row != ("dispatched",):
            raise ValueError("Attempt is not dispatched")
        provenance = {"format_version": 1, "returned_model": None, "returned_provider": None,
                      "provider_request_id": None, "usage": None, "inference_timing": None,
                      "missing_status": "not supplied", **metadata}
        if provenance["format_version"] != 1:
            raise ValueError("Unsupported response provenance version")
        journal = {"format_version": 1, "request_id": request_id, "attempt": number,
                   "response_base64": base64.b64encode(response).decode("ascii"), "metadata": provenance}
        # A single fsynced attempt-bound journal contains everything needed to recover.
        self._artifact(request_id, f"journal-{number}", canonical_json(journal).encode("utf-8"))
        self.recover_response(request_id, number)
        if self._batch_owner is None:
            self.close()

    def recover_response(self, request_id: str, number: int) -> None:
        """Recover an attempt-bound response journal without redispatching."""
        directory = self.root / "raw" / request_id
        journals = list(directory.glob(f"journal-{number}-*.json"))
        if len(journals) != 1:
            raise ValueError("Response journal missing or ambiguous")
        journal_bytes = journals[0].read_bytes()
        if journals[0].stem.rsplit("-", 1)[-1] != hashlib.sha256(journal_bytes).hexdigest():
            raise ValueError("Corrupt response journal")
        journal = parse_exact_json(journal_bytes.decode("utf-8"))
        if journal["format_version"] != 1 or journal["request_id"] != request_id or journal["attempt"] != number:
            raise ValueError("Journal identity mismatch")
        response = base64.b64decode(journal["response_base64"], validate=True)
        metadata = journal["metadata"]
        path, digest = self._artifact(request_id, "response", response)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("UPDATE attempts SET state='response_saved', details=? WHERE request_id=? AND number=? AND state='dispatched'", (canonical_json(metadata), request_id, number))
            if cursor.rowcount != 1:
                raise ValueError("Attempt is not dispatched")
            db.execute("UPDATE requests SET status='response_saved', response_path=?, response_hash=?, response_metadata=? WHERE id=?", (path, digest, canonical_json(metadata), request_id))
            db.commit()

    def complete(self, request_id: str) -> None:
        """Complete or recover a saved response after verifying its artifact."""
        self.response(request_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            billing = db.execute("SELECT r.status FROM attempts a JOIN reservations r ON r.id=a.reservation_id WHERE a.request_id=? AND a.state IN ('response_saved', 'completed')", (request_id,)).fetchone()
            if billing is None or billing[0] != "settled":
                raise ValueError("Response billing must be reconciled before completion")
            cursor = db.execute("UPDATE requests SET status='completed' WHERE id=? AND status IN ('response_saved', 'completed')", (request_id,))
            if cursor.rowcount != 1:
                raise ValueError("Response has not been saved")
            db.execute("UPDATE attempts SET state='completed' WHERE request_id=? AND state='response_saved'", (request_id,))
            db.commit()

    def response(self, request_id: str) -> bytes:
        """Read and verify saved response bytes; missing/corrupt files are errors."""
        with self.connect() as db:
            row = db.execute("SELECT status, response_path, response_hash FROM requests WHERE id=?", (request_id,)).fetchone()
        if row is None or row[0] not in {"response_saved", "completed"}:
            raise ValueError("No saved response")
        data = (self.root / row[1]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row[2]:
            raise ValueError("Corrupt response artifact")
        return data
