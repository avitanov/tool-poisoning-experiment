"""Bounded network concurrency with one durable accounting coordinator."""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Callable

from mcptox_gateway.budget import Budget
from mcptox_gateway.provider import OpenRouterHTTPError, ProviderConfig
from mcptox_gateway.storage import CallStore, canonical_json, request_identity


@dataclass(frozen=True)
class CollectionJob:
    """One immutable request with its already-reserved maximum charge."""

    record: dict[str, Any]
    reservation_id: str
    config: ProviderConfig
    returned_provider: str


class ConcurrentCollector:
    """Run independent requests concurrently; never run actor/reviewer dependencies together.

    The caller prepares case bundles and dependency order. Only network transport
    runs in workers. One coordinator handles cache, claims, billing and persistence.
    A failure stops new dispatches but drains and saves other in-flight responses.
    Explicit HTTP 429 rejections can be retried twice with the same fixed route.
    Ambiguous failures stop; providers, prompts and model outputs never change.
    """

    def __init__(
        self,
        store: CallStore,
        budget: Budget,
        sender: Callable[[dict[str, Any], str, float], dict[str, Any]],
        *,
        max_workers: int = 4,
        min_interval_seconds: float = 3.2,
    ) -> None:
        if (
            not 1 <= max_workers <= 16
            or not math.isfinite(min_interval_seconds)
            or min_interval_seconds < 0
        ):
            raise ValueError("Invalid worker count or dispatch interval")
        self.store = store
        self.budget = budget
        self.sender = sender
        self.max_workers = max_workers
        self.interval = min_interval_seconds
        self._next_dispatch = 0.0

    def _validate_route(self, job: CollectionJob, response: dict[str, Any]) -> None:
        if (
            response.get("model") not in {job.config.model, job.config.model.removesuffix(":free")}
            or response.get("provider") != job.returned_provider
        ):
            raise ValueError("Returned model/provider drift; response retained")

    def _save(self, job: CollectionJob, identity: str, attempt: int, response: dict) -> bytes:
        raw = canonical_json(response).encode()
        self.store.save_response(
            identity,
            attempt,
            raw,
            {
                "returned_model": response.get("model"),
                "returned_provider": response.get("provider"),
                "provider_request_id": response.get("id"),
                "usage": response.get("usage"),
            },
        )
        usage = response.get("usage") or {}
        if "cost" not in usage:
            raise ValueError("Returned billing is missing")
        cost = Decimal(str(usage["cost"]))
        self.budget.settle(
            job.reservation_id,
            str(cost),
            "actual",
            {"usage": usage, "response_id": response.get("id")},
        )
        self.store.complete(identity)
        self._validate_route(job, response)
        return raw

    def _record_error(self, job: CollectionJob, error: Exception) -> None:
        with self.store.connect() as db:
            status = db.execute(
                "SELECT status FROM reservations WHERE id=?", (job.reservation_id,)
            ).fetchone()
        if status == ("reserved",):
            self.budget.unresolved(
                job.reservation_id,
                {
                    "error_type": type(error).__name__,
                    "transport_details": getattr(error, "details", None),
                },
            )

    def collect(
        self,
        jobs: list[CollectionJob],
        *,
        collect_missing: bool = False,
        maximum_new_requests: int | None = None,
    ) -> dict[str, bytes]:
        """Replay by default, or collect only missing reserved requests.

        Args:
            jobs: Independent requests with complete case reservations.
            collect_missing: Explicit permission to use transport for cache misses.
            maximum_new_requests: Remaining provider quota; applies to unique misses.

        Raises:
            ValueError: On unavailable cache, invalid configuration, or a stopped batch.
        """
        unique: dict[str, CollectionJob] = {}
        for job in jobs:
            identity = request_identity(job.record)
            if identity in unique and unique[identity] != job:
                raise ValueError("Duplicate request has conflicting reservation/configuration")
            unique[identity] = job
        results, missing = {}, deque()
        for identity, job in unique.items():
            with self.store.connect() as db:
                row = db.execute("SELECT status FROM requests WHERE id=?", (identity,)).fetchone()
            if row == ("completed",):
                results[identity] = self.store.response(identity)
                self._validate_route(job, json.loads(results[identity]))
            elif row is not None and row != ("planned",):
                raise ValueError(
                    "Prior incomplete attempt needs reconciliation; no automatic retry"
                )
            else:
                missing.append((identity, job))
        if not missing:
            return results
        if not collect_missing:
            raise ValueError(f"Missing {len(missing)} saved calls; replay cannot use network")
        if maximum_new_requests is not None and len(missing) > maximum_new_requests:
            raise ValueError("Missing calls exceed remaining request quota")
        credentials = {}
        for identity, job in missing:
            payload = job.record["payload"]
            routing = payload.get("provider", {})
            if (
                payload.get("model") != job.config.model
                or job.record["model"] != job.config.model
                or job.record["provider"] != job.config.provider
                or len(routing.get("only", [])) != 1
                or routing.get("allow_fallbacks") is not False
                or job.config.timeout_seconds <= 0
            ):
                raise ValueError(
                    "Requests must pin one matching model/provider with fallback disabled"
                )
            if job.config.model.endswith(":free") and self.interval < 3.2:
                raise ValueError("Free requests require at least 3.2 seconds between dispatches")
            credentials[identity] = os.environ.get(job.config.api_key_env)
            if not credentials[identity]:
                raise ValueError("Credential environment variable is missing")
            self.store.plan(job.record, canonical_json(payload).encode())

        failures: list[str] = []
        dispatched_count = 0
        active: dict[Future, tuple[str, CollectionJob, int]] = {}
        with (
            self.store.collection_session(),
            ThreadPoolExecutor(max_workers=self.max_workers) as pool,
        ):
            while active or (missing and not failures):
                # Handle all observed failures before permitting another dispatch.
                for future in [future for future in active if future.done()]:
                    identity, job, attempt = active.pop(future)
                    try:
                        results[identity] = self._save(job, identity, attempt, future.result())
                    except OpenRouterHTTPError as error:
                        if (
                            error.details.get("http_status") == 429
                            and attempt < 3
                            and (
                                maximum_new_requests is None
                                or dispatched_count < maximum_new_requests
                            )
                        ):
                            try:
                                retry_id = self.store.retry_rate_limited(
                                    identity, attempt, error.details
                                )
                                missing.appendleft(
                                    (identity, replace(job, reservation_id=retry_id))
                                )
                                try:
                                    delay = float(error.details.get("retry_after") or 1)
                                except (TypeError, ValueError):
                                    delay = 1.0
                                delay = delay if math.isfinite(delay) else 1.0
                                self._next_dispatch = max(
                                    self._next_dispatch, time.monotonic() + min(max(delay, 0), 60)
                                )
                            except Exception as retry_error:
                                self._record_error(job, retry_error)
                                failures.append(f"{identity}: {type(retry_error).__name__}")
                        else:
                            self._record_error(job, error)
                            failures.append(f"{identity}: {type(error).__name__}")
                    except Exception as error:
                        self._record_error(job, error)
                        failures.append(f"{identity}: {type(error).__name__}")
                if not failures and missing and len(active) < self.max_workers:
                    if (
                        maximum_new_requests is not None
                        and dispatched_count >= maximum_new_requests
                    ):
                        failures.append("maximum new request quota exhausted")
                        continue
                    delay = self._next_dispatch - time.monotonic()
                    if delay <= 0:
                        identity, job = missing.popleft()
                        try:
                            attempt = self.store.claim(identity, job.reservation_id)
                            self.store.dispatched(identity, attempt)
                            dispatched_count += 1
                            future = pool.submit(
                                self.sender,
                                job.record["payload"],
                                credentials[identity],
                                job.config.timeout_seconds,
                            )
                            active[future] = (identity, job, attempt)
                            self._next_dispatch = time.monotonic() + self.interval
                        except Exception as error:
                            self._record_error(job, error)
                            failures.append(f"{identity}: {type(error).__name__}")
                        continue
                if active:
                    timeout = None
                    if missing and not failures and len(active) < self.max_workers:
                        timeout = max(0, self._next_dispatch - time.monotonic())
                    wait(active, timeout=timeout, return_when=FIRST_COMPLETED)
                elif missing and not failures:
                    time.sleep(max(0, self._next_dispatch - time.monotonic()))
        if failures:
            raise ValueError(
                "Collection stopped; in-flight responses retained: " + "; ".join(failures)
            )
        return results
