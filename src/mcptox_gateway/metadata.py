"""Deterministic description-integrity evidence; never an authorization decision."""

from __future__ import annotations

from collections import Counter
from typing import Any

from mcptox_gateway.dataset import DatasetError, sha256_text


REPORT_SCHEMA_VERSION = 1


def build_reference_catalog(references: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    """Record approved and explicitly unavailable identities for comparisons.

    Source extraction is evidence, not approval. Pending or missing reviews become
    ``unavailable`` rather than an unregistered identity, preserving the reason
    without exposing the unapproved text as a trusted baseline.

    Raises:
        DatasetError: If identities are missing/duplicated or approved text is absent.
    """

    catalog: list[dict[str, Any]] = []
    identities: set[tuple[object, object]] = set()
    for reference in references:
        identity = (reference.get("server_id"), reference.get("tool_name"))
        if None in identity or identity in identities:
            raise DatasetError(f"Invalid or ambiguous reference identity: {identity!r}.")
        identities.add(identity)
        approved = reference.get("approval_status") == "approved"
        if approved and not isinstance(reference.get("description"), str):
            raise DatasetError(f"Approved reference has no description: {identity!r}.")
        catalog.append(
            {
                "server_id": identity[0],
                "tool_name": identity[1],
                "description": reference.get("description") if approved else None,
                "availability": "approved" if approved else "unavailable",
                "missing_reason": None if approved else "reference_not_independently_approved",
                "source": reference.get("source"),
                "registry_version": version,
            }
        )
    return catalog


def compare_descriptions(
    inventory: list[dict[str, Any]], registry: list[dict[str, Any]], registry_version: str
) -> dict[str, Any]:
    """Compare every actor-visible description with exact approved source strings.

    ``inventory`` entries require ``server_id``, ``tool_name``, and a string
    ``observed_description``. The result is audit evidence only and has no allow
    or block field.
    """

    index: dict[tuple[object, object], dict[str, Any]] = {}
    for reference in registry:
        identity = (reference.get("server_id"), reference.get("tool_name"))
        if identity in index:
            raise DatasetError(f"Ambiguous registry identity: {identity!r}.")
        index[identity] = reference
    entries: list[dict[str, Any]] = []
    seen: set[tuple[object, object]] = set()
    for observed in inventory:
        identity = (observed.get("server_id"), observed.get("tool_name"))
        description = observed.get("observed_description")
        if None in identity or not isinstance(description, str):
            raise DatasetError(f"Cannot compare unavailable identity or description: {identity!r}.")
        if identity in seen:
            raise DatasetError(f"Ambiguous observed identity: {identity!r}.")
        seen.add(identity)
        reference = index.get(identity)
        if reference is None:
            status, approved_description, missing_reason = (
                "unregistered",
                None,
                "no_approved_reference",
            )
            expected_digest = None
            provenance = None
        elif reference.get("availability") == "unavailable":
            status, approved_description = "unavailable", None
            missing_reason = str(reference.get("missing_reason") or "reference_unavailable")
            expected_digest = None
            provenance = reference.get("source")
        else:
            approved_description = reference["description"]
            expected_digest = sha256_text(approved_description)
            status = "match" if description == approved_description else "mismatch"
            missing_reason = None
            provenance = reference.get("source")
        entries.append(
            {
                "server_id": identity[0],
                "tool_name": identity[1],
                "selected": bool(observed.get("selected", False)),
                "status": status,
                "approved_description": approved_description,
                "observed_description": description,
                "approved_sha256": expected_digest,
                "observed_sha256": sha256_text(description),
                "reference_version": registry_version,
                "reference_provenance": provenance,
                "missing_reason": missing_reason,
                "observed_source_span": observed.get("source_span"),
            }
        )
    entries.sort(key=lambda entry: (str(entry["server_id"]), str(entry["tool_name"])))
    counts = Counter(entry["status"] for entry in entries)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "registry_version": registry_version,
        "entries": entries,
        "coverage": {
            key: counts.get(key, 0) for key in ("match", "mismatch", "unregistered", "unavailable")
        },
    }
