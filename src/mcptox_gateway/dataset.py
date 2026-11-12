"""Parse the pinned MCPTox archive without executing any upstream material."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


SNAPSHOT_COMMIT = "f85189f9ad12504c197c7f920ab818a40657b1fa"
RESPONSE_FILENAME = "response_all.json"
DESCRIPTION_PATTERN = re.compile(
    r"^Tool: (?P<tool>[^\n]+)\nDescription: (?P<description>.*?)(?=\nArguments:)",
    re.DOTALL | re.MULTILINE,
)


class DatasetError(ValueError):
    """Raised when the immutable archive has an unexpected or ambiguous shape."""


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of an exact UTF-8 string without normalization."""

    return sha256_bytes(value.encode("utf-8"))


def load_archive(path: Path | str) -> dict[str, Any]:
    """Load the response archive as data only.

    Args:
        path: Path to the pinned ``response_all.json`` bytes.

    Returns:
        The archive's JSON object retaining all original field spellings.

    Raises:
        DatasetError: If the JSON object or required server inventory is absent.
    """

    archive_path = Path(path)
    try:
        raw = archive_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"Cannot load response archive {archive_path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("servers"), dict):
        raise DatasetError("response archive must contain a 'servers' object.")
    return payload


def verify_manifest(manifest_path: Path | str, workspace_root: Path | str = ".") -> None:
    """Verify byte counts, SHA-256 digests, and Git blob IDs from a manifest.

    Args:
        manifest_path: JSON manifest produced during immutable source acquisition.
        workspace_root: Root to resolve the manifest's portable file paths.

    Raises:
        DatasetError: If any listed file differs from the recorded immutable bytes.
    """

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DatasetError(f"Invalid source manifest: {error}") from error
    for entry in files:
        file_path = Path(workspace_root) / entry["path"]
        try:
            data = file_path.read_bytes()
        except OSError as error:
            raise DatasetError(f"Cannot read manifest file {file_path}: {error}") from error
        if len(data) != entry["bytes"] or sha256_bytes(data) != entry["sha256"]:
            raise DatasetError(f"Immutable source verification failed for {entry['path']}.")
        header = f"blob {len(data)}\0".encode("ascii")
        blob_sha1 = hashlib.sha1(header + data).hexdigest()
        if blob_sha1 != entry["git_blob_sha1"]:
            raise DatasetError(f"Git blob verification failed for {entry['path']}.")


def _description_matches(prompt: str) -> Iterator[re.Match[str]]:
    """Validate each tool block once, retaining exact description spans.

    Bound matching by the next tool header so a malformed description cannot
    consume a later tool. Ambiguous headers are errors, never silently repaired.
    """

    if not isinstance(prompt, str):
        raise DatasetError("Tool prompt must be a string.")
    headers = list(re.finditer(r"^Tool:", prompt, re.MULTILINE))
    if not headers:
        raise DatasetError("Tool prompt contains no tool declarations.")
    names: set[str] = set()
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(prompt)
        match = DESCRIPTION_PATTERN.match(prompt, header.start(), end)
        if match is None:
            raise DatasetError(f"Malformed tool block at character {header.start()}.")
        name = match.group("tool")
        if name in names:
            raise DatasetError(f"Ambiguous duplicate tool {name!r}.")
        names.add(name)
        yield match


def extract_descriptions(prompt: str) -> list[dict[str, str]]:
    """Extract exact tool descriptions from a released prompt.

    The parser intentionally preserves whitespace and Unicode inside descriptions.
    Raises:
        DatasetError: If tool declarations are absent, malformed, or duplicated.
    """

    return [match.groupdict() for match in _description_matches(prompt)]


def actor_visible_inventory(
    case: dict[str, Any], selected_tool: str | None = None
) -> list[dict[str, Any]]:
    """Return actor-visible description evidence from one exact released prompt.

    Args:
        case: A row returned by :func:`build_cases`.
        selected_tool: An optional untrusted proposed tool name for display only.

    Returns:
        Source-ordered inventory entries with exact strings and source spans.

    Raises:
        DatasetError: If the prompt is missing, malformed, or has duplicate tools.
    """

    prompt = case.get("system")
    if not isinstance(prompt, str):
        raise DatasetError("Case has no actor-visible system prompt.")
    server_id = case.get("server_id")
    inventory: list[dict[str, Any]] = []
    for match in _description_matches(prompt):
        name = match.group("tool")
        inventory.append(
            {
                "server_id": server_id,
                "tool_name": name,
                "observed_description": match.group("description"),
                "selected": name == selected_tool,
                "source_span": {
                    "start": match.start("description"),
                    "end": match.end("description"),
                },
            }
        )
    return inventory


def build_cases(archive_path: Path | str) -> list[dict[str, Any]]:
    """Build lossless case rows and deterministic scenario-family identifiers.

    A family groups records with the same released server identity and legitimate
    query. Source positions, not repeated upstream numeric IDs, distinguish cases.
    """

    path = Path(archive_path)
    archive = load_archive(path)
    snapshot_hash = sha256_bytes(path.read_bytes())
    rows: list[dict[str, Any]] = []
    for server_position, (server_key, server) in enumerate(archive["servers"].items()):
        if not isinstance(server, dict):
            raise DatasetError(f"Server {server_key!r} is not an object.")
        instances = server.get("malicious_instance")
        if not isinstance(instances, list):
            raise DatasetError(f"Server {server_key!r} has no malicious_instance list.")
        for instance_position, instance in enumerate(instances):
            if not isinstance(instance, dict) or not isinstance(instance.get("datas"), list):
                raise DatasetError(
                    f"Invalid malicious instance at {server_key}[{instance_position}]."
                )
            metadata = instance.get("metadata")
            if not isinstance(metadata, dict):
                raise DatasetError(f"Missing metadata at {server_key}[{instance_position}].")
            for data_position, data in enumerate(instance["datas"]):
                if not isinstance(data, dict):
                    raise DatasetError("Instance data must be an object.")
                for key in ("id", "query", "system"):
                    if key not in data:
                        raise DatasetError(
                            f"Missing data field {key!r} at {server_key}[{instance_position}]."
                        )
                query = data["query"]
                system = data["system"]
                if not isinstance(query, str) or not isinstance(system, str):
                    raise DatasetError("Released query and system prompt must be strings.")
                source_position = f"{server_position}:{instance_position}:{data_position}"
                case_key = f"{snapshot_hash}\x1f{source_position}"
                family_key = "\x1f".join(
                    (str(server.get("server_id")), str(server.get("server_name")), query)
                )
                case_id = f"case_{sha256_text(case_key)[:20]}"
                family_id = f"family_{sha256_text(family_key)[:20]}"
                rows.append(
                    {
                        "case_id": case_id,
                        "family_id": family_id,
                        "snapshot_sha256": snapshot_hash,
                        "source_position": source_position,
                        "server_key": server_key,
                        "server_id": server.get("server_id"),
                        "server_name": server.get("server_name"),
                        "original_numeric_id": data["id"],
                        "query": query,
                        "system": system,
                        "prompt_sha256": sha256_text(system),
                        "variant": {
                            "instance_position": instance_position,
                            "data_position": data_position,
                        },
                        "template": metadata.get("paradigm"),
                        "security risk": metadata.get("security risk"),
                        "wrong_data": instance.get("wrong_data"),
                        "poisoned_tool": instance.get("poisoned_tool"),
                    }
                )
    if len({row["case_id"] for row in rows}) != len(rows):
        raise DatasetError("Case IDs are not unique.")
    return rows


def extract_clean_references(archive_path: Path | str) -> list[dict[str, Any]]:
    """Extract clean-prompt descriptions as unapproved source evidence.

    This extraction deliberately does not convert source text into trusted records;
    an independent reviewer must approve a reference before it enters a registry.
    """

    archive = load_archive(archive_path)
    references: list[dict[str, Any]] = []
    for server_position, (server_key, server) in enumerate(archive["servers"].items()):
        if not isinstance(server, dict):
            raise DatasetError(f"Server {server_key!r} is not an object.")
        prompt = server.get("clean_system_promot")
        if not isinstance(prompt, str):
            raise DatasetError(f"Server {server_key!r} has no clean_system_promot string.")
        descriptions = extract_descriptions(prompt)
        for item in descriptions:
            tool_name = item["tool"]
            description = item["description"]
            references.append(
                {
                    "server_id": server.get("server_id"),
                    "server_name": server.get("server_name"),
                    "tool_name": tool_name,
                    "description": description,
                    "description_sha256": sha256_text(description),
                    "source": {
                        "file": RESPONSE_FILENAME,
                        "server_position": server_position,
                        "field": "clean_system_promot",
                    },
                    "approval_status": "pending_independent_review",
                    "missing_reason": None,
                }
            )
    return references


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> None:
    """Write JSONL deterministically, retaining non-ASCII source text."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    destination.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
    )
