"""Build lossless cases and unapproved reference candidates from pinned bytes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcptox_gateway.dataset import SNAPSHOT_COMMIT, build_cases, extract_clean_references
from mcptox_gateway.pipelines.common import stage_root, write_once
from mcptox_gateway.pipelines.download_data import download_sources


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()


def ingest(root: Path) -> None:
    """Verify sources and persist deterministic preparation data without overwrites."""
    download_sources(root)
    archive = root / "data/upstream" / SNAPSHOT_COMMIT / "response_all.json"
    write_once(root / "data/prepared/cases.jsonl", _jsonl(build_cases(archive)))
    write_once(
        root / "data/prepared/reference_candidates.jsonl", _jsonl(extract_clean_references(archive))
    )


if __name__ == "__main__":
    ingest(stage_root(__doc__))
