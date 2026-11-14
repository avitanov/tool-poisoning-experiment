"""Reconstruct the existing proposed subset without changing selection semantics."""

from __future__ import annotations

import json
from pathlib import Path

from mcptox_gateway.dataset import SNAPSHOT_COMMIT, load_archive, sha256_bytes
from mcptox_gateway.pipelines.common import encode_json, stage_root, verify_inputs, write_once
from mcptox_gateway.pipelines.download_data import download_sources
from mcptox_gateway.sampling import select_attack_samples, select_controls
from mcptox_gateway.settings import load_settings


def create_splits(root: Path) -> None:
    """Select fixed pilot/main plans from verified cases and the configured seed.

    Control plans remain pending review; this stage does not author an answer key
    or approve references. The reproduction lock is a preparation lock, not a
    main-experiment freeze.
    """
    download_sources(root)
    lock = verify_inputs(root)
    cases_path = root / "data/prepared/cases.jsonl"
    if sha256_bytes(cases_path.read_bytes()) != lock["outputs"]["data/prepared/cases.jsonl"]:
        raise ValueError("Ingested cases differ from the preparation lock")
    settings = load_settings(root / "config/experiment.json")
    seed = settings.raw["sampling"]["selection_seed"]
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    archive = load_archive(root / "data/upstream" / SNAPSHOT_COMMIT / "response_all.json")
    write_once(
        root / "data/prepared/attack_selection.json",
        encode_json(select_attack_samples(cases, seed)),
    )
    write_once(
        root / "data/prepared/control_plans.json", encode_json(select_controls(archive, seed))
    )


if __name__ == "__main__":
    create_splits(stage_root(__doc__))
