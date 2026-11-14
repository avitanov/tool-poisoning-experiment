"""Create a credential-free reproducibility snapshot with SQLite consistency."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path


def snapshot_bundle(project_root: Path, destination: Path) -> dict[str, str]:
    """Create a replay bundle using SQLite backup rather than a live file copy.

    The bundle omits dotenv files and never downloads restricted upstream data.
    """

    if destination.exists():
        raise ValueError("Bundle destination already exists")
    destination.mkdir(parents=True)
    included = ["config", "data/derived", "data/subsets", "notebooks", "prompts", "src", "tests", "pyproject.toml", "uv.lock", "README.md", "IMPLEMENTATION_PLAN.md"]
    copied: list[str] = []
    for relative in included:
        source = project_root / relative
        if not source.exists():
            continue
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".env", "*.pyc", "__pycache__"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied.append(relative)
    calls = project_root / "results" / "calls.sqlite"
    if calls.exists():
        target = destination / "results" / "calls.sqlite"
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(calls) as source_db, sqlite3.connect(target) as target_db:
            source_db.backup(target_db)
        copied.append("results/calls.sqlite")
    manifest = {"format_version": 1, "included": copied, "excluded": [".env", "data/upstream (license-sensitive; acquire using manifest)"], "restore": "Install with uv sync --locked, then run the notebook in replay mode with network and API keys unavailable."}
    (destination / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
