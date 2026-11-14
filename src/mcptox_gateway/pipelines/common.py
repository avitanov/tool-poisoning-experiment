"""Small shared helpers for immutable preparation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcptox_gateway.dataset import sha256_bytes

OUTPUTS = (
    "data/prepared/cases.jsonl",
    "data/prepared/reference_candidates.jsonl",
    "data/prepared/attack_selection.json",
    "data/prepared/control_plans.json",
)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact without interpreting its contents as code."""
    return json.loads(path.read_text(encoding="utf-8"))


def encode_json(value: Any) -> bytes:
    """Serialize an artifact consistently across operating systems."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_once(path: Path, data: bytes) -> None:
    """Reuse identical artifacts and refuse to replace different evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"Artifact differs: {path}; preserve it and record an amendment.")


def input_hashes(root: Path) -> dict[str, str]:
    """Hash code and configuration with Git checkout line endings normalized.

    Dataset and generated-artifact hashes always use exact bytes instead.
    """
    files = [root / name for name in (".python-version", "pyproject.toml", "uv.lock")]
    files += [root / "data/manifest.json"]
    for directory, pattern in (
        ("config", "*.json"),
        ("prompts", "*.txt"),
        ("src/mcptox_gateway", "*.py"),
    ):
        files += sorted((root / directory).rglob(pattern))
    return {
        path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        for path in files
        if path.name != "reproduction.lock.json"
    }


def verify_inputs(root: Path) -> dict[str, Any]:
    """Stop if the checked-in preparation contract or its input inventory drifted."""
    lock = read_json(root / "config/reproduction.lock.json")
    if lock["inputs"] != input_hashes(root):
        raise ValueError(
            "Preparation inputs differ from reproduction.lock.json; amendment required."
        )
    return lock


def stage_root(description: str) -> Path:
    """Parse the common root option for an independently runnable stage."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args().root.resolve()
