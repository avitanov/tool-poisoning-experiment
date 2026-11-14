"""Download only the immutable source files named by the reviewed manifest."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from mcptox_gateway.dataset import SNAPSHOT_COMMIT, sha256_bytes, verify_manifest
from mcptox_gateway.pipelines.common import read_json, stage_root, verify_inputs, write_once


def download_sources(root: Path, *, allow_download: bool = False) -> None:
    """Verify existing sources; fetch missing bytes only with explicit opt-in.

    Existing corrupted files are never replaced. Downloads are inert data, checked
    for size, SHA-256 and Git blob identity before being saved. No upstream code runs.
    """
    verify_inputs(root)
    manifest_path = root / "data/manifest.json"
    manifest = read_json(manifest_path)
    if manifest["commit"] != SNAPSHOT_COMMIT or not manifest["files"]:
        raise ValueError("Unexpected or empty source manifest")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        expected = Path("data/upstream") / SNAPSHOT_COMMIT / relative.name
        url = (
            "https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/"
            f"{SNAPSHOT_COMMIT}/{relative.name}"
        )
        if relative != expected or entry["source_url"] != url:
            raise ValueError("Source path or URL is outside the pinned archive")
        destination = root / relative
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source path escapes project root")
        if destination.exists():
            data = destination.read_bytes()
        elif allow_download:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(entry["bytes"] + 1)
        else:
            raise ValueError(f"Missing {relative}; run download_data explicitly to acquire it.")
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if (
            len(data) != entry["bytes"]
            or sha256_bytes(data) != entry["sha256"]
            or blob != entry["git_blob_sha1"]
        ):
            raise ValueError(f"Source checksum mismatch: {relative}")
        write_once(destination, data)
    verify_manifest(manifest_path, root)


def main() -> None:
    """Acquire and verify missing pinned sources, without any inference calls."""
    download_sources(stage_root(__doc__), allow_download=True)


if __name__ == "__main__":
    main()
