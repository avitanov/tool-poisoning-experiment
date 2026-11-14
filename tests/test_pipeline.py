"""Offline clone reproduction and failure-boundary checks for preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mcptox_gateway.dataset import SNAPSHOT_COMMIT, build_cases, extract_clean_references
from mcptox_gateway.pipelines.common import OUTPUTS, encode_json, input_hashes, write_once
from mcptox_gateway.pipelines.create_splits import create_splits
from mcptox_gateway.pipelines.download_data import download_sources
from mcptox_gateway.pipelines.ingest import _jsonl, ingest
from mcptox_gateway.pipelines.verify import verify
from mcptox_gateway.sampling import select_attack_samples, select_controls


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """Create a tiny independent checkout with enough families for fixed quotas."""
    project = Path(__file__).resolve().parents[1]
    for name in (".python-version", "pyproject.toml", "uv.lock", "config/experiment.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((project / name).read_bytes())
    servers = {}
    for number in range(1, 4):
        prompt = "Tool: help\nDescription: Provides help.\nArguments:\n"
        servers[str(number)] = {
            "server_id": number,
            "server_name": str(number),
            "clean_system_promot": prompt,
            "clean_querys": [f"clean {number}-{i}" for i in range(10)],
            "malicious_instance": [
                {
                    "metadata": {"paradigm": f"Template-{number}"},
                    "wrong_data": 0,
                    "poisoned_tool": "help",
                    "datas": [
                        {"id": i, "query": f"request {i}", "system": prompt} for i in range(50)
                    ],
                }
            ],
        }
    archive = {"servers": servers}
    data = encode_json(archive)
    relative = f"data/upstream/{SNAPSHOT_COMMIT}/response_all.json"
    write_once(tmp_path / relative, data)
    manifest = {
        "commit": SNAPSHOT_COMMIT,
        "files": [
            {
                "path": relative,
                "source_url": "https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/"
                f"{SNAPSHOT_COMMIT}/response_all.json",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            }
        ],
    }
    write_once(tmp_path / "data/manifest.json", encode_json(manifest))
    cases = build_cases(tmp_path / relative)
    expected = [
        _jsonl(cases),
        _jsonl(extract_clean_references(tmp_path / relative)),
        encode_json(select_attack_samples(cases)),
        encode_json(select_controls(archive)),
    ]
    lock = {
        "inputs": input_hashes(tmp_path),
        "outputs": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in zip(OUTPUTS, expected, strict=True)
        },
    }
    write_once(tmp_path / "config/reproduction.lock.json", encode_json(lock))
    return tmp_path


def test_fresh_clone_and_repeat_are_identical_without_network(clone: Path) -> None:
    with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        for _ in range(2):
            download_sources(clone)
            ingest(clone)
            create_splits(clone)
            verify(clone)
    selected = json.loads((clone / OUTPUTS[2]).read_text())
    assert len(selected["pilot"]) == 10
    assert len(selected["main"]) == 100
    assert not (clone / "results").exists()


def test_changed_settings_and_extra_code_are_rejected(clone: Path) -> None:
    path = clone / "config/experiment.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="inputs differ"):
        download_sources(clone)
    path.write_bytes(original)
    write_once(clone / "src/mcptox_gateway/extra.py", b"# changed implementation\n")
    with pytest.raises(ValueError, match="inputs differ"):
        download_sources(clone)


def test_tampered_output_stops_downstream_and_is_not_overwritten(clone: Path) -> None:
    ingest(clone)
    path = clone / OUTPUTS[0]
    path.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="cases differ"):
        create_splits(clone)
    with pytest.raises(ValueError, match="Artifact differs"):
        ingest(clone)
    assert path.read_bytes() == b"tampered\n"


def test_missing_source_needs_opt_in_and_bad_download_is_not_saved(clone: Path) -> None:
    path = clone / f"data/upstream/{SNAPSHOT_COMMIT}/response_all.json"
    original = path.read_bytes()
    path.unlink()
    with patch("urllib.request.urlopen") as request:
        with pytest.raises(ValueError, match="Missing"):
            download_sources(clone)
        request.assert_not_called()
        request.return_value.__enter__.return_value.read.return_value = b"bad download"
        with pytest.raises(ValueError, match="checksum mismatch"):
            download_sources(clone, allow_download=True)
        assert not path.exists()
        request.return_value.__enter__.return_value.read.return_value = original
        download_sources(clone, allow_download=True)
        assert path.read_bytes() == original


def test_corrupt_existing_source_is_never_refetched(clone: Path) -> None:
    path = clone / f"data/upstream/{SNAPSHOT_COMMIT}/response_all.json"
    path.write_bytes(b"corrupt")
    with patch("urllib.request.urlopen", side_effect=AssertionError("must not fetch")):
        with pytest.raises(ValueError, match="checksum mismatch"):
            download_sources(clone, allow_download=True)
