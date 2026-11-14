"""Offline checks for credential-free bundle boundaries."""

from __future__ import annotations

from pathlib import Path

from mcptox_gateway.packaging import snapshot_bundle


def test_bundle_excludes_credentials_and_snapshots_sqlite(tmp_path: Path) -> None:
    """A local replay bundle cannot accidentally copy dotenv credentials."""

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "providers.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=not-copied", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    destination = tmp_path / "bundle"
    manifest = snapshot_bundle(tmp_path, destination)
    assert "config" in manifest["included"]
    assert not (destination / ".env").exists()
