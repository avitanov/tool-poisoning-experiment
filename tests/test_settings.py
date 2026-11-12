"""Offline tests for the Unit 1 settings boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcptox_gateway.settings import SettingsError, load_settings


def _settings_payload() -> dict[str, object]:
    return {
        "mode": "replay",
        "budget": {"project_ceiling_usd": "10.00", "operational_stop_usd": "9.50"},
        "models": {
            "actor": {"provider": None, "model": None, "api_key_env": None},
            "reviewer": {"provider": None, "model": None, "api_key_env": None},
        },
        "generation": {"max_output_tokens": None},
    }


def test_replay_template_needs_no_credential(tmp_path: Path) -> None:
    """An unqualified replay template loads and clearly blocks collection."""

    settings_path = tmp_path / "experiment.json"
    settings_path.write_text(json.dumps(_settings_payload()), encoding="utf-8")

    settings = load_settings(settings_path)

    assert settings.mode == "replay"
    assert (
        "actor provider is unset; qualify it before collection."
        in settings.collection_readiness_issues()
    )


def test_missing_settings_field_has_actionable_error(tmp_path: Path) -> None:
    """Missing budget data does not silently select a spending limit."""

    settings_path = tmp_path / "experiment.json"
    payload = _settings_payload()
    del payload["budget"]
    settings_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SettingsError, match="settings.budget"):
        load_settings(settings_path)


def test_operational_stop_cannot_exceed_ceiling(tmp_path: Path) -> None:
    """The conservative stop threshold is checked before future collection."""

    settings_path = tmp_path / "experiment.json"
    payload = _settings_payload()
    payload["budget"] = {"project_ceiling_usd": "10.00", "operational_stop_usd": "10.01"}
    settings_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SettingsError, match="cannot exceed"):
        load_settings(settings_path)


@pytest.mark.parametrize(
    "section,value,error_path",
    [
        ("budget", None, "settings.budget"),
        ("models", {"actor": {}, "reviewer": {}}, "settings.models.actor.provider"),
        ("generation", {}, "settings.generation.max_output_tokens"),
    ],
)
def test_invalid_settings_sections_remain_explicit_errors(
    tmp_path: Path, section: str, value: object, error_path: str
) -> None:
    """Simpler validation must still reject incomplete configuration."""

    payload = _settings_payload()
    payload[section] = value
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SettingsError, match=error_path):
        load_settings(path)
