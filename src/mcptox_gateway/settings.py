"""Read the small, safe experiment settings template used by the notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class SettingsError(ValueError):
    """Raised when the local experiment settings file is missing or invalid."""


@dataclass(frozen=True)
class ExperimentSettings:
    """Validated settings that Unit 1 needs to make offline replay explicit.

    The class intentionally retains unqualified model fields as ``None``. They are
    required only for a future, explicit collection workflow and must not make
    replay depend on credentials or network access.

    Args:
        mode: The configured execution mode, currently replay or fixture.
        project_ceiling_usd: Project-wide maximum paid API spend.
        operational_stop_usd: Conservative stopping threshold below the ceiling.
        raw: Complete parsed JSON object for notebook display and later units.
    """

    mode: str
    project_ceiling_usd: Decimal
    operational_stop_usd: Decimal
    raw: dict[str, Any]

    def collection_readiness_issues(self) -> list[str]:
        """Return explicit reasons why paid collection is unavailable.

        Collection is deliberately unavailable until Unit 5 qualifies provider
        identifiers and cost bounds. This method reads only local configuration.
        """

        issues: list[str] = []
        for role in ("actor", "reviewer"):
            model = self.raw["models"][role]
            if not model["provider"]:
                issues.append(f"{role} provider is unset; qualify it before collection.")
            if not model["model"]:
                issues.append(f"{role} model ID is unset; qualify it before collection.")
            if not model["api_key_env"]:
                issues.append(
                    f"{role} credential variable name is unset; configure it after qualification."
                )
        if self.raw["generation"]["max_output_tokens"] is None:
            issues.append("generation.max_output_tokens is unset; bound outputs before collection.")
        return issues


def _require_fields(value: object, path: str, fields: tuple[str, ...]) -> dict[str, Any]:
    """Check an object's required fields and report their configuration path."""

    if not isinstance(value, dict):
        raise SettingsError(f"{path} must be a JSON object.")
    for field in fields:
        if field not in value:
            raise SettingsError(f"Missing required setting: {path}.{field}.")
    return value


def _parse_decimal(value: object, path: str) -> Decimal:
    """Parse a non-negative JSON money value without floating-point rounding."""

    if not isinstance(value, str):
        raise SettingsError(f"{path} must be a decimal string.")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise SettingsError(f"{path} must be a valid decimal string.") from error
    if not amount.is_finite() or amount < 0:
        raise SettingsError(f"{path} must be a finite non-negative amount.")
    return amount


def load_settings(path: Path | str) -> ExperimentSettings:
    """Load and validate the Unit 1 settings template without reading credentials.

    Args:
        path: Path to ``config/experiment.json`` or a compatible local file.

    Returns:
        Parsed settings suitable for offline notebook checks.

    Raises:
        SettingsError: If the file is absent, invalid JSON, or lacks Unit 1 fields.
    """

    settings_path = Path(path)
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SettingsError(
            f"Settings file not found: {settings_path}. Copy or restore config/experiment.json."
        ) from error
    except json.JSONDecodeError as error:
        raise SettingsError(
            f"Settings file is not valid JSON: {settings_path}: {error.msg}."
        ) from error

    root = _require_fields(data, "settings", ("mode", "budget", "models", "generation"))
    mode = root["mode"]
    if mode not in {"replay", "fixture"}:
        raise SettingsError("settings.mode must be 'replay' or 'fixture' in Unit 1.")

    budget = _require_fields(
        root["budget"], "settings.budget", ("project_ceiling_usd", "operational_stop_usd")
    )
    ceiling = _parse_decimal(budget["project_ceiling_usd"], "settings.budget.project_ceiling_usd")
    stop = _parse_decimal(budget["operational_stop_usd"], "settings.budget.operational_stop_usd")
    if stop > ceiling:
        raise SettingsError(
            "settings.budget.operational_stop_usd cannot exceed the project ceiling."
        )

    models = _require_fields(root["models"], "settings.models", ("actor", "reviewer"))
    for role in ("actor", "reviewer"):
        _require_fields(
            models[role], f"settings.models.{role}", ("provider", "model", "api_key_env")
        )
    _require_fields(root["generation"], "settings.generation", ("max_output_tokens",))

    return ExperimentSettings(
        mode=mode,
        project_ceiling_usd=ceiling,
        operational_stop_usd=stop,
        raw=root,
    )
