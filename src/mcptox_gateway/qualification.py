"""Load the optional OpenRouter credential from a local dotenv file."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_key(path: Path | str, variable: str = "OPENROUTER_API_KEY") -> str | None:
    """Read one locally supplied dotenv variable without logging its value."""

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    prefix = f"{variable}="
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix) :].strip().strip('"').strip("'")
            return value or None
    return None
