"""Read OpenRouter account limits without making an inference request."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from mcptox_gateway.qualification import load_dotenv_key


def check_openrouter_limits(api_key: str) -> dict[str, Any]:
    """Return only quota and spending limits; exclude key labels and identifiers."""
    request = Request(
        "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}
    )
    with urlopen(request, timeout=30) as response:
        data = json.load(response)["data"]
    effective = data.get("effective_budget")
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://openrouter.ai/api/v1/key",
        "key_limit_usd": data.get("limit"),
        "key_limit_remaining_usd": data.get("limit_remaining"),
        "effective_budget_remaining_usd": (
            effective.get("remaining_usd") if isinstance(effective, dict) else None
        ),
        "free_model_daily_requests": data.get("free_model_daily_requests"),
        "documented_free_requests_per_minute": 20,
        "documentation": "https://openrouter.ai/pricing",
        "paid_rate_limit": "provider-dependent; no fixed platform rate inferred",
        "note": "Key cap remaining is not account balance. Deprecated rate_limit is ignored.",
    }


def main() -> None:
    """Inspect limits using the local key; print no credentials or key metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or load_dotenv_key(args.root / ".env")
    if not key:
        parser.error("OPENROUTER_API_KEY is unavailable")
    print(json.dumps(check_openrouter_limits(key), indent=2))


if __name__ == "__main__":
    main()
