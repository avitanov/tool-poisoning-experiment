"""Build actor messages without rewriting the benchmark's tool descriptions."""

from pathlib import Path


def build_actor_messages(
    source_system: str, user_request: str, *, prompt_root: Path | None = None
) -> list[dict[str, str]]:
    """Append the user-requested light warning to the original system context."""
    root = prompt_root or Path(__file__).resolve().parents[2] / "prompts"
    warning = (root / "actor_safety.txt").read_text(encoding="utf-8").strip()
    if not warning:
        raise ValueError("Actor safety prompt is empty")
    return [
        {"role": "system", "content": source_system + "\n\n" + warning},
        {"role": "user", "content": user_request},
    ]
