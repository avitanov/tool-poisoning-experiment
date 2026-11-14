"""Synthetic in-memory execution only; never touches the host system."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulationResult:
    """An inert result from a supported fake operation."""

    status: str
    effect: dict[str, Any] | None


def fresh_state(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an independent synthetic state copy."""

    return deepcopy(state if state is not None else {"files": {}, "messages": []})


def _virtual_path(path: object) -> str | None:
    if not isinstance(path, str) or not path.startswith("/") or "\\" in path or ".." in path.split("/"):
        return None
    return path


def simulate_action(state: dict[str, Any], action: dict[str, Any]) -> SimulationResult:
    """Apply one supported action to a supplied in-memory state.

    Only ``write_file`` and ``send_message`` are intentionally modeled. Unknown
    operations remain explicit rather than being approximated with host I/O.
    """

    tool, arguments = action.get("tool"), action.get("arguments")
    if not isinstance(arguments, dict):
        return SimulationResult("unsupported", None)
    if tool == "write_file":
        path, content = _virtual_path(arguments.get("path")), arguments.get("content")
        if path is None or not isinstance(content, str):
            return SimulationResult("unsupported", None)
        state.setdefault("files", {})[path] = content
        return SimulationResult("executed", {"kind": "write_file", "path": path})
    if tool == "send_message":
        recipient, content = arguments.get("recipient"), arguments.get("content")
        if not isinstance(recipient, str) or not isinstance(content, str):
            return SimulationResult("unsupported", None)
        state.setdefault("messages", []).append({"recipient": recipient, "content": content})
        return SimulationResult("executed", {"kind": "send_message", "recipient": recipient})
    return SimulationResult("unsupported", None)
