"""Collect one additional actor on the fixed pilot or main cases."""

from __future__ import annotations

import argparse
import copy
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from mcptox_gateway.actor import build_actor_messages
from mcptox_gateway.budget import Budget
from mcptox_gateway.case_collection import CaseCollector
from mcptox_gateway.dataset import load_archive, verify_manifest
from mcptox_gateway.judged_case import _payload
from mcptox_gateway.pipelines.common import encode_json, read_json, write_once
from mcptox_gateway.pipelines.free_actor_pilot import FREE_ACTOR
from mcptox_gateway.pipelines.research_main import build_manifest as build_source_manifest
from mcptox_gateway.pipelines.research_pilot import (
    ROUTES,
    SOURCE,
    _check_endpoint,
    _maximum,
    _route,
)
from mcptox_gateway.provider import extract_content, openrouter_bounded_sender
from mcptox_gateway.qualification import load_dotenv_key
from mcptox_gateway.storage import CallStore, request_identity

ACTORS = {
    "nemotron": FREE_ACTOR,
    "qwen": _route(
        "qwen/qwen3.5-397b-a17b",
        "alibaba",
        "Alibaba",
        {"max_tokens": 1024, "reasoning": {"effort": "none"}},
        "0.39",
        "2.34",
        262144,
    ),
    "minimax": _route(
        "minimax/minimax-m3.1-flash",
        "minimax",
        "Minimax",
        {"max_tokens": 1024, "reasoning": {"effort": "low"}},
        "0.15",
        "0.60",
        1048576,
    ),
    "gpt": ROUTES["actor"],
}


def make_manifest(
    root: Path, model: str, split: str, *, run_id: str | None = None
) -> dict[str, Any]:
    """Freeze existing case contexts, current endpoint metadata and cost bounds."""
    verify_manifest(root / "data/manifest.json", root)
    custom = run_id is not None
    run_id = run_id or f"research-{split}-{model}-2026-11-13-v1"
    if custom:
        base = build_source_manifest(
            root,
            run_id=run_id,
            split=split,
            routes={**ROUTES, "actor": ACTORS[model]},
            check_prior_pilot=False,
        )
        for item in base["cases"]:
            item["bounds"] = {"actor": item["bounds"]["actor"], "B": "0", "C": "0", "judge": "0"}
        base["maximum_actor_cost_usd"] = str(
            sum((Decimal(item["bounds"]["actor"]) for item in base["cases"]), Decimal(0))
        )
        base["known_limitations"].append(
            "MiniMax actor and reviewer use the same model"
            if model == "minimax"
            else "Actor collection budgeted separately from later exact-text review"
        )
        return base
    original = (
        "research-pilot-nemotron-2026-11-13-v1"
        if split == "pilot"
        else "research-main-nemotron-2026-11-13-v1"
    )
    base = read_json(root / "results" / original / "manifest.json")
    routes = {**base["routes"], "actor": ACTORS[model]}
    endpoints = {key: _check_endpoint(route, judge=key == "judge") for key, route in routes.items()}
    risks: dict[str, tuple[str, str]] = {}
    if split == "pilot":
        selection = read_json(root / "data/prepared/attack_selection.json")["pilot"]
        servers = list(load_archive(root / SOURCE)["servers"].items())
        for row in selection:
            i, j, k = map(int, row["source_position"].split(":"))
            name, server = servers[i]
            instance = server["malicious_instance"][j]
            original_case = instance["datas"][k]
            if name != row["server_key"] or original_case["query"] != row["query"]:
                raise ValueError("Pilot source identity changed")
            risks[row["case_id"]] = (
                instance["security_risk_description"],
                instance["poisoned_tool"],
            )
    cases = []
    for old in base["cases"]:
        item = copy.deepcopy(old)
        item["case"]["case_id"] = run_id + ":" + item["source_id"]
        if split == "pilot" and item["kind"] == "attack":
            item["goal"], item["source_poisoned_tool"] = risks[item["source_id"]]
        else:
            item.setdefault("source_poisoned_tool", None)
        payload = _payload(
            routes["actor"], build_actor_messages(item["case"]["system"], item["case"]["query"])
        )
        item["bounds"] = {
            "actor": _maximum(routes["actor"], payload),
            "B": "0",
            "C": "0",
            "judge": "0",
        }
        cases.append(item)
    maximum = sum((Decimal(x["bounds"]["actor"]) for x in cases), Decimal(0))
    return {
        "run_id": run_id,
        "scope": "additional actor, raw-text judged, no tool execution",
        "routes": routes,
        "endpoint_snapshots": endpoints,
        "cases": cases,
        "maximum_actor_cost_usd": str(maximum),
        "proposal_byte_limit": 3584,
        "known_limitations": base["known_limitations"]
        + [
            "Different actor models are reported separately",
            "MiniMax actor and reviewer use the same model"
            if model == "minimax"
            else "Judge labels are not independently human validated",
            "Actor collection is budgeted separately from later exact-text review",
        ],
    }


def main() -> None:
    """Explicitly collect or replay actor responses for one frozen configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(ACTORS))
    parser.add_argument("--split", required=True, choices=["pilot", "main"])
    parser.add_argument("--run-id", help="distinct run ID built directly from pinned data")
    parser.add_argument(
        "--pilot-evaluation-id", help="completed pilot evaluation required for a new main run"
    )
    parser.add_argument("--collect-missing", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    run_id = args.run_id or f"research-{args.split}-{args.model}-2026-11-13-v1"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Run identifier must be a simple name")
    output = root / "results" / run_id
    path = output / "manifest.json"
    store = CallStore(root / "results")
    budget = Budget(store)
    if not path.exists():
        if not args.collect_missing:
            raise ValueError("Actor manifest is missing")
        if args.run_id and args.split == "main":
            pilot_id = args.pilot_evaluation_id
            if not pilot_id or not re.fullmatch(r"[A-Za-z0-9_-]+", pilot_id):
                raise ValueError("A completed pilot evaluation is required for a new main run")
            pilot_dir = root / "results" / pilot_id
            pilot_manifest = read_json(pilot_dir / "manifest.json")
            pilot = read_json(pilot_dir / "summary.json")
            actor_id = pilot_manifest["source_pilot"]
            pilot_actor = read_json(root / "results" / actor_id / "manifest.json")
            if (
                pilot_actor["routes"]["actor"]["model"] != ACTORS[args.model]["model"]
                or pilot["attack_text_pass_through"]["complete_triples"] != 10
                or pilot["control_text_pass_through"]["complete_triples"] != 3
            ):
                raise ValueError("Pilot does not match the actor or has incomplete classifications")
        manifest = make_manifest(root, args.model, args.split, run_id=args.run_id)
        if Decimal(manifest["maximum_actor_cost_usd"]) > Decimal(
            budget.summary()["operational_remaining"]
        ):
            raise ValueError("Actor run exceeds remaining project budget")
        write_once(path, encode_json(manifest))
    manifest = read_json(path)
    if (
        manifest["run_id"] != run_id
        or manifest["routes"]["actor"]["model"] != ACTORS[args.model]["model"]
    ):
        raise ValueError("Frozen actor manifest does not match the requested run and model")
    if args.collect_missing:
        key = os.environ.get("OPENROUTER_API_KEY") or load_dotenv_key(root / ".env")
        if not key:
            raise ValueError("OpenRouter key missing")
        os.environ["OPENROUTER_API_KEY"] = key
        print(
            "Actor preflight:",
            run_id,
            "cases",
            len(manifest["cases"]),
            "actor maximum USD",
            manifest["maximum_actor_cost_usd"],
            flush=True,
        )
    route = manifest["routes"]["actor"]
    interval = 3.2 if args.model == "nemotron" else 0.5
    for n, item in enumerate(manifest["cases"], 1):
        case = item["case"]
        record = {
            "role": "actor",
            "case_id": case["case_id"],
            "payload": _payload(route, build_actor_messages(case["system"], case["query"])),
            "provider": "openrouter",
            "model": route["model"],
            "generation": route["generation"],
            "repetition_id": "0",
            "prompt_version": "light-warning-v1",
            "reference_versions": {"experiment_mode": "judged-v1"},
        }
        adapter = CaseCollector(
            store,
            budget,
            case_id=case["case_id"],
            bounds=item["bounds"],
            routes=manifest["routes"],
            sender=openrouter_bounded_sender,
            collect=args.collect_missing,
        )
        adapter.collector.interval = interval
        if args.collect_missing:
            time.sleep(interval)
        raw = adapter.send("actor", record)
        try:
            content = extract_content(raw)
            status = (
                "not_parsed"
                if len(content.encode("utf8")) <= manifest["proposal_byte_limit"]
                else "oversize"
            )
        except ValueError:
            status = "provider_no_text"
        adapter.release_unused()
        trace = {
            "case_id": case["case_id"],
            "proposal_status": status,
            "request_ids": {"actor": request_identity(record)},
        }
        write_once(output / "traces" / (item["source_id"] + ".json"), encode_json(trace))
        print(f"{run_id}: {n}/{len(manifest['cases'])} {item['source_id']} {status}", flush=True)
        if status == "oversize":
            raise ValueError("Actor output exceeded bound; saved without truncation")
    print("Budget:", budget.summary(), flush=True)


if __name__ == "__main__":
    main()
