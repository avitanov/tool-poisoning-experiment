"""Collect the fixed pilot actor outputs with one zero-price NVIDIA route."""

from __future__ import annotations

import argparse
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from mcptox_gateway.actor import build_actor_messages
from mcptox_gateway.budget import Budget
from mcptox_gateway.case_collection import CaseCollector
from mcptox_gateway.dataset import actor_visible_inventory
from mcptox_gateway.judged_case import _payload
from mcptox_gateway.metadata import compare_descriptions
from mcptox_gateway.pipelines.common import encode_json, read_json, write_once
from mcptox_gateway.pipelines.pilot_raw_comparison import REVIEW_PROMPT
from mcptox_gateway.pipelines.pilot_raw_judge import CONTEXT_INSTRUCTION
from mcptox_gateway.pipelines.research_pilot import ROUTES, _maximum, _route, build_manifest
from mcptox_gateway.provider import extract_content, openrouter_bounded_sender
from mcptox_gateway.qualification import load_dotenv_key
from mcptox_gateway.storage import CallStore, canonical_json, request_identity

RUN = "research-pilot-nemotron-2026-11-13-v1"
FREE_ACTOR = _route(
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia",
    "Nvidia",
    {"max_tokens": 1024, "temperature": 0, "reasoning": {"effort": "none"}},
    "0",
    "0",
    1000000,
)


def prepare_manifest(root: Path) -> dict[str, Any]:
    """Freeze the same pilot subset and conservative raw-review cost bounds."""
    routes = {**ROUTES, "actor": FREE_ACTOR}
    manifest = build_manifest(root, run_id=RUN, routes=routes)
    dummy = "\u0000" * 8192
    for item in manifest["cases"]:
        case = item["case"]
        shared = {
            "user_request": case["query"],
            "clean_tool_semantics": item["clean_semantics"],
            "raw_actor_output": dummy,
        }
        b = _payload(
            routes["reviewer"],
            [
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": canonical_json(shared)},
            ],
        )
        report = compare_descriptions(
            actor_visible_inventory(case), item["registry"], "pilot-source-v1"
        )
        c = _payload(
            routes["reviewer"],
            [
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": canonical_json(shared)},
                {"role": "user", "content": canonical_json(report) + " " * 2000},
            ],
        )
        judge_data = {
            "user_request": case["query"],
            "clean_semantics": item["clean_semantics"],
            "attack_goal": item["goal"],
            "actor_text": dummy,
            "actor_visible_tool_context": case["system"],
        }
        judge = _payload(
            routes["judge"],
            [
                {"role": "system", "content": CONTEXT_INSTRUCTION},
                {"role": "user", "content": canonical_json(judge_data)},
            ],
            judge=True,
        )
        item["bounds"].update(
            B=_maximum(routes["reviewer"], b),
            C=_maximum(routes["reviewer"], c),
            judge=_maximum(routes["judge"], judge),
        )
    manifest["maximum_missing_cost_usd"] = str(
        sum((Decimal(v) for x in manifest["cases"] for v in x["bounds"].values()), Decimal(0))
    )
    manifest["scope"] = "free-actor pilot; assistant review, no human validation"
    return manifest


def main() -> None:
    """Collect missing actor calls explicitly, or replay saved calls offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-missing", action="store_true")
    collect = parser.parse_args().collect_missing
    root = Path.cwd()
    output = root / "results" / RUN
    path = output / "manifest.json"
    store = CallStore(root / "results")
    budget = Budget(store)
    if not path.exists():
        if not collect:
            raise ValueError("Free actor pilot manifest is missing")
        write_once(path, encode_json(prepare_manifest(root)))
    manifest = read_json(path)
    if collect:
        if Decimal(manifest["maximum_missing_cost_usd"]) > Decimal(
            budget.summary()["operational_remaining"]
        ):
            raise ValueError("Full pilot upper bound exceeds the project budget")
        key = os.environ.get("OPENROUTER_API_KEY") or load_dotenv_key(root / ".env")
        if not key:
            raise ValueError("OpenRouter key missing")
        os.environ["OPENROUTER_API_KEY"] = key
        print(
            f"Free actor pilot: {len(manifest['cases'])} cases, "
            f"full pilot bound ${manifest['maximum_missing_cost_usd']}",
            flush=True,
        )
    route = manifest["routes"]["actor"]
    next_dispatch = time.monotonic() + (3.2 if collect else 0)
    for number, item in enumerate(manifest["cases"], 1):
        case = item["case"]
        payload = _payload(route, build_actor_messages(case["system"], case["query"]))
        record = {
            "role": "actor",
            "case_id": case["case_id"],
            "payload": payload,
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
            collect=collect,
        )
        with store.connect() as db:
            status = db.execute("SELECT status FROM requests WHERE id=?", (request_identity(record),)).fetchone()
        if collect and status != ("completed",):
            time.sleep(max(0, next_dispatch - time.monotonic()))
        raw = adapter.send("actor", record)
        if collect and status != ("completed",):
            next_dispatch = time.monotonic() + 3.2
        content = extract_content(raw)
        if len(content.encode("utf-8")) > manifest["proposal_byte_limit"]:
            raise ValueError("Actor output exceeded frozen bound; preserved without truncation")
        adapter.release_unused()
        trace = {
            "case_id": case["case_id"],
            "proposal_status": "not_parsed",
            "request_ids": {"actor": request_identity(record)},
        }
        write_once(output / "traces" / (item["source_id"] + ".json"), encode_json(trace))
        print(f"Free actor {number}/{len(manifest['cases'])}: {item['source_id']}", flush=True)
    print(
        json.dumps(
            {
                "scope": "free actor pilot; outputs require blinded review and judging",
                "budget": budget.summary(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
