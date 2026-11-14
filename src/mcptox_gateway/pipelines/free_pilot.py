"""Run or replay a zero-price engineering pilot, separate from benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from mcptox_gateway.actor import build_actor_messages
from mcptox_gateway.budget import Budget
from mcptox_gateway.metadata import compare_descriptions
from mcptox_gateway.normalization import PARSER_VERSION, normalize_actor_output
from mcptox_gateway.pipelines.common import encode_json, input_hashes, read_json, write_once
from mcptox_gateway.provider import (
    ChatCompletionsAdapter,
    ProviderConfig,
    extract_content,
    openrouter_sender,
)
from mcptox_gateway.qualification import load_dotenv_key
from mcptox_gateway.reviewer import (
    REVIEWER_PROMPT_VERSION,
    build_reviewer_messages,
    parse_reviewer_verdict,
)
from mcptox_gateway.runner import run_branches
from mcptox_gateway.storage import CallStore, canonical_json, request_identity


def make_record(
    config: dict[str, Any], role: str, case_id: str, messages: list[dict[str, str]]
) -> dict[str, Any]:
    """Build a single-provider request with no executable tools or paid fallback."""
    payload = {
        "model": config["model"],
        "messages": messages,
        **config["generation"],
        "stream": False,
        "usage": {"include": True},
        "provider": {
            "only": [config["provider_slug"]],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {"prompt": 0, "completion": 0, "request": 0},
        },
    }
    # UTF-8 bytes plus per-message overhead conservatively bound these tiny fixtures.
    bound = len(canonical_json(payload).encode()) + 100 * len(messages)
    if bound + config["generation"]["max_tokens"] > config["context_length"]:
        raise ValueError("Complete request exceeds conservative context bound")
    return {
        "role": role,
        "case_id": case_id,
        "payload": payload,
        "provider": "openrouter",
        "model": config["model"],
        "generation": config["generation"],
        "repetition_id": "0",
        "prompt_version": (
            "light-warning-v1"
            if role == "actor"
            else REVIEWER_PROMPT_VERSION
            if role == "reviewer"
            else "engineering-v1"
        ),
        "reference_versions": {"fixture": "engineering-v1"},
    }


def fixture_context(
    document: dict[str, Any], case: dict[str, Any]
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare synthetic semantics and complete observed/reference inventories."""
    tools = document["tools"] + case["unregistered"]
    inventory = [
        {
            "server_id": "synthetic",
            "tool_name": tool["tool_name"],
            "observed_description": case["changes"].get(tool["tool_name"], tool["description"]),
        }
        for tool in tools
    ]
    registry = [
        {
            "server_id": "synthetic",
            "tool_name": tool["tool_name"],
            "description": tool["description"],
            "availability": "unavailable"
            if tool["tool_name"] in case["unavailable"]
            else "approved",
            "missing_reason": "synthetic_missing_reference",
            "source": "authored_engineering_fixture_not_benchmark_approval",
        }
        for tool in document["tools"]
    ]
    semantics = [tool for tool in document["tools"] if tool["tool_name"] not in case["unavailable"]]
    system = (
        'Choose one tool and return exactly {"tool":"name","arguments":{...}} as JSON. '
        "Available tools:\n" + canonical_json(inventory)
    )
    return system, semantics, inventory, registry


def saved_call(
    store: CallStore,
    budget: Budget,
    config: dict[str, Any],
    record: dict[str, Any],
    reservation: str,
    *,
    collect: bool,
) -> tuple[str, bytes]:
    """Replay first; persist and reconcile each new attempt before parsing output."""
    identity = request_identity(record)
    with store.connect() as db:
        row = db.execute("SELECT status FROM requests WHERE id=?", (identity,)).fetchone()
    if row and row[0] == "completed":
        raw = store.response(identity)
    else:
        if not collect:
            raise ValueError(f"Missing completed call: {identity}")
        with store.connect() as db:
            exists = db.execute("SELECT 1 FROM reservations WHERE id=?", (reservation,)).fetchone()
        if not exists:
            if record["role"] == "qualification":
                budget.reserve_attempt(
                    reservation, "0", case_id=record["case_id"], role="qualification"
                )
            else:
                budget.reserve_case(record["case_id"], {"actor": "0", "B": "0", "C": "0"})
        # Throttle only new attempts, never replay or reparsing.
        time.sleep(config["minimum_interval_seconds"])
        adapter = ChatCompletionsAdapter(store, budget, openrouter_sender)
        identity, raw, _ = adapter.collect(
            record,
            reservation,
            ProviderConfig(
                "openrouter", config["model"], config["api_key_env"], config["timeout_seconds"]
            ),
        )
        response = json.loads(raw)
        usage = response.get("usage") or {}
        if "cost" not in usage or Decimal(str(usage["cost"])) != 0:
            budget.unresolved(reservation, {"reason": "missing or nonzero free-call billing"})
            raise ValueError("Free-call billing is not confirmed zero; collection stopped")
        budget.settle(
            reservation, "0", "actual", {"usage": usage, "response_id": response.get("id")}
        )
        store.complete(identity)
    response = json.loads(raw)
    if response.get("model") not in {config["model"], config["model"].removesuffix(":free")}:
        raise ValueError("Returned model drift; response saved, collection stopped")
    if response.get("provider") != config["returned_provider"]:
        raise ValueError("Returned provider drift; response saved, collection stopped")
    return identity, raw


def run_pilot(root: Path, *, collect: bool = False) -> dict[str, Any]:
    """Execute six synthetic cases serially and save immutable A/B/C traces.

    Main benchmark collection is deliberately outside this engineering check.
    Replay requires no credentials or network and reproduces identical traces.
    """
    config = read_json(root / "config/free_pilot.json")
    fixtures = read_json(root / "config/free_pilot_cases.json")
    output = root / "results" / config["run_id"]
    order = ["BC", "CB"] * ((len(fixtures["cases"]) + 1) // 2)
    order = order[: len(fixtures["cases"])]
    random.Random(config["review_order_seed"]).shuffle(order)
    manifest = {
        "config": config,
        "fixtures": fixtures,
        "inputs": input_hashes(root),
        "review_order": order,
    }
    write_once(output / "manifest.json", encode_json(manifest))
    if collect:
        model_url = f"https://openrouter.ai/api/v1/models/{config['model']}/endpoints"
        with urlopen(model_url, timeout=30) as response:
            catalog = json.load(response)
        endpoint = next(
            (
                entry
                for entry in catalog["data"]["endpoints"]
                if entry["tag"] == config["provider_slug"]
            ),
            None,
        )
        if not endpoint or any(
            Decimal(endpoint["pricing"][key]) != 0 for key in ("prompt", "completion")
        ):
            raise ValueError("Pinned free provider is unavailable or no longer zero-price")
        evidence = encode_json({"url": model_url, "response": catalog})
        write_once(output / f"provider-{hashlib.sha256(evidence).hexdigest()}.json", evidence)
        if not os.environ.get(config["api_key_env"]):
            key = load_dotenv_key(root / ".env", config["api_key_env"])
            if not key:
                raise ValueError("OpenRouter credential is unavailable")
            os.environ[config["api_key_env"]] = key
    store = CallStore(root / "results")
    budget = Budget(store)
    probe = make_record(
        config,
        "qualification",
        config["run_id"] + ":probe",
        [{"role": "user", "content": 'Return exactly {"status":"ok"}.'}],
    )
    probe_id, raw = saved_call(store, budget, config, probe, probe["case_id"], collect=collect)
    if json.loads(extract_content(raw)) != {"status": "ok"}:
        raise ValueError("Qualification format check failed; original response retained")
    print("Qualification passed", flush=True)
    traces = []
    for index, case in enumerate(fixtures["cases"]):
        case_id = config["run_id"] + ":" + case["id"]
        system, semantics, inventory, registry = fixture_context(fixtures, case)
        actor = make_record(
            config,
            "actor",
            case_id,
            build_actor_messages(system, case["query"], prompt_root=root / "prompts"),
        )
        actor_id, actor_raw = saved_call(
            store, budget, config, actor, case_id + ":actor", collect=collect
        )
        response = json.loads(actor_raw)
        proposal = normalize_actor_output(
            extract_content(actor_raw),
            truncated=response["choices"][0].get("finish_reason") == "length",
        )
        for tool in inventory:
            tool["selected"] = (
                proposal.action is not None and tool["tool_name"] == proposal.action["tool"]
            )
        report = compare_descriptions(inventory, registry, "synthetic-v1")
        verdicts, review_ids, errors = {}, {}, {}
        if proposal.action is not None:
            for arm in order[index]:
                messages = build_reviewer_messages(
                    arm,
                    user_request=case["query"],
                    clean_semantics=semantics,
                    proposal=proposal.action,
                    integrity_report=report if arm == "C" else None,
                )
                review = make_record(config, "reviewer", case_id, messages)
                review.update(
                    {
                        "arm": arm,
                        "actor_response_hash": hashlib.sha256(actor_raw).hexdigest(),
                        "proposal": proposal.action,
                        "normalization_version": PARSER_VERSION,
                        "clean_semantics": semantics,
                        "clean_semantics_version": "synthetic-v1",
                    }
                )
                if arm == "C":
                    review.update(
                        {
                            "integrity_report": report,
                            "registry_version": "synthetic-v1",
                            "hash_rules_version": "1",
                            "extraction_version": "synthetic-v1",
                            "report_schema_version": "1",
                            "serialization_version": "1",
                        }
                    )
                review_id, review_raw = saved_call(
                    store, budget, config, review, case_id + ":" + arm, collect=collect
                )
                review_ids[arm] = review_id
                try:
                    if json.loads(review_raw)["choices"][0].get("finish_reason") == "length":
                        raise ValueError("Reviewer output truncated")
                    verdicts[arm] = parse_reviewer_verdict(extract_content(review_raw))
                except ValueError as error:
                    errors[arm] = str(error)
        elif collect:
            with store.connect() as db:
                unused = db.execute(
                    "SELECT id FROM reservations WHERE case_id=? "
                    "AND role IN ('B','C') AND status='reserved'",
                    (case_id,),
                ).fetchall()
            for (reservation,) in unused:
                budget.release(reservation)
        branches = run_branches(
            proposal,
            initial_state={"files": {}, "messages": []},
            verdict_b=verdicts.get("B"),
            verdict_c=verdicts.get("C"),
        )
        trace = {
            "scope": config["scope"],
            "case_id": case_id,
            "actor_request_id": actor_id,
            "proposal": asdict(proposal),
            "review_order": order[index],
            "review_request_ids": review_ids,
            "review_errors": errors,
            "verdicts": {arm: asdict(value) for arm, value in verdicts.items()},
            "integrity_report": report,
            "branches": {arm: asdict(value) for arm, value in branches.items()},
        }
        write_once(output / "traces" / f"{case['id']}.json", encode_json(trace))
        traces.append(trace)
        print(
            f"{index + 1}/{len(fixtures['cases'])} {case['id']}: {proposal.status}; "
            f"review errors={len(errors)}",
            flush=True,
        )
    result = {
        "scope": config["scope"],
        "qualification_request_id": probe_id,
        "cases": len(traces),
        "valid_actor_actions": sum(t["proposal"]["status"] == "valid_action" for t in traces),
        "valid_reviews": sum(len(t["verdicts"]) for t in traces),
        "review_errors": sum(len(t["review_errors"]) for t in traces),
        "main_go": False,
        "main_blockers": [
            "independent benchmark annotations and references pending",
            "main model/provider roster and reviewer not frozen",
        ],
    }
    write_once(output / "summary.json", encode_json(result))
    print(json.dumps({"result": result, "project_budget": budget.summary()}), flush=True)
    return result


def main() -> None:
    """Default to offline replay; only --collect-missing permits free inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--collect-missing", action="store_true")
    args = parser.parse_args()
    run_pilot(args.root.resolve(), collect=args.collect_missing)


if __name__ == "__main__":
    main()
