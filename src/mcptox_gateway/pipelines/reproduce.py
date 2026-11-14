"""Run the fixed MCPTox pipeline in explicit fresh or offline replay mode."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from mcptox_gateway.pipelines.common import encode_json, read_json, write_once
from mcptox_gateway.pipelines.create_splits import create_splits
from mcptox_gateway.pipelines.download_data import download_sources
from mcptox_gateway.pipelines.ingest import ingest
from mcptox_gateway.pipelines.verify import verify

MODELS = ("gpt", "qwen", "minimax", "nemotron")


def _stage(root: Path, module: str, arguments: list[str]) -> None:
    """Run an existing stage with its own cache and ledger safeguards."""
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=root, check=True)


def run_pipeline(
    root: Path, *, mode: str, models: list[str], experiment_id: str | None, download: bool = False
) -> None:
    """Prepare and collect a named experiment, or replay persisted historical runs.

    Fresh mode makes paid model calls. Replay uses only saved model calls and
    refuses to create a missing manifest or dispatch a request.
    """
    if mode not in {"fresh", "replay"} or not models or any(m not in MODELS for m in models):
        raise ValueError("Invalid reproduction mode or actor selection")
    if len(set(models)) != len(models):
        raise ValueError("Select each actor at most once")
    if experiment_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Experiment identifier must contain only letters, digits, _ or -")
    if mode == "fresh" and not experiment_id:
        raise ValueError("Fresh mode requires a new experiment identifier")
    if mode == "replay" and download:
        raise ValueError("Replay never downloads source data")
    if mode == "fresh":
        comparison_path = root / "results" / f"{experiment_id}-comparison.json"
        if comparison_path.exists() and set(read_json(comparison_path)["actors"]) != set(models):
            raise ValueError("Experiment identifier already belongs to a different model selection")
        download_sources(root, allow_download=download)
        ingest(root)
        create_splits(root)
        verify(root)
    summaries = {}
    for model in models:
        if experiment_id:
            for split in ("pilot", "main") if mode == "fresh" else ("main",):
                actor_id = f"{experiment_id}-{split}-{model}"
                evaluation_id = f"{actor_id}-evaluation"
                collect = ["--collect-missing"] if mode == "fresh" else []
                pilot = (
                    ["--pilot-evaluation-id", f"{experiment_id}-pilot-{model}-evaluation"]
                    if split == "main"
                    else []
                )
                _stage(
                    root,
                    "mcptox_gateway.pipelines.research_multi_actor",
                    ["--model", model, "--split", split, "--run-id", actor_id, *pilot, *collect],
                )
                _stage(
                    root,
                    "mcptox_gateway.pipelines.pilot_raw_comparison",
                    [
                        "--pilot",
                        actor_id,
                        "--run-id",
                        evaluation_id,
                        "--benchmark-schema",
                        "--parallel",
                        *collect,
                    ],
                )
                result = read_json(root / "results" / evaluation_id / "summary.json")
                if split == "pilot" and (
                    result["attack_text_pass_through"]["complete_triples"] != 10
                    or result["control_text_pass_through"]["complete_triples"] != 3
                ):
                    raise ValueError(f"{model} pilot has incomplete classifications; main stopped")
            summaries[model] = evaluation_id
        else:
            if model == "nemotron":
                actor_id = "research-main-nemotron-2026-11-13-v1"
                evaluation_id = "research-main-raw-comparison-nemotron-2026-11-13-v1"
                _stage(root, "mcptox_gateway.pipelines.research_main", [])
            else:
                actor_id = f"research-main-{model}-2026-11-13-v1"
                evaluation_id = f"research-main-{model}-evaluation-2026-11-13-v1"
                _stage(
                    root,
                    "mcptox_gateway.pipelines.research_multi_actor",
                    ["--model", model, "--split", "main"],
                )
            _stage(
                root,
                "mcptox_gateway.pipelines.pilot_raw_comparison",
                [
                    "--pilot",
                    actor_id,
                    "--run-id",
                    evaluation_id,
                    "--benchmark-schema",
                    "--parallel",
                ],
            )
            summaries[model] = evaluation_id
    if mode == "fresh":
        result = {
            "experiment_id": experiment_id,
            "actors": {
                model: {
                    "run_id": run_id,
                    "summary": read_json(root / "results" / run_id / "summary.json"),
                }
                for model, run_id in summaries.items()
            },
            "scope": "separate judged raw-text comparisons; no tool execution",
        }
        write_once(root / "results" / f"{experiment_id}-comparison.json", encode_json(result))
    print("Completed", mode, "for", ", ".join(models), flush=True)


def main() -> None:
    """Provide an offline default; collection requires an explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "replay"), default="replay")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--experiment-id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--collect-missing", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.mode == "fresh" and not args.collect_missing:
        parser.error("fresh mode requires --collect-missing; model calls can cost money")
    if args.mode == "replay" and args.collect_missing:
        parser.error("replay cannot collect model calls")
    run_pipeline(
        args.root.resolve(),
        mode=args.mode,
        models=args.models,
        experiment_id=args.experiment_id,
        download=args.download,
    )


if __name__ == "__main__":
    main()
