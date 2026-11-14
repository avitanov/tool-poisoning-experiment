"""Orchestrate the preparation, inference and evaluation stages."""

import argparse
import subprocess
import sys
from pathlib import Path

from mcptox_gateway.pipelines.create_splits import create_splits
from mcptox_gateway.pipelines.download_data import download_sources
from mcptox_gateway.pipelines.ingest import ingest
from mcptox_gateway.pipelines.reproduce import MODELS, run_pipeline
from mcptox_gateway.pipelines.verify import verify


def _run_module(root: Path, module: str, arguments: list[str]) -> None:
    """Run an existing stage in its own process, preserving its replay safeguards."""
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=root, check=True)


def main() -> None:
    """Dispatch individual stages or replay the completed main experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "download",
            "ingest",
            "split",
            "prepare",
            "verify",
            "infer",
            "evaluate",
            "reproduce",
        ],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--download", action="store_true", help="fetch missing pinned sources")
    parser.add_argument(
        "--collect-missing",
        action="store_true",
        help="explicitly collect missing calls for the frozen main run",
    )
    parser.add_argument("--mode", choices=["fresh", "replay"], default="replay")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.download and args.command not in {"prepare", "reproduce"}:
        parser.error("--download is supported with prepare or fresh reproduce")
    if args.collect_missing and args.command not in {"infer", "evaluate", "reproduce"}:
        parser.error("--collect-missing is only supported for model stages")
    if args.command == "reproduce" and args.mode == "fresh" and not args.collect_missing:
        parser.error("fresh reproduction requires --collect-missing; model calls can cost money")
    if args.command == "reproduce" and args.mode == "replay" and args.collect_missing:
        parser.error("replay cannot collect model calls")
    try:
        if args.command == "download":
            download_sources(root, allow_download=True)
        elif args.command == "ingest":
            ingest(root)
        elif args.command == "split":
            create_splits(root)
        elif args.command == "prepare":
            download_sources(root, allow_download=args.download)
            print("1/4 Sources verified", flush=True)
            ingest(root)
            print("2/4 Cases and reference candidates reconstructed", flush=True)
            create_splits(root)
            print("3/4 Proposed attack/control subsets reconstructed", flush=True)
            verify(root)
            print("4/4 Preparation verified", flush=True)
        elif args.command == "verify":
            verify(root)
            print("Preparation verified against config/reproduction.lock.json.", flush=True)
        elif args.command == "reproduce":
            run_pipeline(
                root,
                mode=args.mode,
                models=args.models,
                experiment_id=args.experiment_id,
                download=args.download,
            )
        else:
            actor = "mcptox_gateway.pipelines.research_main"
            evaluation = "mcptox_gateway.pipelines.pilot_raw_comparison"
            comparison_args = [
                "--pilot",
                "research-main-nemotron-2026-11-13-v1",
                "--run-id",
                "research-main-raw-comparison-nemotron-2026-11-13-v1",
                "--benchmark-schema",
                "--parallel",
            ]
            collect = ["--collect-missing"] if args.collect_missing else []
            if args.command == "infer":
                _run_module(root, actor, collect)
            if args.command == "evaluate":
                _run_module(root, evaluation, comparison_args + collect)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Pipeline stopped: {error}\n")


if __name__ == "__main__":
    main()
