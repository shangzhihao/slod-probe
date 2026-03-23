"""Linear probe training and evaluation entry point."""

from __future__ import annotations

import argparse
import json

from probing.runner import run_probes
from shared.utils import load_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the linear probe script.

    Supports flags for experiment family and compatibility with assignment commands.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate linear probes on frozen embeddings."
    )
    parser.add_argument(
        "--condition",
        choices=("all", "in_domain"),
        default="all",
        help="Which experiment family to run.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train probes and save them to the configured model directory.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate probes, loading saved models when training is not requested.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the probe training and evaluation pipeline.

    This script runs the configured probe experiments (in-domain, cross-domain,
    controlled) and outputs a JSON summary of the results.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.train and not args.eval:
        print("no jobs")
        return 0

    settings = load_settings()
    payload = run_probes(
        settings,
        condition=args.condition,
        train=args.train,
        evaluate=args.eval,
    )
    if args.train and not args.eval:
        print(
            json.dumps(
                {
                    "models_dir": str(settings.probe.models_dir),
                    "saved_models": len(payload["model_paths"]),
                },
                ensure_ascii=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "results_path": payload["results_path"],
                    "runs": len(payload["runs"]),
                },
                ensure_ascii=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
