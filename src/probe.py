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
        help="Compatibility flag for the assignment command; training is always performed.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Compatibility flag for the assignment command; evaluation is always performed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the probe training and evaluation pipeline.

    This script runs the configured probe experiments (in-domain, cross-domain,
    controlled) and outputs a JSON summary of the results.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    payload = run_probes(settings, condition=args.condition)
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
