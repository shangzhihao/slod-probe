"""Linear probe training and evaluation runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.utils import SLoDSettings, slugify

from .controls import LengthControlStrategy
from .evaluation import evaluate_split
from .metrics import LABELS
from .io import load_model_domain_artifacts, load_or_build_controlled_artifact
from .split import DomainArtifact, split_paper_ids, subset_artifact

type DomainSplit = dict[str, Any]


def _build_run_record(
    *,
    model_name: str,
    model_slug: str,
    condition: str,
    train_domain: str,
    test_domain: str,
    split_result: dict[str, Any],
    controlled: bool,
    control_strategy: str | None,
    settings: SLoDSettings,
) -> dict[str, Any]:
    """Combine metadata and evaluation results into a single run record."""
    return {
        "model_name": model_name,
        "model_slug": model_slug,
        "condition": condition,
        "train_domain": train_domain,
        "test_domain": test_domain,
        "controlled": controlled,
        "control_strategy": control_strategy,
        "token_length": settings.probe.token_length if controlled else None,
        "seed": settings.probe.seed,
        **split_result,
    }


def _build_domain_splits(
    artifacts: dict[str, DomainArtifact],
    settings: SLoDSettings,
) -> dict[str, DomainSplit]:
    """Create reusable paper-level train/test splits for each domain artifact.

    Args:
        artifacts: Dictionary mapping domain names to DomainArtifacts.
        settings: Project configuration settings.

    Returns:
        A dictionary mapping domain names to splits (train/test paper IDs and artifacts).
    """
    domain_splits: dict[str, DomainSplit] = {}
    for domain, artifact in artifacts.items():
        train_ids, test_ids = split_paper_ids(
            artifact.records,
            seed=settings.probe.seed,
            train_fraction=settings.probe.train_fraction,
            labels=LABELS,
        )
        domain_splits[domain] = {
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_artifact": subset_artifact(artifact, train_ids),
            "test_artifact": subset_artifact(artifact, test_ids),
        }
    return domain_splits


def _evaluate_domain_split(
    train_artifact: DomainArtifact,
    test_artifact: DomainArtifact,
    *,
    settings: SLoDSettings,
) -> dict[str, Any]:
    """Internal helper to trigger the evaluation pipeline for a specific split."""
    return evaluate_split(
        train_artifact,
        test_artifact,
        seed=settings.probe.seed,
        learning_rate=settings.probe.learning_rate,
        num_steps=settings.probe.num_steps,
        weight_decay=settings.probe.weight_decay,
    )


def run_in_domain(
    *,
    model_name: str,
    model_slug: str,
    domain_splits: dict[str, DomainSplit],
    settings: SLoDSettings,
) -> list[dict[str, Any]]:
    """Run the in-domain probe evaluation for one embedding model.

    Evaluates probes trained and tested on the same domain (e.g., CV -> CV).
    """
    runs: list[dict[str, Any]] = []
    for domain, split in domain_splits.items():
        print(f"  in-domain {domain}", flush=True)
        split_result = _evaluate_domain_split(
            split["train_artifact"],
            split["test_artifact"],
            settings=settings,
        )
        runs.append(
            _build_run_record(
                model_name=model_name,
                model_slug=model_slug,
                condition="in_domain",
                train_domain=domain,
                test_domain=domain,
                split_result=split_result,
                controlled=False,
                control_strategy=None,
                settings=settings,
            )
        )
    return runs


def run_cross_domain(
    *,
    model_name: str,
    model_slug: str,
    artifacts: dict[str, DomainArtifact],
    domain_splits: dict[str, DomainSplit],
    settings: SLoDSettings,
) -> list[dict[str, Any]]:
    """Run cross-domain transfer probes for one embedding model."""
    runs: list[dict[str, Any]] = []
    domains = list(artifacts.keys())
    for train_domain in domains:
        for test_domain in domains:
            if train_domain == test_domain:
                continue
            print(f"  cross-domain {train_domain} -> {test_domain}", flush=True)
            split_result = _evaluate_domain_split(
                domain_splits[train_domain]["train_artifact"],
                artifacts[test_domain],
                settings=settings,
            )
            runs.append(
                _build_run_record(
                    model_name=model_name,
                    model_slug=model_slug,
                    condition="cross_domain",
                    train_domain=train_domain,
                    test_domain=test_domain,
                    split_result=split_result,
                    controlled=False,
                    control_strategy=None,
                    settings=settings,
                )
            )
    return runs


def run_controlled(
    *,
    model_name: str,
    model_slug: str,
    artifacts: dict[str, DomainArtifact],
    domain_splits: dict[str, DomainSplit],
    settings: SLoDSettings,
    results_dir: Path,
    batch_size: int,
    control_strategy: LengthControlStrategy,
) -> list[dict[str, Any]]:
    """Run length-controlled in-domain probes for one embedding model."""
    runs: list[dict[str, Any]] = []
    for domain, artifact in artifacts.items():
        print(f"  controlled in-domain {domain}", flush=True)
        controlled_artifact = load_or_build_controlled_artifact(
            model_name=model_name,
            model_slug=model_slug,
            domain=domain,
            original=artifact,
            settings=settings,
            results_dir=results_dir,
            batch_size=batch_size,
            strategy=control_strategy,
        )
        train_ids = domain_splits[domain]["train_ids"]
        test_ids = domain_splits[domain]["test_ids"]
        split_result = _evaluate_domain_split(
            subset_artifact(controlled_artifact, train_ids),
            subset_artifact(controlled_artifact, test_ids),
            settings=settings,
        )
        runs.append(
            _build_run_record(
                model_name=model_name,
                model_slug=model_slug,
                condition="controlled",
                train_domain=domain,
                test_domain=domain,
                split_result=split_result,
                controlled=True,
                control_strategy=control_strategy,
                settings=settings,
            )
        )
    return runs


def run_all(
    *,
    model_name: str,
    model_slug: str,
    artifacts: dict[str, DomainArtifact],
    domain_splits: dict[str, DomainSplit],
    settings: SLoDSettings,
    results_dir: Path,
    batch_size: int,
    control_strategy: LengthControlStrategy,
) -> list[dict[str, Any]]:
    """Run every probe condition for one embedding model."""
    runs = run_in_domain(
        model_name=model_name,
        model_slug=model_slug,
        domain_splits=domain_splits,
        settings=settings,
    )

    runs.extend(
        run_cross_domain(
            model_name=model_name,
            model_slug=model_slug,
            artifacts=artifacts,
            domain_splits=domain_splits,
            settings=settings,
        )
    )

    runs.extend(
        run_controlled(
            model_name=model_name,
            model_slug=model_slug,
            artifacts=artifacts,
            domain_splits=domain_splits,
            settings=settings,
            results_dir=results_dir,
            batch_size=batch_size,
            control_strategy=control_strategy,
        )
    )

    return runs


def save_probe_results(
    results_dir: Path,
    settings: SLoDSettings,
    embeddings_dir: Path,
    condition: str,
    runs: list[dict[str, Any]],
) -> str:
    """Serialize the full probe results to a JSON artifact."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": settings.model_dump(mode="json"),
        "embeddings_dir": str(embeddings_dir),
        "results_dir": str(results_dir),
        "condition": condition,
        "runs": runs,
    }
    results_path = results_dir / "probe_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return str(results_path)


def _validate_probe_condition(condition: str) -> None:
    """Reject unsupported experiment families early."""
    if condition not in {"all", "in_domain"}:
        raise ValueError(f"unsupported condition: {condition}")


def _run_model_probes(
    *,
    model_name: str,
    settings: SLoDSettings,
    embeddings_dir: Path,
    results_dir: Path,
    batch_size: int,
    control_strategy: LengthControlStrategy,
    condition: str,
) -> list[dict[str, Any]]:
    """Run the requested probe family for a single embedding model."""
    model_slug = slugify(model_name)
    print(f"probing model {model_name} -> {model_slug}", flush=True)
    artifacts = load_model_domain_artifacts(
        embeddings_dir, model_slug, settings.dataset.domains
    )
    domain_splits = _build_domain_splits(artifacts, settings)

    if condition == "in_domain":
        return run_in_domain(
            model_name=model_name,
            model_slug=model_slug,
            domain_splits=domain_splits,
            settings=settings,
        )
    return run_all(
        model_name=model_name,
        model_slug=model_slug,
        artifacts=artifacts,
        domain_splits=domain_splits,
        settings=settings,
        results_dir=results_dir,
        batch_size=batch_size,
        control_strategy=control_strategy,
    )


def run_probes(
    settings: SLoDSettings,
    *,
    condition: str = "all",
) -> dict[str, Any]:
    """Train and evaluate configured probe conditions for every embedding model."""
    _validate_probe_condition(condition)
    embeddings_dir = settings.embedding.output_dir
    results_dir = settings.probe.results_dir
    batch_size = settings.pipeline.batch_size
    control_strategy = settings.pipeline.control_strategy
    results_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for model_name in settings.embedding.model_name:
        runs.extend(
            _run_model_probes(
                model_name=model_name,
                settings=settings,
                embeddings_dir=embeddings_dir,
                results_dir=results_dir,
                batch_size=batch_size,
                control_strategy=control_strategy,
                condition=condition,
            )
        )

    results_path = save_probe_results(
        results_dir, settings, embeddings_dir, condition, runs
    )
    return {"results_path": results_path, "runs": runs}
