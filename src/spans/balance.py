"""Balancing and writing helpers for dataset generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Mapping

from shared.schema import SpanRecord

LABELS = ("macro", "meso", "micro")
TARGET_DOMAINS = ("nlp", "cv")
DEFAULT_OUTPUT_DIR = Path("data/spans")


def group_spans_by_domain(
    spans: Iterable[SpanRecord],
    *,
    target_domains: tuple[str, ...] = TARGET_DOMAINS,
) -> dict[str, list[SpanRecord]]:
    """Partition a list of spans by their inferred paper domain.

    This takes an iterable of SpanRecords and groups them into a dictionary
    keyed by domain (e.g., 'nlp', 'cv').

    Args:
        spans: Iterable of span records to be grouped.
        target_domains: A tuple of expected domain labels.

    Returns:
        A dictionary mapping domain strings to a list of SpanRecords.
    """
    grouped: dict[str, list[SpanRecord]] = {domain: [] for domain in target_domains}
    for span in spans:
        if span.paper_domain in grouped:
            grouped[span.paper_domain].append(span)
    return grouped


def balance_spans_grouped(
    spans: Iterable[SpanRecord],
    *,
    target_domains: tuple[str, ...] = TARGET_DOMAINS,
    seed: int = 13,
) -> dict[str, list[SpanRecord]]:
    """Balance each domain independently so label frequencies stay comparable.

    This first groups spans by domain and then downsamples the labels within
    each domain so that the macro, meso, and micro counts are equal (to the
    smallest class size) within that domain.

    Args:
        spans: Iterable of span records to be balanced and grouped.
        target_domains: A tuple of expected domain labels.
        seed: Random seed for deterministic downsampling.

    Returns:
        A dictionary mapping domain strings to balanced lists of SpanRecords.
    """
    grouped = group_spans_by_domain(spans, target_domains=target_domains)
    balanced: dict[str, list[SpanRecord]] = {}
    for domain, domain_spans in grouped.items():
        balanced[domain] = balance_spans(domain_spans, seed=seed)
    return balanced


def balance_spans(spans: list[SpanRecord], seed: int = 13) -> list[SpanRecord]:
    """Downsample each label bucket to the smallest class size.

    This function ensures that the number of macro, meso, and micro label
    records are equal by downsampling the larger classes to the size of
    the smallest class in the input list.

    Args:
        spans: A list of span records to balance.
        seed: Random seed for deterministic downsampling.

    Returns:
        A balanced list of SpanRecords.
    """
    buckets: dict[str, list[SpanRecord]] = {label: [] for label in LABELS}
    for span in spans:
        if span.label in buckets:
            buckets[span.label].append(span)

    if any(not bucket for bucket in buckets.values()):
        return spans

    target = min(len(bucket) for bucket in buckets.values())
    rng = random.Random(seed)

    balanced: list[SpanRecord] = []
    for label in LABELS:
        bucket = buckets[label][:]
        # Sort before shuffling so the seeded sample is stable across runs.
        bucket.sort(
            key=lambda span: (
                span.paper_id,
                span.source_file,
                span.section_name,
                span.source_kind,
                span.text,
            )
        )
        rng.shuffle(bucket)
        balanced.extend(bucket[:target])

    balanced.sort(
        key=lambda span: (
            span.label,
            span.paper_id,
            span.source_file,
            span.section_name,
            span.source_kind,
            span.text,
        )
    )
    return balanced


def write_spans(spans: Iterable[SpanRecord], output_path: Path) -> Path:
    """Write a sequence of span records to a JSONL file.

    This dumps each record into a separate JSON line in the output file.

    Args:
        spans: Iterable of span records.
        output_path: Path to the target JSONL file.

    Returns:
        The Path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for span in spans:
            handle.write(json.dumps(span.model_dump(), ensure_ascii=True))
            handle.write("\n")
    return output_path


def write_domain_spans(
    spans_by_domain: Mapping[str, Iterable[SpanRecord]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write a sorted JSONL artifact for each domain in the input mapping.

    Args:
        spans_by_domain: Mapping of domain names to an iterable of span records.
        output_dir: Parent directory where the JSONL files should be written.

    Returns:
        A dictionary mapping domain strings to their respective file Paths.
    """
    output_paths: dict[str, Path] = {}
    for domain, domain_spans in spans_by_domain.items():
        domain_spans = sorted(
            list(domain_spans),
            key=lambda span: (
                span.label,
                span.paper_id,
                span.source_file,
                span.section_name,
                span.source_kind,
                span.text,
            ),
        )
        output_paths[domain] = write_spans(domain_spans, output_dir / f"{domain}.jsonl")
    return output_paths


def summarize_spans(
    spans: Iterable[SpanRecord],
    *,
    target_domains: tuple[str, ...] = TARGET_DOMAINS,
) -> dict[str, object]:
    """Generate a summary of label, domain, and paper counts for the provided spans.

    Useful for quick reporting in the CLI after dataset generation.

    Args:
        spans: Iterable of span records.
        target_domains: Tuple of domain labels to track.

    Returns:
        A dictionary containing total span count, per-label counts,
        per-domain counts, and the number of unique papers.
    """
    counts: dict[str, int] = {label: 0 for label in LABELS}
    papers: set[int] = set()
    domains: dict[str, int] = {domain: 0 for domain in target_domains}
    for span in spans:
        counts[span.label] = counts.get(span.label, 0) + 1
        papers.add(span.paper_id)
        if span.paper_domain in domains:
            domains[span.paper_domain] += 1
    return {
        "total_spans": sum(counts.values()),
        "label_counts": counts,
        "domain_counts": domains,
        "papers": len(papers),
    }
