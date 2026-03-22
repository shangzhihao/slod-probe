"""Span extraction helpers for dataset generation."""

from __future__ import annotations

from pathlib import Path

from spans.balance import balance_spans
from spans.domain import load_metadata_index
from spans.extract import collect_spans_from_paper
from spans.parse import dedupe_spans
from shared.schema import DomainInferenceSettings, SectionRuleSettings, SpanRecord
from shared.utils import load_settings, read_jsonl_gz


def build_spans(
    raw_dir: Path,
    *,
    target_domains: tuple[str, ...] | None = None,
    balance: bool = True,
    seed: int | None = None,
    domain_inference: DomainInferenceSettings | None = None,
    section_rules: SectionRuleSettings | None = None,
    intro_lead_paragraphs: int | None = None,
) -> list[SpanRecord]:
    """Walk the raw shard files and materialize weakly labeled span records.

    This function iterates through all raw paper shards in the specified directory,
    extracts spans from each paper based on the provided settings, dedupes them,
    and optionally balances the dataset by label.

    Args:
        raw_dir: Path to the directory containing raw paper shards (papers_part_*.gz).
        target_domains: Tuple of domain names to include (e.g., ("nlp", "cv")).
            If None, defaults to the domains specified in the settings.
        balance: Whether to balance the spans by label (macro, meso, micro)
            after extraction.
        seed: Random seed for balancing. If None, defaults to the settings seed.
        domain_inference: Settings for inferring the domain of a paper.
            If None, uses defaults from load_settings().
        section_rules: Rules for identifying section types (intro, micro, etc.).
            If None, uses defaults from load_settings().
        intro_lead_paragraphs: Number of lead paragraphs in the introduction to
            treat as macro labels. If None, uses defaults from load_settings().

    Returns:
        A list of SpanRecord objects extracted and processed from the shards.
    """
    dataset_settings = load_settings().dataset
    target_domains = (
        dataset_settings.domains if target_domains is None else target_domains
    )
    seed = dataset_settings.seed if seed is None else seed
    metadata_index = load_metadata_index(raw_dir)
    paper_paths = sorted(raw_dir.glob("papers_part_*.gz"))
    spans: list[SpanRecord] = []

    for path in paper_paths:
        for record in read_jsonl_gz(path):
            spans.extend(
                collect_spans_from_paper(
                    record,
                    path.name,
                    metadata_index,
                    target_domains=target_domains,
                    domain_inference=domain_inference,
                    section_rules=section_rules,
                    intro_lead_paragraphs=intro_lead_paragraphs,
                )
            )

    spans = dedupe_spans(spans)
    if balance:
        # Optional balancing happens after global deduping so duplicates do not skew counts.
        return balance_spans(spans, seed=seed)
    spans.sort(
        key=lambda span: (
            span.label,
            span.paper_id,
            span.source_file,
            span.section_name,
            span.source_kind,
            span.text,
        )
    )
    return spans
