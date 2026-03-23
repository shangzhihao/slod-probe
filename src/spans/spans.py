"""Span extraction helpers for dataset generation."""

from __future__ import annotations

from pathlib import Path

from spans.balance import balance_spans
from spans.domain import load_metadata_index
from spans.extract import collect_spans_from_paper
from spans.parse import dedupe_spans
from shared.schema import DomainInferenceSettings, SectionRuleSettings, SpanRecord
from shared.utils import load_settings, read_jsonl_gz


def _collect_raw_spans(
    raw_dir: Path,
    *,
    target_domains: tuple[str, ...],
    domain_inference: DomainInferenceSettings | None,
    section_rules: SectionRuleSettings | None,
    intro_lead_paragraphs: int | None,
) -> list[SpanRecord]:
    """Read shard files and collect raw span records before finalization."""
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
    return spans


def _span_sort_key(span: SpanRecord) -> tuple[str, int, str, str, str, str]:
    """Provide the stable ordering used for unbalanced output artifacts."""
    return (
        span.label,
        span.paper_id,
        span.source_file,
        span.section_name,
        span.source_kind,
        span.text,
    )


def _finalize_spans(
    spans: list[SpanRecord],
    *,
    balance: bool,
    seed: int,
) -> list[SpanRecord]:
    """Apply deduping plus the chosen post-processing policy."""
    deduped = dedupe_spans(spans)
    if balance:
        # Optional balancing happens after global deduping so duplicates do not skew counts.
        return balance_spans(deduped, seed=seed)

    deduped.sort(key=_span_sort_key)
    return deduped


def build_spans(
    raw_dir: Path,
    *,
    balance: bool = True,
) -> list[SpanRecord]:
    """Walk the raw shard files and materialize weakly labeled span records.

    This function iterates through all raw paper shards in the specified directory,
    extracts spans from each paper based on the provided settings, dedupes them,
    and optionally balances the dataset by label.

    Args:
        raw_dir: Path to the directory containing raw paper shards (papers_part_*.gz).
        balance: Whether to balance the spans by label (macro, meso, micro)
            after extraction.

    Returns:
        A list of SpanRecord objects extracted and processed from the shards.
    """
    dataset_settings = load_settings().dataset
    spans = _collect_raw_spans(
        raw_dir,
        target_domains=dataset_settings.domains,
        domain_inference=dataset_settings.domain_inference,
        section_rules=dataset_settings.section_rules,
        intro_lead_paragraphs=dataset_settings.intro_lead_paragraphs,
    )
    return _finalize_spans(spans, balance=balance, seed=dataset_settings.seed)
