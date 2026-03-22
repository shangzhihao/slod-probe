"""Dataset loading and weak-label generation entry point."""

from __future__ import annotations

import json

from spans.balance import (
    balance_spans_grouped,
    group_spans_by_domain,
    summarize_spans,
    write_domain_spans,
)
from spans.spans import build_spans
from shared.utils import load_settings


def main() -> int:
    """Execute the dataset generation pipeline.

    This script loads configuration settings, extracts spans from raw shards,
    optionally balances the labels, and writes the results to per-domain
    JSONL artifacts in the data/spans directory.
    """
    settings = load_settings()
    raw_dir = settings.dataset.raw_dir
    output_dir = settings.dataset.output_dir
    target_domains = tuple(settings.dataset.domains)
    spans = build_spans(
        raw_dir,
        target_domains=target_domains,
        balance=False,
        domain_inference=settings.dataset.domain_inference,
        section_rules=settings.dataset.section_rules,
        intro_lead_paragraphs=settings.dataset.intro_lead_paragraphs,
    )
    if settings.dataset.balance:
        balanced_by_domain = balance_spans_grouped(
            spans,
            target_domains=target_domains,
            seed=settings.dataset.seed,
        )
    else:
        balanced_by_domain = group_spans_by_domain(spans, target_domains=target_domains)

    output_paths = write_domain_spans(balanced_by_domain, output_dir)
    summary = summarize_spans(
        [span for domain_spans in balanced_by_domain.values() for span in domain_spans],
        target_domains=target_domains,
    )
    print(
        json.dumps(
            {
                "outputs": {domain: str(path) for domain, path in output_paths.items()},
                **summary,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
