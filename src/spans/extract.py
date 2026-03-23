"""Single-paper span extraction helpers for dataset generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from spans.domain import paper_domain_from_metadata
from spans.parse import (
    dedupe_spans,
    parse_annotation_list,
    section_for_offset,
    sorted_spans,
    span_text,
    span_start,
)
from spans.sections import (
    clean_section_name,
    is_conclusion_section,
    is_intro_section,
    is_micro_section,
    is_spurious_section,
    normalize_whitespace,
    token_count,
)
from shared.schema import DomainInferenceSettings, SectionRuleSettings, SpanRecord
from shared.utils import load_settings, split_first_sentence

TARGET_DOMAINS = ("nlp", "cv")


@dataclass(frozen=True)
class PaperSpanContext:
    paper_id: int
    paper_domain: str
    source_file: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedPaperExtraction:
    paper_id: int
    text: str
    ctx: PaperSpanContext
    title_spans: list[dict[str, int]]
    abstract_spans: list[dict[str, int]]
    section_spans: list[dict[str, int]]
    paragraph_spans: list[dict[str, int]]


def append_span(
    spans: list[SpanRecord],
    ctx: PaperSpanContext,
    *,
    section_name: str,
    label: str,
    text: str,
    source_kind: str,
) -> None:
    """Create a SpanRecord and append it to the provided list.

    Args:
        spans: List to which the new SpanRecord should be appended.
        ctx: Context information about the current paper.
        section_name: Name of the section where the span was found.
        label: Weak label for the span (macro, meso, micro).
        text: The text content of the span.
        source_kind: A string describing the origin of the span
            (e.g., 'title', 'abstract', 'paragraph').
    """
    spans.append(
        SpanRecord(
            paper_id=ctx.paper_id,
            section_name=section_name,
            label=label,
            text=text,
            token_count=token_count(text),
            source_file=ctx.source_file,
            source_kind=source_kind,
            paper_domain=ctx.paper_domain,
            paper_title=ctx.metadata.get("paper_title"),
            paper_year=ctx.metadata.get("paper_year"),
            paper_venue=ctx.metadata.get("paper_venue"),
        )
    )


def collect_spans_from_paper(
    record: dict[str, Any],
    source_file: str,
    metadata_index: dict[int, dict[str, Any]],
    *,
    target_domains: tuple[str, ...] = TARGET_DOMAINS,
    domain_inference: DomainInferenceSettings | None = None,
    section_rules: SectionRuleSettings | None = None,
    intro_lead_paragraphs: int | None = None,
) -> list[SpanRecord]:
    """Apply the assignment's weak-label rules to one parsed paper record.

    This function extracts title, abstract, and body paragraphs from a paper,
    assigning them weak labels (macro, meso, micro) based on predefined rules
    (e.g., lead paragraphs of introduction are macro, first sentence of
    non-intro sections are meso).

    Args:
        record: The raw paper content and annotation record from the shard.
        source_file: The name of the shard file where this paper was found.
        metadata_index: A dictionary for looking up metadata by paper corpus ID.
        target_domains: Tuple of domains that should be included.
        domain_inference: Settings for domain inference.
        section_rules: Rules for classifying sections.
        intro_lead_paragraphs: How many lead paragraphs of the introduction to
            consider as macro labels.

    Returns:
        A list of SpanRecord objects extracted from the paper.
    """
    section_rules = _resolve_section_rules(section_rules)
    prepared = _prepare_paper_extraction(
        record,
        source_file,
        metadata_index,
        target_domains=target_domains,
        domain_inference=domain_inference,
        section_rules=section_rules,
    )
    if prepared is None:
        return []

    intro_lead_paragraphs = _resolve_intro_lead_paragraphs(intro_lead_paragraphs)
    spans = _collect_front_matter_spans(prepared)
    if not prepared.paragraph_spans or not prepared.section_spans:
        return dedupe_spans(spans)

    spans.extend(
        _collect_body_spans(
            prepared,
            section_rules=section_rules,
            intro_lead_paragraphs=intro_lead_paragraphs,
        )
    )
    return dedupe_spans(spans)


def _prepare_paper_extraction(
    record: dict[str, Any],
    source_file: str,
    metadata_index: dict[int, dict[str, Any]],
    *,
    target_domains: tuple[str, ...],
    domain_inference: DomainInferenceSettings | None,
    section_rules: SectionRuleSettings,
) -> PreparedPaperExtraction | None:
    """Validate paper domain and extract base annotation spans."""
    paper_id = int(record["corpusid"])
    content = record["content"]
    text = content["text"]
    annotations = content["annotations"]
    metadata = metadata_index[paper_id]
    paper_domain = paper_domain_from_metadata(metadata, domain_inference)
    if paper_domain not in target_domains:
        return None
    ctx = PaperSpanContext(
        paper_id=paper_id,
        paper_domain=paper_domain,
        source_file=source_file,
        metadata=metadata,
    )

    title_spans = sorted_spans(parse_annotation_list(annotations.get("title")))
    abstract_spans = sorted_spans(parse_annotation_list(annotations.get("abstract")))
    section_spans = [
        span
        for span in sorted_spans(
            parse_annotation_list(annotations.get("sectionheader"))
        )
        # Ignore boilerplate section headers like appendices or acknowledgments.
        if not is_spurious_section(span_text(text, span), section_rules)
    ]
    paragraph_spans = sorted_spans(parse_annotation_list(annotations.get("paragraph")))

    return PreparedPaperExtraction(
        paper_id=paper_id,
        text=text,
        ctx=ctx,
        title_spans=title_spans,
        abstract_spans=abstract_spans,
        section_spans=section_spans,
        paragraph_spans=paragraph_spans,
    )


def _resolve_intro_lead_paragraphs(intro_lead_paragraphs: int | None) -> int:
    """Determine the number of lead paragraphs to use from settings if not provided."""
    if intro_lead_paragraphs is not None:
        return intro_lead_paragraphs
    return load_settings().dataset.intro_lead_paragraphs


def _resolve_section_rules(
    section_rules: SectionRuleSettings | None,
) -> SectionRuleSettings:
    """Determine section-matching rules from settings if not provided."""
    if section_rules is not None:
        return section_rules
    return load_settings().dataset.section_rules


def _collect_front_matter_spans(prepared: PreparedPaperExtraction) -> list[SpanRecord]:
    """Extract macro spans from the paper title and abstract."""
    spans: list[SpanRecord] = []

    # Title and abstract spans are always weak macro labels.
    if prepared.title_spans:
        title_text = normalize_whitespace(
            span_text(prepared.text, prepared.title_spans[0])
        )
        if title_text:
            append_span(
                spans,
                prepared.ctx,
                section_name="title",
                label="macro",
                text=title_text,
                source_kind="title",
            )

    for span in prepared.abstract_spans:
        abstract_text = normalize_whitespace(span_text(prepared.text, span))
        if not abstract_text:
            continue
        append_span(
            spans,
            prepared.ctx,
            section_name="abstract",
            label="macro",
            text=abstract_text,
            source_kind="abstract",
        )

    return spans


def _assign_paragraph_label(
    *,
    section_name: str,
    paragraph_index: int,
    section_rules: SectionRuleSettings,
    intro_lead_paragraphs: int,
) -> tuple[str, str] | None:
    """Determine the weak label and source kind for a paragraph based on its section."""
    if is_intro_section(section_name, section_rules):
        # The introduction gets its lead paragraphs as macro spans.
        if paragraph_index < intro_lead_paragraphs:
            return "macro", "paragraph"
        return None

    if is_conclusion_section(section_name, section_rules):
        # Conclusions are treated as macro spans regardless of paragraph order.
        return "macro", "paragraph"

    if paragraph_index == 0:
        # Assignment meso labels are only the first sentence of real
        # non-intro, non-conclusion sections.
        return "meso", "paragraph_lead"

    if is_micro_section(section_name, section_rules):
        return "micro", "paragraph_detail"

    return None


def _collect_body_spans(
    prepared: PreparedPaperExtraction,
    *,
    section_rules: SectionRuleSettings | None,
    intro_lead_paragraphs: int,
) -> list[SpanRecord]:
    """Extract weak labels from paper paragraphs based on section position."""
    # Track how many paragraphs we have seen per section so the first-paragraph
    # and intro rules stay deterministic within each section.
    section_paragraph_counts: dict[tuple[int, int, str], int] = defaultdict(int)
    spans: list[SpanRecord] = []

    for paragraph in prepared.paragraph_spans:
        paragraph_text = normalize_whitespace(span_text(prepared.text, paragraph))
        if not paragraph_text:
            continue

        paragraph_start = span_start(paragraph)
        # Paragraphs inherit the closest preceding section header annotation.
        section = section_for_offset(prepared.section_spans, paragraph_start)
        if section is None:
            continue

        section_name = clean_section_name(span_text(prepared.text, section))
        section_bucket = (prepared.paper_id, span_start(section), section_name)
        paragraph_index = section_paragraph_counts[section_bucket]
        section_paragraph_counts[section_bucket] += 1

        assignment = _assign_paragraph_label(
            section_name=section_name,
            paragraph_index=paragraph_index,
            section_rules=section_rules,
            intro_lead_paragraphs=intro_lead_paragraphs,
        )
        if assignment is None:
            continue

        label, source_kind = assignment
        final_text = paragraph_text
        if source_kind == "paragraph_lead":
            lead_sentence, _ = split_first_sentence(paragraph_text)
            if not lead_sentence:
                continue
            final_text = lead_sentence

        append_span(
            spans,
            prepared.ctx,
            section_name=section_name,
            label=label,
            text=final_text,
            source_kind=source_kind,
        )

    return spans
