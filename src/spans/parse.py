"""Low-level span parsing helpers for dataset generation."""

from __future__ import annotations

import json
from typing import Any, Iterable

from shared.schema import SpanRecord


def parse_annotation_list(raw_value: Any) -> list[dict[str, Any]]:
    """Accept either already parsed lists or JSON-encoded annotation payloads.

    Args:
        raw_value: The annotation data to parse. Can be None, a JSON string,
            or a list of dictionaries.

    Returns:
        A list of dictionaries representing the annotations.

    Raises:
        TypeError: If the raw_value is not of an expected type.
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    if isinstance(raw_value, list):
        return raw_value
    raise TypeError(f"expected annotation list payload, got {type(raw_value).__name__}")


def span_start(span: dict[str, Any]) -> int:
    """Get the start offset of a span."""
    return int(span["start"])


def span_end(span: dict[str, Any]) -> int:
    """Get the end offset of a span."""
    return int(span["end"])


def sorted_spans(raw_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort spans by start and then end offset.

    Sorting ensures that later section lookups can efficiently scan from
    left-to-right through the document.

    Args:
        raw_spans: A list of span dictionaries.

    Returns:
        A new list of spans sorted by offset.
    """
    return sorted(raw_spans, key=lambda item: (span_start(item), span_end(item)))


def section_for_offset(
    sections: list[dict[str, Any]], offset: int
) -> dict[str, Any] | None:
    """Return the most recent section header that starts before the given offset.

    This is used to determine which section a paragraph belongs to by finding
     the nearest preceding section header annotation.

    Args:
        sections: A sorted list of section header spans.
        offset: The start offset of the paragraph.

    Returns:
        The section header span dictionary that applies to the offset,
        or None if no section header precedes it.
    """
    candidate: dict[str, Any] | None = None
    for section in sections:
        if span_start(section) <= offset:
            candidate = section
        else:
            break
    return candidate


def span_text(text: str, span: dict[str, Any]) -> str:
    """Extract the substring from the text corresponding to the span offsets."""
    return text[span_start(span) : span_end(span)]


def dedupe_spans(spans: Iterable[SpanRecord]) -> list[SpanRecord]:
    """Drop exact duplicate spans while preserving the first observed ordering.

    Duplicates are identified based on paper ID, section name, label, text,
    source file, and source kind.

    Args:
        spans: An iterable of span records.

    Returns:
        A list of unique span records.
    """
    seen: set[tuple[Any, ...]] = set()
    deduped: list[SpanRecord] = []
    for span in spans:
        key = (
            span.paper_id,
            span.section_name,
            span.label,
            span.text,
            span.source_file,
            span.source_kind,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped
