"""Domain inference helpers for dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from shared.schema import DomainInferenceSettings
from spans.sections import normalize_whitespace
from shared.utils import load_settings, read_jsonl_gz


def contains_keyword(text: str, keyword: str) -> bool:
    """Match a keyword either literally or on word boundaries for simple tokens.

    If the keyword contains non-alphanumeric characters, it performs a simple
    substring check. Otherwise, it uses regex to match the keyword as a whole word.

    Args:
        text: The text to search within.
        keyword: The keyword to search for.

    Returns:
        True if the keyword is found according to the rules, False otherwise.
    """
    if not keyword:
        return False

    if re.search(r"[^a-z0-9]", keyword):
        return keyword in text

    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Check if any of the provided keywords are present in the text.

    Args:
        text: The text to search within.
        keywords: A tuple of keywords to search for.

    Returns:
        True if any keyword is found, False otherwise.
    """
    return any(contains_keyword(text, keyword) for keyword in keywords)


def load_metadata_index(raw_dir: Path) -> dict[int, dict[str, Any]]:
    """Index metadata rows by corpus id so paper shards can join against them quickly.

    This function reads all metadata shards (meta_part_*.gz) in the raw directory
    and builds a dictionary keyed by paper corpus ID for fast lookups during
    span extraction.

    Args:
        raw_dir: Path to the directory containing metadata shards.

    Returns:
        A dictionary mapping corpus IDs to paper metadata dictionaries.
    """
    metadata_index: dict[int, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("meta_part_*.gz")):
        for record in read_jsonl_gz(path):
            metadata_index[int(record["corpusid"])] = {
                "paper_title": record["title"],
                "paper_year": record["year"],
                "paper_venue": record["venue"],
                "s2fieldsofstudy": record["s2fieldsofstudy"],
                "externalids": record["externalids"],
            }
    return metadata_index


def paper_text_metadata(metadata: dict[str, Any]) -> str:
    """Flatten title, venue, ids, and fields-of-study into one scoring string.

    This combines various metadata fields into a single lowercased string used
    for keyword-based domain inference.

    Args:
        metadata: A dictionary of paper metadata.

    Returns:
        A concatenated string of relevant metadata fields.
    """
    title = normalize_whitespace(str(metadata["paper_title"])).lower()
    venue = normalize_whitespace(str(metadata["paper_venue"])).lower()
    externalids = metadata["externalids"]
    dbpl = normalize_whitespace(
        str(externalids.get("DBLP") or externalids.get("dblp") or "")
    ).lower()
    acl = normalize_whitespace(
        str(
            externalids.get("ACL")
            or externalids.get("Acl")
            or externalids.get("acl")
            or ""
        )
    ).lower()
    fos = metadata["s2fieldsofstudy"] or []
    fos_text = " ".join(
        normalize_whitespace(str(item.get("category") or "")).lower()
        for item in fos
        if isinstance(item, dict)
    )
    return " ".join(part for part in (title, venue, dbpl, acl, fos_text) if part)


def paper_domain_from_metadata(
    metadata: dict[str, Any],
    domain_inference: DomainInferenceSettings | None = None,
) -> str | None:
    """Infer whether a paper belongs to 'nlp' or 'cv' based on metadata keywords.

    This function calculates scores for NLP and CV domains by checking for
    keywords in the paper's title, venue, and other metadata. The domain with
    the higher score is returned.

    Args:
        metadata: A dictionary of paper metadata.
        domain_inference: Settings for domain inference keywords and weights.
            If None, defaults from load_settings() are used.

    Returns:
        'nlp', 'cv', or None if no domain could be confidently inferred.
    """
    text = paper_text_metadata(metadata)
    settings = domain_inference or load_settings().dataset.domain_inference
    nlp_score = 0
    cv_score = 0

    if contains_any(text, settings.nlp_venue_keywords):
        nlp_score += settings.venue_weight
    if contains_any(text, settings.cv_venue_keywords):
        cv_score += settings.venue_weight

    if contains_any(text, settings.nlp_title_keywords):
        nlp_score += settings.title_weight
    if contains_any(text, settings.cv_title_keywords):
        cv_score += settings.title_weight

    if contains_any(text, settings.nlp_bonus_keywords):
        nlp_score += settings.bonus_weight
    if contains_any(text, settings.cv_bonus_keywords):
        cv_score += settings.bonus_weight

    if nlp_score == 0 and cv_score == 0:
        return None
    if nlp_score > cv_score:
        return "nlp"
    if cv_score > nlp_score:
        return "cv"
    return None
