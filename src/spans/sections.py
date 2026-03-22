"""Section and text normalization helpers for dataset generation."""

from __future__ import annotations

import re

from shared.schema import SectionRuleSettings
from shared.utils import load_settings


def _section_rules(section_rules: SectionRuleSettings | None) -> SectionRuleSettings:
    """Helper to resolve section rules from either input or project settings."""
    return section_rules or load_settings().dataset.section_rules


def normalize_whitespace(text: str) -> str:
    """Replace all whitespace sequences with a single space and trim the ends."""
    return re.sub(r"\s+", " ", text).strip()


def token_count(text: str) -> int:
    """Approximate the token count of a string using whitespace splitting.

    This is a fast, approximate method used for simple dataset statistics.

    Args:
        text: The string to count tokens in.

    Returns:
        The number of whitespace-separated tokens.
    """
    normalized = normalize_whitespace(text)
    return 0 if not normalized else len(normalized.split(" "))


def clean_section_name(text: str) -> str:
    """Normalize whitespace and remove surrounding punctuation from a section name.

    Returns 'unknown' if the resulting string is empty.
    """
    cleaned = normalize_whitespace(text)
    cleaned = cleaned.strip(" :-")
    return cleaned or "unknown"


def match_section(
    section_name: str,
    keywords: tuple[str, ...],
    *,
    prefix: bool = False,
) -> bool:
    """Match a normalized section name against a set of keywords or prefixes.

    Args:
        section_name: The name of the section to check.
        keywords: A tuple of lowercase keywords to search for.
        prefix: If True, only matches if the section name starts with a keyword.
            Otherwise, performs a substring check.

    Returns:
        True if a match is found, False otherwise.
    """
    key = normalize_whitespace(section_name).lower()
    if prefix:
        return any(key.startswith(keyword) for keyword in keywords)
    return any(keyword in key for keyword in keywords)


def is_spurious_section(
    section_name: str, section_rules: SectionRuleSettings | None = None
) -> bool:
    """Check if a section header represents boilerplate or non-content (e.g. Appendix)."""
    rules = _section_rules(section_rules)
    return match_section(section_name, rules.spurious_section_prefixes, prefix=True)


def is_intro_section(
    section_name: str, section_rules: SectionRuleSettings | None = None
) -> bool:
    """Check if a section header represents an introduction."""
    rules = _section_rules(section_rules)
    return match_section(section_name, rules.intro_keywords)


def is_conclusion_section(
    section_name: str, section_rules: SectionRuleSettings | None = None
) -> bool:
    """Check if a section header represents a conclusion."""
    rules = _section_rules(section_rules)
    return match_section(section_name, rules.conclusion_keywords)


def is_micro_section(
    section_name: str, section_rules: SectionRuleSettings | None = None
) -> bool:
    """Check if a section header represents a technical or detailed content (micro)."""
    rules = _section_rules(section_rules)
    return match_section(section_name, rules.micro_section_keywords)
