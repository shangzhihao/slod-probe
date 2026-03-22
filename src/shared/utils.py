"""Shared utilities for the SLoD probe prototype."""

from __future__ import annotations

import gzip
import json
import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .schema import SLoDSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.toml"
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def slugify(text: str) -> str:
    """Convert a model name or string into a filesystem-friendly slug.

    Replaces non-alphanumeric characters with underscores and lowercases the result.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return slug.lower() or "model"


def split_first_sentence(text: str) -> tuple[str, str]:
    """Split a string into its first sentence and the remaining text.

    Uses a simple heuristic tuned for scientific paper prose (looking for
    punctuation followed by an uppercase letter or bracket).

    Args:
        text: The text to split.

    Returns:
        A tuple of (first_sentence, remaining_text).
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return "", ""

    sentence_end = SENTENCE_BOUNDARY_RE.search(normalized)
    if sentence_end:
        first_sentence = normalized[: sentence_end.start() + 1].strip()
        tail = normalized[sentence_end.end() :].strip()
        return first_sentence, tail
    return normalized, ""


def read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    """Stream dictionaries from a gzipped JSONL file.

    Args:
        path: Path to the .jsonl.gz file.

    Yields:
        Each line in the file as a parsed dictionary.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


@lru_cache(maxsize=1)
def load_settings(config_path: Path | None = None) -> SLoDSettings:
    """Load and cache the repository configuration from a TOML file.

    The configuration is cached so that every part of the pipeline sees
    a consistent view of the settings.

    Args:
        config_path: Optional path to the config.toml file.
            Defaults to the root config.toml.

    Returns:
        An SLoDSettings Pydantic model.
    """
    resolved_path = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with resolved_path.open("rb") as config_file:
        return SLoDSettings.model_validate(tomllib.load(config_file))
