"""Length-control experiment helpers for the SLoD probe."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Literal

from shared.schema import SpanRecord

LengthControlStrategy = Literal["truncate", "sample", "middle"]


def _token_ids(tokenizer: Any, text: str) -> list[Any]:
    """Extract token IDs or strings from text using a transformer tokenizer.

    Args:
        tokenizer: A Hugging Face AutoTokenizer.
        text: The text string to tokenize.

    Returns:
        A list of tokens or token IDs.
    """
    if hasattr(tokenizer, "tokenize"):
        return list(tokenizer.tokenize(text))
    return tokenizer.encode(text, add_special_tokens=False)


def _select_window_start(
    token_ids: list[int],
    *,
    max_tokens: int,
    seed: int,
    key: str,
    strategy: LengthControlStrategy,
) -> int:
    """Pick the deterministic start offset for a controlled token window.

    Ensures that for a given paper/section/strategy/seed, the same window
    is always selected across different experiment runs.

    Args:
        token_ids: Full list of tokens.
        max_tokens: Target window size.
        seed: Random seed.
        key: A string identifying the record (e.g., paper_id:section).
        strategy: Window selection strategy ('truncate', 'middle', 'sample').

    Returns:
        The starting integer offset for the window.
    """
    if len(token_ids) <= max_tokens:
        return 0
    span = len(token_ids) - max_tokens
    if strategy == "truncate":
        return 0
    if strategy == "middle":
        return span // 2

    digest = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (span + 1)


def control_text(
    text: str,
    tokenizer: Any,
    *,
    max_tokens: int,
    seed: int,
    key: str,
    strategy: LengthControlStrategy = "sample",
) -> str:
    """Trim a string to a fixed token budget without affecting short spans.

    Args:
        text: Original text content.
        tokenizer: Transformer tokenizer used for measurement.
        max_tokens: The exact token budget to enforce.
        seed: Seed for the 'sample' strategy.
        key: Record-level key for deterministic sampling.
        strategy: 'truncate' (first N), 'middle' (center), or 'sample' (deterministic random).

    Returns:
        The trimmed text string.
    """
    token_ids = _token_ids(tokenizer, text)
    if len(token_ids) <= max_tokens:
        return text

    start = _select_window_start(
        token_ids,
        max_tokens=max_tokens,
        seed=seed,
        key=key,
        strategy=strategy,
    )
    window = token_ids[start : start + max_tokens]
    if hasattr(tokenizer, "convert_tokens_to_string"):
        return tokenizer.convert_tokens_to_string(window)
    return tokenizer.decode(
        window, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def control_record(
    record: SpanRecord,
    tokenizer: Any,
    *,
    max_tokens: int,
    seed: int,
    strategy: LengthControlStrategy = "sample",
) -> SpanRecord:
    """Apply length-control to a SpanRecord and update its token count.

    Args:
        record: The original SpanRecord.
        tokenizer: Transformer tokenizer.
        max_tokens: Target length.
        seed: Seed for random selection.
        strategy: Window selection strategy.

    Returns:
        A new SpanRecord with the controlled text.
    """
    key = f"{record.paper_id}:{record.section_name}:{record.source_kind}:{record.label}"
    controlled_text = control_text(
        record.text,
        tokenizer,
        max_tokens=max_tokens,
        seed=seed,
        key=key,
        strategy=strategy,
    )
    token_count = len(_token_ids(tokenizer, controlled_text))
    return record.model_copy(
        update={"text": controlled_text, "token_count": token_count}
    )


def control_records(
    records: Iterable[SpanRecord],
    tokenizer: Any,
    *,
    max_tokens: int,
    seed: int,
    strategy: LengthControlStrategy = "sample",
) -> list[SpanRecord]:
    """Enforce a fixed token budget across a sequence of SpanRecords.

    This ensures that differences in span length do not confound the probe
    results when comparing models or domains.

    Args:
        records: Iterable of SpanRecords.
        tokenizer: Transformer tokenizer.
        max_tokens: Target length.
        seed: Seed for random selection.
        strategy: Window selection strategy.

    Returns:
        A list of updated SpanRecords.
    """
    return [
        control_record(
            record,
            tokenizer,
            max_tokens=max_tokens,
            seed=seed,
            strategy=strategy,
        )
        for record in records
    ]
