"""Embedding I/O and transformer helpers for the SLoD probe."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from shared.schema import SpanRecord


@dataclass(frozen=True)
class TransformerBundle:
    model_name: str
    tokenizer: Any
    model: Any
    device: torch.device
    hidden_size: int


def configure_hf_cache(cache_dir: Path) -> None:
    """Point Hugging Face environment variables to the project's local cache directory.

    This ensures that models are downloaded and loaded from the configured
    location rather than the default ~/.cache/huggingface.
    """
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a small JSONL file fully into memory.

    This helper is used for per-domain span records.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_spans(path: Path) -> list[SpanRecord]:
    """Load and validate SpanRecords from a JSONL file."""
    return [SpanRecord.model_validate(record) for record in _read_jsonl(path)]


def load_domain_spans(
    spans_dir: Path, domains: tuple[str, ...]
) -> dict[str, list[SpanRecord]]:
    """Load the per-domain span files expected by the embedding pipeline.

    Args:
        spans_dir: The directory where per-domain JSONL files are stored.
        domains: A tuple of domain strings to load.

    Returns:
        A dictionary mapping domain strings to lists of SpanRecords.

    Raises:
        FileNotFoundError: If a required domain file is missing.
    """
    domain_spans: dict[str, list[SpanRecord]] = {}
    for domain in domains:
        path = spans_dir / f"{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing span file for domain '{domain}': {path}")
        domain_spans[domain] = _load_spans(path)
    return domain_spans


def _load_transformer_bundle(model_name: str, *, cache_dir: Path) -> TransformerBundle:
    """Instantiate a tokenizer/model pair and pin it to the active device.

    Args:
        model_name: The Hugging Face model hub identifier.
        cache_dir: Local path to cache the model weights and tokenizer.

    Returns:
        A TransformerBundle containing the model, tokenizer, and metadata.

    Raises:
        ValueError: If the model's hidden size could not be determined.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading model {model_name} on {device}", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    model.eval()
    model.to(device)
    # Different encoder families expose the embedding width under different config keys.
    hidden_size = int(
        getattr(model.config, "hidden_size", 0) or getattr(model.config, "dim", 0)
    )
    if hidden_size <= 0:
        raise ValueError(f"could not determine hidden size for model {model_name}")
    return TransformerBundle(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
        hidden_size=hidden_size,
    )


def load_transformer_bundle(model_name: str, *, cache_dir: Path) -> TransformerBundle:
    """Load one transformer bundle after pointing HF at the project cache."""
    resolved_cache_dir = cache_dir / "hub"
    configure_hf_cache(cache_dir)
    return _load_transformer_bundle(model_name, cache_dir=resolved_cache_dir)


def _pool_hidden_states(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling: str,
) -> torch.Tensor:
    """Apply CLS or mean pooling to a batch of transformer hidden states.

    Args:
        last_hidden_state: The raw hidden state tensor from the encoder.
        attention_mask: Tensor indicating non-padded token positions.
        pooling: The strategy to use ('cls' or 'mean').

    Returns:
        A single vector per batch element.
    """
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling != "mean":
        raise ValueError(f"unsupported pooling strategy: {pooling}")

    # Mean pooling should ignore padded tokens so longer padding does not dilute vectors.
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _embed_batch(
    bundle: TransformerBundle,
    texts: list[str],
    *,
    max_tokens: int,
    pooling: str,
) -> torch.Tensor:
    """Tokenize, encode, and pool one minibatch of texts.

    Moves tensors to the device specified in the bundle and returns a CPU tensor.
    """
    batch = bundle.tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    batch = {key: value.to(bundle.device) for key, value in batch.items()}
    with torch.inference_mode():
        outputs = bundle.model(**batch)
    pooled = _pool_hidden_states(
        outputs.last_hidden_state,
        batch["attention_mask"],
        pooling=pooling,
    )
    return pooled.detach().to("cpu")


def embed_texts(
    bundle: TransformerBundle,
    texts: list[str],
    *,
    max_tokens: int,
    pooling: str,
    batch_size: int,
) -> torch.Tensor:
    """Embed all texts using fixed-size minibatches and concatenate the outputs.

    Args:
        bundle: The loaded transformer tokenizer and model.
        texts: A list of strings to embed.
        max_tokens: Maximum sequence length.
        pooling: Strategy for converting hidden states to vectors.
        batch_size: Minibatch size for inference.

    Returns:
        A concatenated PyTorch tensor of all embeddings.
    """
    if not texts:
        return torch.empty((0, bundle.hidden_size), dtype=torch.float32)

    batches: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batches.append(
            _embed_batch(bundle, batch_texts, max_tokens=max_tokens, pooling=pooling)
        )
    return torch.cat(batches, dim=0)


def write_embedding_artifact(
    output_path: Path,
    *,
    model_name: str,
    pooling: str,
    max_tokens: int,
    seed: int,
    domain: str,
    records: list[SpanRecord],
    embeddings: torch.Tensor,
) -> Path:
    """Persist embeddings and source span metadata into a single PyTorch file.

    Args:
        output_path: Path where the .pt file should be saved.
        model_name: The name of the transformer backbone used.
        pooling: The pooling strategy used.
        max_tokens: The token budget used during extraction.
        seed: The seed used for embedding generation.
        domain: The domain of the papers in the artifact.
        records: List of SpanRecords whose text matches the embeddings row-for-row.
        embeddings: Tensor of shape (count, dim).

    Returns:
        The Path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "pooling": pooling,
        "max_tokens": max_tokens,
        "seed": seed,
        "domain": domain,
        "span_count": len(records),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "records": [record.model_dump() for record in records],
        "embeddings": embeddings,
    }
    torch.save(payload, output_path)
    return output_path
