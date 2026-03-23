"""Probe artifact I/O helpers for the SLoD probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from embedding.core import embed_texts, load_transformer_bundle
from shared.schema import SpanRecord
from shared.utils import SLoDSettings

from .controls import LengthControlStrategy, control_records
from .split import DomainArtifact


def controlled_source_signature(records: list[SpanRecord]) -> str:
    """Generate a stable hash of the source spans used for a length-control rerun.

    This ensures that cached controlled embeddings are invalidated if the
    source dataset generation rules change.

    Args:
        records: List of SpanRecords.

    Returns:
        A 16-byte hex digest.
    """
    digest = hashlib.blake2b(digest_size=16)
    for record in records:
        payload = json.dumps(
            {
                "paper_id": record.paper_id,
                "section_name": record.section_name,
                "label": record.label,
                "text": record.text,
                "token_count": record.token_count,
                "source_file": record.source_file,
                "source_kind": record.source_kind,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def controlled_cache_matches(
    artifact: dict,
    *,
    original: DomainArtifact,
    settings: SLoDSettings,
    strategy: LengthControlStrategy,
) -> bool:
    """Validate that a cached controlled artifact matches the current settings.

    Args:
        artifact: The dictionary loaded from a cached .pt file.
        original: The uncropped domain artifact.
        settings: Current project settings.
        strategy: Current length control strategy.

    Returns:
        True if the cache is still valid, False otherwise.
    """
    return (
        artifact.get("token_length") == settings.probe.token_length
        and artifact.get("strategy") == strategy
        and artifact.get("source_signature")
        == controlled_source_signature(original.records)
    )


def load_embedding_artifact(path: Path) -> DomainArtifact:
    """Load a PyTorch embedding artifact and convert components to standard types.

    Ensures that embeddings are on CPU and records are validated against
    the SpanRecord schema.

    Args:
        path: Path to the .pt file.

    Returns:
        A DomainArtifact container.
    """
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    records = [SpanRecord.model_validate(record) for record in artifact["records"]]
    embeddings = artifact["embeddings"].detach().to(torch.float32).cpu()
    return DomainArtifact(records=records, embeddings=embeddings)


def load_model_domain_artifacts(
    embeddings_dir: Path,
    model_slug: str,
    domains: tuple[str, ...],
) -> dict[str, DomainArtifact]:
    """Load all per-domain embedding artifacts associated with a model.

    Args:
        embeddings_dir: Directory containing model subdirectories.
        model_slug: Slug of the transformer model to load.
        domains: Tuple of domains to include.

    Returns:
        A dictionary mapping domain strings to DomainArtifacts.

    Raises:
        FileNotFoundError: If a required .pt artifact is missing.
    """
    model_dir = embeddings_dir / model_slug
    artifacts: dict[str, DomainArtifact] = {}
    for domain in domains:
        path = model_dir / f"{domain}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing embedding artifact: {path}")
        artifacts[domain] = load_embedding_artifact(path)
    return artifacts


def controlled_artifact_path(results_dir: Path, model_slug: str, domain: str) -> Path:
    """Determine the cache path for a length-controlled embedding artifact."""
    return results_dir / "controlled_embeddings" / model_slug / f"{domain}.pt"


def _artifact_to_domain_artifact(artifact: dict) -> DomainArtifact:
    """Convert a serialized artifact payload into the standard container type."""
    records = [SpanRecord.model_validate(record) for record in artifact["records"]]
    embeddings = artifact["embeddings"].detach().to(torch.float32).cpu()
    return DomainArtifact(records=records, embeddings=embeddings)


def _load_cached_controlled_artifact(
    cache_path: Path,
    *,
    original: DomainArtifact,
    settings: SLoDSettings,
    strategy: LengthControlStrategy,
) -> DomainArtifact | None:
    """Load a cached controlled artifact when the cache matches current inputs."""
    if not cache_path.exists():
        return None

    artifact = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not controlled_cache_matches(
        artifact,
        original=original,
        settings=settings,
        strategy=strategy,
    ):
        return None
    return _artifact_to_domain_artifact(artifact)


def _build_controlled_artifact(
    *,
    model_name: str,
    original: DomainArtifact,
    settings: SLoDSettings,
    batch_size: int,
    strategy: LengthControlStrategy,
) -> DomainArtifact:
    """Apply length control and re-embed the controlled records."""
    bundle = load_transformer_bundle(model_name, cache_dir=settings.embedding.cache_dir)
    controlled_records = control_records(
        original.records,
        bundle.tokenizer,
        # The probe may ask for a shorter view than the encoder supports, but never longer.
        max_tokens=min(settings.probe.token_length, settings.embedding.max_tokens),
        seed=settings.probe.seed,
        strategy=strategy,
    )
    embeddings = embed_texts(
        bundle,
        [record.text for record in controlled_records],
        max_tokens=settings.embedding.max_tokens,
        pooling=settings.embedding.pooling,
        batch_size=batch_size,
    )
    return DomainArtifact(records=controlled_records, embeddings=embeddings)


def _controlled_artifact_payload(
    *,
    model_name: str,
    model_slug: str,
    domain: str,
    original: DomainArtifact,
    controlled: DomainArtifact,
    settings: SLoDSettings,
    strategy: LengthControlStrategy,
) -> dict[str, object]:
    """Build the serialized cache payload for a controlled artifact."""
    return {
        "model_name": model_name,
        "model_slug": model_slug,
        "domain": domain,
        "token_length": settings.probe.token_length,
        "strategy": strategy,
        "source_signature": controlled_source_signature(original.records),
        "records": [record.model_dump() for record in controlled.records],
        "embeddings": controlled.embeddings,
    }


def _save_controlled_artifact(
    cache_path: Path,
    *,
    model_name: str,
    model_slug: str,
    domain: str,
    original: DomainArtifact,
    controlled: DomainArtifact,
    settings: SLoDSettings,
    strategy: LengthControlStrategy,
) -> None:
    """Persist a rebuilt controlled artifact to its cache location."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _controlled_artifact_payload(
            model_name=model_name,
            model_slug=model_slug,
            domain=domain,
            original=original,
            controlled=controlled,
            settings=settings,
            strategy=strategy,
        ),
        cache_path,
    )


def load_or_build_controlled_artifact(
    *,
    model_name: str,
    model_slug: str,
    domain: str,
    original: DomainArtifact,
    settings: SLoDSettings,
    results_dir: Path,
    batch_size: int,
    strategy: LengthControlStrategy,
) -> DomainArtifact:
    """Efficiently retrieve or regenerate length-controlled embeddings.

    Re-embedding is expensive, so this function caches results in the
    results_dir and checks a source signature before rebuilding.

    Args:
        model_name: The Hugging Face model identifier.
        model_slug: Filesystem-safe name for the model.
        domain: Domain name.
        original: The full, uncropped domain artifact.
        settings: Project settings.
        results_dir: Directory where the cache is stored.
        batch_size: Minibatch size for transformer inference.
        strategy: Strategy for picking text windows (e.g., 'sample', 'truncate').

    Returns:
        A length-controlled DomainArtifact.
    """
    cache_path = controlled_artifact_path(results_dir, model_slug, domain)
    cached = _load_cached_controlled_artifact(
        cache_path,
        original=original,
        settings=settings,
        strategy=strategy,
    )
    if cached is not None:
        return cached

    controlled = _build_controlled_artifact(
        model_name=model_name,
        original=original,
        settings=settings,
        batch_size=batch_size,
        strategy=strategy,
    )
    _save_controlled_artifact(
        cache_path,
        model_name=model_name,
        model_slug=model_slug,
        domain=domain,
        original=original,
        controlled=controlled,
        settings=settings,
        strategy=strategy,
    )
    return controlled
