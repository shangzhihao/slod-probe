"""Frozen embedding extraction for the SLoD probe."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from embedding.core import (
    configure_hf_cache,
    embed_texts,
    load_domain_spans,
    load_transformer_bundle,
    write_embedding_artifact,
)
from shared.schema import SpanRecord
from shared.utils import SLoDSettings, load_settings, slugify


def _resolve_embedding_inputs(
    settings: SLoDSettings,
    *,
    spans_dir: Path | None,
    output_dir: Path | None,
    batch_size: int | None,
) -> tuple[Path, Path, int]:
    """Resolve optional embedding pipeline overrides from project settings."""
    resolved_batch_size = (
        settings.pipeline.batch_size if batch_size is None else batch_size
    )
    resolved_spans_dir = settings.dataset.output_dir if spans_dir is None else spans_dir
    resolved_output_dir = (
        settings.embedding.output_dir if output_dir is None else output_dir
    )
    return resolved_spans_dir, resolved_output_dir, resolved_batch_size


def _embed_domain_records(
    records: list[SpanRecord],
    *,
    bundle: object,
    settings: SLoDSettings,
    batch_size: int,
) -> object:
    """Embed one domain's records while preserving their row order."""
    return embed_texts(
        bundle,
        [record.text for record in records],
        max_tokens=settings.embedding.max_tokens,
        pooling=settings.embedding.pooling,
        batch_size=batch_size,
    )


def _write_domain_embedding_artifact(
    artifact_path: Path,
    *,
    model_name: str,
    domain: str,
    records: list[SpanRecord],
    embeddings: object,
    settings: SLoDSettings,
) -> Path:
    """Persist one domain embedding artifact."""
    return write_embedding_artifact(
        artifact_path,
        model_name=model_name,
        pooling=settings.embedding.pooling,
        max_tokens=settings.embedding.max_tokens,
        seed=settings.embedding.seed,
        domain=domain,
        records=records,
        embeddings=embeddings,
    )


def _embed_model_domains(
    model_name: str,
    *,
    settings: SLoDSettings,
    domain_spans: dict[str, list[SpanRecord]],
    output_dir: Path,
    batch_size: int,
) -> dict[str, Path]:
    """Embed all domain splits for a single transformer model."""
    model_slug = slugify(model_name)
    print(f"embedding model {model_name} -> {model_slug}", flush=True)
    bundle = load_transformer_bundle(model_name, cache_dir=settings.embedding.cache_dir)
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    model_outputs: dict[str, Path] = {}
    for domain, records in domain_spans.items():
        print(f"  embedding domain {domain}: {len(records)} spans", flush=True)
        embeddings = _embed_domain_records(
            records,
            bundle=bundle,
            settings=settings,
            batch_size=batch_size,
        )
        artifact_path = model_dir / f"{domain}.pt"
        model_outputs[domain] = _write_domain_embedding_artifact(
            artifact_path,
            model_name=model_name,
            domain=domain,
            records=records,
            embeddings=embeddings,
            settings=settings,
        )
        print(f"  wrote {artifact_path}", flush=True)

    return model_outputs


def _embed_configured_models(
    settings: SLoDSettings,
    *,
    domain_spans: dict[str, list[SpanRecord]],
    output_dir: Path,
    batch_size: int,
) -> dict[str, dict[str, Path]]:
    """Run per-model embedding generation for the configured backbone list."""
    output_paths: dict[str, dict[str, Path]] = {}
    for model_name in settings.embedding.model_name:
        output_paths[model_name] = _embed_model_domains(
            model_name,
            settings=settings,
            domain_spans=domain_spans,
            output_dir=output_dir,
            batch_size=batch_size,
        )
    return output_paths


def build_embeddings(
    settings: SLoDSettings,
    *,
    spans_dir: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, dict[str, Path]]:
    """Embed each configured domain split for every configured backbone model.

    Args:
        settings: Project configuration settings.
        spans_dir: Directory containing input span JSONL files.
        output_dir: Parent directory where embedding artifacts should be saved.
        batch_size: Minibatch size for transformer inference.

    Returns:
        A nested dictionary mapping model names and domains to artifact paths.
    """
    spans_dir, output_dir, batch_size = _resolve_embedding_inputs(
        settings,
        spans_dir=spans_dir,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    configure_hf_cache(settings.embedding.cache_dir)
    domain_spans = load_domain_spans(spans_dir, settings.dataset.domains)
    return _embed_configured_models(
        settings,
        domain_spans=domain_spans,
        output_dir=output_dir,
        batch_size=batch_size,
    )


def main() -> int:
    """Execute the frozen embedding extraction pipeline.

    Loads per-domain span records and uses configured transformer models
    to generate vector representations, which are saved as .pt artifacts.
    """
    settings = load_settings()
    spans_dir = settings.dataset.output_dir
    output_dir = settings.embedding.output_dir
    batch_size = settings.pipeline.batch_size
    outputs = build_embeddings(
        settings,
        spans_dir=spans_dir,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    print(
        json.dumps(
            {
                "outputs": {
                    model_name: {
                        domain: str(path) for domain, path in domain_outputs.items()
                    }
                    for model_name, domain_outputs in outputs.items()
                },
                "models": list(settings.embedding.model_name),
                "domains": list(settings.dataset.domains),
                "spans_dir": str(spans_dir),
                "output_dir": str(output_dir),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
