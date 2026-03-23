"""Shared schema models for the SLoD probe prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DatasetSettings(BaseModel):
    """Configuration for dataset generation and span extraction."""

    class DomainInferenceSettings(BaseModel):
        """Keywords and weights used to infer if a paper is 'nlp' or 'cv'."""

        nlp_title_keywords: tuple[str, ...]
        nlp_venue_keywords: tuple[str, ...]
        cv_title_keywords: tuple[str, ...]
        cv_venue_keywords: tuple[str, ...]
        nlp_bonus_keywords: tuple[str, ...]
        cv_bonus_keywords: tuple[str, ...]
        venue_weight: int
        title_weight: int
        bonus_weight: int

    class SectionRuleSettings(BaseModel):
        """Keywords used to classify paper sections into label buckets."""

        spurious_section_prefixes: tuple[str, ...]
        intro_keywords: tuple[str, ...]
        conclusion_keywords: tuple[str, ...]
        micro_section_keywords: tuple[str, ...]

    raw_dir: Path
    output_dir: Path
    domains: tuple[str, ...]
    seed: int
    balance: bool
    intro_lead_paragraphs: int
    domain_inference: DomainInferenceSettings
    section_rules: SectionRuleSettings


class EmbeddingSettings(BaseModel):
    """Configuration for frozen embedding extraction."""

    model_name: tuple[str, ...]
    pooling: Literal["mean", "cls"]
    max_tokens: int
    seed: int
    output_dir: Path
    cache_dir: Path


class PipelineSettings(BaseModel):
    """Shared settings for the overall experiment pipeline."""

    batch_size: int
    control_strategy: Literal["truncate", "sample", "middle"]


class ProbeSettings(BaseModel):
    """Hyperparameters for linear probe training and evaluation."""

    seed: int
    token_length: int
    train_fraction: float
    learning_rate: float
    num_steps: int
    weight_decay: float
    results_dir: Path
    models_dir: Path = Path("models")


DomainInferenceSettings = DatasetSettings.DomainInferenceSettings
SectionRuleSettings = DatasetSettings.SectionRuleSettings


class SpanRecord(BaseModel):
    """A single weakly-labeled text span and its associated metadata."""

    paper_id: int
    section_name: str
    label: str
    text: str
    token_count: int
    source_file: str
    source_kind: str
    paper_domain: str | None = None
    paper_title: str | None = None
    paper_year: int | None = None
    paper_venue: str | None = None


class SLoDSettings(BaseModel):
    """Root configuration model for the SLoD probe prototype."""

    dataset: DatasetSettings
    embedding: EmbeddingSettings
    pipeline: PipelineSettings
    probe: ProbeSettings

    model_config = ConfigDict(extra="ignore")
