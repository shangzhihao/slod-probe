"""Paper-level train/test splitting helpers for the SLoD probe."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from shared.schema import SpanRecord


@dataclass(frozen=True)
class DomainArtifact:
    records: list[SpanRecord]
    embeddings: torch.Tensor


def select_indices(records: list[SpanRecord], paper_ids: set[int]) -> list[int]:
    """Return the row indices of records belonging to the selected paper IDs.

    Args:
        records: List of span records.
        paper_ids: A set of corpus IDs representing one partition of the split.

    Returns:
        A list of integer row indices.
    """
    return [
        index for index, record in enumerate(records) if record.paper_id in paper_ids
    ]


def subset_artifact(artifact: DomainArtifact, paper_ids: set[int]) -> DomainArtifact:
    """Slice records and embeddings together to maintain row-level alignment.

    Args:
        artifact: The original DomainArtifact (records + embeddings tensor).
        paper_ids: A set of corpus IDs representing the partition to slice.

    Returns:
        A new DomainArtifact containing only the subset of records and vectors.

    Raises:
        ValueError: If the resulting partition is empty.
    """
    indices = select_indices(artifact.records, paper_ids)
    if not indices:
        raise ValueError("paper split produced an empty subset")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return DomainArtifact(
        records=[artifact.records[index] for index in indices],
        embeddings=artifact.embeddings.index_select(0, index_tensor),
    )


def labels_present(
    records: list[SpanRecord], paper_ids: set[int], labels: tuple[str, ...]
) -> bool:
    """Check if a subset of papers contains every target label.

    Ensures that a train/test split is valid for multiclass classification.

    Args:
        records: Full list of span records.
        paper_ids: Subset of corpus IDs.
        labels: Tuple of label strings that must all be present.

    Returns:
        True if all labels are found within the subset, False otherwise.
    """
    present = {record.label for record in records if record.paper_id in paper_ids}
    return all(label in present for label in labels)


def split_paper_ids(
    records: list[SpanRecord],
    seed: int,
    train_fraction: float,
    labels: tuple[str, ...],
) -> tuple[set[int], set[int]]:
    """Randomly split unique paper IDs into train and test sets.

    This function performs a paper-level split to prevent document leakage.
    It retries the split until every label is present in both partitions.

    Args:
        records: List of all available span records for the domain.
        seed: Random seed for deterministic shuffling.
        train_fraction: Proportion of papers to use for training (0.0 to 1.0).
        labels: Tuple of labels that must be present in both splits.

    Returns:
        A tuple of (train_paper_ids, test_paper_ids).

    Raises:
        ValueError: If there are too few papers or a valid split cannot be found.
    """
    paper_ids = sorted({record.paper_id for record in records})
    if len(paper_ids) < 2:
        raise ValueError("need at least two papers to create a train/test split")

    rng = random.Random(seed)
    for _ in range(512):
        # Retry a bounded number of times because some paper mixes make valid
        # label-complete train/test splits impossible after shuffling.
        shuffled = paper_ids[:]
        rng.shuffle(shuffled)
        cutoff = max(
            1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction)))
        )
        train_ids = set(shuffled[:cutoff])
        test_ids = set(shuffled[cutoff:])
        if labels_present(records, train_ids, labels) and labels_present(
            records, test_ids, labels
        ):
            return train_ids, test_ids
    raise ValueError(
        "unable to create a paper split with all labels present in both partitions"
    )
