# `data/` Layout

This directory contains both the compact raw fixtures used to generate weak
labels and the generated span files consumed by the embedding and probe stages.

## `data/raw/`

`data/raw/` holds small gzip-compressed JSONL fixture shards that mimic the
shape of the prepared S2ORC-derived input without requiring the full raw
downloads.

The files are:

- `data/raw/papers_part_1.gz`
- `data/raw/papers_part_2.gz`
- `data/raw/meta_part_1.gz`
- `data/raw/meta_part_2.gz`

Each line in these shards is one JSON object. The paper shards contain paper
records, and the metadata shards contain the companion metadata records used by
domain filtering and weak-label generation.

These fixtures were produced locally as a small stand-in for the much larger
prepared dataset used by the full workflow. The goal is to keep a portable
sample that behaves like raw input while making local iteration, smoke tests,
and code review practical.

## `data/spans/`

`data/spans/` contains the generated weak-label span files written by:

```bash
uv run python -m src.dataset
```

The main outputs are:

- `data/spans/nlp.jsonl`
- `data/spans/cv.jsonl`

Each line is one labeled span record with the text, label, token count, paper
metadata, and source provenance needed by the later embedding and probing
stages.

## Relationship Between the Two

The intended flow is:

1. Read fixture shards from `data/raw/`.
2. Generate weak labels into `data/spans/`.

`data/raw/` is input. `data/spans/` is derived output.
