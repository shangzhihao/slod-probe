# SLoD Probe

This repository contains the Semantic Level of Detail (SLoD) probing pipeline.

## What’s included

- Weak-label generation from prepared S2ORC shards
- Frozen embedding extraction with two models
- Linear probe training and evaluation
- Length-controlled experiments
- A runnable analysis notebook
- A report draft and literature review notes

## Data and outputs

The pipeline produces:

- `data/spans/nlp.jsonl`
- `data/spans/cv.jsonl`
- `embeddings/<model_slug>/*.pt`
- `results/probe_results.json`
- `results/probe_results.png`
- `results/probe_class_f1.png`

See [`data/README.md`](data/README.md) for the layout and purpose of both `data/raw/` and `data/spans/`.

## Project layout

- `src/dataset.py`: config-driven script entry point for weak-label generation
- `src/spans/`: weak-label extraction and balancing helpers
- `src/embed.py`: config-driven script entry point for frozen embedding extraction
- `src/embedding/`: core embedding and transformer logic
- `src/probe.py`: script entry point for probe evaluation with assignment CLI flags
- `src/probing/`: linear probe training, evaluation, metrics, and splitting logic
- `src/shared/`: shared Pydantic models and utility helpers
- `data/README.md`: dataset fixture provenance and generated span layout
- `notebooks/analysis.ipynb`: full analysis notebook
- `report/report.md`: technical report
- `literature_review/literature_review.md`: literature review

## Configuration

Runtime configuration lives in `config.toml`.
It includes `dataset`, `embedding`, `pipeline`, and `probe` sections, plus
nested dataset heuristics in `dataset.domain_inference` and
`dataset.section_rules`.
The `dataset` section now controls both artifact paths and weak-label policy,
including introduction lead-paragraph count, domain-keyword tables, section
matching rules, and balancing.
The `embedding` section controls model selection, pooling, truncation, output
directory, and the local Hugging Face cache directory.
The `probe` section controls split and optimizer settings plus the results
output directory.
The `pipeline` section stores shared execution settings such as batch size,
control strategy, and shared runtime defaults.

## Installation

```bash
# preferred
uv sync

# or with venv + pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the pipeline
```bash
# Cached intermediate artifacts are included,
# so you can run the probes directly.
uv run python src/probe.py --train --eval --condition all

# Open the analysis notebook.
uv run jupyter lab notebooks/analysis.ipynb

# Or regenerate the full pipeline from scratch.
# This may take longer because the embedding models need to be downloaded.
uv run python src/dataset.py
uv run python src/embed.py
uv run python src/probe.py --train --eval --condition all

# To run only the in-domain experiment:
uv run python src/probe.py --train --eval --condition in_domain
```


## Reproduce the pipeline (without uv)

```bash
# Cached intermediate artifacts are included,
# so you can run the probes directly.
python src/probe.py --train --eval --condition all

# Open the analysis notebook.
jupyter lab notebooks/analysis.ipynb

# Or regenerate the full pipeline from scratch.
# This may take longer because the embedding models need to be downloaded.
python src/dataset.py
python src/embed.py
python src/probe.py --train --eval --condition all

# To run only the in-domain experiment:
python src/probe.py --train --eval --condition in_domain
```

Run the entrypoints as direct scripts, for example `python src/dataset.py`.

## Notes

- The probe uses a linear classifier only.
- The controlled experiment uses a fixed-length token window to reduce the
  length confound.
- `src/dataset.py` and `src/embed.py` reads its generation settings from `config.toml`.
- `src/probe.py` reads its probe settings from `config.toml`; the CLI
  is limited to the assignment-facing flags `--train`, `--eval`, and
  `--condition`.
- `python src/probe.py --train --eval --condition in_domain` runs only
  the in-domain probe.
- `python src/probe.py --train --eval --condition all`
  runs the full in-domain, cross-domain, and controlled result set.
