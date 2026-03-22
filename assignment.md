# Assessment Research Task from Andrey Ustyuzhanin

Source email:

- From: Andrey Ustyuzhanin <Andrey.Ustyuzhanin@constructor.org>
- To: `zhihao.shang@gmail.com`
- Subject: `assessment research task`
- Received: March 17, 2026

## Overview

This assessment evaluates:

- ability to read and synthesize research papers on representation
  learning and probing classifiers
- ability to write functional code for embedding extraction and linear probing
- ability to reason critically about experimental design, especially
  confounds and controls

Duration: 5-7 days  
Format: remote work with deliverables submission

## Project Context

The task is framed around the Temporal Knowledge Hypergraph (TKH)
project. The core question is whether the Semantic Level of Detail
(SLoD) of scientific text is already encoded in frozen language model
embeddings.

The concrete research question:

Can a simple linear classifier trained on frozen embeddings, using weak
labels from paper structure rather than manual annotation, distinguish:

- macro-level text: abstract, introduction, conclusion
- meso-level text: section lead paragraphs
- micro-level text: methods, experiments, detailed results

## Part 1: Literature Review

Objective: survey probing classifiers, representation learning, and
abstraction in NLP.

Read and analyze 3 papers from this list, or equivalent recent papers:

- Tenney, Das, Pavlick (2019), *BERT Rediscovers the Classical NLP Pipeline*
- Belinkov (2022), *Probing Classifiers: Promises, Shortcomings, and Advances*
- Nickel, Kiela (2017), *Poincare Embeddings for Learning Hierarchical Representations*
- Vilnis, McCallum (2015), *Word Representations via Gaussian Embedding*
- Lo et al. (2020), *S2ORC: The Semantic Scholar Open Research Corpus*
- Ratner et al. (2017), *Snorkel: Rapid Training Data Creation with Weak Supervision*

Deliverable 1:

- literature summary in PDF or Markdown
- 2-3 pages maximum

It should cover:

- what probing classifiers can and cannot tell us
- what is known about hierarchical structure in embeddings
- how weak supervision can replace manual annotation
- what is relevant to SLoD detection and what the main risks are

## Part 2: Prototype Implementation

Objective: build a pipeline that:

- generates weak SLoD labels from document structure
- trains and evaluates a linear probe on frozen embeddings

### 2.1 Weak Label Generation

Use structured scientific papers and assign labels from section structure:

- `macro`: title, abstract, introduction (first 2 paragraphs), conclusion
- `meso`: first sentence of each non-intro, non-conclusion section
- `micro`: methods, experiments, and result details from non-lead paragraphs

Dataset options:

- `allenai/s2orc`
- `allenai/qasper`

Minimum data target:

- at least 500 text spans per label class
- across 50+ papers
- from 1-2 domains such as NLP and Computer Vision

Required metadata per span:

- `paper_id`
- `section_name`
- `label`
- `text`
- `token_count`

Also required:

- document where each span came from
- keep class sizes balanced within 10%

### 2.2 Embedding Extraction

Use one frozen embedding model:

- `allenai/scibert_scivocab_uncased`
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

Representation:

- mean-pooled token embeddings, or
- `[CLS]` token

Constraint:

- no fine-tuning; weights must remain frozen

### 2.3 Linear Probe Training and Evaluation

Use a linear classifier only:

- logistic regression, or
- linear SVM

Run these conditions:

- in-domain: train and test within the same domain
- cross-domain: train on one domain and test on another
- controlled: length-matched version of the in-domain setup

Report:

- macro-averaged F1 for 3-way classification
- confusion matrix
- per-class precision and recall
- accuracy for all three conditions

### 2.4 Required Control Experiment

The email explicitly flags text length as a likely confound.

Required control:

- truncate or sample all spans to a fixed token range before embedding
- rerun the in-domain evaluation
- report whether F1 drops, stays the same, or improves
- discuss what that says about what the probe actually learned

## Suggested Code Organization

```text
slod-probe/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── embed.py
│   ├── probe.py
│   ├── controls.py
│   └── utils.py
├── data/
│   └── spans/
├── embeddings/
├── results/
└── notebooks/
    └── analysis.ipynb
```

Deliverable 2:

- GitHub repository or ZIP
- runnable prototype, with command:
  `python src/probe.py --train --eval --condition in_domain`
- cached embeddings or clear reproduction instructions
- saved metrics in a `results/` directory

## Part 3: Evaluation and Report

Objective: analyze whether SLoD is encoded in frozen embeddings.

Required quantitative analysis:

- F1 across in-domain, cross-domain, and length-controlled conditions
- confusion matrices for each condition
- comparison to a majority-class baseline

Required qualitative analysis:

- 3 high-confidence correct examples per class
- 3 failure examples per class
- optional but recommended: UMAP or t-SNE visualization colored by label

Deliverable 3:

- technical report in PDF or Markdown
- 2-4 pages maximum

Suggested report sections:

- introduction
- methodology
- results
- error analysis
- discussion and future work

## Evaluation Criteria

The email lists five evaluation buckets:

- Literature Review: 20%
- Code Quality: 35%
- Technical Report: 25%
- Research Maturity: 15%
- Critical Thinking: 5%

Important emphasis from the rubric:

- handle the length confound rigorously
- avoid leakage and keep experiments clean
- justify design choices
- report honestly, including failures and limitations

Bonus points are available for:

- embedding-space visualization
- comparing more than one embedding model
- deeper confound analysis across domains or paper ages
- synthetic paired-data alternatives to structural labels
- continuous SLoD regression instead of only 3-class classification

## Submission Instructions

Deadline:

- March 24, 2026
- stated as 7 days from task assignment on March 17, 2026

Submit:

- GitHub repository link or ZIP
- literature summary PDF/Markdown
- technical report PDF/Markdown

Send to:

- `andrey.ustyuzhanin@constructor.org`

Email subject:

- `PhD Assessment - [Your Name] - SLoD Probe`

Submission note:

- include the GitHub link if possible, otherwise attach a ZIP and the PDFs
