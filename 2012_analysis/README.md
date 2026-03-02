# 2012 Phenomenon — Quantitative Analysis Scripts

This folder contains scripts used in Chapter 3 of:

Kevin Whitesides  
PhD Dissertation, Religious Studies  
University of California, Santa Barbara  

**Networks of Heterodoxy: Shared Dissent and the Dynamics of Counter-Discourse**

---

## Overview

These scripts were developed to analyze a corpus of 220 English-language books
associated with the “2012 phenomenon.”

The dataset encodes the presence/absence of recurring tropes across texts.
Each trope represents an item appearing in a book's index identified through
manual coding and iterative refinement.

The analysis supports the dissertation’s broader theoretical model of
**networked heterodoxy**, specifically:

- Measuring trope frequency
- Identifying cross-text overlap
- Modeling similarity between texts
- Generating one-mode and bimodal projection networks
- Detecting structural clustering patterns
- Topic modeling based on target tropes

---

## Dataset Structure

The scripts assume a CSV file structured as follows:

```csv
book_title,plato,pleiades,galactic_alignment,solar_flare
Fingerprints of the Gods,1,1,1,0
The Mayan Factor,0,1,0,1
```

Where:

- Each row = one book
- Each trope column = binary (1 = trope present, 0 = absent)
- Tropes were coded manually and validated iteratively
- There should be no totals rows or columns--only a header row, book rows, and trope columns

The dataset used for the dissertation is not included--only the scripts used to analyze it.
Researchers wishing to replicate the analysis should email the author: kevinwhitesides@ucsb.edu
Researchers wishing to run similar analyses should create a similar matrix for their own data and adapt the scripts.

---

## Scripts Included

---

### 1. `book_trope_matrix_cleaner.py`

**Purpose:**
Preprocesses raw spreadsheet exports into analysis-ready format.

**Operations May Include:**
- Removing empty rows
- Normalizing headers
- Ensuring numeric encoding (0/1)
- Dropping totals columns

---

### 2. `jaccard_similarity_matrix.py`

**Purpose:**
Computes pairwise Jaccard similarity between books.

Jaccard index defined as:

J(A,B) = |A ∩ B| / |A ∪ B|

**Outputs:**
- Full similarity matrix (CSV)
- Optional thresholded matrix
- Heatmap visualization (PNG)

**Key Parameters:**
- Minimum trope frequency threshold (e.g., >= 2 occurrences)
- Binary vs weighted toggle (if applicable)

**Used in Dissertation For:**
- Figure X: Jaccard heatmap
- Discussion of cross-text similarity
- Detecting structural proximity

---

### 4. `network_projection_export.py`

**Purpose:**
Constructs one-mode projection networks from bipartite book–trope matrix.

**Modes:**
- Book–book similarity network
- Trope–trope co-occurrence network

**Edge Weight Options:**
- Raw shared trope count
- Jaccard similarity
- Thresholded edges only

**Outputs:**
- Edge list (CSV)
- Node list (CSV)
- Optional GEXF file for Gephi

**Used in Dissertation For:**
- Gephi visualizations
- Louvain modularity clustering
- Structural density analysis

---

### 5. `trope_cooccurrence_analysis.py`

**Purpose:**
Measures how often tropes co-occur across texts.

**Outputs:**
- Co-occurrence matrix
- Edge-weighted trope network
- Centrality metrics (optional)

**Interpretive Function:**
Identifies trope clusters that may reflect shared heterodox strategies.

---

## Thresholding Logic

Several scripts use adjustable parameters such as:

- Minimum trope frequency (e.g., appear in ≥2 books)
- Minimum edge weight
- Removal of singleton nodes

Thresholding decisions are documented in:
`/docs/workflow_overview.md`

These decisions were made to:

- Reduce noise
- Avoid spurious similarity inflation
- Highlight structurally meaningful overlap

---

## Reproducibility Notes

- All scripts written in Python 3.10+
- Core libraries:
  - pandas
  - numpy
  - networkx
  - matplotlib
  - seaborn
  - scikit-learn

Exact package versions used for dissertation release:
(see repository release tag: v1.0-dissertation)

---

## Relationship to Theoretical Framework

These scripts operationalize key distinctions in the dissertation:

- Recurrence vs dominance
- Frequency vs structural centrality
- Numerical spread vs domain-specific authority

The analysis is not intended as purely quantitative modeling,
but as structural support for interpretive argument.

---

## Important Limitations

- No copyrighted texts are included
- Coding decisions involve interpretive judgment
- Similarity metrics reflect trope presence, not semantic nuance
- Results depend on binary encoding rather than weighted discourse intensity

---

## Citation

If referencing these scripts, please cite:

Whitesides, Kevin.  
*Networked Heterodoxy — Dissertation Scripts* (v1.0).  
GitHub repository. [URL]

---

## Contact

Questions or replication inquiries:
kevin.whitesides [institutional email]
