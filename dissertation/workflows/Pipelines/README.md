# Pipelines

This directory documents the **pipeline implementations** in the repository.

Pipelines are multi-stage workflows that combine individual scripts into
reproducible analytical sequences. Each pipeline corresponds to a specific
analytical question and follows a defined sequence of steps.

---

## Pipeline Overview

The diagram below shows the pipeline structure of the repository, with each
pipeline represented as a distinct analytical pathway.

![Networked Heterodoxy pipelines](networked_heterodoxy_pipelines.png)

---

## Available Pipelines

### Pipeline 2: Jaccard Matrix + Clustering

**Purpose:**  
Identify similarity structure and group cases based on shared feature repertoires.

**Pipeline script:**  
`02_jaccard_pipeline.py`

---

## Steps

1. **Jaccard similarity computation**  
   → `03_similarity/01_jaccard_similarity_heatmap.py`  
   - Computes case × case Jaccard similarity matrix  
   - Optionally generates heatmap  
   - Writes analysis summary  

2. **Hierarchical clustering**  
   → `03_similarity/02_cluster_from_jaccard.py`  
   - Converts similarity to distance  
   - Performs clustering  
   - Assigns cases to clusters
   - Computes per-case similarity to assigned cluster     
   - Writes cluster outputs and summary  

---

## Inputs

- Binary incidence matrix (case × feature)
- Metadata columns are allowed at the beginning of the file
- Presence is marked using a specified token (default: `"X"`)

---

## Outputs

The pipeline creates a timestamped output directory containing:

### Stage 1 (`01_jaccard/`)
- Jaccard similarity matrix (CSV)
- Heatmap (PNG, optional)
- analysis summary

### Stage 2 (`02_clustering/`)
- Cluster summary (CSV)
- Case similarity-to-cluster (CSV)
- Dendrogram (PNG, optional)
- analysis summary

### Pipeline-level
- `pipeline_summary.txt` (top-level summary with timestamp)

---

## Clustering Modes

The clustering stage supports two modes. Use **only one at a time**.

---

### Fixed number of clusters

Set:

    N_CLUSTERS = 3
    DISTANCE_CUTOFF = None

Meaning:
- Forces the data into a fixed number of clusters

---

### Distance cutoff

Set:

    N_CLUSTERS = None
    DISTANCE_CUTOFF = 0.65

Meaning:
- Cuts the dendrogram at a specified distance threshold
- The number of clusters emerges from the data

For Jaccard similarity:

    distance = 1 - similarity

So:
- `DISTANCE_CUTOFF = 0.65` → minimum similarity = 0.35
- Lower cutoff → more clusters
- Higher cutoff → fewer clusters

---

## Running the Pipeline

From the repository root:

    python workflows/pipelines/02_jaccard_pipeline.py

The pipeline will:

- run both stages
- organize outputs into subfolders
- generate a top-level summary file

---

## Notes

- Pipelines call the underlying scripts directly via their `run(...)` functions
- Each stage remains independently usable as a standalone script
- Some analytical decisions (e.g., parameter selection) are made prior to running pipelines