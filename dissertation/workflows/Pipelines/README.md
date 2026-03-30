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

### Pipeline 1: One-Mode + Topology

**Purpose:**  
Construct one-mode networks (feature × feature and/or case × case) and analyze
their structural properties using topology metrics.

**Pipeline script:**  
`01_one_mode_topology_pipeline.py`

---

### Pipeline 2: Jaccard Matrix + Clustering

**Purpose:**  
Identify similarity structure and group cases based on shared feature repertoires.

**Pipeline script:**  
`02_jaccard_pipeline.py`

---

# Pipeline 1: One-Mode + Topology

## Steps

1. **One-mode network construction**  
   → `02_networks/01_build_one_mode_projection.py`  
   - Builds feature × feature and/or case × case networks  
   - Applies node and edge thresholds  
   - Outputs thresholded networks (GEXF + edge list)  

2. **k-component analysis**  
   → `04_topology/01_k_components_from_gexf.py`  
   - Identifies structurally cohesive subgraphs  
   - Computes k-components and k-core structure  
   - Writes component summaries and subgraphs  

3. **Burt brokerage metrics**  
   → `04_topology/02_burt_brokerage_metrics.py`  
   - Computes constraint, effective size, efficiency, and degree  
   - Annotates the network with node-level metrics  

---

## Inputs

- Binary incidence matrix (case × feature)
- Metadata columns are allowed at the beginning of the file
- Presence is marked using a specified token (default: `"X"`)

---

## Outputs

The pipeline creates a timestamped output directory:

```
01_one_mode_topology_pipeline_YYYYMMDD_HHMMSS/
    feature/
        thrX/
            01_one_mode_network/
            02_k_components/
            03_burt/
    case/
        thrY/
            01_one_mode_network/
            02_k_components/
            03_burt/
    pipeline_summary.txt
```

### Per-threshold outputs

#### 01_one_mode_network/
- Network file (GEXF)
- Edge list (CSV)

#### 02_k_components/
- Component summary (CSV)
- Node core numbers (CSV)
- Node summary (CSV)
- Component subgraphs (GEXF)
- analysis summary

#### 03_burt/
- Burt metrics (CSV)
- Network annotated with metrics (GEXF)
- analysis summary

---

## Configuration Notes

- Feature and case projections are both enabled by default, but users can restrict to one if desired
- Each projection type has **independent node and edge thresholds**
- Threshold selection can be determined using the diagnostics scripts prior to running the pipeline

---

## Running the Pipeline

From the repository root:

```
python workflows/pipelines/01_one_mode_topology_pipeline.py
```

The pipeline will:

- build one-mode networks
- run topology analyses for each thresholded network
- organize outputs into structured subfolders
- generate a top-level summary file

---

# Pipeline 2: Jaccard Matrix + Clustering

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

```
python workflows/pipelines/02_jaccard_pipeline.py
```

The pipeline will:

- run both stages
- organize outputs into subfolders
- generate a top-level summary file

---

## Notes

- Pipelines call the underlying scripts directly via their `run(...)` functions
- Each stage remains independently usable as a standalone script
- Some analytical decisions (e.g., parameter selection) are made prior to running pipelines