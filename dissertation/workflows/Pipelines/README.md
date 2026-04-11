# Pipelines

This directory documents the **pipeline implementations** in the repository.

Pipelines are multi-stage workflows that combine individual scripts into
reproducible analytical sequences. Each pipeline corresponds to a specific
analytical question and follows a defined sequence of steps.

Together, the current pipelines address four complementary analytical problems:

- **Pipeline 1:** network structure
- **Pipeline 2:** similarity and clustering
- **Pipeline 3:** case-level absence, gradients, and recurrent mediation
- **Pipeline 4:** feature-level absence, gradients, and recurrent mediation

---

## Pipeline Overview

The diagram below shows the pipeline structure of the repository, with each
pipeline represented as a distinct analytical pathway.

![Networked Heterodoxy pipelines](networked_heterodoxy_pipelines.png)

---

## Available Pipelines

### Pipeline 1: One-Mode + Topology

**Purpose:**  
Construct one-mode networks (feature × feature and/or case × case) from the
incidence matrix and analyze their structural properties using network topology
metrics.

**Pipeline script:**  
`01_one_mode_topology_pipeline.py`

---

### Pipeline 2: Jaccard Matrix + Clustering

**Purpose:**  
Compute pairwise similarity between cases using Jaccard metrics and identify
group structure through clustering and similarity-based analysis.

**Pipeline script:**  
`02_jaccard_pipeline.py`

---

### Pipeline 3: Zero-Overlap Case Suite

**Purpose:**  
Identify case pairs with no shared features, analyze absence-based network
structure, explore discursive gradients connecting otherwise disconnected
cases, and identify recurring mediating vocabularies across those gradients.

**Pipeline script:**  
`03_case_zero_overlap_pipeline.py`

---

### Pipeline 4: Zero-Overlap Feature Suite

**Purpose:**  
Identify feature pairs that never co-occur, analyze feature-level absence
structure, explore feature gradients connecting otherwise disjoint regions
of the corpus, and identify recurring mediating cases across those gradients.

**Pipeline script:**  
`04_feature_zero_overlap_pipeline.py`

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

```text
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

```bash
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

### Fixed number of clusters

Set:

    N_CLUSTERS = 3
    DISTANCE_CUTOFF = None

Meaning:
- Forces the data into a fixed number of clusters

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

```bash
python workflows/pipelines/02_jaccard_pipeline.py
```

The pipeline will:

- run both stages
- organize outputs into subfolders
- generate a top-level summary file

---

# Pipeline 3: Zero-Overlap Case Suite

## Purpose

Identify, evaluate, and structurally analyze **case pairs with no shared
features**, and explore the **discursive pathways (gradients)** that connect
them.

This pipeline operationalizes **absence structure** in the dataset, moving from:

- zero-overlap detection
- absence networks
- graded transitional chains between otherwise disconnected cases
- recurring mediating vocabularies across those gradient chains

---

## Steps

1. **Significant zero-overlap analysis**  
   → `03_similarity/04_significant_zero_case_overlap.py`  
   - Identifies case pairs with zero shared features  
   - Uses null-model sampling to estimate expected zero-overlap frequency  
   - Computes empirical p-values and FDR-corrected significance  
   - Outputs full zero-overlap table with significance columns

2. **Absence network construction**  
   → `02_networks/03_build_case_absence_networks.py`  
   - Builds case × case absence graph from significant zero-overlap pairs  
   - Filters cases by minimum number of zero-overlap neighbors  
   - Builds complementary case × feature bipartite graph  
   - Outputs network files and summary

3. **Case gradient search**  
   → `03_similarity/05_find_case_gradients.py`  
   - Identifies chains connecting zero-overlap endpoint pairs  
   - Uses Jaccard similarity as a continuous bridge metric  
   - Supports:
     - strict gradient mode (strong monotonic constraints)
     - ranked gradient mode (scored transitions)
   - Can:
     - use an existing Jaccard matrix
     - or compute one internally
   - Outputs ranked gradient chains and summary

4. **Gradient network construction (optional)**  
   → `02_networks/04_build_case_gradient_networks.py`  
   - Selects a single gradient chain  
   - Builds case × feature bipartite graph for that chain  
   - Computes within-chain Jaccard diagnostics  
   - Outputs network, matrices, and visualization
   
5. **Gradient recurrence analysis (optional)**  
   → `03_similarity/08_gradient_recurrence_analyzer.py`  
   - Aggregates across all retained gradients  
   - Identifies features that recur across multiple gradient chains  
   - Computes co-recurrence relationships among mediating features  
   - Outputs summary tables of recurrent mediators and co-occurrence structure

---

## Inputs

- Binary incidence matrix (case × feature)
- Metadata columns allowed at the beginning of the file
- Presence marked with a token (default: `"X"`)

---

## Outputs

The pipeline creates a timestamped output directory:

```text
03_case_zero_overlap_pipeline_YYYYMMDD_HHMMSS/
    01_significant_zero_overlap/
    02_absence_networks/
    03_case_gradients/
    04_gradient_network/    # optional
	05_gradient_recurrence/ # optional
    pipeline_summary.txt
```

---

## Gradient Modes

The gradient stage supports flexible execution.

### Jaccard handling

- `"compute"` → compute Jaccard internally (default)
- `"existing"` → use precomputed Jaccard matrix (e.g. from Pipeline 2)

### Endpoint selection

- `"all"` → all zero-overlap pairs
- `"significant"` → only statistically significant pairs (default)
- `"specific"` → user-defined pair

### Gradient search modes

- **Strict**
  - Enforces monotonic progression between endpoints
  - Produces fewer, highly constrained chains

- **Ranked**
  - Scores candidate chains by:
    - adjacency strength
    - monotonicity quality
    - positional smoothness
  - Produces richer exploratory output

### Gradient network stage (optional)

- `"skip"` → do not construct a network
- `"top_row"` → use highest-ranked gradient (default)
- `"specific_endpoints"` → build network for chosen pair

## Recurrence Analysis Notes

- This stage operates **after gradient identification**
- It aggregates across **multiple gradient chains**, rather than analyzing a single chain
- It identifies:
  - frequently reused mediating features
  - co-recurrent mediator sets
- Conceptually, it shifts analysis from:
  - individual gradient paths  
  to  
  - **stable cross-gradient mediation patterns**

---

## Running the Pipeline

From the repository root:

```bash
python workflows/pipelines/03_case_zero_overlap_pipeline.py
```

The pipeline will:

- compute significant zero-overlap pairs
- build absence networks
- identify gradient chains
- optionally run recurrence analysis
- optionally construct a gradient network
- organize outputs into structured subfolders
- generate a top-level summary file

---

## Notes

- This pipeline complements Pipeline 2 by focusing on **absence rather than similarity**
- It enables analysis of **structural disconnection and transitional pathways**
- The gradient stage provides a bridge between:
  - discrete absence (zero overlap)
  - continuous similarity (Jaccard space)
- The recurrence stage identifies **stable mediating vocabularies across gradients**

---

# Pipeline 4: Zero-Overlap Feature Suite

## Purpose

Identify, evaluate, and structurally analyze **feature pairs that never
co-occur**, and explore the **feature gradients** that connect otherwise
disjoint regions of the corpus.

This pipeline operationalizes **feature-level absence structure**, moving from:

- zero-overlap feature detection
- feature absence networks
- graded transitions across feature space
- recurring mediating cases across those gradients

---

## Steps

1. **Significant zero-overlap feature analysis**  
   → `03_similarity/06_significant_zero_feature_overlap.py`  
   - Identifies feature pairs that never co-occur in the same case  
   - Uses a degree-preserving null model to estimate expected zero-overlap frequency  
   - Computes empirical p-values and FDR-corrected significance  
   - Outputs full zero-overlap feature table with significance columns

2. **Feature absence network construction**  
   → `02_networks/05_build_feature_absence_network.py`  
   - Builds a one-mode feature absence graph from significant zero-overlap pairs  
   - Filters features by minimum number of zero-overlap neighbors  
   - Builds a complementary feature × case bipartite graph  
   - Outputs network files and summary

3. **Feature gradient search**  
   → `03_similarity/07_find_feature_gradients.py`  
   - Identifies chains connecting zero-overlap feature endpoint pairs  
   - Computes feature × feature similarity internally from the incidence matrix  
   - Uses both Jaccard similarity and raw co-occurrence as adjacency constraints  
   - Supports:
     - strict gradient mode (strong monotonic constraints)
     - ranked gradient mode (scored transitions)
   - Outputs ranked feature-gradient chains and summary

4. **Feature gradient network construction (optional)**  
   → `02_networks/06_build_feature_gradient_networks.py`  
   - Selects a single feature gradient chain  
   - Builds a feature × case bipartite graph for that chain  
   - Computes within-chain feature Jaccard diagnostics  
   - Outputs network, matrices, ranked pairs, and visualization
   
5. **Gradient recurrence analysis (optional)**  
   → `03_similarity/08_gradient_recurrence_analyzer.py`  
   - Aggregates across retained feature gradients  
   - Identifies cases that recur across multiple gradients  
   - Computes co-recurrence relationships among mediating cases  
   - Outputs summary tables of recurrent mediators and co-occurrence structure

---

## Inputs

- Binary incidence matrix (case × feature)
- Metadata columns allowed at the beginning of the file
- Presence marked with a token (default: `"X"`)

---

## Outputs

The pipeline creates a timestamped output directory:

```text
04_feature_zero_overlap_pipeline_YYYYMMDD_HHMMSS/
    01_significant_zero_overlap/
    02_absence_networks/
    03_feature_gradients/
    04_gradient_network/    # optional
	05_gradient_recurrence/ # optional
    pipeline_summary.txt
```

---

## Gradient Modes

The gradient stage supports flexible execution.

### Endpoint selection

- `"all"` → all zero-overlap feature pairs
- `"significant"` → only statistically significant pairs (default)
- `"specific"` → user-defined feature pair

### Gradient search modes

- **Strict**
  - Enforces monotonic progression between endpoint features
  - Produces fewer, highly constrained chains

- **Ranked**
  - Scores candidate chains by:
    - adjacency strength
    - monotonicity quality
    - positional smoothness
  - Produces richer exploratory output

### Gradient network stage (optional)

- `"skip"` → do not construct a network
- `"top_row"` → use highest-ranked gradient (default)
- `"specific_endpoints"` → build network for chosen feature pair
- `"specific_chain"` → build network for an exact stored chain string

## Recurrence Analysis Notes

- This stage operates **after feature gradient identification**
- It aggregates across multiple feature-gradient chains
- It identifies:
  - recurrent mediating cases
  - co-recurrent case sets
- It highlights cases that function as **structural bridges across feature space**

---

## Feature-Space Notes

Pipeline 4 uses a somewhat stricter set of defaults than the case pipeline
because feature-space structure is typically sparser and more vulnerable to
trivial absences.

In particular:

- `MIN_FEATURE_FREQ` helps suppress zero-overlap patterns driven only by rarity
- `MIN_ADJ_JACCARD` and `MIN_ADJ_COOCC` ensure that adjacent gradient steps are
  both proportionally meaningful and supported by real shared cases
- `MIN_GRADIENT_FEATURES_PER_CASE` keeps only cases that substantively support
  the selected feature gradient in the final support graph

---

## Running the Pipeline

From the repository root:

```bash
python workflows/pipelines/04_feature_zero_overlap_pipeline.py
```

The pipeline will:

- compute significant zero-overlap feature pairs
- build feature absence networks
- identify feature-gradient chains
- optionally construct a feature-gradient network
- optionally run recurrence analysis
- organize outputs into structured subfolders
- generate a top-level summary file

---

## Notes

- This pipeline is the feature-level complement to Pipeline 3
- It focuses on **feature disjunction**, rather than case disjunction
- It enables analysis of:
  - structurally disjoint feature regions
  - feature-space transitions
  - support patterns across cases
- The recurrence stage identifies **cases that repeatedly bridge feature regions**

---

## General Notes

- Pipelines call the underlying scripts directly via their `run(...)` functions
- Each stage remains independently usable as a standalone script
- Some analytical decisions (e.g., parameter selection) are made prior to running pipelines