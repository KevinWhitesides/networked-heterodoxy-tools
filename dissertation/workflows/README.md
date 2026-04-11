# Workflows

This directory documents the **analytical workflows** supported by this
repository and how individual scripts fit together into larger pipelines.

The repository is organized into modular stages that can be used
independently or combined into multi-step analyses.

---

## Workflow Overview

The diagram below shows the overall analytical workflow structure of the toolkit.
It presents the relationships between analytical components without designating
specific pipelines.

![Networked Heterodoxy workflow overview](networked_heterodoxy_workflows.png)

---

## Analytical Components

This repository is organized into modular **analytical components**, not a single linear pipeline.

Each component corresponds to a folder of scripts that perform a specific type of operation.  
Different workflows combine these components in different orders.

### 01_diagnostics
- Parameter exploration tool (co-occurrence threshold diagnostics)
- Typically used prior to network construction to identify ideal outputs
- Only used for the One-Mode Network workflow

### 02_networks
- Construction of network representations
- Includes multiple types of networks:
  - one-mode projection networks
  - bipartite networks
  - case absence networks
  - feature absence networks
  - gradient networks
- Network construction may occur **early or late**, depending on the workflow
  - e.g. zero-overlap and gradient identification networks use 03_similarity outputs

### 03_similarity
- Direct comparison of cases or features without constructing a network
- Includes:
  - Jaccard similarity
  - clustering
  - zero-overlap detection
  - gradient identification
  - gradient recurrence analysis

### 04_topology
- Structural analysis of networks
- Includes:
  - k-components
  - brokerage metrics

### 05_distribution
- Analysis of feature frequency and distribution among producers

### 06_topic_modeling
- Probabilistic topic modeling of textual corpora

---

## Workflows

Analyses are performed by combining components into **specific workflows**.

Examples include:

### One-Mode Projection Workflow
1. (Optional) Diagnostics (01_diagnostics)
2. Network construction (02_networks → projection)
3. Topological analysis (04_topology)

---

### Jaccard Similarity → Clustering Workflow
1. Similarity computation (03_similarity)
2. Clustering (03_similarity)

---

### Case Zero-Overlap → Gradient Workflow

1. Zero-overlap detection (03_similarity)  
2. Significance testing (03_similarity)  
3. Gradient identification (03_similarity)  
4. (Optional) network construction (02_networks → gradient networks)  
5. (Optional) **gradient recurrence analysis (03_similarity)**  

This workflow moves from:

- **direct similarity**
→ to **complete disjunction**
→ to **indirect continuity (gradients)**
→ to **recurrent mediating vocabularies across gradients**

---

### Feature Zero-Overlap → Gradient Workflow

1. Zero-overlap detection (03_similarity)  
2. Significance testing (03_similarity)  
3. Gradient identification (03_similarity)  
4. (Optional) network construction (02_networks → gradient networks)  
5. (Optional) **gradient recurrence analysis (03_similarity)**  

This workflow mirrors the case-level analysis along the **feature dimension**, allowing
identification of:

- feature disjunctions
- intermediary feature pathways
- recurrent mediating **cases** across gradients

---

## Gradient Recurrence as an Analytical Layer

Gradient recurrence analysis is a **post-gradient step** that aggregates across
many retained gradients to identify:

- recurring mediators (features or cases)
- co-recurrent mediator sets
- potential **meta-boundary vocabularies**

Conceptually, it shifts analysis from:

- **single gradient chains (local structure)**  
to  
- **recurring mediation patterns across the corpus (global structure)**  

This step is optional because:

- it requires sufficient gradient density to be meaningful
- it is more interpretive than structural
- it aggregates across results rather than producing primary relationships

---

## Pipelines vs Standalone Scripts

All scripts in this repository can be used as **standalone tools**.

However, many scripts are also designed to function as components in
**multi-stage pipelines**, where outputs from one stage are passed to
the next.

For example:

- Jaccard similarity → clustering  
- zero-overlap detection → gradient search → recurrence analysis  
- zero-overlap detection → gradient search → network construction  
- projection networks → topological analysis  

---

## Pre-Pipeline vs Pipeline Steps

Some steps that require user decisions within a workflow are intentionally **not automated within pipelines**:

- Parameter diagnostics (e.g., threshold selection)
- Interpretive decisions (e.g., selecting specific endpoints for gradients)
- Selection of which gradients to visualize or emphasize

These steps are performed manually before or after running a pipeline,
ensuring that pipelines remain:

- reproducible  
- deterministic  
- non-interactive following initial parameter setting  

---

## Pipeline Implementations

Runnable pipeline scripts and detailed instructions are documented in:

`workflows/pipelines/`

That directory also includes a color-coded version of the workflow diagram,
showing how specific pipelines are defined within the overall structure.

See that directory for:

- available pipelines  
- execution instructions  
- expected inputs and outputs  