# 02_networks

This folder contains scripts for constructing network graphs from binary incidence matrices.

Networks are generated either as:

- Projected one-mode networks (item × item weighted co-occurrence)
- Bipartite incidence networks (case × item)
- Topic-derived networks (one-mode & bipartite)

These scripts assume a cleaned “no metadata” input matrix unless otherwise specified.

---

## Scripts

### 01_build_one_mode_projection.py

#### Purpose

Constructs a weighted one-mode projection from a binary incidence matrix and exports thresholded networks suitable for Gephi.

#### Input

- `.xlsx` or `.csv`
- Rows = cases (e.g., books, songs)
- Columns = items/tropes
- Presence marked with `"X"`
- No totals rows
- No metadata columns (recommended)

Optional safeguard: `DROP_COLUMNS = []`

#### Global Node Filter

Applies `MIN_NODE_FREQ` (default = 2) prior to projection.

This removes items appearing in fewer than two cases, ensuring recurrence.

#### Edge Thresholds

For each value in `EDGE_THRESHOLDS`, the script:

- Retains edges with co-occurrence ≥ threshold
- Builds a weighted undirected graph
- Exports both:
  - Edge list CSV
  - GEXF network

#### Node Attributes Added

- `frequency`
- `degree`
- `weighted_degree`

#### Typical Use

Edit configuration variables at the top of the script, then run:

`python build_one_mode_projection.py`

Multiple thresholded graphs can be produced in a single run.

---

## Notes

- Node filtering is applied globally before edge thresholding.
- Edge thresholding is applied per output graph.
- This script is dataset-agnostic and was used across multiple case studies (2012 literature corpus, hip hop corpus).

### 02_build_bipartite_network.py

#### Purpose

Constructs a bipartite (case × item) network directly from a binary incidence matrix and exports it in Gephi-ready format.

This preserves the original incidence structure without projecting it into a one-mode co-occurrence network.

The script also produces a pairwise comparison table identifying shared and unshared items between all case pairs.

#### Input

- `.xlsx` or `.csv`
- Rows = cases (e.g., books, songs)
- Columns = items/tropes
- Presence marked with `"X"`

By default the script assumes:

- The first `N_METADATA_COLS` columns contain metadata
- Item columns begin immediately afterward

Example structure:

| Source Title | Author | Year | Publisher | Plato | Atlantis | Aztec |
|--------------|-------|------|-----------|-------|---------|------|
| Book A | ... | ... | ... | X | | X |
| Book B | ... | ... | ... | | X | X |

Configuration variables at the top of the script allow adjustment of:

- metadata column count
- presence token
- column containing case identifiers

#### Outputs

**1. Bipartite network**

Exports a GEXF graph:

- Nodes:
  - cases (e.g., books, songs)
  - items/tropes
- Edges:
  - case–item incidence
- Edge weights: none (binary incidence)

Node attributes include:

- `type` (`case` or `item`)
- `bipartite` partition identifier

This network preserves the original dataset structure and can be used for:

- visualization
- bipartite projections
- two-mode network analysis

**2. Pairwise overlap table**

Exports a CSV listing every case pair with:

- number of shared items
- list of shared items
- number of unshared items
- list of unshared items

This CSV file is primarily intended as a qualitative aid for exploring specific overlaps between cases.

#### Typical Use

Edit configuration variables at the top of the script, then run:

`python build_bipartite_network.py`

The script will generate:

- a bipartite `.gexf` network
- a pairwise overlap `.csv`

### 03_build_absence_networks.py

Constructs network representations of **statistically significant zero-overlap relationships** between cases.

This script is designed to work downstream of the similarity analysis performed by
`03_similarity/04_significant_zero_overlap.py`. While that script identifies pairs of
cases that share **no features and whose absence of overlap is unlikely under a
degree-preserving null model**, the present script converts those results into
network structures suitable for exploration and visualization.

The script produces two complementary graphs:

1. **Absence graph (case × case)**  
   - Nodes represent cases (books, songs, etc.).  
   - Edges represent **statistically significant zero-overlap relationships** between cases.  
   - The resulting graph provides a quick structural overview of how cases
     diverge from one another within the dataset.

2. **Bipartite graph of the retained subset (case × feature)**  
   - Nodes represent both cases and features (tropes).  
   - Edges represent the presence of a feature in a case.  
   - Only cases participating meaningfully in the absence structure are retained.
     Specifically, cases must have at least a configurable number of significant
     zero-overlap relationships (default: **two**).  
   - Features are also filtered to those appearing in at least a configurable number of retained cases (default:**two**).

This graph shows the **feature repertoires of the subset of cases that define
the absence network**, making it possible to see which features cluster within
different regions of the retained discourse field.

While the absence graph shows **which cases diverge**, the bipartite graph reveals
**why they diverge** by displaying the feature repertoires that structure those
differences. In many analyses, the bipartite graph provides the most
substantively informative representation of the retained discourse field.

---

#### Outputs

The script generates three files:

`absence_graph_sig.gexf`  
: A case × case network in which edges represent statistically significant
  zero-overlap relationships.

`bipartite_thr2.gexf`  
: A bipartite case × feature graph built from the subset of cases retained
  in the absence network. Features are filtered to those appearing in at least
  two retained cases in order to reduce noise.

`analysis_summary.txt`  
: A compact record of the run, including input files, filtering parameters,
  and network statistics.

---

#### Analytical role

This script forms the **network construction stage** of the absence-analysis
workflow:

03_similarity/04_significant_zero_overlap.py
↓
03_build_absence_networks.py


The first script identifies statistically meaningful absence relationships.
The present script converts those results into network structures that allow
the retained subset of cases to be explored visually and structurally.

The resulting graphs can be opened directly in **Gephi** or other network
analysis software for further exploration.

### build_topic_network.py
(placeholder)