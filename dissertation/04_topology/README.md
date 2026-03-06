# 04_topology

Scripts for analyzing the **structural topology of networks**.

Where similarity analyses compare the **feature repertoires of cases**, topology
analysis examines the **connectivity structure of the network itself**.

Topology measures help identify three structural roles in networks:

- **cohesive clusters** of nodes
- **deeply embedded nodes** within dense regions of the network
- **brokerage nodes** that connect otherwise separate regions

Most scripts in this folder operate on **existing network graphs (`.gexf`)**
produced by the scripts in `02_networks/`.

---

## Analytical Focus

Topology analyses address questions about **structural cohesion** and
**brokerage** within networks.

| Method | Research question |
|------|-------------------|
| **k-components** | Which groups of nodes remain connected even if multiple nodes are removed? |
| **k-core decomposition** | Which nodes lie within dense regions of the network based on their degree connections? |
| **Burt brokerage metrics** | Which nodes connect otherwise separate regions of the network? |

These measures describe **different but complementary structural properties**
of a network.

---

## Structural Cohesion

Structural cohesion describes **how strongly groups of nodes hold together**.

Two related methods help measure cohesion.

### k-components

k-component analysis identifies **cohesive subgraphs** that require the
removal of at least *k* nodes to disconnect them.

Higher *k* values indicate:

- stronger structural cohesion
- greater redundancy of connections
- tightly integrated clusters of nodes

In discourse networks, k-components can reveal **robust thematic clusters**
that remain connected even if key tropes or concepts are removed.

Because k-components measure **node-connectivity robustness**, they identify
groups that remain structurally intact even under disruption.

---

### k-core decomposition

A k-core identifies nodes that remain after recursively removing nodes with
degree less than *k*.

The **core number** of a node indicates the deepest k-core in which that node
appears.

Core numbers therefore measure **node embeddedness**, identifying nodes that
lie within dense regions of the network.

Unlike k-components, k-cores are based on **degree thresholds** rather than
connectivity robustness. A node may lie within a high k-core even if the
surrounding region is not highly resilient to node removal.

---

## Structural Brokerage

While cohesion measures how strongly nodes cluster together, **brokerage
metrics identify nodes that connect otherwise separate parts of a network**.

### Burt brokerage metrics

Burt’s structural hole theory identifies nodes that bridge **gaps between
network regions**.

Key metrics include:

- **constraint** — the degree to which a node’s neighbors are themselves
  interconnected. Lower values indicate stronger brokerage positions.

- **effective size** — the number of **non-redundant neighbors** connected
  to a node.

- **efficiency** — the proportion of a node’s ties that provide
  non-redundant connections.

In discourse networks, brokerage nodes often correspond to **concepts or
entities that link otherwise distinct thematic clusters**.

These nodes frequently play important roles in **connecting discourse
communities or conceptual domains**.

---

## Interpreting Topology Measures Together

These topology measures describe **different structural properties of a
network**.

- **k-components** identify *structurally robust clusters of nodes*
- **k-core numbers** identify *nodes embedded within dense regions of the network*
- **brokerage metrics** identify *nodes that bridge otherwise separate regions*

Together they help distinguish between:

- **cohesive clusters**
- **structurally embedded nodes**
- **bridging nodes connecting different discourse domains**

---

## Current Scripts

### 01_k_components_from_gexf.py

Computes **k-components** and **k-core numbers** from an existing network.

Input:

- `.gexf` network file  
- typically produced by scripts in `02_networks/`

The script:

1. Reads a network graph from GEXF.
2. Computes **k-core numbers** for all nodes.
3. Computes **k-components** of the graph.
4. Exports:

   - node-level **k-core numbers**
   - node-level **degree summaries**
   - **GEXF subgraphs** for each k-component
   - a **summary table** listing all components

The exported component graphs can be opened directly in Gephi, while
accompanying CSV files provide tabular summaries of component structure.

---

### 02_burt_brokerage_metrics.py

Computes **Burt-style brokerage metrics** for nodes in an existing network.

Input:

- `.gexf` network file  
- typically produced by scripts in `02_networks/`

The script:

1. Reads the network graph from GEXF.
2. Ensures the network is undirected.
3. Computes node-level brokerage metrics:

   - **constraint**
   - **effective size**
   - **efficiency**
   - **degree**

4. Exports:

   - a CSV table of brokerage metrics for all nodes
   - a new `.gexf` network with metrics added as node attributes
   - an `analysis_summary.txt` file documenting the run

These outputs allow brokerage metrics to be explored both
**numerically** (via the CSV) and **visually** in network analysis
software such as Gephi.

---

## Visualizing Brokerage in Gephi

After running the brokerage script, the exported `.gexf` network can be
opened directly in **Gephi** for visual exploration.

A useful workflow for identifying brokerage nodes is:

1. Run **Modularity** to detect communities.
2. Color nodes by **modularity class**.
3. Size nodes by: 1 − constraint


Lower constraint indicates stronger brokerage positions, so this
transformation makes strong brokers appear larger.

Brokerage nodes will often appear as **large nodes positioned between
differently colored communities**, connecting otherwise separate regions
of the network.

---

## Sanity Checks

When interpreting brokerage metrics, several simple checks help confirm
that the analysis ran correctly:

- **Constraint values** should lie between approximately **0 and 1**.
  Strong brokers typically have values close to **0**.

- Nodes with **degree = 1** will always have **constraint = 1**, because
  they have no opportunity to broker between neighbors.

- Brokerage metrics often correlate with **betweenness centrality**, but
  they are not identical.  
  Betweenness measures **global shortest paths**, while constraint is an
  **ego-network measure** based on the redundancy of a node’s neighbors.

---

## Relationship to Gephi

Gephi includes built-in tools for **k-core decomposition**, but does not
directly compute several of the other topology measures used here.

The scripts in this folder extend Gephi’s capabilities by:

- computing **k-components**, which Gephi does not provide
- computing and exporting **k-core numbers** programmatically
- computing **Burt brokerage metrics**

The resulting `.gexf` files can be opened directly in Gephi for visual
exploration of structural patterns.

---

## Data Assumptions

Scripts in this folder assume that the input network:

- is stored as a **GEXF graph**
- represents a previously constructed network (e.g., feature co-occurrence)
- has typically been generated using the scripts in `02_networks/`

Topology analysis therefore represents a **downstream analytical step**
following network construction.