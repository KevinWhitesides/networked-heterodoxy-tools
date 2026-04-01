# Sample Files

This folder contains curated files for exploring and testing the workflows and concepts in the **networked heterodoxy tools** repository.

These files serve two complementary purposes:

- **Pipeline input** (incidence matrices): for running the repository’s analytical workflows  
- **Direct exploration** (Gephi network): for immediately engaging with a pre-constructed network without having to run any code

---

## 1. `first_7_books.xlsx`

A small binary incidence matrix derived from the seven earliest books appearing in the full 2012 phenomenon dataset.

### Purpose
This file provides a simple, manageable dataset for running and testing the repository’s pipelines and stand-alone processes, including:

- One-mode network diagnostics and construction (feature × feature, case × case) and topology metrics
- Bipartite network construction
- Features distribution
- Case-pair features comparison
- Jaccard similarity computation and clustering

### Important Note
This dataset **does not contain any zero-overlap book pairs**.

As a result:
- Zero-overlap detection scripts will return no results  
- Gradient-finding workflows based on zero-overlap pairs will not produce meaningful output  

This is expected behavior and reflects the structure of this particular subset.

### Recommended Use
Use this dataset to:
- Learn the basic workflow of the repository  
- Verify that scripts are running correctly  
- Generate interpretable small-scale network outputs
- Compare the results with dissertation outputs and analyses in Chapter 3  

---

## 2. *(Coming Soon)* Curated Balanced Sample Dataset

A larger sample incidence matrix (approximately 20–30 books) will be added to this folder.

### Purpose
This dataset is being designed to support **all major workflows in the repository**, including:

- Zero-overlap detection  
- Gradient identification  
- Jaccard similarity and clustering  
- One-mode network construction and topological analysis  

### Design Goals
The dataset will intentionally include:

- At least one pair of books with zero trope overlap  
- Intermediate “bridge” books forming gradient paths between them  
- Locally dense clusters to support community detection  
- Variation in trope counts (including more central and more peripheral cases)  

### Intended Role
This dataset will serve as the **primary all-purpose sample** for:

- Demonstrating the full analytical set of workflows and pipelines  
- Producing non-trivial, interpretable outputs across methods  
- Helping users understand how different analytical components relate to one another  

---

## 3. `39_book_no_overlap_demo_network.gephi`

A pre-constructed Gephi network derived from a 39-book subset of the corpus consisting of books involved in **significant zero-overlap relationships**.
This network was also used in the dissertation and can be explored more deeply and compared with the analyses in Chapter 3.

### Purpose
This file is intended for **immediate exploration in Gephi**, allowing users to:

- Examine structurally distinct discourse regions  
- Identify bridge books and tropes connecting otherwise disconnected areas  
- Explore modularity-based clustering  
- Experiment with k-core decomposition and network filtering  

### Important Note
This file is a **ready-made network**, not a pipeline input.

It is **not intended to be used as input to the repository’s scripts**. Instead, it provides a direct and immediate way to explore the kinds of structures those scripts are designed to reveal.

### Suggested Exploration (in Gephi)
- Look at the general shape of the network. Explore the contents of its clusters.
- Explore modularity class assignments
- Use the k-core filter (e.g., k = 7) to examine the network core
	- On the right-side "Filters" panel, "K-Core" should already appear in the "Queries" box. Click it.
	- In the K-Core settings just below, set the # to 1 (the full network) and click the "Filter" button.
	- Slowly start raising the number one at a time, and observe what happens (peripheral nodes drop away, revealing the most interconnected core of the discourse space)
- Inspect edges between clusters to identify bridging structures 
- Hover over individual nodes to explore their "ego networks" (the set of nodes directly connected to them)
- Look inside the "Data Laboratory" to find a detailed map of nodes, edges, and their attributes
- Explore the options in "Preview" mode to see how the network can be visually transformed and exported for presentation/publication

---

## Summary

- **`first_7_books.xlsx`** → Simple pipeline testing (no zero-overlap structure)
- **Curated sample (forthcoming)** → Full-featured dataset supporting all workflows  
- **`39_book_no_overlap_demo_network.gephi`** → Immediate network exploration in Gephi  

Together, these files provide complementary entry points into the repository’s methods and the broader model of networked heterodoxy.
Alternately, users are encouraged to build and explore their own datasets. Ultimately, these tools are intended to aid in research
and allow them to perform their own networked analyses of their own data. If you do so, please share your results with kevin@kevinwhitesides.com.