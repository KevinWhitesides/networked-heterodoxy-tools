# 05_distribution

Scripts for analyzing the **distribution of features across producers** within a corpus.

While network and similarity analyses examine relationships among features or cases,
distribution analysis examines **how features are used by different producers**
(e.g., artists, authors, speakers).

These scripts help determine whether a feature's prominence in a corpus arises from:

- **broad diffusion** across many producers, or
- **concentrated usage** within a smaller subset of producers.

This distinction can reveal important differences between:

- **field-wide vocabulary** shared across a discourse community
- **producer signatures** associated with specific individuals
- **clique-diagnostic features** concentrated within small producer clusters

Distribution analysis therefore complements network approaches by examining
**who is producing which features**, rather than how features co-occur.

---

## Analytical Focus

These scripts address questions about **how features circulate within a producer field**.

| Measure | Research question |
|------|-------------------|
| **Prominence** | How frequently does a feature appear in the corpus overall? |
| **Diffusion** | How many producers use the feature? |
| **Concentration** | Is the feature's prominence driven by a small subset of producers? |
| **Producer dominance** | Does a single producer account for a large share of the feature's usage? |
| **Clique concentration** | Is the feature strongly associated with a small cluster of producers? |

Together, these measures distinguish several different patterns of feature circulation:

- **Field-wide features** used broadly across producers
- **Signature features** strongly associated with a single producer
- **Clique-diagnostic features** concentrated within a small producer cluster
- **Rare or peripheral features** appearing only sporadically

---

## Current Scripts

### 01_feature_distribution_suite.py

Measures whether a feature's prominence in a corpus is driven by **broad diffusion across producers**
or by **concentrated use within a smaller producer subset**.

The script expects a **binary incidence matrix** where:

- rows represent cases (songs, books, documents, etc.)
- one metadata column identifies the producer (artist, author, etc.)
- feature columns mark presence using `"X"` or another configurable token

For each feature, the script computes:

- **case_count** — total number of cases using the feature (**prominence**)
- **producer_count** — number of unique producers using the feature (**diffusion**)
- **concentration_ratio** — prominence relative to diffusion (`case_count / producer_count`)
- **producer_diffusion_pct** — proportion of all producers using the feature
- **top_producer** — producer most strongly associated with the feature
- **top_producer_count** — number of cases by that producer using the feature
- **top_producer_share** — proportion of total feature usage accounted for by the top producer
- **top_3_producer_share** — proportion of usage accounted for by the top three producers
- **top_3_minus_top_1_share** — concentration attributable to a small producer cluster beyond the single top producer

The script exports:

- **feature_distribution_metrics.csv** — full table of features and distribution metrics
- **analysis_summary.txt** — documentation of the run and suggested sorting workflows

---

## Interpreting Results

The output table can be sorted in several ways to answer different analytical questions.

Sort by **case_count (descending)**  
→ identifies the most **prominent features** in the corpus.

Sort by **producer_diffusion_pct (descending)**  
→ identifies the most **widely diffused features** across producers.

Sort by **concentration_ratio (descending)**  
→ identifies features whose prominence is **concentrated within a smaller producer subset**.

Sort by **top_producer_share (descending)**  
→ identifies features functioning as **strong producer signatures**.

Sort by **top_3_minus_top_1_share (descending)**  
→ identifies features associated with **small producer clusters or cliques**.

These patterns help distinguish whether a feature functions primarily as:

- shared vocabulary of the discourse field
- a producer-specific signature
- a marker of a small discourse community within the larger corpus

---

## Relationship to Other Analyses

Distribution analysis complements other analytical approaches in the toolkit:

| Analytical lens | Folder |
|---|---|
| Co-occurrence networks | `02_networks` |
| Case similarity | `03_similarity` |
| Network topology | `04_topology` |
| Feature distribution across producers | `05_distribution` |

Together these methods provide multiple perspectives on the structure of a corpus:

- **what appears with what**
- **which cases resemble each other**
- **how networks are structurally organized**
- **who produces which features**

---

## Data Assumptions

Scripts in this folder assume a **binary incidence matrix** where:

- rows represent cases (songs, books, documents, etc.)
- one metadata column identifies the **producer** (artist, author, etc.)
- all metadata columns appear to the **left** of the feature columns
- feature columns contain a presence marker such as `"X"`
- absence is represented by a blank cell

The producer column must fall within the metadata block on the left.

This means that the script uses two configuration settings to identify the relevant columns:

- `PRODUCER_COL` — the metadata column containing the producer
- `N_METADATA_COLS` — the number of metadata columns preceding the feature columns

All columns to the right of the metadata block are interpreted as feature columns.

Example:

| Artist | Album | Song | Year | feature_1 | feature_2 | feature_3 |
|------|------|------|------|------|------|------|

In this case:

- `PRODUCER_COL = "Artist"`
- `N_METADATA_COLS = 4`

If your dataset uses a different metadata structure, adjust 'PRODUCER_COL' and `N_METADATA_COLS` accordingly.

This format allows the script to compute feature-level prominence,
diffusion, and concentration across the producer field.