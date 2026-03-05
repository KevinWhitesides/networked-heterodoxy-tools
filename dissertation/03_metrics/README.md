\# 03\_metrics



Scripts for computing structural and similarity measures derived from network or incidence data.



These tools typically operate on \*\*binary incidence matrices\*\*

(case × feature/trope), from which similarity measures or structural

statistics are derived.



\## Analytical Focus



The scripts in this folder are designed to answer specific research

questions about relationships within binary incidence datasets

(case × feature/trope matrices).



| Metric | Research question |

|------|-------------------|

| \*\*Jaccard similarity\*\* | Which cases have similar feature repertoires? |

| \*\*Brokerage (Burt’s metrics)\*\* | Which nodes connect otherwise separate clusters or discourse regions? |

| \*\*Non-overlap detection\*\* | Which pairs of cases have completely non-overlapping repertoires? |

| \*\*Discourse gradient identification\*\* | Which intermediary cases link otherwise non-overlapping cases within the broader network? |



These measures help identify patterns of similarity, mediation, and

distinctiveness within cultural datasets.



---



\## Current Scripts



\### jaccard\_similarity\_heatmap.py



Computes pairwise \*\*Jaccard similarity\*\* between cases based on shared feature/trope presence.



The script:



1\. Reads a binary incidence matrix from `.xlsx` or `.csv`.

2\. Optionally filters features appearing in fewer than a specified number of cases (default: ≥2).

3\. Computes a case × case Jaccard similarity matrix.

4\. Exports:

&nbsp;  - a CSV similarity matrix

&nbsp;  - a heatmap visualization (PNG)



Jaccard similarity measures overlap in \*\*feature repertoires\*\* while ignoring shared absences, making it well suited for sparse cultural feature datasets.



---



\## Data Assumptions



Most scripts in this folder assume data structured as a \*\*binary incidence matrix\*\*:



\- rows = cases (books, songs, etc.)

\- columns = features/tropes

\- presence marked by `"X"`

\- absence left blank



This format allows straightforward transformation into similarity matrices or network measures.

