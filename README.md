\# Networked Heterodoxy Dissertation Scripts



This repository contains general-purpose scripts used in the computational

analysis for the dissertation \*Networks of Heterodoxy: Shared Dissent and the Dynamics of Counter-Discourse\*.



The scripts are organized by methodological task rather than dataset.



\## Repository Structure



\- \*\*01\_diagnostics/\*\*  

&nbsp; Threshold and structural diagnostics for network calibration.



\- \*\*02\_networks/\*\*  

&nbsp; Construction of one-mode and bipartite networks and similarity matrices.



\- \*\*03\_metrics/\*\*  

&nbsp; Network metrics including brokerage, centrality, and structural analysis.



\- \*\*04\_topic\_modeling/\*\*  

&nbsp; MALLET pipelines, pyLDAvis outputs, and topic dendrograms.



\- \*\*docs/\*\*

  Methodological notes and data format specifications.



\- \*\*workflows/\*\*  

&nbsp; Notes describing how scripts were used for specific datasets (e.g., 2012, Hip Hop).



\## Data Format Assumptions



Most network scripts assume starting from a binary incidence matrix, a spreadsheet wherein:



\- Rows = cases (books, songs, etc.)

\- Columns = tropes or features

\- Presence marked by "X"

\- Absence left blank

