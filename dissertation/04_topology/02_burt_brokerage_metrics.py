#!/usr/bin/env python3
"""
02_burt_brokerage_metrics.py

Compute Burt-style brokerage / structural hole metrics on an existing network.

Inputs
------
- A one-mode network file in GEXF format (typically produced by scripts in 02_networks)

Outputs
-------
1) burt_metrics.csv
   Node-level table including:
   - constraint
   - effective_size
   - efficiency
   - degree

2) network_with_burt.gexf
   The input network with brokerage metrics added as node attributes

3) analysis_summary.txt
   A compact human-readable record of the run

Notes
-----
- This script assumes the network has already been constructed.
- It does not build or threshold the network itself.
- Constraint is often the most directly interpretable brokerage metric.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import networkx as nx


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

INPUT_GEXF = Path("input_network.gexf")
OUTPUT_DIR = Path(".")

WEIGHT_ATTR = "weight"

OUT_CSV = "burt_metrics.csv"
OUT_GEXF = "network_with_burt.gexf"
OUT_SUMMARY = "analysis_summary.txt"

PROGRESS_EVERY = 200


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_undirected(G: nx.Graph) -> nx.Graph:
    """Convert directed graphs to undirected if necessary."""
    if G.is_directed():
        return G.to_undirected()
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_GEXF.exists():
        raise FileNotFoundError(f"Input GEXF not found: {INPUT_GEXF}")

    # Load graph
    G = nx.read_gexf(INPUT_GEXF)
    G = ensure_undirected(G)

    if G.number_of_nodes() == 0:
        raise ValueError("Input network has 0 nodes.")
    if G.number_of_edges() == 0:
        raise ValueError("Input network has 0 edges.")

    nodes = list(G.nodes())
    degree = dict(G.degree())

    constraint = {}
    effective_size = {}

    # Compute Burt metrics node by node
    for i, node in enumerate(nodes, start=1):
        constraint[node] = nx.constraint(G, nodes=[node], weight=WEIGHT_ATTR)[node]
        effective_size[node] = nx.effective_size(G, nodes=[node], weight=WEIGHT_ATTR)[node]

        if PROGRESS_EVERY and i % PROGRESS_EVERY == 0:
            print(f"{i}/{len(nodes)} nodes processed...")

    # Build results table
    df = pd.DataFrame({
        "Id": nodes,
        "constraint": [constraint[n] for n in nodes],
        "effective_size": [effective_size[n] for n in nodes],
        "degree": [degree[n] for n in nodes],
    })

    # Efficiency = effective size / degree
    df["efficiency"] = df["effective_size"] / df["degree"].replace(0, pd.NA)

    # Export CSV
    out_csv_path = OUTPUT_DIR / OUT_CSV
    df.to_csv(out_csv_path, index=False, encoding="utf-8")

    # Add attributes back into graph
    nx.set_node_attributes(G, constraint, "constraint")
    nx.set_node_attributes(G, effective_size, "effective_size")
    nx.set_node_attributes(
        G,
        {n: (effective_size[n] / degree[n] if degree[n] != 0 else None) for n in nodes},
        "efficiency"
    )
    nx.set_node_attributes(G, degree, "degree")

    # Export annotated GEXF
    out_gexf_path = OUTPUT_DIR / OUT_GEXF
    nx.write_gexf(G, out_gexf_path)

    # Summary stats
    min_constraint_node = df.loc[df["constraint"].idxmin(), "Id"]
    max_effective_size_node = df.loc[df["effective_size"].idxmax(), "Id"]

    # Write summary
    out_summary_path = OUTPUT_DIR / OUT_SUMMARY
    with open(out_summary_path, "w", encoding="utf-8") as f:
        f.write("=== Burt Brokerage Metrics Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Input\n")
        f.write("-----\n")
        f.write(f"Network file: {INPUT_GEXF}\n")
        f.write(f"Weight attribute used: {WEIGHT_ATTR}\n\n")

        f.write("Network statistics\n")
        f.write("------------------\n")
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Directed input converted to undirected: {'yes' if G.is_directed() else 'no'}\n\n")

        f.write("Metric summaries\n")
        f.write("----------------\n")
        f.write(f"Lowest constraint node: {min_constraint_node}\n")
        f.write(f"Highest effective size node: {max_effective_size_node}\n")
        f.write(f"Mean constraint: {df['constraint'].mean():.6f}\n")
        f.write(f"Mean effective size: {df['effective_size'].mean():.6f}\n")
        f.write(f"Mean efficiency: {df['efficiency'].dropna().mean():.6f}\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{OUT_CSV}\n")
        f.write(f"{OUT_GEXF}\n")
        f.write(f"{OUT_SUMMARY}\n")

    # Console summary
    print("[✓] Burt brokerage analysis complete.")
    print(f"    Input network:      {INPUT_GEXF}")
    print(f"    Nodes / edges:      {G.number_of_nodes()} / {G.number_of_edges()}")
    print(f"    Output CSV:         {out_csv_path}")
    print(f"    Output GEXF:        {out_gexf_path}")
    print(f"    Analysis summary:   {out_summary_path}")


if __name__ == "__main__":
    main()