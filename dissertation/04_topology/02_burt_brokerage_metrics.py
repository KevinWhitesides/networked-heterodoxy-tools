#!/usr/bin/env python3
"""
02_burt_brokerage_metrics.py

Compute Burt-style brokerage / structural hole metrics on an existing one-mode
network in GEXF format.

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

Standalone use:
    Edit the CONFIG block below, then run:
        python 02_burt_brokerage_metrics.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import pandas as pd
import numpy as np


# =============================================================================
# CONFIG (standalone defaults)
# =============================================================================

INPUT_GEXF = Path("feature/first_7_books_feature_thr3.gexf")
OUTPUT_DIR = Path(".")

WEIGHT_ATTR = "weight"

OUT_CSV = "burt_metrics.csv"
OUT_GEXF = "network_with_burt.gexf"
OUT_SUMMARY = "analysis_summary.txt"

PROGRESS_EVERY = 200


# =============================================================================
# Progress-print helpers
# =============================================================================

def _timestamp() -> str:
    """Return a compact timestamp for console progress messages."""
    return datetime.now().strftime("%H:%M:%S")


def _print_stage_start(message: str) -> None:
    """Print a standardized stage-start message."""
    print(f"[{_timestamp()}] [→] {message}")


def _print_stage_done(message: str) -> None:
    """Print a standardized stage-complete message."""
    print(f"[{_timestamp()}] [✓] {message}")


def _print_info(message: str) -> None:
    """Print a standardized informational message."""
    print(f"[{_timestamp()}] [i] {message}")


# =============================================================================
# Helpers
# =============================================================================

def _ensure_undirected(G: nx.Graph) -> tuple[nx.Graph, bool]:
    """
    Convert directed graphs to undirected if necessary.

    Returns:
        graph, was_directed
    """
    was_directed = G.is_directed()
    if was_directed:
        _print_info("Input graph is directed; converting to undirected")
        return G.to_undirected(), True
    return G, False


def _compute_burt_metrics(
    G: nx.Graph,
    *,
    weight_attr: str,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, int]]:
    """
    Compute Burt metrics node-by-node and return the results table plus raw dicts.
    """
    nodes = list(G.nodes())
    degree = dict(G.degree())

    constraint: dict[str, float] = {}
    effective_size: dict[str, float] = {}

    _print_info(f"Beginning node-by-node Burt metrics for {len(nodes):,} node(s)")

    for i, node in enumerate(nodes, start=1):
        constraint[node] = nx.constraint(G, nodes=[node], weight=weight_attr)[node]
        effective_size[node] = nx.effective_size(G, nodes=[node], weight=weight_attr)[node]

        if progress_every and i % progress_every == 0:
            print(f"[{_timestamp()}] [i] {i}/{len(nodes)} nodes processed...")

    _print_stage_done("Node-by-node Burt metric computation complete")

    df = pd.DataFrame(
        {
            "Id": nodes,
            "constraint": [constraint[n] for n in nodes],
            "effective_size": [effective_size[n] for n in nodes],
            "degree": [degree[n] for n in nodes],
        }
    )

    # Efficiency = effective size / degree
    df["efficiency"] = df["effective_size"] / df["degree"].replace(0, np.nan)

    return df, constraint, effective_size, degree


def _annotate_graph(
    G: nx.Graph,
    *,
    constraint: dict[str, float],
    effective_size: dict[str, float],
    degree: dict[str, int],
) -> None:
    """Add Burt metrics back into graph node attributes."""
    nx.set_node_attributes(G, constraint, "constraint")
    nx.set_node_attributes(G, effective_size, "effective_size")
    nx.set_node_attributes(
        G,
        {n: (effective_size[n] / degree[n] if degree[n] != 0 else float("nan")) for n in degree},
        "efficiency",
    )
    nx.set_node_attributes(G, degree, "degree")


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_gexf: Path,
    weight_attr: str,
    was_directed: bool,
    graph_num_nodes: int,
    graph_num_edges: int,
    min_constraint_node: str,
    max_effective_size_node: str,
    mean_constraint: float,
    mean_effective_size: float,
    mean_efficiency: float,
    out_csv: Path,
    out_gexf: Path,
) -> None:
    """Write a plain-text summary of the run."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Burt Brokerage Metrics Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Input\n")
        f.write("-----\n")
        f.write(f"Network file: {input_gexf}\n")
        f.write(f"Weight attribute used: {weight_attr}\n\n")

        f.write("Network statistics\n")
        f.write("------------------\n")
        f.write(f"Nodes: {graph_num_nodes}\n")
        f.write(f"Edges: {graph_num_edges}\n")
        f.write(f"Directed input converted to undirected: {'yes' if was_directed else 'no'}\n\n")

        f.write("Metric summaries\n")
        f.write("----------------\n")
        f.write(f"Lowest constraint node: {min_constraint_node}\n")
        f.write(f"Highest effective size node: {max_effective_size_node}\n")
        f.write(f"Mean constraint: {mean_constraint:.6f}\n")
        f.write(f"Mean effective size: {mean_effective_size:.6f}\n")
        f.write(f"Mean efficiency: {mean_efficiency:.6f}\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_csv.name}\n")
        f.write(f"{out_gexf.name}\n")
        f.write(f"{out_path.name}\n")


# =============================================================================
# Pipeline-ready entry point
# =============================================================================

def run(
    *,
    input_gexf: Path,
    output_dir: Path,
    weight_attr: str = "weight",
    out_csv_name: str = "burt_metrics.csv",
    out_gexf_name: str = "network_with_burt.gexf",
    out_summary_name: str = "analysis_summary.txt",
    progress_every: int = 200,
) -> Dict[str, Any]:
    """
    Compute Burt brokerage metrics on an existing GEXF network and return a
    structured result dictionary.

    This is the entry point pipeline runners should call.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_gexf.exists():
        raise FileNotFoundError(f"Input GEXF not found: {input_gexf}")

    _print_stage_start(f"Loading GEXF: {input_gexf}")
    G_in = nx.read_gexf(input_gexf)
    G, was_directed = _ensure_undirected(G_in)
    _print_stage_done("Graph loaded")

    if G.number_of_nodes() == 0:
        raise ValueError("Input network has 0 nodes.")
    if G.number_of_edges() == 0:
        raise ValueError("Input network has 0 edges.")

    _print_info(
        f"Graph stats: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges"
    )

    _print_stage_start("Computing Burt brokerage metrics")
    df, constraint, effective_size, degree = _compute_burt_metrics(
        G,
        weight_attr=weight_attr,
        progress_every=progress_every,
    )
    _print_stage_done("Burt brokerage metrics computed")

    out_csv_path = output_dir / out_csv_name
    _print_stage_start(f"Writing Burt metrics CSV: {out_csv_path.name}")
    df.to_csv(out_csv_path, index=False, encoding="utf-8")
    _print_stage_done("Burt metrics CSV written")

    _print_stage_start("Annotating graph with Burt node attributes")
    _annotate_graph(
        G,
        constraint=constraint,
        effective_size=effective_size,
        degree=degree,
    )
    _print_stage_done("Graph annotation complete")

    out_gexf_path = output_dir / out_gexf_name
    _print_stage_start(f"Writing annotated GEXF: {out_gexf_path.name}")
    nx.write_gexf(G, out_gexf_path)
    _print_stage_done("Annotated GEXF written")

    _print_stage_start("Computing summary statistics")
    min_constraint_node = str(df.loc[df["constraint"].idxmin(), "Id"])
    max_effective_size_node = str(df.loc[df["effective_size"].idxmax(), "Id"])

    mean_constraint = float(df["constraint"].mean())
    mean_effective_size = float(df["effective_size"].mean())
    mean_efficiency = float(df["efficiency"].dropna().mean())
    _print_stage_done("Summary statistics computed")

    out_summary_path = output_dir / out_summary_name
    _print_stage_start(f"Writing analysis summary: {out_summary_path.name}")
    _write_summary(
        out_path=out_summary_path,
        run_timestamp=run_timestamp,
        input_gexf=input_gexf,
        weight_attr=weight_attr,
        was_directed=was_directed,
        graph_num_nodes=G.number_of_nodes(),
        graph_num_edges=G.number_of_edges(),
        min_constraint_node=min_constraint_node,
        max_effective_size_node=max_effective_size_node,
        mean_constraint=mean_constraint,
        mean_effective_size=mean_effective_size,
        mean_efficiency=mean_efficiency,
        out_csv=out_csv_path,
        out_gexf=out_gexf_path,
    )
    _print_stage_done("Analysis summary written")

    return {
        "run_timestamp": run_timestamp,
        "input_gexf": str(input_gexf),
        "output_dir": str(output_dir),
        "graph_num_nodes": G.number_of_nodes(),
        "graph_num_edges": G.number_of_edges(),
        "burt_metrics_csv": str(out_csv_path),
        "annotated_gexf": str(out_gexf_path),
        "summary_txt": str(out_summary_path),
        "min_constraint_node": min_constraint_node,
        "max_effective_size_node": max_effective_size_node,
        "mean_constraint": mean_constraint,
        "mean_effective_size": mean_effective_size,
        "mean_efficiency": mean_efficiency,
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Burt brokerage metrics from an existing GEXF network."
    )

    parser.add_argument(
        "--input-gexf",
        type=Path,
        default=INPUT_GEXF,
        help="Path to input GEXF network",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs",
    )
    parser.add_argument(
        "--weight-attr",
        type=str,
        default=WEIGHT_ATTR,
        help='Edge weight attribute name (default: "weight")',
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=OUT_CSV,
        help="Output filename for Burt metrics CSV",
    )
    parser.add_argument(
        "--out-gexf",
        type=str,
        default=OUT_GEXF,
        help="Output filename for annotated GEXF",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=OUT_SUMMARY,
        help="Output filename for analysis summary",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help="Print progress every N nodes (0 disables progress messages)",
    )

    return parser.parse_args()


# =============================================================================
# Main (standalone execution)
# =============================================================================

def main() -> None:
    """Run the script using CONFIG defaults or CLI overrides."""
    args = _parse_args()

    result = run(
        input_gexf=args.input_gexf,
        output_dir=args.output_dir,
        weight_attr=args.weight_attr,
        out_csv_name=args.out_csv,
        out_gexf_name=args.out_gexf,
        out_summary_name=args.out_summary,
        progress_every=args.progress_every,
    )

    print("[✓] Burt brokerage analysis complete.")
    print(f"    Input network:      {result['input_gexf']}")
    print(f"    Nodes / edges:      {result['graph_num_nodes']} / {result['graph_num_edges']}")
    print(f"    Output CSV:         {result['burt_metrics_csv']}")
    print(f"    Output GEXF:        {result['annotated_gexf']}")
    print(f"    Analysis summary:   {result['summary_txt']}")
    
if __name__ == "__main__":
    main()