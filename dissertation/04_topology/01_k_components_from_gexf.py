#!/usr/bin/env python3
"""
01_k_components_from_gexf.py

Compute k-components (k-vertex-connected components) on an existing one-mode
network file in GEXF format, and export:

1) A summary CSV listing all k-components
2) One GEXF subgraph per k-component
3) One CSV node list per k-component
4) A node-level CSV of k-core numbers (optional)
5) A node-level CSV of degree / weighted degree (optional)
6) analysis_summary.txt

Why this script exists:
- Network construction (incidence -> projection -> thresholded GEXF) belongs in 02_networks
- This script is purely topology analysis on an already-built network

Standalone use:
    Edit the CONFIG block below, then run:
        python 01_k_components_from_gexf.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

# NetworkX import paths can vary; try robustly.
try:
    from networkx.algorithms.connectivity import k_components  # NetworkX 3.x
except Exception:  # pragma: no cover
    try:
        from networkx.algorithms.connectivity.kcomponents import k_components
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import k_components from NetworkX. "
            "Please install/upgrade NetworkX (>= 2.8 recommended; 3.x ideal)."
        ) from e


# =============================================================================
# CONFIG (standalone defaults)
# =============================================================================

# Input network file (GEXF)
INPUT_GEXF = Path("feature/first_7_books_feature_thr3.gexf")

# Output directory
OUTPUT_DIR = Path("k_components_output")

# Optionally export only certain k-levels (e.g. [2, 3, 4]).
# None = export all k-levels returned by k_components()
EXPORT_ONLY_K: Optional[List[int]] = None

# Node-level exports
EXPORT_CORE_NUMBERS = True
CORE_NUMBERS_CSV = "node_core_numbers.csv"

EXPORT_NODE_SUMMARY = True
NODE_SUMMARY_CSV = "node_summary.csv"

# Edge weight attribute name (used for weighted degree if present)
WEIGHT_ATTR = "weight"

# Filename prefix for per-component outputs
COMPONENT_PREFIX = "kcomp"

# Summary filename
OUT_SUMMARY = "analysis_summary.txt"


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

def _ensure_undirected(G: nx.Graph, weight_attr: str) -> nx.Graph:
    """
    k_components is defined for undirected graphs.

    If a directed graph is read, convert it to undirected, summing reciprocal
    edge weights when both directions exist.
    """
    was_directed = G.is_directed()
    if not was_directed:
        return G

    _print_info("Input graph is directed; converting to undirected graph")

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        w = data.get(weight_attr, 1)
        if H.has_edge(u, v):
            H[u][v][weight_attr] = H[u][v].get(weight_attr, 1) + w
        else:
            H.add_edge(u, v, **data)

    return H


def _infer_has_weights(G: nx.Graph, weight_attr: str) -> bool:
    """Return True if any edge contains the given weight attribute."""
    for _, _, d in G.edges(data=True):
        if weight_attr in d:
            return True
    return False


def _normalize_export_only_k(export_only_k: Optional[List[int]]) -> Optional[List[int]]:
    """Validate and normalize EXPORT_ONLY_K."""
    if export_only_k is None:
        return None

    out = sorted({int(k) for k in export_only_k})
    if any(k < 1 for k in out):
        raise ValueError("All k-levels in EXPORT_ONLY_K must be >= 1.")
    return out


def _write_node_summaries(
    G: nx.Graph,
    outdir: Path,
    *,
    export_core_numbers: bool,
    core_numbers_csv: str,
    export_node_summary: bool,
    node_summary_csv: str,
    weight_attr: str,
) -> dict[str, Optional[str]]:
    """Export core numbers and basic node summary stats."""
    outdir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Optional[str]] = {
        "core_numbers_csv": None,
        "node_summary_csv": None,
    }

    if export_core_numbers:
        _print_stage_start("Computing node core numbers")
        core = nx.core_number(G)
        out_path = outdir / core_numbers_csv
        pd.DataFrame(
            {"node": list(core.keys()), "core_number": list(core.values())}
        ).sort_values(["core_number", "node"], ascending=[False, True]).to_csv(
            out_path, index=False, encoding="utf-8"
        )
        outputs["core_numbers_csv"] = str(out_path)
        _print_stage_done(f"Node core numbers written: {out_path.name}")

    if export_node_summary:
        _print_stage_start("Computing node degree / weighted-degree summary")
        has_w = _infer_has_weights(G, weight_attr)
        deg = dict(G.degree())
        if has_w:
            wdeg = dict(G.degree(weight=weight_attr))
        else:
            wdeg = {n: float("nan") for n in G.nodes()}

        out_path = outdir / node_summary_csv
        pd.DataFrame(
            {
                "node": list(G.nodes()),
                "degree": [deg[n] for n in G.nodes()],
                "weighted_degree": [wdeg[n] for n in G.nodes()],
            }
        ).sort_values(["degree", "node"], ascending=[False, True]).to_csv(
            out_path, index=False, encoding="utf-8"
        )
        outputs["node_summary_csv"] = str(out_path)
        _print_stage_done(f"Node summary written: {out_path.name}")

    return outputs


def _write_k_components(
    G: nx.Graph,
    kcomp: Dict[int, List[set]],
    outdir: Path,
    *,
    export_only_k: Optional[List[int]],
    component_prefix: str,
) -> tuple[Path, List[int], int]:
    """
    Export each k-component as GEXF + node list CSV, and write a summary CSV.

    Additions preserved here:
    - If a given k has more than one component, also export a combined same-k GEXF.
    - Also export an isolated same-k GEXF containing only intra-component edges.
    - Also export an all-nodes CSV for that combined same-k graph, including:
        * component_id
        * component_ids
        * component_count
        * adjacent_overlap_count
        * overlap_internal_degree

    Returns:
        summary_path, exported_k_levels, total_components_exported
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    ks = sorted(kcomp.keys())
    if export_only_k is not None:
        ks = [k for k in ks if k in set(export_only_k)]

    total_components = 0

    _print_info(f"k-levels available for export: {ks}")

    for k in ks:
        comps = kcomp[k]
        _print_stage_start(f"Exporting k={k} components ({len(comps)} component(s))")

        k_level_gexf_name = ""
        k_level_nodes_csv_name = ""
        k_level_isolated_gexf_name = ""

        # Export one combined same-k graph + all-nodes CSV if multiple components exist
        if len(comps) > 1:
            _print_stage_start(f"Creating combined same-k exports for k={k}")

            union_nodes = set().union(*comps)
            combined_sub = G.subgraph(union_nodes).copy()

            # Track all same-k memberships per node
            membership_map = {}
            for i, node_set in enumerate(comps, start=1):
                for node in node_set:
                    membership_map.setdefault(node, []).append(i)

            component_ids_attr = {
                node: "|".join(str(cid) for cid in sorted(ids))
                for node, ids in membership_map.items()
            }
            component_count_attr = {
                node: len(ids)
                for node, ids in membership_map.items()
            }

            nx.set_node_attributes(combined_sub, component_ids_attr, "component_ids")
            nx.set_node_attributes(combined_sub, component_count_attr, "component_count")

            k_level_gexf_name = f"{component_prefix}_k{k}_all_components.gexf"
            k_level_nodes_csv_name = f"{component_prefix}_k{k}_all_nodes.csv"
            k_level_isolated_gexf_name = f"{component_prefix}_k{k}_all_components_isolated.gexf"

            # Combined induced graph: keeps all original edges among unioned nodes
            nx.write_gexf(combined_sub, outdir / k_level_gexf_name)

            # Isolated combined graph: keeps only intra-component edges
            isolated_sub = nx.Graph()
            isolated_sub.graph.update(G.graph)

            for i, node_set in enumerate(comps, start=1):
                sub = G.subgraph(node_set).copy()
                sub_component_ids = {
                    n: "|".join(str(cid) for cid in sorted(membership_map[n]))
                    for n in sub.nodes()
                }
                sub_component_count = {
                    n: len(membership_map[n])
                    for n in sub.nodes()
                }

                nx.set_node_attributes(sub, sub_component_ids, "component_ids")
                nx.set_node_attributes(sub, sub_component_count, "component_count")
                isolated_sub.add_nodes_from(sub.nodes(data=True))
                isolated_sub.add_edges_from(sub.edges(data=True))

            nx.write_gexf(isolated_sub, outdir / k_level_isolated_gexf_name)

            # Overlap-node sets and derived metrics
            overlap_nodes = {
                node for node, count in component_count_attr.items() if count > 1
            }

            # For any node: how many overlap nodes does it touch?
            adjacent_overlap_count_map = {
                node: sum(1 for nbr in combined_sub.neighbors(node) if nbr in overlap_nodes)
                for node in combined_sub.nodes()
            }

            # For overlap nodes only: how many overlap nodes does it touch?
            overlap_internal_degree_map = {
                node: (
                    sum(1 for nbr in combined_sub.neighbors(node) if nbr in overlap_nodes)
                    if node in overlap_nodes
                    else 0
                )
                for node in combined_sub.nodes()
            }

            all_nodes_rows = []
            for i, node_set in enumerate(comps, start=1):
                for node in sorted(node_set):
                    all_nodes_rows.append(
                        {
                            "node": node,
                            "component_id": i,
                            "component_ids": "|".join(str(cid) for cid in sorted(membership_map[node])),
                            "component_count": len(membership_map[node]),
                            "adjacent_overlap_count": adjacent_overlap_count_map[node],
                            "overlap_internal_degree": overlap_internal_degree_map[node],
                        }
                    )

            pd.DataFrame(all_nodes_rows).sort_values(
                ["component_id", "node"]
            ).to_csv(
                outdir / k_level_nodes_csv_name,
                index=False,
                encoding="utf-8",
            )

            _print_stage_done(f"Combined same-k exports written for k={k}")

        for i, node_set in enumerate(comps, start=1):
            sub = G.subgraph(node_set).copy()

            gexf_name = f"{component_prefix}_k{k}_component_{i}.gexf"
            csv_name = f"{component_prefix}_k{k}_component_{i}_nodes.csv"

            nx.write_gexf(sub, outdir / gexf_name)

            pd.DataFrame({"node": sorted(node_set)}).to_csv(
                outdir / csv_name, index=False, encoding="utf-8"
            )

            rows.append(
                {
                    "k": k,
                    "component_id": i,
                    "num_nodes": sub.number_of_nodes(),
                    "num_edges": sub.number_of_edges(),
                    "gexf_file": gexf_name,
                    "nodes_csv": csv_name,
                    "k_level_gexf_file": k_level_gexf_name,
                    "k_level_nodes_csv": k_level_nodes_csv_name,
                    "k_level_isolated_gexf_file": k_level_isolated_gexf_name,
                }
            )
            total_components += 1

        _print_stage_done(f"k={k} export complete")

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["k", "component_id"])

    summary_path = outdir / f"{component_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    _print_stage_done(f"Component summary CSV written: {summary_path.name}")

    return summary_path, ks, total_components


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_gexf: Path,
    output_dir: Path,
    export_only_k: Optional[List[int]],
    export_core_numbers: bool,
    export_node_summary: bool,
    weight_attr: str,
    graph_was_directed: bool,
    graph_num_nodes: int,
    graph_num_edges: int,
    exported_k_levels: List[int],
    total_components_exported: int,
    component_summary_csv: Path,
    core_numbers_csv: Optional[str],
    node_summary_csv: Optional[str],
) -> None:
    """Write a plain-text summary of the run."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== k-Components Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Input GEXF: {input_gexf}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write("Settings\n")
        f.write("--------\n")
        f.write(f"EXPORT_ONLY_K: {export_only_k}\n")
        f.write(f"EXPORT_CORE_NUMBERS: {export_core_numbers}\n")
        f.write(f"EXPORT_NODE_SUMMARY: {export_node_summary}\n")
        f.write(f"WEIGHT_ATTR: {weight_attr}\n")
        f.write(f"Directed input converted to undirected: {'yes' if graph_was_directed else 'no'}\n\n")

        f.write("Graph statistics\n")
        f.write("----------------\n")
        f.write(f"Nodes: {graph_num_nodes}\n")
        f.write(f"Edges: {graph_num_edges}\n")
        f.write(f"Exported k-levels: {exported_k_levels}\n")
        f.write(f"Total k-components exported: {total_components_exported}\n\n")

        f.write("Outputs\n")
        f.write("-------\n")
        f.write(f"Component summary CSV: {component_summary_csv.name}\n")
        if core_numbers_csv is not None:
            f.write(f"Core numbers CSV: {Path(core_numbers_csv).name}\n")
        else:
            f.write("Core numbers CSV: (not generated)\n")
        if node_summary_csv is not None:
            f.write(f"Node summary CSV: {Path(node_summary_csv).name}\n")
        else:
            f.write("Node summary CSV: (not generated)\n")
        f.write(f"Analysis summary: {out_path.name}\n")


def _parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers into a list."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    try:
        return [int(item) for item in items]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integers."
        ) from e


# =============================================================================
# Pipeline-ready entry point
# =============================================================================

def run(
    *,
    input_gexf: Path,
    output_dir: Path,
    export_only_k: Optional[List[int]] = None,
    export_core_numbers: bool = True,
    core_numbers_csv: str = "node_core_numbers.csv",
    export_node_summary: bool = True,
    node_summary_csv: str = "node_summary.csv",
    weight_attr: str = "weight",
    component_prefix: str = "kcomp",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Compute k-components on an existing GEXF network and return a structured
    result dictionary.

    This is the entry point pipeline runners should call.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not input_gexf.exists():
        raise FileNotFoundError(f"Input GEXF not found: {input_gexf}")

    export_only_k = _normalize_export_only_k(export_only_k)

    _print_stage_start(f"Loading GEXF: {input_gexf}")
    G_in = nx.read_gexf(input_gexf)
    graph_was_directed = G_in.is_directed()
    G = _ensure_undirected(G_in, weight_attr=weight_attr)
    _print_stage_done("Graph loaded")

    if G.number_of_nodes() == 0:
        raise ValueError("Input network has 0 nodes. Nothing to analyze.")
    if G.number_of_edges() == 0:
        raise ValueError("Input network has 0 edges. k-components are not meaningful on an edgeless graph.")

    output_dir.mkdir(parents=True, exist_ok=True)

    _print_info(
        f"Graph stats: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges"
    )

    node_outputs = _write_node_summaries(
        G,
        output_dir,
        export_core_numbers=export_core_numbers,
        core_numbers_csv=core_numbers_csv,
        export_node_summary=export_node_summary,
        node_summary_csv=node_summary_csv,
        weight_attr=weight_attr,
    )

    _print_stage_start("Computing k-components")
    kcomp = k_components(G)
    _print_stage_done("k-components computed")

    component_summary_csv, exported_k_levels, total_components_exported = _write_k_components(
        G,
        kcomp,
        output_dir,
        export_only_k=export_only_k,
        component_prefix=component_prefix,
    )

    out_summary = output_dir / out_summary_name
    _print_stage_start("Writing analysis summary")
    _write_summary(
        out_path=out_summary,
        run_timestamp=run_timestamp,
        input_gexf=input_gexf,
        output_dir=output_dir,
        export_only_k=export_only_k,
        export_core_numbers=export_core_numbers,
        export_node_summary=export_node_summary,
        weight_attr=weight_attr,
        graph_was_directed=graph_was_directed,
        graph_num_nodes=G.number_of_nodes(),
        graph_num_edges=G.number_of_edges(),
        exported_k_levels=exported_k_levels,
        total_components_exported=total_components_exported,
        component_summary_csv=component_summary_csv,
        core_numbers_csv=node_outputs["core_numbers_csv"],
        node_summary_csv=node_outputs["node_summary_csv"],
    )
    _print_stage_done(f"Analysis summary written: {out_summary.name}")

    return {
        "run_timestamp": run_timestamp,
        "input_gexf": str(input_gexf),
        "output_dir": str(output_dir),
        "graph_num_nodes": G.number_of_nodes(),
        "graph_num_edges": G.number_of_edges(),
        "exported_k_levels": exported_k_levels,
        "total_components_exported": total_components_exported,
        "component_summary_csv": str(component_summary_csv),
        "core_numbers_csv": node_outputs["core_numbers_csv"],
        "node_summary_csv": node_outputs["node_summary_csv"],
        "summary_txt": str(out_summary),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute k-components from an existing GEXF network."
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
        "--export-only-k",
        type=_parse_int_list,
        default=EXPORT_ONLY_K,
        help="Comma-separated list of k-levels to export (e.g. 2,3,4). Default: export all",
    )
    parser.add_argument(
        "--no-core-numbers",
        action="store_true",
        help="Disable export of node core numbers CSV",
    )
    parser.add_argument(
        "--core-numbers-csv",
        type=str,
        default=CORE_NUMBERS_CSV,
        help="Filename for core numbers CSV",
    )
    parser.add_argument(
        "--no-node-summary",
        action="store_true",
        help="Disable export of node degree / weighted degree CSV",
    )
    parser.add_argument(
        "--node-summary-csv",
        type=str,
        default=NODE_SUMMARY_CSV,
        help="Filename for node summary CSV",
    )
    parser.add_argument(
        "--weight-attr",
        type=str,
        default=WEIGHT_ATTR,
        help='Edge weight attribute name (default: "weight")',
    )
    parser.add_argument(
        "--component-prefix",
        type=str,
        default=COMPONENT_PREFIX,
        help="Filename prefix for per-component outputs",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=OUT_SUMMARY,
        help="Output summary filename",
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
        export_only_k=args.export_only_k,
        export_core_numbers=not args.no_core_numbers,
        core_numbers_csv=args.core_numbers_csv,
        export_node_summary=not args.no_node_summary,
        node_summary_csv=args.node_summary_csv,
        weight_attr=args.weight_attr,
        component_prefix=args.component_prefix,
        out_summary_name=args.out_summary,
    )

    print("[✓] k-components computed.")
    print(f"    Input:            {result['input_gexf']}")
    print(f"    Graph:            {result['graph_num_nodes']:,} nodes | {result['graph_num_edges']:,} edges")
    print(f"    k-levels:         {result['exported_k_levels']}")
    print(f"    Components:       {result['total_components_exported']}")
    print(f"    Output dir:       {result['output_dir']}")
    print(f"    Component CSV:    {result['component_summary_csv']}")
    if result["core_numbers_csv"] is not None:
        print(f"    Core numbers:     {result['core_numbers_csv']}")
    else:
        print("    Core numbers:     (not generated)")
    if result["node_summary_csv"] is not None:
        print(f"    Node summary:     {result['node_summary_csv']}")
    else:
        print("    Node summary:     (not generated)")
    print(f"    Summary:          {result['summary_txt']}")


if __name__ == "__main__":
    main()