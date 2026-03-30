#!/usr/bin/env python3
"""
01_build_one_mode_projection.py

Build one-mode projection networks from a binary incidence matrix.

This script supports:

- feature × feature projection
- case × case projection
- or both in a single run

For each selected projection mode, the script:

1) Reads a binary incidence matrix from .xlsx/.xls or .csv
2) Ignores leading metadata columns via N_METADATA_COLS
3) Applies a node-frequency filter appropriate to the projection mode
4) Computes a weighted one-mode projection
5) Generates one network per specified edge threshold
6) Exports:
   - edge list CSV
   - GEXF network with node attributes
   - analysis_summary.txt

Projection modes
----------------
- "feature":
    Nodes are features.
    MIN_FEATURE_NODE_FREQ filters out features appearing in too few cases.
    FEATURE_EDGE_THRESHOLDS sets co-occurrence thresholds for feature pairs.

- "case":
    Nodes are cases.
    MIN_CASE_NODE_FREQ filters out cases containing too few features.
    CASE_EDGE_THRESHOLDS sets overlap thresholds for case pairs.

- "both":
    Build both projection types in one run.

Standalone use:
    Edit the CONFIG block below, then run:
        python 01_build_one_mode_projection.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG (standalone defaults)
# Edit these values for normal standalone use.
# Pipeline runners can override them via run(...) or CLI arguments.
# =============================================================================

INPUT_PATH = Path("dissertation/sample_data/first_7_books.xlsx")

# Case-label column. If None, the script will prefer "Source Title", then
# "Title", then fall back to row numbers.
TITLE_COL: Optional[str] = None

# Number of metadata columns before feature columns begin.
N_METADATA_COLS = 4

PRESENCE_TOKEN = "X"

# Projection modes:
#   ["feature"]
#   ["case"]
#   ["feature", "case"]
PROJECTION_MODES = ["feature"]

# Minimum node frequency filters
# Feature mode: minimum number of cases a feature must appear in
MIN_FEATURE_NODE_FREQ = 4

# Case mode: minimum number of features a case must contain
MIN_CASE_NODE_FREQ = 2

# Edge thresholds by projection mode
# Feature mode: minimum number of shared cases for feature-feature edges
FEATURE_EDGE_THRESHOLDS = [3, 4]

# Case mode: minimum number of shared features for case-case edges
CASE_EDGE_THRESHOLDS = [1, 15]

# Output directory
OUTPUT_DIR = Path(".")

# Summary filename
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# Helpers
# =============================================================================

def _read_table(path: Path) -> pd.DataFrame:
    """Read .xlsx/.xls or .csv as strings; preserve blanks literally."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .xlsx, .xls, or .csv.")

    df.columns = df.columns.map(str)
    return df


def _choose_title_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    """Choose the case-label column, if any."""
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(
                f"TITLE_COL was set to {preferred!r}, but that column was not found.\n"
                f"Available columns include: {list(df.columns)[:10]} ..."
            )
        return preferred

    if "Source Title" in df.columns:
        return "Source Title"
    if "Title" in df.columns:
        return "Title"
    return None


def _extract_case_names(df: pd.DataFrame, title_col: Optional[str]) -> list[str]:
    """Return unique case names aligned to dataframe rows."""
    if title_col is None:
        return [f"row_{i + 1}" for i in range(len(df))]

    names = df[title_col].astype(str).tolist()

    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name} ({seen[name]})")

    return out


def _build_incidence(
    df: pd.DataFrame,
    n_metadata_cols: int,
    token: str,
    case_names: list[str],
) -> pd.DataFrame:
    """
    Build a binary incidence matrix (cases × features).

    Metadata columns are allowed; the first n_metadata_cols columns are ignored
    when constructing the feature matrix.
    """
    if n_metadata_cols < 0 or n_metadata_cols >= df.shape[1]:
        raise ValueError(
            f"N_METADATA_COLS={n_metadata_cols} is invalid for a table with {df.shape[1]} columns."
        )

    feature_cols = list(df.columns[n_metadata_cols:])
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS and your input file.")

    incidence = df[feature_cols].eq(token).astype(np.int8)
    incidence.index = case_names
    return incidence


def _normalize_projection_modes(modes: list[str]) -> list[str]:
    """Validate and normalize projection mode list."""
    allowed = {"feature", "case"}
    normalized = [m.strip().lower() for m in modes]

    if not normalized:
        raise ValueError("At least one projection mode must be specified.")

    invalid = [m for m in normalized if m not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid projection mode(s): {invalid}. Allowed values are: 'feature', 'case'."
        )

    seen = set()
    out = []
    for m in normalized:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _compute_density(num_nodes: int, num_edges: int) -> float:
    """Compute simple undirected graph density."""
    possible_pairs = num_nodes * (num_nodes - 1) // 2
    if possible_pairs == 0:
        return 0.0
    return num_edges / possible_pairs


def _build_feature_projection(
    incidence: pd.DataFrame,
    *,
    min_feature_node_freq: int,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """
    Build weighted feature × feature projection inputs.

    Returns:
        cooccurrence matrix, retained node frequencies, original node count
    """
    raw_freq = incidence.sum(axis=0)
    original_node_count = int(len(raw_freq))

    if min_feature_node_freq > 1:
        keep_cols = raw_freq[raw_freq >= min_feature_node_freq].index
        filtered = incidence.loc[:, keep_cols]
        freq = raw_freq.loc[keep_cols]
    else:
        filtered = incidence
        freq = raw_freq

    cooc = filtered.T.dot(filtered)
    return cooc, freq, original_node_count


def _build_case_projection(
    incidence: pd.DataFrame,
    *,
    min_case_node_freq: int,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """
    Build weighted case × case projection inputs.

    Returns:
        cooccurrence matrix, retained case sizes, original node count
    """
    raw_case_sizes = incidence.sum(axis=1)
    original_node_count = int(len(raw_case_sizes))

    if min_case_node_freq > 1:
        keep_rows = raw_case_sizes[raw_case_sizes >= min_case_node_freq].index
        filtered = incidence.loc[keep_rows, :]
        case_sizes = raw_case_sizes.loc[keep_rows]
    else:
        filtered = incidence
        case_sizes = raw_case_sizes

    cooc = filtered.dot(filtered.T)
    return cooc, case_sizes, original_node_count


def _export_thresholded_networks(
    *,
    cooc: pd.DataFrame,
    node_measure: pd.Series,
    edge_thresholds: list[int],
    dataset_stem: str,
    projection_mode: str,
    output_dir: Path,
) -> dict[int, dict[str, Any]]:
    """
    Build and export one graph per edge threshold.

    Returns:
        threshold-indexed results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = cooc.shape[0]
    upper_mask = np.triu(np.ones((n_total, n_total), dtype=bool), k=1)

    results: dict[int, dict[str, Any]] = {}

    for thr in edge_thresholds:
        rows, cols = np.where((cooc.values >= thr) & upper_mask)
        weights = cooc.values[rows, cols]

        edges_df = pd.DataFrame(
            {
                "source": cooc.index[rows],
                "target": cooc.columns[cols],
                "weight": weights,
            }
        )

        G = nx.Graph()
        G.add_nodes_from(cooc.index.tolist())

        for _, row in edges_df.iterrows():
            G.add_edge(row["source"], row["target"], weight=int(row["weight"]))

        measure_attr = "frequency" if projection_mode == "feature" else "feature_count"
        node_attr_dict = {node: int(node_measure[node]) for node in cooc.index}
        nx.set_node_attributes(G, node_attr_dict, measure_attr)
        nx.set_node_attributes(G, {node: projection_mode for node in G.nodes()}, "projection_mode")

        degree_dict = dict(G.degree())
        weighted_degree_dict = dict(G.degree(weight="weight"))
        nx.set_node_attributes(G, degree_dict, "degree")
        nx.set_node_attributes(G, weighted_degree_dict, "weighted_degree")

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = _compute_density(num_nodes, num_edges)

        base_name = f"{dataset_stem}_{projection_mode}_thr{thr}"
        edge_csv_path = output_dir / f"{base_name}_edges.csv"
        gexf_path = output_dir / f"{base_name}.gexf"

        edges_df.to_csv(edge_csv_path, index=False, encoding="utf-8")
        nx.write_gexf(G, gexf_path)

        results[int(thr)] = {
            "edge_threshold": int(thr),
            "edge_csv": str(edge_csv_path),
            "gexf": str(gexf_path),
            "node_count": num_nodes,
            "edge_count": num_edges,
            "density": density,
        }

    return results


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_path: Path,
    title_col: Optional[str],
    n_metadata_cols: int,
    presence_token: str,
    projection_modes: list[str],
    min_feature_node_freq: int,
    min_case_node_freq: int,
    feature_edge_thresholds: list[int],
    case_edge_thresholds: list[int],
    results: dict[str, Any],
) -> None:
    """Write a plain-text summary of the run."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== One-Mode Projection Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Title column: {title_col if title_col is not None else '(auto / fallback)'}\n")
        f.write(f"N_METADATA_COLS: {n_metadata_cols}\n")
        f.write(f"Presence token: {presence_token}\n\n")

        f.write("Settings\n")
        f.write("--------\n")
        f.write(f"Projection modes: {projection_modes}\n")
        f.write(f"MIN_FEATURE_NODE_FREQ: {min_feature_node_freq}\n")
        f.write(f"MIN_CASE_NODE_FREQ: {min_case_node_freq}\n")
        f.write(f"FEATURE_EDGE_THRESHOLDS: {feature_edge_thresholds}\n")
        f.write(f"CASE_EDGE_THRESHOLDS: {case_edge_thresholds}\n\n")

        f.write("Projection results\n")
        f.write("------------------\n")

        for mode in projection_modes:
            mode_result = results["projections"][mode]
            f.write(f"\n[{mode} projection]\n")
            f.write(f"Original nodes: {mode_result['original_node_count']}\n")
            f.write(f"Retained nodes after node filtering: {mode_result['retained_node_count']}\n")
            f.write(f"Edge thresholds used: {mode_result['edge_thresholds_used']}\n")

            for thr, thr_result in mode_result["thresholds"].items():
                f.write(f"  Threshold ≥ {thr}\n")
                f.write(f"    Nodes: {thr_result['node_count']}\n")
                f.write(f"    Edges: {thr_result['edge_count']}\n")
                f.write(f"    Density: {thr_result['density']:.6f}\n")
                f.write(f"    Edge CSV: {Path(thr_result['edge_csv']).name}\n")
                f.write(f"    GEXF: {Path(thr_result['gexf']).name}\n")


def _parse_csv_list(value: str) -> list[str]:
    """Parse comma-separated strings into a list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers into a list."""
    items = _parse_csv_list(value)
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
    input_path: Path,
    output_dir: Path,
    title_col: Optional[str] = None,
    n_metadata_cols: int = 0,
    presence_token: str = "X",
    projection_modes: list[str] | tuple[str, ...] = ("feature",),
    min_feature_node_freq: int = 2,
    min_case_node_freq: int = 2,
    feature_edge_thresholds: list[int] | tuple[int, ...] = (20, 30, 40),
    case_edge_thresholds: list[int] | tuple[int, ...] = (5, 10, 15),
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Build one-mode projection networks and return a structured result dictionary.

    This is the entry point pipeline runners should call.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    projection_modes = _normalize_projection_modes(list(projection_modes))
    feature_edge_thresholds = [int(x) for x in feature_edge_thresholds]
    case_edge_thresholds = [int(x) for x in case_edge_thresholds]

    df = _read_table(input_path)
    chosen_title_col = _choose_title_col(df, title_col)
    case_names = _extract_case_names(df, chosen_title_col)
    incidence = _build_incidence(df, n_metadata_cols, presence_token, case_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_stem = input_path.stem
    results: Dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "title_col": chosen_title_col,
        "n_cases_total": int(incidence.shape[0]),
        "n_features_total": int(incidence.shape[1]),
        "projections": {},
    }

    for mode in projection_modes:
        mode_dir = output_dir / mode

        if mode == "feature":
            cooc, node_measure, original_node_count = _build_feature_projection(
                incidence,
                min_feature_node_freq=min_feature_node_freq,
            )
            edge_thresholds = feature_edge_thresholds

        elif mode == "case":
            cooc, node_measure, original_node_count = _build_case_projection(
                incidence,
                min_case_node_freq=min_case_node_freq,
            )
            edge_thresholds = case_edge_thresholds

        else:
            raise ValueError(f"Unsupported projection mode: {mode}")

        retained_node_count = int(cooc.shape[0])

        threshold_results = _export_thresholded_networks(
            cooc=cooc,
            node_measure=node_measure,
            edge_thresholds=edge_thresholds,
            dataset_stem=dataset_stem,
            projection_mode=mode,
            output_dir=mode_dir,
        )

        results["projections"][mode] = {
            "output_dir": str(mode_dir),
            "original_node_count": original_node_count,
            "retained_node_count": retained_node_count,
            "edge_thresholds_used": edge_thresholds,
            "thresholds": threshold_results,
        }

    out_summary = output_dir / out_summary_name
    _write_summary(
        out_path=out_summary,
        run_timestamp=run_timestamp,
        input_path=input_path,
        title_col=chosen_title_col,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        projection_modes=projection_modes,
        min_feature_node_freq=min_feature_node_freq,
        min_case_node_freq=min_case_node_freq,
        feature_edge_thresholds=feature_edge_thresholds,
        case_edge_thresholds=case_edge_thresholds,
        results=results,
    )

    results["summary_txt"] = str(out_summary)
    return results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build one-mode projection networks from a binary incidence matrix."
    )

    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Path to input .xlsx/.xls or .csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--title-col", type=str, default=TITLE_COL, help="Column containing case names")
    parser.add_argument(
        "--n-metadata-cols",
        type=int,
        default=N_METADATA_COLS,
        help="Number of metadata columns before feature columns begin",
    )
    parser.add_argument(
        "--presence-token",
        type=str,
        default=PRESENCE_TOKEN,
        help='Presence token in the matrix (default: "X")',
    )
    parser.add_argument(
        "--projection-modes",
        type=_parse_csv_list,
        default=PROJECTION_MODES,
        help='Comma-separated projection modes: "feature", "case", or both (e.g. feature,case)',
    )
    parser.add_argument(
        "--min-feature-node-freq",
        type=int,
        default=MIN_FEATURE_NODE_FREQ,
        help="Minimum number of cases a feature must appear in (feature projection)",
    )
    parser.add_argument(
        "--min-case-node-freq",
        type=int,
        default=MIN_CASE_NODE_FREQ,
        help="Minimum number of features a case must contain (case projection)",
    )
    parser.add_argument(
        "--feature-edge-thresholds",
        type=_parse_int_list,
        default=FEATURE_EDGE_THRESHOLDS,
        help="Comma-separated edge thresholds for feature projection (e.g. 20,30,40)",
    )
    parser.add_argument(
        "--case-edge-thresholds",
        type=_parse_int_list,
        default=CASE_EDGE_THRESHOLDS,
        help="Comma-separated edge thresholds for case projection (e.g. 5,10,15)",
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
        input_path=args.input,
        output_dir=args.output_dir,
        title_col=args.title_col,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        projection_modes=args.projection_modes,
        min_feature_node_freq=args.min_feature_node_freq,
        min_case_node_freq=args.min_case_node_freq,
        feature_edge_thresholds=args.feature_edge_thresholds,
        case_edge_thresholds=args.case_edge_thresholds,
        out_summary_name=args.out_summary,
    )

    print("[✓] One-mode projection complete.")
    print(f"    Input:              {result['input_path']}")
    print(f"    Cases total:        {result['n_cases_total']:,}")
    print(f"    Features total:     {result['n_features_total']:,}")
    print(f"    Output dir:         {result['output_dir']}")

    for mode, mode_result in result["projections"].items():
        print(f"    [{mode}]")
        print(f"      Original nodes:   {mode_result['original_node_count']:,}")
        print(f"      Retained nodes:   {mode_result['retained_node_count']:,}")
        print(f"      Thresholds:       {mode_result['edge_thresholds_used']}")
        for thr, thr_result in mode_result["thresholds"].items():
            print(
                f"      Threshold ≥ {thr}: "
                f"{thr_result['node_count']:,} nodes | {thr_result['edge_count']:,} edges"
            )

    print(f"    Summary:            {result['summary_txt']}")


if __name__ == "__main__":
    main()