#!/usr/bin/env python3
"""
04_build_case_gradient_networks.py

Build network and diagnostic outputs for a selected discourse gradient.

Outputs
-------
1) gradient_bipartite.gexf
2) gradient_jaccard_subset.csv
3) gradient_jaccard_pairs_ranked.csv
4) gradient_jaccard_heatmap.png
5) analysis_summary.txt

This script is typically used after 05_find_case_gradients.py. It selects one
gradient chain either by row number or by exact chain string, then builds a
bipartite case-feature graph plus within-chain Jaccard diagnostics.

Standalone use:
    Edit the CONFIG block below, then run:
        python 04_build_case_gradient_networks.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

GRADIENTS_CSV = Path("case_gradients.csv")
INCIDENCE_PATH = Path("input_incidence_matrix.xlsx")
SHEET_NAME = 0

CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

SELECTION_MODE = "row"   # "row" or "chain"
SELECTED_ROW = 0
EXACT_CHAIN_STRING = ""

MIN_CASES_FOR_FEATURE = 2

OUTPUT_DIR = Path("gradient_output")

OUT_GEXF = "gradient_bipartite.gexf"
OUT_JACCARD_MATRIX = "gradient_jaccard_subset.csv"
OUT_JACCARD_PAIRS = "gradient_jaccard_pairs_ranked.csv"
OUT_HEATMAP = "gradient_jaccard_heatmap.png"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# TITLE SHORTENING
# =============================================================================

def _shorten_title(title: str) -> str:
    """Remove subtitle after colon."""
    return str(title).split(":")[0].strip()


def _ensure_unique_titles(titles: List[str]) -> List[str]:
    """Ensure shortened titles are unique."""
    seen: Dict[str, int] = {}
    result: List[str] = []

    for t in titles:
        base = _shorten_title(t)

        if base not in seen:
            seen[base] = 1
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base} [{seen[base]}]")

    return result


# =============================================================================
# DATA LOADING
# =============================================================================

def _read_incidence_matrix(path: Path, *, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read incidence matrix from CSV or Excel, preserving blanks."""
    if not path.exists():
        raise FileNotFoundError(f"Incidence matrix not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv, .xlsx, or .xls.")

    df = df.fillna("")
    df.columns = df.columns.map(str)
    return df


def _get_feature_columns(df: pd.DataFrame, n_metadata_cols: int) -> List[str]:
    """Assume feature columns begin after the first n_metadata_cols columns."""
    if n_metadata_cols < 0 or n_metadata_cols >= df.shape[1]:
        raise ValueError(
            f"N_METADATA_COLS={n_metadata_cols} invalid for table with {df.shape[1]} columns."
        )
    return list(df.columns[n_metadata_cols:])


def _binarize(df: pd.DataFrame, feature_cols: List[str], *, presence_token: str) -> pd.DataFrame:
    """Convert feature columns to 0/1 using the configured presence token."""
    truthy = {"x", "✓", "check", "true", "1", "y", "yes"}

    def to_bin(x: object) -> int:
        s = str(x).strip().lower()
        if s == presence_token.lower() or s in truthy:
            return 1
        if s.isdigit() and int(s) > 0:
            return 1
        return 0

    return df[feature_cols].applymap(to_bin).astype(np.uint8)


# =============================================================================
# JACCARD
# =============================================================================

def _compute_jaccard_matrix(X: np.ndarray) -> np.ndarray:
    """Compute case × case Jaccard matrix from binary matrix X."""
    n = X.shape[0]
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            inter = int(np.bitwise_and(X[i], X[j]).sum())
            union = int(np.bitwise_or(X[i], X[j]).sum())
            jacc = inter / union if union > 0 else 0.0
            mat[i, j] = jacc
            mat[j, i] = jacc

    return mat


def _build_ranked_pairs(X: np.ndarray, titles: List[str]) -> pd.DataFrame:
    """Build ranked Jaccard pair list within the selected gradient subset."""
    rows = []

    for i, j in combinations(range(len(titles)), 2):
        inter = int(np.bitwise_and(X[i], X[j]).sum())
        union = int(np.bitwise_or(X[i], X[j]).sum())
        jacc = inter / union if union > 0 else 0.0

        rows.append(
            {
                "case_1": titles[i],
                "case_2": titles[j],
                "shared_features": inter,
                "union_features": union,
                "jaccard": jacc,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("jaccard", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# HEATMAP
# =============================================================================

def _write_heatmap(matrix: np.ndarray, labels: List[str], path: Path) -> None:
    """Write heatmap image for within-gradient Jaccard similarities."""
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Jaccard Similarity Within Gradient")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# =============================================================================
# BIPARTITE GRAPH
# =============================================================================

def _build_bipartite(
    subset_df: pd.DataFrame,
    case_titles: Dict[str, str],
    feature_cols: List[str],
    *,
    case_id_column: str,
    min_cases_for_feature: int,
) -> tuple[nx.Graph, List[str]]:
    """
    Build gradient bipartite graph and return:
        graph, kept_features
    """
    trope_freq = subset_df[feature_cols].sum()
    keep_features = trope_freq[trope_freq >= min_cases_for_feature].index.tolist()

    G = nx.Graph()

    # Case nodes
    for full, short in case_titles.items():
        G.add_node(
            short,
            label=short,
            full_title=full,
            node_type="case",
        )

    # Feature nodes
    for feat in keep_features:
        G.add_node(
            feat,
            label=feat,
            node_type="feature",
            degree_in_gradient_subset=int(trope_freq[feat]),
        )

    # Edges
    for _, row in subset_df.iterrows():
        case = case_titles[row[case_id_column]]
        for feat in keep_features:
            if int(row[feat]) == 1:
                G.add_edge(case, feat)

    return G, keep_features


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run(
    *,
    gradients_csv: Path,
    incidence_path: Path,
    output_dir: Path,
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    selection_mode: str = "row",      # "row" or "chain"
    selected_row: int = 0,
    exact_chain_string: str = "",
    min_cases_for_feature: int = 2,
    out_gexf_name: str = "gradient_bipartite.gexf",
    out_jaccard_matrix_name: str = "gradient_jaccard_subset.csv",
    out_jaccard_pairs_name: str = "gradient_jaccard_pairs_ranked.csv",
    out_heatmap_name: str = "gradient_jaccard_heatmap.png",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Build network and diagnostic outputs for one selected case gradient.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if selection_mode not in {"row", "chain"}:
        raise ValueError("selection_mode must be 'row' or 'chain'.")

    if min_cases_for_feature < 1:
        raise ValueError("min_cases_for_feature must be >= 1.")

    # --------------------------------------------------------
    # Load gradients
    # --------------------------------------------------------
    if not gradients_csv.exists():
        raise FileNotFoundError(f"Gradients CSV not found: {gradients_csv}")

    grad_df = pd.read_csv(gradients_csv)

    if "chain" not in grad_df.columns:
        raise ValueError(f"{gradients_csv} must contain a 'chain' column.")

    if grad_df.empty:
        raise ValueError("Gradients CSV is empty.")

    if selection_mode == "row":
        if selected_row < 0 or selected_row >= len(grad_df):
            raise ValueError(
                f"selected_row={selected_row} is out of bounds for gradients CSV with {len(grad_df)} rows."
            )
        row = grad_df.iloc[selected_row]

    else:  # selection_mode == "chain"
        if not exact_chain_string.strip():
            raise ValueError("For selection_mode='chain', exact_chain_string must be provided.")

        matches = grad_df[grad_df["chain"].astype(str) == exact_chain_string]
        if matches.empty:
            raise ValueError("No row in gradients CSV matched exact_chain_string.")
        row = matches.iloc[0]

    chain = [c.strip() for c in str(row["chain"]).split("|")]
    if len(chain) < 2:
        raise ValueError("Selected chain must contain at least 2 cases.")

    # --------------------------------------------------------
    # Load incidence matrix
    # --------------------------------------------------------
    df = _read_incidence_matrix(incidence_path, sheet_name=sheet_name)

    if case_id_column not in df.columns:
        raise ValueError(f"CASE_ID_COLUMN '{case_id_column}' not found in {incidence_path}")

    feature_cols = _get_feature_columns(df, n_metadata_cols)
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    bin_df = _binarize(df, feature_cols, presence_token=presence_token)
    df_bin = pd.concat([df[[case_id_column]].copy(), bin_df], axis=1)
    df_bin[case_id_column] = df_bin[case_id_column].astype(str)

    # --------------------------------------------------------
    # Subset to gradient chain
    # --------------------------------------------------------
    subset = df_bin[df_bin[case_id_column].isin(chain)].copy()

    if len(subset) != len(chain):
        found_cases = set(subset[case_id_column].astype(str).tolist())
        missing = [c for c in chain if c not in found_cases]
        raise ValueError(
            "Some chain cases were not found in the incidence matrix: "
            f"{missing[:10]}"
        )

    subset = subset.set_index(case_id_column).loc[chain].reset_index()

    # --------------------------------------------------------
    # Title shortening
    # --------------------------------------------------------
    full_titles = subset[case_id_column].astype(str).tolist()
    short_titles = _ensure_unique_titles(full_titles)
    title_map = dict(zip(full_titles, short_titles))

    # --------------------------------------------------------
    # Build bipartite graph
    # --------------------------------------------------------
    G, keep_features = _build_bipartite(
        subset,
        title_map,
        feature_cols,
        case_id_column=case_id_column,
        min_cases_for_feature=min_cases_for_feature,
    )

    out_gexf_path = output_dir / out_gexf_name
    nx.write_gexf(G, out_gexf_path)

    # --------------------------------------------------------
    # Jaccard matrix
    # --------------------------------------------------------
    X = subset[feature_cols].to_numpy(dtype=np.uint8)

    jaccard = _compute_jaccard_matrix(X)
    jacc_df = pd.DataFrame(jaccard, index=short_titles, columns=short_titles)

    out_jaccard_matrix_path = output_dir / out_jaccard_matrix_name
    jacc_df.to_csv(out_jaccard_matrix_path, encoding="utf-8")

    # --------------------------------------------------------
    # Ranked pair list
    # --------------------------------------------------------
    ranked = _build_ranked_pairs(X, short_titles)

    out_jaccard_pairs_path = output_dir / out_jaccard_pairs_name
    ranked.to_csv(out_jaccard_pairs_path, index=False, encoding="utf-8")

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------
    out_heatmap_path = output_dir / out_heatmap_name
    _write_heatmap(jaccard, short_titles, out_heatmap_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    out_summary_path = output_dir / out_summary_name
    with open(out_summary_path, "w", encoding="utf-8") as f:
        f.write("=== Gradient Network Construction Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Gradients CSV: {gradients_csv}\n")
        f.write(f"Incidence matrix: {incidence_path}\n\n")

        f.write("Selection\n")
        f.write("---------\n")
        f.write(f"SELECTION_MODE: {selection_mode}\n")
        if selection_mode == "row":
            f.write(f"SELECTED_ROW: {selected_row}\n")
        else:
            f.write(f"EXACT_CHAIN_STRING: {exact_chain_string}\n")
        f.write(f"Selected chain:\n    {' -> '.join(short_titles)}\n\n")

        f.write("Settings\n")
        f.write("--------\n")
        f.write(f"MIN_CASES_FOR_FEATURE: {min_cases_for_feature}\n\n")

        f.write("Output statistics\n")
        f.write("-----------------\n")
        f.write(f"Cases in gradient: {len(short_titles)}\n")
        f.write(f"Features retained in bipartite graph: {len(keep_features)}\n")
        f.write(f"Bipartite graph nodes: {G.number_of_nodes()}\n")
        f.write(f"Bipartite graph edges: {G.number_of_edges()}\n")
        f.write(f"Ranked Jaccard pairs: {len(ranked)}\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_gexf_name}\n")
        f.write(f"{out_jaccard_matrix_name}\n")
        f.write(f"{out_jaccard_pairs_name}\n")
        f.write(f"{out_heatmap_name}\n")
        f.write(f"{out_summary_name}\n")

    return {
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "selected_chain_full": full_titles,
        "selected_chain_short": short_titles,
        "cases_in_gradient": len(short_titles),
        "features_retained": len(keep_features),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "gradient_bipartite_gexf": str(out_gexf_path),
        "gradient_jaccard_subset_csv": str(out_jaccard_matrix_path),
        "gradient_jaccard_pairs_csv": str(out_jaccard_pairs_path),
        "gradient_jaccard_heatmap_png": str(out_heatmap_path),
        "summary_txt": str(out_summary_path),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build network and diagnostic outputs for a selected discourse gradient."
    )

    parser.add_argument("--gradients-csv", type=Path, default=GRADIENTS_CSV)
    parser.add_argument("--incidence-path", type=Path, default=INCIDENCE_PATH)
    parser.add_argument("--sheet-name", default=SHEET_NAME)

    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)

    parser.add_argument("--selection-mode", type=str, default=SELECTION_MODE)
    parser.add_argument("--selected-row", type=int, default=SELECTED_ROW)
    parser.add_argument("--exact-chain-string", type=str, default=EXACT_CHAIN_STRING)

    parser.add_argument("--min-cases-for-feature", type=int, default=MIN_CASES_FOR_FEATURE)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out-gexf", type=str, default=OUT_GEXF)
    parser.add_argument("--out-jaccard-matrix", type=str, default=OUT_JACCARD_MATRIX)
    parser.add_argument("--out-jaccard-pairs", type=str, default=OUT_JACCARD_PAIRS)
    parser.add_argument("--out-heatmap", type=str, default=OUT_HEATMAP)
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    result = run(
        gradients_csv=args.gradients_csv,
        incidence_path=args.incidence_path,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        selection_mode=args.selection_mode,
        selected_row=args.selected_row,
        exact_chain_string=args.exact_chain_string,
        min_cases_for_feature=args.min_cases_for_feature,
        out_gexf_name=args.out_gexf,
        out_jaccard_matrix_name=args.out_jaccard_matrix,
        out_jaccard_pairs_name=args.out_jaccard_pairs,
        out_heatmap_name=args.out_heatmap,
        out_summary_name=args.out_summary,
    )

    print("[✓] Gradient network built.")
    print(f"    Cases:      {result['cases_in_gradient']}")
    print(f"    Features:   {result['features_retained']}")
    print(f"    Graph:      {result['graph_nodes']} nodes | {result['graph_edges']} edges")
    print(f"    Chain:      {' -> '.join(result['selected_chain_short'])}")
    print(f"    Output:     {result['gradient_bipartite_gexf']}")
    print(f"    Summary:    {result['summary_txt']}")


if __name__ == "__main__":
    main()