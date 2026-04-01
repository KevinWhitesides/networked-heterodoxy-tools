#!/usr/bin/env python3
"""
03_build_case_absence_networks.py

Build network outputs from a zero-overlap significance table.

Creates:
1) absence graph (case × case)
2) bipartite graph (case × trope)
3) analysis summary

Pipeline stage:
    Consumes output from 04_significant_zero_case_overlap.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import networkx as nx


# =============================================================================
# CONFIG (standalone defaults)
# =============================================================================

ZERO_OVERLAP_CSV = Path("zero_overlap_pairs_with_significance.csv")
INPUT_PATH = Path("input_incidence_matrix.xlsx")
SHEET_NAME = 0

CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

SIGNIFICANCE_COLUMN = "sig_0.05"
MIN_ZERO_NEIGHBORS = 2
BIPARTITE_TROPE_MIN_CASES = 2

SHORTEN_CASE_LABELS = True
TITLE_MAX_LEN = 36
APPEND_ID_FOR_UNIQUENESS = True

OUTPUT_DIR = Path(".")
OUT_ABS_GEXF = "absence_graph_sig.gexf"
OUT_BIP_GEXF = "bipartite_thr2.gexf"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# Helpers
# =============================================================================

def _read_table(path: Path, sheet_name=0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name, dtype=str, keep_default_na=False)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Unsupported file format")


def _get_feature_columns(df: pd.DataFrame, n_metadata_cols: int) -> List[str]:
    return list(df.columns[n_metadata_cols:])


def _binarize_presence(df: pd.DataFrame, feature_cols: Sequence[str], token: str) -> pd.DataFrame:
    truthy = {"x", "✓", "check", "true", "1", "y", "yes"}

    def to_bin(col):
        if pd.api.types.is_numeric_dtype(col):
            return (col.fillna(0).astype(float) > 0).astype(np.uint8)
        s = col.fillna("").astype(str).str.strip().str.lower()
        return s.apply(lambda v: 1 if v == token.lower() or v in truthy else 0).astype(np.uint8)

    return df[feature_cols].apply(to_bin)


def _make_short_titles(titles: List[str]) -> Dict[str, str]:
    mapping = {}
    for i, t in enumerate(titles):
        short = t[:TITLE_MAX_LEN] + ("…" if len(t) > TITLE_MAX_LEN else "")
        if APPEND_ID_FOR_UNIQUENESS:
            short = f"{short} [{i+1}]"
        mapping[t] = short
    return mapping


# =============================================================================
# Pipeline entry point
# =============================================================================

def run(
    *,
    zero_overlap_csv: Path,
    input_path: Path,
    output_dir: Path,
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    significance_column: str = "sig_0.05",
    min_zero_neighbors: int = 2,
    bipartite_trope_min_cases: int = 2,
    shorten_case_labels: bool = True,
    out_abs_gexf: str = "absence_graph_sig.gexf",
    out_bip_gexf: str = "bipartite_thr2.gexf",
    out_summary: str = "analysis_summary.txt",
) -> Dict[str, Any]:

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load zero-overlap table ---
    pairs_df = pd.read_csv(zero_overlap_csv)

    sig_df = pairs_df[pairs_df[significance_column].astype(bool)].copy()

    # --- Build absence graph ---
    G_abs = nx.Graph()
    for _, row in sig_df.iterrows():
        G_abs.add_edge(row["case_A"], row["case_B"], p_emp=float(row["p_emp"]))

    keep_cases = {n for n, d in G_abs.degree() if d >= min_zero_neighbors}
    G_abs = G_abs.subgraph(keep_cases).copy()

    abs_path = output_dir / out_abs_gexf
    nx.write_gexf(G_abs, abs_path)

    # --- Load incidence matrix ---
    df_raw = _read_table(input_path, sheet_name)
    feature_cols = _get_feature_columns(df_raw, n_metadata_cols)

    df = pd.concat([df_raw[[case_id_column]], df_raw[feature_cols]], axis=1)
    bin_df = _binarize_presence(df, feature_cols, presence_token)

    df_bin = pd.concat([df[[case_id_column]], bin_df], axis=1)

    retained_df = df_bin[df_bin[case_id_column].isin(keep_cases)].copy()

    # --- Build bipartite graph ---
    trope_freq = retained_df.drop(columns=[case_id_column]).sum(axis=0)
    keep_tropes = trope_freq[trope_freq >= bipartite_trope_min_cases].index.tolist()

    G_bip = nx.Graph()

    titles = retained_df[case_id_column].tolist()
    label_map = _make_short_titles(titles) if shorten_case_labels else {t: t for t in titles}

    for t in titles:
        G_bip.add_node(f"case::{t}", label=label_map[t], type="case")

    for trope in keep_tropes:
        G_bip.add_node(f"trope::{trope}", type="trope")

    for _, row in retained_df.iterrows():
        for trope in keep_tropes:
            if int(row[trope]) == 1:
                G_bip.add_edge(f"case::{row[case_id_column]}", f"trope::{trope}")

    bip_path = output_dir / out_bip_gexf
    nx.write_gexf(G_bip, bip_path)

    # --- Summary ---
    summary_path = output_dir / out_summary
    with open(summary_path, "w") as f:
        f.write(f"Run timestamp: {run_timestamp}\n")
        f.write(f"Absence graph: {G_abs.number_of_nodes()} nodes / {G_abs.number_of_edges()} edges\n")
        f.write(f"Bipartite graph: {len(titles)} cases / {len(keep_tropes)} tropes\n")

    return {
        "absence_gexf": str(abs_path),
        "bipartite_gexf": str(bip_path),
        "summary_txt": str(summary_path),
        "cases_retained": len(keep_cases),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--zero-overlap-csv", type=Path, default=ZERO_OVERLAP_CSV)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    args = parser.parse_args()

    result = run(
        zero_overlap_csv=args.zero_overlap_csv,
        input_path=args.input,
        output_dir=args.output_dir,
    )

    print("[✓] Absence network construction complete.")
    print(f"    Absence graph: {result['absence_gexf']}")
    print(f"    Bipartite graph: {result['bipartite_gexf']}")


if __name__ == "__main__":
    main()