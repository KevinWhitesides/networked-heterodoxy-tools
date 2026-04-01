#!/usr/bin/env python3
"""
05_build_feature_absence_network.py

Construct network representations of statistically significant zero-overlap
relationships between features.

This script is designed to work downstream of:

    03_similarity/06_significant_zero_feature_overlap.py

While that script identifies feature pairs that never co-occur in the same case
and evaluates whether those absences are unusual under a degree-preserving null
model, the present script converts those results into network structures
suitable for exploration and visualization.

Outputs
-------
1) feature_absence_graph_sig.gexf
   One-mode feature × feature graph where edges represent statistically
   significant zero-overlap relationships.

2) feature_absence_bipartite.gexf
   Bipartite case × feature graph built from the retained subset of significant
   absence features and the cases that contain them.

3) analysis_summary.txt
   Human-readable record of the run, inputs, thresholds, and graph statistics.

Standalone use:
    Edit the CONFIG block below, then run:
        python 05_build_feature_absence_network.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Inputs
ZERO_FEATURE_OVERLAP_CSV = Path("zero_feature_overlap_with_significance.csv")
INCIDENCE_PATH = Path("input_incidence_matrix.xlsx")   # .xlsx/.xls or .csv
SHEET_NAME = 0

# Matrix structure
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Significance selection
SIGNIFICANCE_COLUMN = "sig_0.05"

# Retention filters
MIN_ZERO_NEIGHBORS = 2
MIN_FEATURES_PER_CASE = 1
MIN_CASES_PER_FEATURE = 2

# Optional shortening for long case titles in the bipartite graph
SHORTEN_CASE_LABELS = True
TITLE_MAX_LEN = 36
APPEND_ID_FOR_UNIQUENESS = True

# Output
OUTPUT_DIR = Path(".")
OUT_ABS_GRAPH = "feature_absence_graph_sig.gexf"
OUT_BIP_GRAPH = "feature_absence_bipartite.gexf"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# HELPERS
# =============================================================================

def _read_table(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=str, keep_default_na=False)
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use .xlsx/.xls or .csv.")

    df.columns = df.columns.map(str)
    return df


def _get_feature_columns(df: pd.DataFrame, n_metadata_cols: int) -> List[str]:
    if n_metadata_cols < 0 or n_metadata_cols >= df.shape[1]:
        raise ValueError(
            f"N_METADATA_COLS={n_metadata_cols} invalid for table with {df.shape[1]} columns."
        )
    return list(df.columns[n_metadata_cols:])


def _binarize_presence(df: pd.DataFrame, feature_cols: Sequence[str], token: str) -> pd.DataFrame:
    truthy = {"x", "✓", "check", "true", "1", "y", "yes"}

    def to_bin(col: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(col):
            return (col.fillna(0).astype(float) > 0).astype(np.uint8)

        s = col.fillna("").astype(str).str.strip().str.lower()
        return s.apply(
            lambda v: 1
            if v == token.lower() or v in truthy or (v.isdigit() and int(v) > 0)
            else 0
        ).astype(np.uint8)

    return df[list(feature_cols)].apply(to_bin)


def _make_short_titles(titles: List[str], max_len: int, append_id: bool) -> dict[str, str]:
    mapping: dict[str, str] = {}
    seen = set()

    for idx, full in enumerate(titles, start=1):
        t = " ".join(str(full).split()).strip()

        if ":" in t:
            t = t.split(":", 1)[0].strip()

        if len(t) <= max_len:
            cand = t
        else:
            cut = t[:max_len]
            if " " in cut:
                cut = cut[:cut.rfind(" ")]
            cand = cut + "…"

        if append_id:
            cand = f"{cand} [{idx}]"

        base = cand
        k = 2
        while cand in seen:
            suffix = f"({k})"
            trunc = max(0, max_len - len(suffix) - 1)
            trunk = base if len(base) <= trunc else base[:trunc] + "…"
            cand = f"{trunk} {suffix}"
            k += 1

        mapping[full] = cand
        seen.add(cand)

    return mapping


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    zero_feature_overlap_csv: Path,
    incidence_path: Path,
    significance_column: str,
    min_zero_neighbors: int,
    min_features_per_case: int,
    min_cases_per_feature: int,
    shorten_case_labels: bool,
    title_max_len: int,
    append_id_for_uniqueness: bool,
    significant_pairs_used: int,
    retained_absence_features: int,
    absence_graph_edges: int,
    supporting_cases_retained: int,
    bipartite_features_retained: int,
    bipartite_edges: int,
    out_abs_graph_name: str,
    out_bip_graph_name: str,
    out_summary_name: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Feature Absence Network Construction Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Feature zero-overlap CSV: {zero_feature_overlap_csv}\n")
        f.write(f"Incidence matrix: {incidence_path}\n\n")

        f.write("Selection settings\n")
        f.write("------------------\n")
        f.write(f"SIGNIFICANCE_COLUMN: {significance_column}\n")
        f.write(f"MIN_ZERO_NEIGHBORS: {min_zero_neighbors}\n")
        f.write(f"MIN_FEATURES_PER_CASE: {min_features_per_case}\n")
        f.write(f"MIN_CASES_PER_FEATURE: {min_cases_per_feature}\n")
        f.write(f"SHORTEN_CASE_LABELS: {shorten_case_labels}\n")
        f.write(f"TITLE_MAX_LEN: {title_max_len}\n")
        f.write(f"APPEND_ID_FOR_UNIQUENESS: {append_id_for_uniqueness}\n\n")

        f.write("Feature absence graph\n")
        f.write("---------------------\n")
        f.write(f"Significant zero-overlap feature pairs in input table: {significant_pairs_used}\n")
        f.write(f"Features retained after MIN_ZERO_NEIGHBORS filter: {retained_absence_features}\n")
        f.write(f"Edges retained: {absence_graph_edges}\n\n")

        f.write("Retained bipartite graph\n")
        f.write("------------------------\n")
        f.write(f"Supporting cases retained: {supporting_cases_retained}\n")
        f.write(f"Features retained in bipartite subset: {bipartite_features_retained}\n")
        f.write(f"Edges (case–feature links): {bipartite_edges}\n\n")

        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("The one-mode feature absence graph shows statistically significant feature\n")
        f.write("pairs that never co-occur. The bipartite retained-subset graph shows the\n")
        f.write("cases supporting those features, making it possible to inspect the feature\n")
        f.write("regions that remain structurally disjoint across the corpus.\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_abs_graph_name}\n")
        f.write(f"{out_bip_graph_name}\n")
        f.write(f"{out_summary_name}\n")


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run(
    *,
    zero_feature_overlap_csv: Path,
    incidence_path: Path,
    output_dir: Path,
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    significance_column: str = "sig_0.05",
    min_zero_neighbors: int = 2,
    min_features_per_case: int = 1,
    min_cases_per_feature: int = 2,
    shorten_case_labels: bool = True,
    title_max_len: int = 36,
    append_id_for_uniqueness: bool = True,
    out_abs_graph_name: str = "feature_absence_graph_sig.gexf",
    out_bip_graph_name: str = "feature_absence_bipartite.gexf",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Build feature absence networks and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_zero_neighbors < 1:
        raise ValueError("min_zero_neighbors must be >= 1.")
    if min_features_per_case < 1:
        raise ValueError("min_features_per_case must be >= 1.")
    if min_cases_per_feature < 1:
        raise ValueError("min_cases_per_feature must be >= 1.")

    # 1) Read significant feature zero-overlap table
    if not zero_feature_overlap_csv.exists():
        raise FileNotFoundError(f"Feature zero-overlap CSV not found: {zero_feature_overlap_csv}")

    zero_df = pd.read_csv(zero_feature_overlap_csv)
    zero_df.columns = zero_df.columns.map(str)

    required_cols = {"feature_a", "feature_b", significance_column}
    if not required_cols.issubset(zero_df.columns):
        raise ValueError(
            f"{zero_feature_overlap_csv} must contain columns: {sorted(required_cols)}"
        )

    # Keep only significant pairs
    sig_df = zero_df[zero_df[significance_column].astype(bool)].copy()

    if sig_df.empty:
        raise ValueError(
            f"No feature pairs were significant under {significance_column}."
        )

    # 2) Build one-mode feature absence graph
    G_abs = nx.Graph()

    for _, row in sig_df.iterrows():
        fa = str(row["feature_a"])
        fb = str(row["feature_b"])

        attrs_a = {}
        attrs_b = {}
        if "count_a" in row.index:
            attrs_a["frequency"] = int(row["count_a"])
        if "count_b" in row.index:
            attrs_b["frequency"] = int(row["count_b"])

        if fa not in G_abs:
            G_abs.add_node(fa, label=fa, type="feature", **attrs_a)
        if fb not in G_abs:
            G_abs.add_node(fb, label=fb, type="feature", **attrs_b)

        edge_attrs = {}
        if "p_emp" in row.index:
            edge_attrs["p_emp"] = float(row["p_emp"])
        edge_attrs["significance_column"] = significance_column

        G_abs.add_edge(fa, fb, **edge_attrs)

    keep_features = {n for n, d in G_abs.degree() if d >= min_zero_neighbors}

    if not keep_features:
        raise ValueError(
            f"No features remain after applying MIN_ZERO_NEIGHBORS={min_zero_neighbors}."
        )

    G_abs_keep = G_abs.subgraph(keep_features).copy()
    out_abs_path = output_dir / out_abs_graph_name
    nx.write_gexf(G_abs_keep, out_abs_path)

    # 3) Read incidence matrix and build retained bipartite graph
    df_raw = _read_table(incidence_path, sheet_name=sheet_name)

    if case_id_column not in df_raw.columns:
        raise ValueError(f"CASE_ID_COLUMN '{case_id_column}' not found in incidence matrix.")

    case_index = df_raw.columns.get_loc(case_id_column)
    if case_index >= n_metadata_cols:
        raise ValueError(
            f"CASE_ID_COLUMN '{case_id_column}' is outside the first {n_metadata_cols} columns.\n"
            "This script assumes that all metadata columns appear to the LEFT of the feature columns."
        )

    feature_cols = _get_feature_columns(df_raw, n_metadata_cols)
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    # Only keep retained significant-absence features that exist in the matrix
    present_keep_features = [f for f in keep_features if f in feature_cols]
    if not present_keep_features:
        raise ValueError("No retained absence features were found in the incidence matrix.")

    df = pd.concat([df_raw[[case_id_column]], df_raw[present_keep_features]], axis=1).copy()
    bin_features = _binarize_presence(df, present_keep_features, presence_token)
    df_bin = pd.concat([df[[case_id_column]].copy(), bin_features], axis=1)
    df_bin[case_id_column] = df_bin[case_id_column].astype(str)

    # Case retention
    case_feature_counts = df_bin[present_keep_features].sum(axis=1)
    subset_df = df_bin.loc[case_feature_counts >= min_features_per_case].copy()

    if subset_df.empty:
        raise ValueError(
            f"No cases remain after applying MIN_FEATURES_PER_CASE={min_features_per_case}."
        )

    # Refine feature retention inside retained subset
    subset_feature_counts = subset_df[present_keep_features].sum(axis=0).astype(int)
    bip_features = subset_feature_counts[subset_feature_counts >= min_cases_per_feature].index.tolist()

    if not bip_features:
        raise ValueError(
            f"No retained features remain after applying MIN_CASES_PER_FEATURE={min_cases_per_feature}."
        )

    subset_df = subset_df[[case_id_column] + bip_features].copy()

    # Shortened case labels
    retained_titles = subset_df[case_id_column].astype(str).tolist()
    if shorten_case_labels:
        short_map = _make_short_titles(retained_titles, title_max_len, append_id_for_uniqueness)
    else:
        short_map = {t: t for t in retained_titles}

    # Build bipartite graph
    G_bip = nx.Graph()

    # Feature nodes
    for feat in bip_features:
        G_bip.add_node(
            f"feature::{feat}",
            label=feat,
            type="feature",
            bipartite=0,
            degree_in_subset=int(subset_feature_counts[feat]),
        )

    # Case nodes
    for full in retained_titles:
        G_bip.add_node(
            f"case::{full}",
            label=short_map.get(full, full),
            full_title=full,
            type="case",
            bipartite=1,
        )

    # Edges
    for _, row in subset_df.iterrows():
        case = str(row[case_id_column])
        for feat in bip_features:
            if int(row[feat]) == 1:
                G_bip.add_edge(f"case::{case}", f"feature::{feat}", weight=1)

    out_bip_path = output_dir / out_bip_graph_name
    nx.write_gexf(G_bip, out_bip_path)

    # 4) Summary
    out_summary = output_dir / out_summary_name
    _write_summary(
        out_summary,
        run_timestamp=run_timestamp,
        zero_feature_overlap_csv=zero_feature_overlap_csv,
        incidence_path=incidence_path,
        significance_column=significance_column,
        min_zero_neighbors=min_zero_neighbors,
        min_features_per_case=min_features_per_case,
        min_cases_per_feature=min_cases_per_feature,
        shorten_case_labels=shorten_case_labels,
        title_max_len=title_max_len,
        append_id_for_uniqueness=append_id_for_uniqueness,
        significant_pairs_used=len(sig_df),
        retained_absence_features=len(keep_features),
        absence_graph_edges=G_abs_keep.number_of_edges(),
        supporting_cases_retained=len(retained_titles),
        bipartite_features_retained=len(bip_features),
        bipartite_edges=G_bip.number_of_edges(),
        out_abs_graph_name=out_abs_graph_name,
        out_bip_graph_name=out_bip_graph_name,
        out_summary_name=out_summary_name,
    )

    return {
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "significant_pairs_used": len(sig_df),
        "retained_absence_features": len(keep_features),
        "absence_graph_nodes": G_abs_keep.number_of_nodes(),
        "absence_graph_edges": G_abs_keep.number_of_edges(),
        "supporting_cases_retained": len(retained_titles),
        "bipartite_features_retained": len(bip_features),
        "bipartite_graph_nodes": G_bip.number_of_nodes(),
        "bipartite_graph_edges": G_bip.number_of_edges(),
        "feature_absence_graph_gexf": str(out_abs_path),
        "feature_absence_bipartite_gexf": str(out_bip_path),
        "summary_txt": str(out_summary),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct network representations of statistically significant zero-overlap relationships between features."
    )

    parser.add_argument("--zero-feature-overlap-csv", type=Path, default=ZERO_FEATURE_OVERLAP_CSV)
    parser.add_argument("--incidence-path", type=Path, default=INCIDENCE_PATH)
    parser.add_argument("--sheet-name", default=SHEET_NAME)

    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)

    parser.add_argument("--significance-column", type=str, default=SIGNIFICANCE_COLUMN)

    parser.add_argument("--min-zero-neighbors", type=int, default=MIN_ZERO_NEIGHBORS)
    parser.add_argument("--min-features-per-case", type=int, default=MIN_FEATURES_PER_CASE)
    parser.add_argument("--min-cases-per-feature", type=int, default=MIN_CASES_PER_FEATURE)

    parser.add_argument("--shorten-case-labels", action="store_true", default=SHORTEN_CASE_LABELS)
    parser.add_argument("--title-max-len", type=int, default=TITLE_MAX_LEN)
    parser.add_argument("--append-id-for-uniqueness", action="store_true", default=APPEND_ID_FOR_UNIQUENESS)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out-abs-graph", type=str, default=OUT_ABS_GRAPH)
    parser.add_argument("--out-bip-graph", type=str, default=OUT_BIP_GRAPH)
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY)

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = _parse_args()

    result = run(
        zero_feature_overlap_csv=args.zero_feature_overlap_csv,
        incidence_path=args.incidence_path,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        significance_column=args.significance_column,
        min_zero_neighbors=args.min_zero_neighbors,
        min_features_per_case=args.min_features_per_case,
        min_cases_per_feature=args.min_cases_per_feature,
        shorten_case_labels=args.shorten_case_labels,
        title_max_len=args.title_max_len,
        append_id_for_uniqueness=args.append_id_for_uniqueness,
        out_abs_graph_name=args.out_abs_graph,
        out_bip_graph_name=args.out_bip_graph,
        out_summary_name=args.out_summary,
    )

    print("[✓] Feature absence network construction complete.")
    print(f"    Significant feature pairs used: {result['significant_pairs_used']}")
    print(f"    Retained absence features:      {result['retained_absence_features']}")
    print(f"    Absence graph:                  {result['feature_absence_graph_gexf']}")
    print(f"    Bipartite graph:                {result['feature_absence_bipartite_gexf']}")
    print(f"    Summary:                        {result['summary_txt']}")


if __name__ == "__main__":
    main()