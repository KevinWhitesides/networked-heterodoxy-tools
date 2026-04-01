#!/usr/bin/env python3
"""
06_build_feature_gradient_networks.py

Build network and diagnostic outputs for a selected feature gradient found by
03_similarity/07_find_feature_gradients.py.

This script takes one selected feature gradient chain (for example
Feature A -> Feature B -> Feature C -> Feature D -> Feature E) and builds:

1) feature_gradient_bipartite.gexf
   A feature × case bipartite network for the selected gradient

2) feature_gradient_jaccard_subset.csv
   The feature × feature Jaccard matrix for the selected gradient features

3) feature_gradient_jaccard_pairs_ranked.csv
   Ranked pairwise similarity table for the selected gradient features

4) feature_gradient_jaccard_heatmap.png
   A heatmap of the selected feature-gradient Jaccard structure

5) analysis_summary.txt
   A compact record of the selected gradient, thresholds, and output statistics

Typical workflow
----------------
1) Run 03_similarity/07_find_feature_gradients.py to produce:
      feature_gradients.csv
2) Select one feature gradient from that table
3) Run this script to build graph / heatmap outputs for it

Standalone use:
    Edit the CONFIG block below, then run:
        python 06_build_feature_gradient_networks.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Inputs
GRADIENTS_CSV = Path("feature_gradients.csv")
INCIDENCE_PATH = Path("input_incidence_matrix.xlsx")   # .xlsx or .csv
SHEET_NAME = 0

# Incidence matrix structure
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Gradient selection mode
#   "row"      -> use SELECTED_ROW (0-based row index in feature_gradients.csv)
#   "endpoint" -> use FEATURE_A and FEATURE_E, then choose the top-ranked row for that pair
#   "chain"    -> use EXACT_CHAIN_STRING exactly as stored in the "chain" column
SELECTION_MODE = "row"          # "row", "endpoint", or "chain"
SELECTED_ROW = 0

FEATURE_A = ""
FEATURE_E = ""

EXACT_CHAIN_STRING = ""

# Case-retention threshold within the selected gradient support graph
# Default = 2, so a case must contain at least two selected gradient features
MIN_GRADIENT_FEATURES_PER_CASE = 2

# Optional label shortening for long case titles
SHORTEN_CASE_LABELS = True
TITLE_MAX_LEN = 36
APPEND_ID_FOR_UNIQUENESS = True

# Output
OUTPUT_DIR = Path(".")
OUT_BIP_GEXF = "feature_gradient_bipartite.gexf"
OUT_JACCARD_CSV = "feature_gradient_jaccard_subset.csv"
OUT_JACCARD_PAIRS = "feature_gradient_jaccard_pairs_ranked.csv"
OUT_JACCARD_PNG = "feature_gradient_jaccard_heatmap.png"
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


def _select_gradient_row(
    df: pd.DataFrame,
    *,
    selection_mode: str,
    selected_row: int,
    feature_a: str,
    feature_e: str,
    exact_chain_string: str,
) -> pd.Series:
    if "chain" not in df.columns:
        raise ValueError("GRADIENTS_CSV must contain a 'chain' column.")

    if selection_mode == "row":
        if selected_row < 0 or selected_row >= len(df):
            raise ValueError(f"SELECTED_ROW {selected_row} is out of range for {len(df)} rows.")
        return df.iloc[selected_row]

    if selection_mode == "endpoint":
        if not feature_a or not feature_e:
            raise ValueError("For selection_mode='endpoint', set feature_a and feature_e.")

        sub = df[
            ((df["feature_A"].astype(str) == str(feature_a)) & (df["feature_E"].astype(str) == str(feature_e))) |
            ((df["feature_A"].astype(str) == str(feature_e)) & (df["feature_E"].astype(str) == str(feature_a)))
        ].copy()

        if sub.empty:
            raise ValueError(f"No feature gradient rows found for endpoint pair ({feature_a}, {feature_e}).")

        return sub.iloc[0]

    if selection_mode == "chain":
        if not exact_chain_string:
            raise ValueError("For selection_mode='chain', set exact_chain_string.")

        sub = df[df["chain"].astype(str) == str(exact_chain_string)].copy()
        if sub.empty:
            raise ValueError(f"No feature gradient row found with chain string:\n{exact_chain_string}")
        return sub.iloc[0]

    raise ValueError("selection_mode must be 'row', 'endpoint', or 'chain'.")


def _parse_chain(chain_string: str) -> List[str]:
    parts = [p.strip() for p in str(chain_string).split("|")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse a valid chain from: {chain_string}")
    return parts


def _build_feature_bipartite(
    subset_df: pd.DataFrame,
    case_id_col: str,
    gradient_features: List[str],
    *,
    short_case_labels: bool = True,
    title_max_len: int = 36,
    append_id: bool = True,
) -> tuple[nx.Graph, list[str], list[str]]:
    """
    Build a feature × case bipartite network from the selected gradient features
    and retained supporting cases.
    """
    retained_titles = subset_df[case_id_col].astype(str).tolist()

    if short_case_labels:
        short_map = _make_short_titles(retained_titles, title_max_len, append_id)
    else:
        short_map = {t: t for t in retained_titles}

    G = nx.Graph()

    # Feature nodes first (focal side)
    for feat in gradient_features:
        G.add_node(
            f"feature::{feat}",
            label=feat,
            type="feature",
            bipartite=0,
        )

    # Case nodes
    for full in retained_titles:
        G.add_node(
            f"case::{full}",
            label=short_map.get(full, full),
            full_title=full,
            type="case",
            bipartite=1,
        )

    # Edges
    for _, row in subset_df.iterrows():
        case = str(row[case_id_col])
        for feat in gradient_features:
            if int(row[feat]) == 1:
                G.add_edge(f"feature::{feat}", f"case::{case}", weight=1)

    return G, retained_titles, gradient_features


def _compute_feature_jaccard_subset(subset_df: pd.DataFrame, gradient_features: List[str]) -> pd.DataFrame:
    """
    Compute feature × feature Jaccard matrix for the selected gradient features.
    """
    X = subset_df[gradient_features].to_numpy(dtype=np.uint8)
    n = len(gradient_features)
    mat = np.zeros((n, n), dtype=float)

    # columns are features
    for i in range(n):
        ai = X[:, i]
        for j in range(i, n):
            bj = X[:, j]
            inter = int(np.bitwise_and(ai, bj).sum())
            union = int(np.bitwise_or(ai, bj).sum())
            jacc = (inter / union) if union > 0 else 0.0
            mat[i, j] = jacc
            mat[j, i] = jacc

    return pd.DataFrame(mat, index=gradient_features, columns=gradient_features)


def _build_ranked_feature_pairs(subset_df: pd.DataFrame, gradient_features: List[str]) -> pd.DataFrame:
    """
    Ranked pair table for the selected gradient features.
    """
    X = subset_df[gradient_features].to_numpy(dtype=np.uint8)
    rows = []

    for i, j in combinations(range(len(gradient_features)), 2):
        ai = X[:, i]
        bj = X[:, j]
        inter = int(np.bitwise_and(ai, bj).sum())
        union = int(np.bitwise_or(ai, bj).sum())
        jacc = (inter / union) if union > 0 else 0.0

        rows.append({
            "feature_1": gradient_features[i],
            "feature_2": gradient_features[j],
            "shared_cases": inter,
            "union_cases": union,
            "jaccard": round(jacc, 6),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["jaccard", "shared_cases", "feature_1", "feature_2"],
            ascending=[False, False, True, True]
        ).reset_index(drop=True)
    return out


def _write_heatmap(jacc_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(jacc_df.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(jacc_df.columns)))
    ax.set_yticks(np.arange(len(jacc_df.index)))
    ax.set_xticklabels(jacc_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(jacc_df.index)

    ax.set_title("Feature Jaccard Similarity Within Selected Gradient")
    ax.set_xlabel("Features")
    ax.set_ylabel("Features")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jaccard similarity")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    gradients_csv: Path,
    incidence_path: Path,
    selection_mode: str,
    selected_row: int,
    feature_a: str,
    feature_e: str,
    exact_chain_string: str,
    selected_chain: List[str],
    endpoint_A: str,
    endpoint_E: str,
    min_gradient_features_per_case: int,
    shorten_case_labels: bool,
    title_max_len: int,
    append_id_for_uniqueness: bool,
    retained_features: int,
    retained_titles: int,
    bip_edges: int,
    jacc_shape: Tuple[int, int],
    min_offdiag: float | None,
    max_offdiag: float | None,
    out_bip_gexf_name: str,
    out_jaccard_csv_name: str,
    out_jaccard_pairs_name: str,
    out_jaccard_png_name: str,
    out_summary_name: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Feature Gradient Network Construction Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Feature gradient table: {gradients_csv}\n")
        f.write(f"Incidence matrix: {incidence_path}\n\n")

        f.write("Gradient selection\n")
        f.write("------------------\n")
        f.write(f"SELECTION_MODE: {selection_mode}\n")
        if selection_mode == "row":
            f.write(f"SELECTED_ROW: {selected_row}\n")
        elif selection_mode == "endpoint":
            f.write(f"FEATURE_A: {feature_a}\n")
            f.write(f"FEATURE_E: {feature_e}\n")
        else:
            f.write(f"EXACT_CHAIN_STRING: {exact_chain_string}\n")
        f.write(f"Selected chain: {' -> '.join(selected_chain)}\n")
        f.write(f"Chain length: {len(selected_chain)}\n")
        f.write(f"Endpoint A: {endpoint_A}\n")
        f.write(f"Endpoint E: {endpoint_E}\n\n")

        f.write("Support-graph settings\n")
        f.write("----------------------\n")
        f.write(f"MIN_GRADIENT_FEATURES_PER_CASE: {min_gradient_features_per_case}\n")
        f.write(f"SHORTEN_CASE_LABELS: {shorten_case_labels}\n")
        f.write(f"TITLE_MAX_LEN: {title_max_len}\n")
        f.write(f"APPEND_ID_FOR_UNIQUENESS: {append_id_for_uniqueness}\n\n")

        f.write("Feature × case bipartite network\n")
        f.write("-------------------------------\n")
        f.write(f"Gradient features retained: {retained_features}\n")
        f.write(f"Supporting cases retained: {retained_titles}\n")
        f.write(f"Edges (feature–case links): {bip_edges}\n\n")

        f.write("Feature Jaccard subset\n")
        f.write("----------------------\n")
        f.write(f"Matrix size: {jacc_shape[0]} × {jacc_shape[1]}\n")
        if min_offdiag is not None and max_offdiag is not None:
            f.write(f"Minimum off-diagonal Jaccard: {min_offdiag:.6f}\n")
            f.write(f"Maximum off-diagonal Jaccard: {max_offdiag:.6f}\n\n")
        else:
            f.write("Only one feature retained; no off-diagonal similarities.\n\n")

        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("The bipartite graph shows the selected feature gradient together with the\n")
        f.write("cases that instantiate it. By default, only cases containing at least two\n")
        f.write("selected gradient features are retained, making the support structure of the\n")
        f.write("feature gradient easier to interpret. The Jaccard outputs provide a compact\n")
        f.write("feature-similarity view of the same selected gradient.\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_bip_gexf_name}\n")
        f.write(f"{out_jaccard_csv_name}\n")
        f.write(f"{out_jaccard_pairs_name}\n")
        f.write(f"{out_jaccard_png_name}\n")
        f.write(f"{out_summary_name}\n")


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
    selection_mode: str = "row",     # "row", "endpoint", or "chain"
    selected_row: int = 0,
    feature_a: str = "",
    feature_e: str = "",
    exact_chain_string: str = "",
    min_gradient_features_per_case: int = 2,
    shorten_case_labels: bool = True,
    title_max_len: int = 36,
    append_id_for_uniqueness: bool = True,
    out_bip_gexf_name: str = "feature_gradient_bipartite.gexf",
    out_jaccard_csv_name: str = "feature_gradient_jaccard_subset.csv",
    out_jaccard_pairs_name: str = "feature_gradient_jaccard_pairs_ranked.csv",
    out_jaccard_png_name: str = "feature_gradient_jaccard_heatmap.png",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Build network and diagnostic outputs for a selected feature gradient.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if selection_mode not in {"row", "endpoint", "chain"}:
        raise ValueError("selection_mode must be 'row', 'endpoint', or 'chain'.")

    if min_gradient_features_per_case < 1:
        raise ValueError("min_gradient_features_per_case must be >= 1.")

    # 1) Read gradient table and select one feature chain
    if not gradients_csv.exists():
        raise FileNotFoundError(f"Feature gradient CSV not found: {gradients_csv}")

    grad_df = pd.read_csv(gradients_csv)
    grad_df.columns = grad_df.columns.map(str)

    selected = _select_gradient_row(
        grad_df,
        selection_mode=selection_mode,
        selected_row=selected_row,
        feature_a=feature_a,
        feature_e=feature_e,
        exact_chain_string=exact_chain_string,
    )
    selected_chain = _parse_chain(str(selected["chain"]))

    endpoint_A = str(selected["feature_A"]) if "feature_A" in selected.index else selected_chain[0]
    endpoint_E = str(selected["feature_E"]) if "feature_E" in selected.index else selected_chain[-1]

    # 2) Read incidence matrix and binarize
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

    df = pd.concat([df_raw[[case_id_column]], df_raw[feature_cols]], axis=1)
    bin_features = _binarize_presence(df, feature_cols, presence_token)
    df_bin = pd.concat([df[[case_id_column]], bin_features], axis=1)
    df_bin[case_id_column] = df_bin[case_id_column].astype(str)

    # Validate selected features
    available_features = set(feature_cols)
    missing = [f for f in selected_chain if f not in available_features]
    if missing:
        raise ValueError(f"These features from the selected gradient were not found in the incidence matrix: {missing}")

    # 3) Retain supporting cases
    # Keep cases containing at least MIN_GRADIENT_FEATURES_PER_CASE selected gradient features
    support_counts = df_bin[selected_chain].sum(axis=1)
    subset_df = df_bin.loc[
        support_counts >= min_gradient_features_per_case,
        [case_id_column] + selected_chain
    ].copy()

    if subset_df.empty:
        raise ValueError(
            "No cases remain after applying MIN_GRADIENT_FEATURES_PER_CASE. "
            "Try lowering the threshold."
        )

    # 4) Build feature × case bipartite graph
    G_bip, retained_titles, retained_features = _build_feature_bipartite(
        subset_df=subset_df,
        case_id_col=case_id_column,
        gradient_features=selected_chain,
        short_case_labels=shorten_case_labels,
        title_max_len=title_max_len,
        append_id=append_id_for_uniqueness,
    )

    out_bip_path = output_dir / out_bip_gexf_name
    nx.write_gexf(G_bip, out_bip_path)

    # 5) Feature-feature Jaccard subset + ranked pairs + heatmap
    jacc_subset_df = _compute_feature_jaccard_subset(subset_df, selected_chain)

    out_jacc_csv = output_dir / out_jaccard_csv_name
    jacc_subset_df.to_csv(out_jacc_csv, encoding="utf-8")

    ranked_pairs_df = _build_ranked_feature_pairs(subset_df, selected_chain)
    out_pairs_csv = output_dir / out_jaccard_pairs_name
    ranked_pairs_df.to_csv(out_pairs_csv, index=False, encoding="utf-8")

    out_jacc_png = output_dir / out_jaccard_png_name
    _write_heatmap(jacc_subset_df, out_jacc_png)

    # 6) Summary
    out_summary = output_dir / out_summary_name

    min_offdiag: float | None = None
    max_offdiag: float | None = None
    if len(jacc_subset_df) > 1:
        mask = ~np.eye(len(jacc_subset_df), dtype=bool)
        offdiag_vals = jacc_subset_df.to_numpy()[mask]
        if offdiag_vals.size > 0:
            min_offdiag = float(offdiag_vals.min())
            max_offdiag = float(offdiag_vals.max())

    _write_summary(
        out_summary,
        run_timestamp=run_timestamp,
        gradients_csv=gradients_csv,
        incidence_path=incidence_path,
        selection_mode=selection_mode,
        selected_row=selected_row,
        feature_a=feature_a,
        feature_e=feature_e,
        exact_chain_string=exact_chain_string,
        selected_chain=selected_chain,
        endpoint_A=endpoint_A,
        endpoint_E=endpoint_E,
        min_gradient_features_per_case=min_gradient_features_per_case,
        shorten_case_labels=shorten_case_labels,
        title_max_len=title_max_len,
        append_id_for_uniqueness=append_id_for_uniqueness,
        retained_features=len(retained_features),
        retained_titles=len(retained_titles),
        bip_edges=G_bip.number_of_edges(),
        jacc_shape=jacc_subset_df.shape,
        min_offdiag=min_offdiag,
        max_offdiag=max_offdiag,
        out_bip_gexf_name=out_bip_gexf_name,
        out_jaccard_csv_name=out_jaccard_csv_name,
        out_jaccard_pairs_name=out_jaccard_pairs_name,
        out_jaccard_png_name=out_jaccard_png_name,
        out_summary_name=out_summary_name,
    )

    return {
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "selected_chain": selected_chain,
        "endpoint_A": endpoint_A,
        "endpoint_E": endpoint_E,
        "gradient_features_retained": len(retained_features),
        "supporting_cases_retained": len(retained_titles),
        "bipartite_nodes": G_bip.number_of_nodes(),
        "bipartite_edges": G_bip.number_of_edges(),
        "feature_gradient_bipartite_gexf": str(out_bip_path),
        "feature_gradient_jaccard_csv": str(out_jacc_csv),
        "feature_gradient_ranked_pairs_csv": str(out_pairs_csv),
        "feature_gradient_heatmap_png": str(out_jacc_png),
        "summary_txt": str(out_summary),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build network and diagnostic outputs for a selected feature gradient."
    )

    parser.add_argument("--gradients-csv", type=Path, default=GRADIENTS_CSV)
    parser.add_argument("--incidence-path", type=Path, default=INCIDENCE_PATH)
    parser.add_argument("--sheet-name", default=SHEET_NAME)

    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)

    parser.add_argument("--selection-mode", type=str, default=SELECTION_MODE)
    parser.add_argument("--selected-row", type=int, default=SELECTED_ROW)
    parser.add_argument("--feature-a", type=str, default=FEATURE_A)
    parser.add_argument("--feature-e", type=str, default=FEATURE_E)
    parser.add_argument("--exact-chain-string", type=str, default=EXACT_CHAIN_STRING)

    parser.add_argument("--min-gradient-features-per-case", type=int, default=MIN_GRADIENT_FEATURES_PER_CASE)

    parser.add_argument("--shorten-case-labels", action="store_true", default=SHORTEN_CASE_LABELS)
    parser.add_argument("--title-max-len", type=int, default=TITLE_MAX_LEN)
    parser.add_argument("--append-id-for-uniqueness", action="store_true", default=APPEND_ID_FOR_UNIQUENESS)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out-bip-gexf", type=str, default=OUT_BIP_GEXF)
    parser.add_argument("--out-jaccard-csv", type=str, default=OUT_JACCARD_CSV)
    parser.add_argument("--out-jaccard-pairs", type=str, default=OUT_JACCARD_PAIRS)
    parser.add_argument("--out-jaccard-png", type=str, default=OUT_JACCARD_PNG)
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY)

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

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
        feature_a=args.feature_a,
        feature_e=args.feature_e,
        exact_chain_string=args.exact_chain_string,
        min_gradient_features_per_case=args.min_gradient_features_per_case,
        shorten_case_labels=args.shorten_case_labels,
        title_max_len=args.title_max_len,
        append_id_for_uniqueness=args.append_id_for_uniqueness,
        out_bip_gexf_name=args.out_bip_gexf,
        out_jaccard_csv_name=args.out_jaccard_csv,
        out_jaccard_pairs_name=args.out_jaccard_pairs,
        out_jaccard_png_name=args.out_jaccard_png,
        out_summary_name=args.out_summary,
    )

    print("[✓] Feature gradient network construction complete.")
    print(f"    Selected chain:      {' -> '.join(result['selected_chain'])}")
    print(f"    Bipartite GEXF:      {result['feature_gradient_bipartite_gexf']}")
    print(f"    Jaccard CSV:         {result['feature_gradient_jaccard_csv']}")
    print(f"    Ranked pairs CSV:    {result['feature_gradient_ranked_pairs_csv']}")
    print(f"    Jaccard heatmap:     {result['feature_gradient_heatmap_png']}")
    print(f"    Summary:             {result['summary_txt']}")


if __name__ == "__main__":
    main()