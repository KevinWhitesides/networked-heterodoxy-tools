#!/usr/bin/env python3
"""
08_gradient_recurrence_analyzer.py

Analyze recurrence across retained gradients to identify a possible
"meta-boundary vocabulary" (or, in feature-gradient mode, a recurring
mediating case set).

Supports two modes:

1) case
   - Input gradients are case gradients (from 05_find_case_gradients.py)
   - Chain members are CASES
   - Mediators counted across gradients are FEATURES

2) feature
   - Input gradients are feature gradients (from 07_find_feature_gradients.py)
   - Chain members are FEATURES
   - Mediators counted across gradients are CASES

Main idea
---------
For each retained gradient row:
- reconstruct the ordered chain
- derive the set of mediators relevant to that chain
- retain only mediators that meet a minimum within-gradient support threshold
- aggregate recurrence across all gradients

Outputs
-------
1) gradient_recurrence_summary.csv
   One row per mediator with recurrence metrics across gradients

2) gradient_recurrence_membership_long.csv
   One row per (gradient, mediator) membership

3) gradient_recurrence_corecurrence_edges.csv
   Pairwise co-recurrence counts of mediators across gradients

4) analysis_summary.txt
   Human-readable summary

Notes
-----
- This script does NOT require Stage 4 GEXFs.
- It works directly from Stage 3 gradients CSV + the original incidence matrix.
- For case gradients, the resulting "mediators" are features/tropes.
- For feature gradients, the resulting "mediators" are cases.

"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Inputs
GRADIENTS_CSV = Path("case_gradients.csv")
INCIDENCE_PATH = Path("input_incidence_matrix.xlsx")   # .xlsx/.xls or .csv
SHEET_NAME = 0

# Mode: "case" or "feature"
GRADIENT_KIND = "case"

# Matrix structure
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# For optional producer-aware summaries (used only if present)
PRODUCER_COL: Optional[str] = None

# Recurrence logic
MIN_WITHIN_GRADIENT_SUPPORT = 2
MIN_GRADIENTS_FOR_META = 2

# Optional weighting by gradient score
WEIGHT_BY_SCORE = True
SCORE_COLUMN = "total_score"

# Output
OUTPUT_DIR = Path(".")
OUT_SUMMARY_CSV = "gradient_recurrence_summary.csv"
OUT_MEMBERSHIP_LONG_CSV = "gradient_recurrence_membership_long.csv"
OUT_CORECURRENCE_EDGES_CSV = "gradient_recurrence_corecurrence_edges.csv"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# HELPERS: reading / binarizing
# =============================================================================

def _read_table(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read .xlsx/.xls or .csv as strings, preserving blanks."""
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


# =============================================================================
# HELPERS: gradients
# =============================================================================

def _parse_chain_string(chain_str: str) -> List[str]:
    """
    Stage 3 case gradients use pipe-delimited chains:
        A | B | C | E
    We assume the feature-gradient script follows the same convention.
    """
    if pd.isna(chain_str):
        return []
    parts = [p.strip() for p in str(chain_str).split("|")]
    return [p for p in parts if p]


def _infer_endpoint_columns(df: pd.DataFrame, gradient_kind: str) -> Tuple[str, str]:
    """
    Infer endpoint columns from the gradient CSV.
    """
    if gradient_kind == "case":
        if {"case_A", "case_E"}.issubset(df.columns):
            return "case_A", "case_E"
    elif gradient_kind == "feature":
        if {"feature_A", "feature_E"}.issubset(df.columns):
            return "feature_A", "feature_E"

    # Fallbacks
    candidates = [
        ("case_A", "case_E"),
        ("feature_A", "feature_E"),
        ("A", "E"),
    ]
    for a_col, e_col in candidates:
        if {a_col, e_col}.issubset(df.columns):
            return a_col, e_col

    raise ValueError(
        "Could not infer endpoint columns from gradients CSV. "
        "Expected columns like case_A/case_E or feature_A/feature_E."
    )


def _make_gradient_id(row: pd.Series, a_col: str, e_col: str) -> str:
    a = str(row[a_col])
    e = str(row[e_col])
    chain = str(row.get("chain", ""))
    return f"{a} -> {e} || {chain}"


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def _prepare_incidence(
    incidence_path: Path,
    *,
    sheet_name: int | str,
    case_id_column: str,
    n_metadata_cols: int,
    presence_token: str,
    producer_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str], Optional[pd.Series]]:
    """
    Returns:
        bin_df indexed by case_id (rows = cases, cols = features)
        feature_cols
        producer_map (Series indexed by case_id) or None
    """
    inc_raw = _read_table(incidence_path, sheet_name=sheet_name)

    if case_id_column not in inc_raw.columns:
        raise ValueError(f"CASE_ID_COLUMN '{case_id_column}' not found in incidence matrix.")

    case_index = inc_raw.columns.get_loc(case_id_column)
    if case_index >= n_metadata_cols:
        raise ValueError(
            f"CASE_ID_COLUMN '{case_id_column}' is outside the first {n_metadata_cols} columns.\n"
            "This script assumes that all metadata columns appear to the LEFT of the feature columns."
        )

    feature_cols = _get_feature_columns(inc_raw, n_metadata_cols)
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    df = pd.concat([inc_raw[[case_id_column]], inc_raw[feature_cols]], axis=1).copy()
    bin_features = _binarize_presence(df, feature_cols, presence_token)
    bin_df = pd.concat([df[[case_id_column]].copy(), bin_features], axis=1)
    bin_df[case_id_column] = bin_df[case_id_column].astype(str)
    bin_df = bin_df.set_index(case_id_column)

    producer_map = None
    if producer_col is not None and producer_col in inc_raw.columns:
        producer_map = (
            inc_raw[[case_id_column, producer_col]]
            .copy()
            .astype(str)
            .set_index(case_id_column)[producer_col]
        )

    return bin_df, feature_cols, producer_map


def _analyze_case_gradient_row(
    chain_cases: List[str],
    *,
    bin_df: pd.DataFrame,
    min_within_gradient_support: int,
) -> Tuple[Counter, Dict[str, int], List[str]]:
    """
    Case-gradient mode:
    - chain members are CASES
    - mediators are FEATURES
    """
    missing_cases = [c for c in chain_cases if c not in bin_df.index]
    if missing_cases:
        raise ValueError(
            f"Cases from gradient not found in incidence matrix: {missing_cases[:10]}"
        )

    sub = bin_df.loc[chain_cases].copy()

    feature_counts = Counter(sub.sum(axis=0).astype(int).to_dict())
    feature_counts = Counter({feat: cnt for feat, cnt in feature_counts.items() if cnt > 0})

    retained = sorted(
        feat for feat, cnt in feature_counts.items()
        if cnt >= min_within_gradient_support
    )

    retained_counts = {feat: int(feature_counts[feat]) for feat in retained}
    return feature_counts, retained_counts, retained


def _analyze_feature_gradient_row(
    chain_features: List[str],
    *,
    bin_df: pd.DataFrame,
    min_within_gradient_support: int,
    producer_map: Optional[pd.Series],
) -> Tuple[Counter, Dict[str, int], List[str], Dict[str, int]]:
    """
    Feature-gradient mode:
    - chain members are FEATURES
    - mediators are CASES
    A case is retained if it contains at least MIN_WITHIN_GRADIENT_SUPPORT
    of the chain's features.
    """
    missing_features = [f for f in chain_features if f not in bin_df.columns]
    if missing_features:
        raise ValueError(
            f"Features from gradient not found in incidence matrix: {missing_features[:10]}"
        )

    sub = bin_df[chain_features].copy()
    case_feature_counts = sub.sum(axis=1).astype(int)

    case_counts = Counter(case_feature_counts.to_dict())
    case_counts = Counter({case: cnt for case, cnt in case_counts.items() if cnt > 0})

    retained_cases = sorted(
        case for case, cnt in case_counts.items()
        if cnt >= min_within_gradient_support
    )

    retained_counts = {case: int(case_counts[case]) for case in retained_cases}

    producer_counts: Dict[str, int] = {}
    if producer_map is not None and retained_cases:
        producers = producer_map.reindex(retained_cases).dropna().astype(str)
        producer_counts = producers.value_counts().to_dict()

    return case_counts, retained_counts, retained_cases, producer_counts


def run(
    *,
    gradients_csv: Path,
    incidence_path: Path,
    output_dir: Path,
    gradient_kind: str = "case",                 # "case" or "feature"
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    producer_col: Optional[str] = None,
    min_within_gradient_support: int = 2,
    min_gradients_for_meta: int = 2,
    weight_by_score: bool = True,
    score_column: str = "total_score",
    out_summary_csv: str = "gradient_recurrence_summary.csv",
    out_membership_long_csv: str = "gradient_recurrence_membership_long.csv",
    out_corecurrence_edges_csv: str = "gradient_recurrence_corecurrence_edges.csv",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Analyze recurrence across retained gradients and return structured outputs.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if gradient_kind not in {"case", "feature"}:
        raise ValueError("gradient_kind must be 'case' or 'feature'.")
    if min_within_gradient_support < 1:
        raise ValueError("min_within_gradient_support must be >= 1.")
    if min_gradients_for_meta < 1:
        raise ValueError("min_gradients_for_meta must be >= 1.")

    if not gradients_csv.exists():
        raise FileNotFoundError(f"Gradients CSV not found: {gradients_csv}")

    grad_df = pd.read_csv(gradients_csv)
    grad_df.columns = grad_df.columns.map(str)

    if "chain" not in grad_df.columns:
        raise ValueError("Gradients CSV must contain a 'chain' column.")

    a_col, e_col = _infer_endpoint_columns(grad_df, gradient_kind)

    bin_df, feature_cols, producer_map = _prepare_incidence(
        incidence_path,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        producer_col=producer_col,
    )

    membership_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    gradient_to_retained_mediators: Dict[str, List[str]] = {}

    for row_idx, row in grad_df.iterrows():
        gradient_id = _make_gradient_id(row, a_col, e_col)
        chain = _parse_chain_string(str(row["chain"]))
        if len(chain) < 2:
            continue

        gradient_score = float(row.get(score_column, np.nan)) if score_column in row.index else np.nan
        score_weight = gradient_score if weight_by_score and pd.notna(gradient_score) else 1.0

        if gradient_kind == "case":
            full_counts, retained_counts, retained = _analyze_case_gradient_row(
                chain,
                bin_df=bin_df,
                min_within_gradient_support=min_within_gradient_support,
            )
            mediator_type = "feature"

            for mediator, cnt in full_counts.items():
                membership_rows.append({
                    "gradient_id": gradient_id,
                    "gradient_row": int(row_idx),
                    "gradient_kind": gradient_kind,
                    "endpoint_A": str(row[a_col]),
                    "endpoint_E": str(row[e_col]),
                    "chain_length": len(chain),
                    "chain": " | ".join(chain),
                    "mediating_entity": mediator,
                    "mediating_entity_type": mediator_type,
                    "within_gradient_support": int(cnt),
                    "retained_by_threshold": mediator in retained_counts,
                    "gradient_score": gradient_score,
                    "score_weight": score_weight,
                })

        else:
            full_counts, retained_counts, retained, producer_counts = _analyze_feature_gradient_row(
                chain,
                bin_df=bin_df,
                min_within_gradient_support=min_within_gradient_support,
                producer_map=producer_map,
            )
            mediator_type = "case"

            for mediator, cnt in full_counts.items():
                membership_rows.append({
                    "gradient_id": gradient_id,
                    "gradient_row": int(row_idx),
                    "gradient_kind": gradient_kind,
                    "endpoint_A": str(row[a_col]),
                    "endpoint_E": str(row[e_col]),
                    "chain_length": len(chain),
                    "chain": " | ".join(chain),
                    "mediating_entity": mediator,
                    "mediating_entity_type": mediator_type,
                    "within_gradient_support": int(cnt),
                    "retained_by_threshold": mediator in retained_counts,
                    "gradient_score": gradient_score,
                    "score_weight": score_weight,
                })

        gradient_to_retained_mediators[gradient_id] = list(retained)

        summary_rows.append({
            "gradient_id": gradient_id,
            "gradient_row": int(row_idx),
            "gradient_kind": gradient_kind,
            "endpoint_A": str(row[a_col]),
            "endpoint_E": str(row[e_col]),
            "chain_length": len(chain),
            "chain": " | ".join(chain),
            "n_mediators_present": len(full_counts),
            "n_mediators_retained": len(retained),
            "gradient_score": gradient_score,
            "score_weight": score_weight,
        })

    if not membership_rows:
        raise ValueError("No valid gradients could be analyzed from the supplied CSV.")

    membership_df = pd.DataFrame(membership_rows)
    gradient_summary_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # Aggregate mediator recurrence
    # -------------------------------------------------------------------------
    retained_membership_df = membership_df[membership_df["retained_by_threshold"]].copy()

    # gradient-level recurrence counts
    gradients_per_mediator = (
        retained_membership_df.groupby("mediating_entity")["gradient_id"]
        .nunique()
        .rename("n_gradients_retained")
    )

    total_gradients = int(gradient_summary_df["gradient_id"].nunique())

    pct_gradients = (gradients_per_mediator / total_gradients).rename("pct_gradients_retained")

    mean_support = (
        retained_membership_df.groupby("mediating_entity")["within_gradient_support"]
        .mean()
        .rename("mean_within_gradient_support")
    )

    median_support = (
        retained_membership_df.groupby("mediating_entity")["within_gradient_support"]
        .median()
        .rename("median_within_gradient_support")
    )

    weighted_score_sum = (
        retained_membership_df.groupby("mediating_entity")["score_weight"]
        .sum()
        .rename("weighted_gradient_score_sum")
    )

    endpoint_pair_count = (
        retained_membership_df.assign(
            endpoint_pair=lambda d: d["endpoint_A"].astype(str) + " || " + d["endpoint_E"].astype(str)
        )
        .groupby("mediating_entity")["endpoint_pair"]
        .nunique()
        .rename("n_endpoint_pairs_spanned")
    )

    mediator_summary_df = pd.concat(
        [
            gradients_per_mediator,
            pct_gradients,
            mean_support,
            median_support,
            weighted_score_sum,
            endpoint_pair_count,
        ],
        axis=1,
    ).reset_index().rename(columns={"mediating_entity": "entity"})

    mediator_summary_df["entity_type"] = (
        "feature" if gradient_kind == "case" else "case"
    )

    mediator_summary_df = mediator_summary_df[
        mediator_summary_df["n_gradients_retained"] >= min_gradients_for_meta
    ].copy()

    mediator_summary_df = mediator_summary_df.sort_values(
        by=["n_gradients_retained", "weighted_gradient_score_sum", "mean_within_gradient_support", "entity"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Co-recurrence edges
    # -------------------------------------------------------------------------
    co_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for _, g_row in gradient_summary_df.iterrows():
        gid = g_row["gradient_id"]
        mediators = sorted(set(gradient_to_retained_mediators.get(gid, [])))
        for a, b in itertools.combinations(mediators, 2):
            co_counts[(a, b)] += 1

    edge_rows = [
        {"entity_a": a, "entity_b": b, "shared_gradients": cnt}
        for (a, b), cnt in co_counts.items()
        if cnt >= min_gradients_for_meta
    ]
    edges_df = pd.DataFrame(edge_rows)
    if not edges_df.empty:
        edges_df = edges_df.sort_values(
            by=["shared_gradients", "entity_a", "entity_b"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    summary_csv_path = output_dir / out_summary_csv
    membership_csv_path = output_dir / out_membership_long_csv
    edges_csv_path = output_dir / out_corecurrence_edges_csv
    summary_txt_path = output_dir / out_summary_name

    mediator_summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    membership_df.to_csv(membership_csv_path, index=False, encoding="utf-8")
    if edges_df.empty:
        pd.DataFrame(columns=["entity_a", "entity_b", "shared_gradients"]).to_csv(
            edges_csv_path, index=False, encoding="utf-8"
        )
    else:
        edges_df.to_csv(edges_csv_path, index=False, encoding="utf-8")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Gradient Recurrence Analyzer Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Gradients CSV: {gradients_csv}\n")
        f.write(f"Incidence matrix: {incidence_path}\n")
        f.write(f"Gradient kind: {gradient_kind}\n")
        f.write(f"Sheet name: {sheet_name}\n")
        f.write(f"Case ID column: {case_id_column}\n")
        f.write(f"N_METADATA_COLS: {n_metadata_cols}\n")
        f.write(f"Presence token: {presence_token}\n")
        f.write(f"Producer column: {producer_col if producer_col is not None else '(none)'}\n\n")

        f.write("Settings\n")
        f.write("--------\n")
        f.write(f"MIN_WITHIN_GRADIENT_SUPPORT: {min_within_gradient_support}\n")
        f.write(f"MIN_GRADIENTS_FOR_META: {min_gradients_for_meta}\n")
        f.write(f"WEIGHT_BY_SCORE: {weight_by_score}\n")
        f.write(f"SCORE_COLUMN: {score_column}\n\n")

        f.write("Run summary\n")
        f.write("-----------\n")
        f.write(f"Total gradients analyzed: {total_gradients}\n")
        f.write(f"Total gradient-membership rows: {len(membership_df)}\n")
        f.write(f"Retained recurring entities written: {len(mediator_summary_df)}\n")
        f.write(f"Co-recurrence edges written: {0 if edges_df.empty else len(edges_df)}\n\n")

        f.write("Interpretation\n")
        f.write("--------------\n")
        if gradient_kind == "case":
            f.write(
                "In case-gradient mode, recurring entities are features/tropes that recur across\n"
                "multiple retained case gradients. These are candidate meta-boundary tropes.\n\n"
            )
        else:
            f.write(
                "In feature-gradient mode, recurring entities are cases that recur across\n"
                "multiple retained feature gradients. These are candidate meta-boundary cases.\n\n"
            )

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_summary_csv}\n")
        f.write(f"{out_membership_long_csv}\n")
        f.write(f"{out_corecurrence_edges_csv}\n")
        f.write(f"{out_summary_name}\n")

    return {
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv_path),
        "membership_long_csv": str(membership_csv_path),
        "corecurrence_edges_csv": str(edges_csv_path),
        "summary_txt": str(summary_txt_path),
        "total_gradients": total_gradients,
        "recurring_entities_written": int(len(mediator_summary_df)),
        "corecurrence_edges_written": 0 if edges_df.empty else int(len(edges_df)),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze recurrence across retained gradients."
    )

    parser.add_argument("--gradients-csv", type=Path, default=GRADIENTS_CSV)
    parser.add_argument("--incidence-path", type=Path, default=INCIDENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    parser.add_argument("--gradient-kind", type=str, default=GRADIENT_KIND, help="'case' or 'feature'")
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)
    parser.add_argument("--producer-col", type=str, default=PRODUCER_COL)

    parser.add_argument("--min-within-gradient-support", type=int, default=MIN_WITHIN_GRADIENT_SUPPORT)
    parser.add_argument("--min-gradients-for-meta", type=int, default=MIN_GRADIENTS_FOR_META)
    parser.add_argument("--no-weight-by-score", action="store_true")
    parser.add_argument("--score-column", type=str, default=SCORE_COLUMN)

    parser.add_argument("--out-summary-csv", type=str, default=OUT_SUMMARY_CSV)
    parser.add_argument("--out-membership-long-csv", type=str, default=OUT_MEMBERSHIP_LONG_CSV)
    parser.add_argument("--out-corecurrence-edges-csv", type=str, default=OUT_CORECURRENCE_EDGES_CSV)
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    producer_col = args.producer_col if args.producer_col not in {"", "None", None} else None

    result = run(
        gradients_csv=args.gradients_csv,
        incidence_path=args.incidence_path,
        output_dir=args.output_dir,
        gradient_kind=args.gradient_kind,
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        producer_col=producer_col,
        min_within_gradient_support=args.min_within_gradient_support,
        min_gradients_for_meta=args.min_gradients_for_meta,
        weight_by_score=not args.no_weight_by_score,
        score_column=args.score_column,
        out_summary_csv=args.out_summary_csv,
        out_membership_long_csv=args.out_membership_long_csv,
        out_corecurrence_edges_csv=args.out_corecurrence_edges_csv,
        out_summary_name=args.out_summary,
    )

    print("[✓] Gradient recurrence analysis complete.")
    print(f"    Total gradients analyzed:      {result['total_gradients']}")
    print(f"    Recurring entities written:    {result['recurring_entities_written']}")
    print(f"    Co-recurrence edges written:   {result['corecurrence_edges_written']}")
    print(f"    Summary CSV:                   {result['summary_csv']}")
    print(f"    Membership CSV:                {result['membership_long_csv']}")
    print(f"    Co-recurrence edges CSV:       {result['corecurrence_edges_csv']}")
    print(f"    Summary TXT:                   {result['summary_txt']}")


if __name__ == "__main__":
    main()