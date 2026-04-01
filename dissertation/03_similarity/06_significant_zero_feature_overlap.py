#!/usr/bin/env python3
"""
06_significant_zero_feature_overlap.py

Identify feature pairs that never co-occur in the same case and evaluate
whether those absences are statistically unusual under a degree-preserving
null model.

This script is the feature-level analogue of the case-level zero-overlap
analysis. It works on a binary incidence matrix where:

- rows = cases (books, songs, documents, etc.)
- columns = features (tropes, topics, entities, etc.)

For each observed zero-overlap feature pair, the script computes:

- feature_a
- count_a
- feature_b
- count_b
- cooc_count              (always 0 for retained rows)
- p_emp                   empirical probability of zero overlap under the null
- sig_0.05                Benjamini–Hochberg FDR flag at alpha = 0.05
- sig_0.01                Benjamini–Hochberg FDR flag at alpha = 0.01

Outputs
-------
1) zero_feature_overlap_with_significance.csv
   Full table of all observed zero-overlap feature pairs and their significance

2) analysis_summary.txt
   Human-readable run summary

Method
------
- Filters features by minimum frequency
- Identifies all observed zero-overlap feature pairs
- Randomizes the incidence matrix using Curveball trades while preserving:
    * number of features per case
    * number of cases per feature
- Estimates empirical p(overlap = 0) for each observed zero-overlap pair
- Applies Benjamini–Hochberg FDR correction

Standalone use:
    Edit the CONFIG block below, then run:
        python 06_significant_zero_feature_overlap.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Input dataset
INPUT_PATH = Path("input_incidence_matrix.xlsx")   # .xlsx/.xls or .csv
SHEET_NAME = 0

# Matrix structure
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Feature filter
MIN_FEATURE_FREQ = 5

# Null model parameters
N_SAMPLES = 250
TRADES_BURN = 20000
TRADES_PER_SAMPLE = 5000
RNG_SEED = 42

# FDR thresholds
FDR_THRESHOLDS = [0.05, 0.01]

# Output
OUTPUT_DIR = Path(".")
OUT_CSV = "zero_feature_overlap_with_significance.csv"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# HELPERS: I/O and matrix prep
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
    """Assume feature columns begin after the first n_metadata_cols columns."""
    if n_metadata_cols < 0 or n_metadata_cols >= df.shape[1]:
        raise ValueError(
            f"N_METADATA_COLS={n_metadata_cols} invalid for table with {df.shape[1]} columns."
        )
    return list(df.columns[n_metadata_cols:])


def _binarize_presence(df: pd.DataFrame, feature_cols: Sequence[str], token: str) -> pd.DataFrame:
    """Convert feature columns to 0/1."""
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
# HELPERS: BH-FDR
# =============================================================================

def _benjamini_hochberg(pvals: Sequence[float], alpha: float) -> List[bool]:
    """
    Return list of significance flags under Benjamini–Hochberg FDR.
    """
    m = len(pvals)
    if m == 0:
        return []

    order = np.argsort(pvals)
    ranked = np.array(pvals, dtype=float)[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = np.where(ranked <= thresh)[0]

    cutoff = ranked[passed.max()] if passed.size else -1.0
    return [(p <= cutoff and cutoff >= 0) for p in pvals]


def _normalize_fdr_thresholds(values: Sequence[float]) -> List[float]:
    """Validate and normalize FDR thresholds."""
    out = sorted({float(v) for v in values})
    if not out:
        raise ValueError("FDR_THRESHOLDS must contain at least one value.")
    for v in out:
        if not (0 < v < 1):
            raise ValueError(f"Invalid FDR threshold: {v}. Must satisfy 0 < alpha < 1.")
    return out


# =============================================================================
# HELPERS: Curveball randomization
# =============================================================================

def _curveball_trade(a: List[int], b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    """
    Perform one Curveball trade between two row adjacency lists.
    """
    sa, sb = set(a), set(b)
    shared = sa & sb
    ua = list(sa - shared)
    ub = list(sb - shared)

    if not ua and not ub:
        return a, b

    pool = ua + ub
    rng.shuffle(pool)

    new_a = list(shared) + pool[:len(ua)]
    new_b = list(shared) + pool[len(ua):]
    return new_a, new_b


def _run_curveball(adj_lists: List[List[int]], trades: int, rng: random.Random) -> None:
    """
    In-place Curveball trades on row adjacency lists.
    """
    n = len(adj_lists)
    for _ in range(trades):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        ai, aj = adj_lists[i], adj_lists[j]
        new_ai, new_aj = _curveball_trade(ai, aj, rng)
        adj_lists[i], adj_lists[j] = new_ai, new_aj


# =============================================================================
# HELPERS: empirical p(overlap=0) for observed feature zero-pairs
# =============================================================================

def _empirical_p_zero_for_feature_pairs(
    adj_lists_init: List[List[int]],
    zero_feature_pairs_idx: Dict[Tuple[int, int], int],
    n_features: int,
    n_samples: int,
    burn_trades: int,
    trades_per_sample: int,
    rng_seed: int,
) -> np.ndarray:
    """
    Returns p_emp for each observed zero-overlap feature pair:
        p_emp = Pr(overlap(feature_i, feature_j) == 0) under a degree-preserving null
    """
    rng = random.Random(rng_seed)

    # Copy case adjacency lists (case -> list of feature IDs)
    adj_lists = [list(sorted(x)) for x in adj_lists_init]

    # Burn-in
    _run_curveball(adj_lists, burn_trades, rng)

    n_pairs = len(zero_feature_pairs_idx)
    zero_counts = np.zeros(n_pairs, dtype=np.int32)

    for _ in range(n_samples):
        _run_curveball(adj_lists, trades_per_sample, rng)

        marks = np.zeros(n_pairs, dtype=bool)

        # For each case, all feature pairs within that case co-occur in this sample
        for feats in adj_lists:
            if len(feats) < 2:
                continue
            feats_sorted = sorted(feats)
            for i, j in combinations(feats_sorted, 2):
                idx = zero_feature_pairs_idx.get((i, j))
                if idx is not None:
                    marks[idx] = True

        # Unmarked observed zero-pairs remained zero in this sample
        zero_counts[~marks] += 1

    return zero_counts / float(n_samples)


# =============================================================================
# HELPERS: summary
# =============================================================================

def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_path: Path,
    sheet_name: int | str,
    case_id_column: str,
    n_metadata_cols: int,
    presence_token: str,
    min_feature_freq: int,
    n_samples: int,
    trades_burn: int,
    trades_per_sample: int,
    rng_seed: int,
    fdr_thresholds: List[float],
    total_cases: int,
    features_before_filtering: int,
    features_after_filtering: int,
    total_possible_pairs: int,
    observed_zero_pairs: int,
    sig_counts: Dict[float, int],
    out_csv_name: str,
) -> None:
    """Write analysis summary."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Significant Zero Feature Overlap Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Input\n")
        f.write("-----\n")
        f.write(f"Dataset: {input_path}\n")
        f.write(f"Sheet name: {sheet_name}\n")
        f.write(f"Case ID column: {case_id_column}\n")
        f.write(f"Metadata columns assumed on the left: {n_metadata_cols}\n")
        f.write(f"Presence token: {presence_token}\n\n")

        f.write("Filtering\n")
        f.write("---------\n")
        f.write(f"MIN_FEATURE_FREQ: {min_feature_freq}\n\n")

        f.write("Corpus statistics\n")
        f.write("-----------------\n")
        f.write(f"Total cases: {total_cases}\n")
        f.write(f"Features before filtering: {features_before_filtering}\n")
        f.write(f"Features after filtering: {features_after_filtering}\n")
        f.write(f"Total possible feature pairs after filtering: {total_possible_pairs:,}\n")
        f.write(f"Observed zero-overlap feature pairs: {observed_zero_pairs:,}\n\n")

        f.write("Null model\n")
        f.write("----------\n")
        f.write(f"N_SAMPLES: {n_samples}\n")
        f.write(f"TRADES_BURN: {trades_burn}\n")
        f.write(f"TRADES_PER_SAMPLE: {trades_per_sample}\n")
        f.write(f"RNG_SEED: {rng_seed}\n\n")

        f.write("Significance results\n")
        f.write("--------------------\n")
        for alpha in fdr_thresholds:
            f.write(f"Pairs significant at FDR {alpha}: {sig_counts.get(alpha, 0)}\n")
        f.write("\n")

        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("Each row in the output CSV is a feature pair that never co-occurs in the same case.\n")
        f.write("The p_emp column gives the empirical probability of observing zero overlap under\n")
        f.write("a degree-preserving null model. Lower p_emp values indicate more surprising absences.\n")
        f.write("The significance columns provide Benjamini–Hochberg FDR flags.\n\n")

        f.write("Suggested sorting workflow\n")
        f.write("--------------------------\n")
        f.write("Sort by p_emp (ascending)\n")
        f.write("    → identifies the most statistically surprising feature disjunctions\n\n")
        f.write("Sort by count_a and count_b (descending)\n")
        f.write("    → highlights disjoint feature pairs that are individually common in the corpus\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_csv_name}\n")
        f.write(f"{out_path.name}\n")


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run(
    *,
    input_path: Path,
    output_dir: Path,
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    min_feature_freq: int = 5,
    n_samples: int = 250,
    trades_burn: int = 20000,
    trades_per_sample: int = 5000,
    rng_seed: int = 42,
    fdr_thresholds: Sequence[float] = (0.05, 0.01),
    out_csv_name: str = "zero_feature_overlap_with_significance.csv",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Run significant zero feature overlap analysis and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_feature_freq < 1:
        raise ValueError("min_feature_freq must be >= 1.")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if trades_burn < 0:
        raise ValueError("trades_burn must be >= 0.")
    if trades_per_sample < 0:
        raise ValueError("trades_per_sample must be >= 0.")

    fdr_thresholds = _normalize_fdr_thresholds(fdr_thresholds)

    # --- Load incidence matrix
    df_raw = _read_table(input_path, sheet_name=sheet_name)

    if case_id_column not in df_raw.columns:
        raise ValueError(f"CASE_ID_COLUMN '{case_id_column}' not found in input columns.")

    case_index = df_raw.columns.get_loc(case_id_column)
    if case_index >= n_metadata_cols:
        raise ValueError(
            f"CASE_ID_COLUMN '{case_id_column}' is outside the first {n_metadata_cols} columns.\n"
            "This script assumes that all metadata columns, including the case identifier, "
            "appear to the LEFT of the feature columns.\n"
            "Increase N_METADATA_COLS or rearrange the dataset so that metadata comes first."
        )

    feature_cols = _get_feature_columns(df_raw, n_metadata_cols)
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    df = pd.concat([df_raw[[case_id_column]], df_raw[feature_cols]], axis=1).copy()
    bin_features = _binarize_presence(df, feature_cols, presence_token)
    df_bin = pd.concat([df[[case_id_column]].copy(), bin_features], axis=1)

    total_cases = len(df_bin)
    features_before_filtering = len(feature_cols)

    # --- Feature frequency filter
    feature_counts_all = df_bin.drop(columns=[case_id_column]).sum(axis=0).astype(int)
    keep_features = feature_counts_all[feature_counts_all >= min_feature_freq].index.tolist()

    if len(keep_features) < 2:
        raise ValueError(
            f"Only {len(keep_features)} features remain after MIN_FEATURE_FREQ={min_feature_freq}. "
            "Need at least 2."
        )

    sub_df = df_bin[[case_id_column] + keep_features].copy()
    X = sub_df[keep_features].to_numpy(dtype=np.uint8)

    feature_counts = sub_df[keep_features].sum(axis=0).astype(int)
    feature_names = list(feature_counts.index)
    counts_arr = feature_counts.values
    n_features = len(feature_names)

    # --- Build co-occurrence matrix and identify observed zero-overlap pairs
    cooc = X.T @ X   # feature × feature raw overlap counts
    np.fill_diagonal(cooc, 0)

    zero_pairs: List[Tuple[int, int]] = []
    for i in range(n_features - 1):
        for j in range(i + 1, n_features):
            if cooc[i, j] == 0:
                zero_pairs.append((i, j))

    out_csv_path = output_dir / out_csv_name
    out_summary_path = output_dir / out_summary_name

    if not zero_pairs:
        empty_cols = [
            "feature_a", "count_a", "feature_b", "count_b",
            "cooc_count", "p_emp", *[f"sig_{alpha}" for alpha in fdr_thresholds]
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df.to_csv(out_csv_path, index=False, encoding="utf-8")

        sig_counts = {alpha: 0 for alpha in fdr_thresholds}
        total_possible = n_features * (n_features - 1) // 2

        _write_summary(
            out_summary_path,
            run_timestamp=run_timestamp,
            input_path=input_path,
            sheet_name=sheet_name,
            case_id_column=case_id_column,
            n_metadata_cols=n_metadata_cols,
            presence_token=presence_token,
            min_feature_freq=min_feature_freq,
            n_samples=n_samples,
            trades_burn=trades_burn,
            trades_per_sample=trades_per_sample,
            rng_seed=rng_seed,
            fdr_thresholds=fdr_thresholds,
            total_cases=total_cases,
            features_before_filtering=features_before_filtering,
            features_after_filtering=n_features,
            total_possible_pairs=total_possible,
            observed_zero_pairs=0,
            sig_counts=sig_counts,
            out_csv_name=out_csv_name,
        )

        return {
            "run_timestamp": run_timestamp,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "total_cases": total_cases,
            "features_before_filtering": features_before_filtering,
            "features_after_filtering": n_features,
            "observed_zero_pairs": 0,
            "fdr_thresholds": list(fdr_thresholds),
            "sig_counts": sig_counts,
            "zero_feature_overlap_csv": str(out_csv_path),
            "summary_txt": str(out_summary_path),
        }

    # --- Build case adjacency lists for Curveball randomization
    adj_lists: List[List[int]] = []
    for row in X:
        feats = list(np.where(row == 1)[0])
        adj_lists.append(feats)

    zero_pairs_idx = {pair: idx for idx, pair in enumerate(zero_pairs)}

    # --- Empirical p(overlap = 0) under degree-preserving null
    p_emp = _empirical_p_zero_for_feature_pairs(
        adj_lists_init=adj_lists,
        zero_feature_pairs_idx=zero_pairs_idx,
        n_features=n_features,
        n_samples=n_samples,
        burn_trades=trades_burn,
        trades_per_sample=trades_per_sample,
        rng_seed=rng_seed,
    )

    # --- Assemble results table
    records = []
    for (i, j), p in zip(zero_pairs, p_emp):
        records.append(
            {
                "feature_a": feature_names[i],
                "count_a": int(counts_arr[i]),
                "feature_b": feature_names[j],
                "count_b": int(counts_arr[j]),
                "cooc_count": 0,
                "p_emp": float(p),
            }
        )

    out_df = pd.DataFrame(records)

    # BH-FDR
    for alpha in fdr_thresholds:
        out_df[f"sig_{alpha}"] = _benjamini_hochberg(out_df["p_emp"].tolist(), alpha)

    # Sort most interesting pairs first
    out_df = out_df.sort_values(
        by=["p_emp", "count_a", "count_b", "feature_a", "feature_b"],
        ascending=[True, False, False, True, True]
    ).reset_index(drop=True)

    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

    sig_counts = {alpha: int(out_df[f"sig_{alpha}"].sum()) for alpha in fdr_thresholds}
    total_possible = n_features * (n_features - 1) // 2

    _write_summary(
        out_summary_path,
        run_timestamp=run_timestamp,
        input_path=input_path,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        min_feature_freq=min_feature_freq,
        n_samples=n_samples,
        trades_burn=trades_burn,
        trades_per_sample=trades_per_sample,
        rng_seed=rng_seed,
        fdr_thresholds=fdr_thresholds,
        total_cases=total_cases,
        features_before_filtering=features_before_filtering,
        features_after_filtering=n_features,
        total_possible_pairs=total_possible,
        observed_zero_pairs=len(out_df),
        sig_counts=sig_counts,
        out_csv_name=out_csv_name,
    )

    return {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_cases": total_cases,
        "features_before_filtering": features_before_filtering,
        "features_after_filtering": n_features,
        "observed_zero_pairs": len(out_df),
        "fdr_thresholds": list(fdr_thresholds),
        "sig_counts": sig_counts,
        "zero_feature_overlap_csv": str(out_csv_path),
        "summary_txt": str(out_summary_path),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    try:
        return [float(item) for item in items]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of floats."
        ) from e


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify significant zero-overlap feature pairs under a degree-preserving null model."
    )

    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)

    parser.add_argument("--min-feature-freq", type=int, default=MIN_FEATURE_FREQ)

    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--trades-burn", type=int, default=TRADES_BURN)
    parser.add_argument("--trades-per-sample", type=int, default=TRADES_PER_SAMPLE)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)

    parser.add_argument("--fdr-thresholds", type=_parse_float_list, default=FDR_THRESHOLDS)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out-csv", type=str, default=OUT_CSV)
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY)

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = _parse_args()

    result = run(
        input_path=args.input,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        min_feature_freq=args.min_feature_freq,
        n_samples=args.n_samples,
        trades_burn=args.trades_burn,
        trades_per_sample=args.trades_per_sample,
        rng_seed=args.rng_seed,
        fdr_thresholds=args.fdr_thresholds,
        out_csv_name=args.out_csv,
        out_summary_name=args.out_summary,
    )

    print("[✓] Significant zero feature overlap analysis complete.")
    print(f"    Input dataset:      {result['input_path']}")
    print(f"    Total cases:        {result['total_cases']}")
    print(f"    Features analyzed:  {result['features_after_filtering']}")
    print(f"    Zero-overlap pairs: {result['observed_zero_pairs']:,}")
    for alpha in result["fdr_thresholds"]:
        print(f"    FDR {alpha} sig pairs: {result['sig_counts'][alpha]}")
    print(f"    Main output:        {result['zero_feature_overlap_csv']}")
    print(f"    Summary:            {result['summary_txt']}")


if __name__ == "__main__":
    main()