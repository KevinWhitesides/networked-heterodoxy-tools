#!/usr/bin/env python3
"""
04_significant_zero_case_overlap.py

Identify observed zero-overlap case pairs in a binary incidence matrix and test
whether those absences are unusually strong under a degree-preserving null model.

Pipeline
--------
1) Read a binary incidence matrix from .xlsx/.xls or .csv
2) Optionally filter features by global frequency
3) Optionally filter cases by minimum number of retained features
4) Identify all observed zero-overlap case pairs
5) Generate a degree-preserving null via Curveball randomization
6) Estimate empirical p(overlap = 0) for each observed zero-overlap pair
7) Apply Benjamini–Hochberg FDR correction
8) Export:
   - zero_overlap_pairs_with_significance.csv
   - analysis_summary.txt

Notes
-----
- This is the similarity / significance stage of the case zero-overlap workflow.
- It does NOT build graphs. Downstream network scripts can consume the CSV output.
- If no observed zero-overlap pairs are present after filtering, the script writes
  an empty output table with the expected columns and records that outcome in the
  summary file.

Standalone use:
    Edit the CONFIG block below, then run:
        python 04_significant_zero_case_overlap.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
import itertools
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG (standalone defaults)
# =============================================================================

# Input incidence matrix (.xlsx/.xls or .csv)
INPUT_PATH = Path("input_incidence_matrix.xlsx")

# If reading Excel, which sheet to use
SHEET_NAME = 0

# Column containing case identifiers / titles
CASE_ID_COLUMN = "Source Title"

# Number of leftmost metadata columns before feature/trope columns begin
N_METADATA_COLS = 4

# Presence token
PRESENCE_TOKEN = "X"

# Filtering
GLOBAL_FEATURE_MIN_CASES = 2
MIN_FEATURES_PER_CASE = 1

# Null model (Curveball) parameters
N_SAMPLES = 300
TRADES_BURN = 20000
TRADES_PER_SAMPLE = 5000
RNG_SEED = 42

# FDR thresholds to report
FDR_THRESHOLDS = [0.05, 0.01]

# Output
OUTPUT_DIR = Path(".")
OUT_CSV = "zero_overlap_pairs_with_significance.csv"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# Helpers: reading / binarization
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
            f"N_METADATA_COLS={n_metadata_cols} is invalid for a table with {df.shape[1]} columns."
        )
    return list(df.columns[n_metadata_cols:])


def _binarize_presence(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    token: str,
) -> pd.DataFrame:
    """
    Convert feature columns to 0/1.

    Truthy values:
      - PRESENCE_TOKEN exactly
      - numeric values > 0
      - common truthy strings like yes / true / check / ✓ / x
    """
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


def _normalize_fdr_thresholds(values: Sequence[float]) -> List[float]:
    """Validate and normalize FDR thresholds."""
    out = sorted({float(v) for v in values}, reverse=False)
    if not out:
        raise ValueError("FDR_THRESHOLDS must contain at least one value.")
    for v in out:
        if not (0 < v < 1):
            raise ValueError(f"Invalid FDR threshold: {v}. Must satisfy 0 < alpha < 1.")
    return out


# =============================================================================
# Helpers: statistics
# =============================================================================

def _benjamini_hochberg(pvals: Sequence[float], alpha: float) -> List[bool]:
    """Return BH/FDR significance flags for the given alpha."""
    m = len(pvals)
    if m == 0:
        return []

    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    k = np.where(ranked <= thresh)[0]
    cutoff = ranked[k.max()] if k.size else -1.0
    return [(p <= cutoff and cutoff >= 0) for p in pvals]


# =============================================================================
# Curveball null model
# =============================================================================

def _curveball_trade(
    a: List[int],
    b: List[int],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """
    Perform one Curveball trade between two rows represented as lists of feature IDs.

    Preserves row sums and column sums in aggregate across the matrix.
    """
    sa, sb = set(a), set(b)
    shared = sa & sb
    ua = list(sa - shared)
    ub = list(sb - shared)

    if not ua and not ub:
        return a, b

    pool = ua + ub
    rng.shuffle(pool)

    new_a = list(shared) + pool[: len(ua)]
    new_b = list(shared) + pool[len(ua) :]
    return new_a, new_b


def _run_curveball(adj_lists: List[List[int]], trades: int, rng: random.Random) -> None:
    """In-place Curveball trades on adjacency lists."""
    n = len(adj_lists)
    for _ in range(trades):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        adj_lists[i], adj_lists[j] = _curveball_trade(adj_lists[i], adj_lists[j], rng)


def _empirical_p_zero_for_pairs(
    adj_lists_init: List[List[int]],
    zero_pairs_idx: Dict[Tuple[int, int], int],
    n_features: int,
    n_samples: int,
    burn_trades: int,
    trades_per_sample: int,
    rng_seed: int,
) -> np.ndarray:
    """
    Estimate empirical p(overlap = 0) for each observed zero-overlap pair
    under a degree-preserving Curveball null model.

    Only observed zero-overlap pairs are tested.
    """
    rng = random.Random(rng_seed)

    # Work on a copy
    adj_lists = [list(sorted(x)) for x in adj_lists_init]

    # Burn-in
    _run_curveball(adj_lists, burn_trades, rng)

    n_pairs = len(zero_pairs_idx)
    zero_counts = np.zeros(n_pairs, dtype=np.int32)

    for _ in range(n_samples):
        _run_curveball(adj_lists, trades_per_sample, rng)

        # Build feature -> cases incidence for this sample
        feature_cases: List[List[int]] = [[] for _ in range(n_features)]
        for case_idx, feats in enumerate(adj_lists):
            for f in feats:
                feature_cases[f].append(case_idx)

        # Mark observed-zero pairs that become nonzero in this sample
        marks = np.zeros(n_pairs, dtype=bool)
        for cases in feature_cases:
            if len(cases) < 2:
                continue
            for i, j in itertools.combinations(cases, 2):
                if i > j:
                    i, j = j, i
                idx = zero_pairs_idx.get((i, j))
                if idx is not None:
                    marks[idx] = True

        # Unmarked pairs stayed at overlap == 0
        zero_counts[~marks] += 1

    return zero_counts / float(n_samples)


# =============================================================================
# Helpers: summary
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
    global_feature_min_cases: int,
    min_features_per_case: int,
    n_samples: int,
    trades_burn: int,
    trades_per_sample: int,
    rng_seed: int,
    fdr_thresholds: List[float],
    cases_retained: int,
    features_retained: int,
    observed_zero_pairs: int,
    sig_counts: Dict[float, int],
    out_csv_name: str,
) -> None:
    """Write a plain-text summary of the run."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Significant Zero-Overlap Case Analysis Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Sheet name: {sheet_name}\n")
        f.write(f"CASE_ID_COLUMN: {case_id_column}\n")
        f.write(f"N_METADATA_COLS: {n_metadata_cols}\n")
        f.write(f"PRESENCE_TOKEN: {presence_token}\n\n")

        f.write("Filtering\n")
        f.write("---------\n")
        f.write(f"GLOBAL_FEATURE_MIN_CASES: {global_feature_min_cases}\n")
        f.write(f"MIN_FEATURES_PER_CASE: {min_features_per_case}\n\n")

        f.write("Null model settings\n")
        f.write("-------------------\n")
        f.write(f"N_SAMPLES: {n_samples}\n")
        f.write(f"TRADES_BURN: {trades_burn}\n")
        f.write(f"TRADES_PER_SAMPLE: {trades_per_sample}\n")
        f.write(f"RNG_SEED: {rng_seed}\n\n")

        f.write("Dataset summary\n")
        f.write("---------------\n")
        f.write(f"Cases retained after filtering: {cases_retained}\n")
        f.write(f"Features retained after filtering: {features_retained}\n")
        f.write(f"Observed zero-overlap pairs: {observed_zero_pairs}\n")
        for alpha in fdr_thresholds:
            f.write(f"Significant zero-overlap pairs @ FDR {alpha}: {sig_counts.get(alpha, 0)}\n")
        f.write("\n")

        f.write("Outputs\n")
        f.write("-------\n")
        f.write(f"CSV: {out_csv_name}\n")
        f.write(f"Summary: {out_path.name}\n")


# =============================================================================
# Pipeline-ready entry point
# =============================================================================

def run(
    *,
    input_path: Path,
    output_dir: Path,
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    global_feature_min_cases: int = 2,
    min_features_per_case: int = 1,
    n_samples: int = 300,
    trades_burn: int = 20000,
    trades_per_sample: int = 5000,
    rng_seed: int = 42,
    fdr_thresholds: Sequence[float] = (0.05, 0.01),
    out_csv_name: str = "zero_overlap_pairs_with_significance.csv",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Run significant zero-overlap case analysis and return a structured result dictionary.

    This is the entry point pipeline runners should call.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if global_feature_min_cases < 1:
        raise ValueError("GLOBAL_FEATURE_MIN_CASES must be >= 1.")
    if min_features_per_case < 1:
        raise ValueError("MIN_FEATURES_PER_CASE must be >= 1.")
    if n_samples < 1:
        raise ValueError("N_SAMPLES must be >= 1.")
    if trades_burn < 0:
        raise ValueError("TRADES_BURN must be >= 0.")
    if trades_per_sample < 0:
        raise ValueError("TRADES_PER_SAMPLE must be >= 0.")

    fdr_thresholds = _normalize_fdr_thresholds(fdr_thresholds)

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    df_raw = _read_table(input_path, sheet_name=sheet_name)

    if case_id_column not in df_raw.columns:
        raise ValueError(
            f"CASE_ID_COLUMN '{case_id_column}' not found in input columns."
        )

    feature_cols = _get_feature_columns(df_raw, n_metadata_cols)
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    # Keep only case id + feature columns
    df = pd.concat([df_raw[[case_id_column]], df_raw[feature_cols]], axis=1)

    # Binarize
    bin_features = _binarize_presence(df, feature_cols, presence_token)
    df_bin = pd.concat([df[[case_id_column]], bin_features], axis=1)

    # Global feature filter
    feature_freq = df_bin.drop(columns=[case_id_column]).sum(axis=0)
    kept_features = feature_freq[feature_freq >= global_feature_min_cases].index.tolist()
    if not kept_features:
        raise ValueError("No features remain after GLOBAL_FEATURE_MIN_CASES filter.")

    df_glob = pd.concat([df_bin[[case_id_column]], df_bin[kept_features]], axis=1)

    # Case floor
    case_feature_counts = df_glob.drop(columns=[case_id_column]).sum(axis=1).astype(int)
    kept_case_mask = case_feature_counts >= min_features_per_case
    df_cases = df_glob.loc[kept_case_mask].reset_index(drop=True)

    if df_cases.empty:
        raise ValueError("No cases remain after MIN_FEATURES_PER_CASE filter.")

    case_names = df_cases[case_id_column].astype(str).tolist()
    kept_feature_cols = [c for c in df_cases.columns if c != case_id_column]

    # Build adjacency lists (case -> feature IDs)
    feature_index = {feat: i for i, feat in enumerate(kept_feature_cols)}
    adj_lists: List[List[int]] = []
    for _, row in df_cases.iterrows():
        present_feats = [feature_index[c] for c in kept_feature_cols if int(row[c]) == 1]
        adj_lists.append(present_feats)

    n_cases = len(adj_lists)
    n_features = len(kept_feature_cols)

    case_counts_after_filter = {
        row[case_id_column]: int(row[kept_feature_cols].sum())
        for _, row in df_cases.iterrows()
    }

    # Observed zero-overlap pairs
    zero_pairs: List[Tuple[int, int]] = []
    for i, j in itertools.combinations(range(n_cases), 2):
        if len(set(adj_lists[i]).intersection(adj_lists[j])) == 0:
            zero_pairs.append((i, j))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = output_dir / out_csv_name
    out_summary_path = output_dir / out_summary_name

    # Handle no zero-overlap case pairs
    if not zero_pairs:
        empty_df = pd.DataFrame(
            columns=[
                "case_A",
                "case_B",
                "case_A_feature_count",
                "case_B_feature_count",
                "observed_overlap",
                "p_emp",
                *[f"sig_{alpha}" for alpha in fdr_thresholds],
            ]
        )
        empty_df.to_csv(out_csv_path, index=False, encoding="utf-8")

        sig_counts = {alpha: 0 for alpha in fdr_thresholds}
        _write_summary(
            out_path=out_summary_path,
            run_timestamp=run_timestamp,
            input_path=input_path,
            sheet_name=sheet_name,
            case_id_column=case_id_column,
            n_metadata_cols=n_metadata_cols,
            presence_token=presence_token,
            global_feature_min_cases=global_feature_min_cases,
            min_features_per_case=min_features_per_case,
            n_samples=n_samples,
            trades_burn=trades_burn,
            trades_per_sample=trades_per_sample,
            rng_seed=rng_seed,
            fdr_thresholds=fdr_thresholds,
            cases_retained=n_cases,
            features_retained=n_features,
            observed_zero_pairs=0,
            sig_counts=sig_counts,
            out_csv_name=out_csv_path.name,
        )

        return {
            "run_timestamp": run_timestamp,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "cases_retained": n_cases,
            "features_retained": n_features,
            "observed_zero_pairs": 0,
            "fdr_thresholds": list(fdr_thresholds),
            "sig_counts": sig_counts,
            "zero_overlap_csv": str(out_csv_path),
            "summary_txt": str(out_summary_path),
        }

    # Null model significance
    zero_pairs_idx = {(i, j): k for k, (i, j) in enumerate(zero_pairs)}
    p_emp = _empirical_p_zero_for_pairs(
        adj_lists_init=adj_lists,
        zero_pairs_idx=zero_pairs_idx,
        n_features=n_features,
        n_samples=n_samples,
        burn_trades=trades_burn,
        trades_per_sample=trades_per_sample,
        rng_seed=rng_seed,
    )

    # Build output table
    records = []
    for (i, j), p in zip(zero_pairs, p_emp):
        a = case_names[i]
        b = case_names[j]
        records.append(
            {
                "case_A": a,
                "case_B": b,
                "case_A_feature_count": case_counts_after_filter[a],
                "case_B_feature_count": case_counts_after_filter[b],
                "observed_overlap": 0,
                "p_emp": float(p),
            }
        )

    out_df = pd.DataFrame(records)

    # BH/FDR flags
    for alpha in fdr_thresholds:
        out_df[f"sig_{alpha}"] = _benjamini_hochberg(out_df["p_emp"].tolist(), alpha)

    out_df = out_df.sort_values(["p_emp", "case_A", "case_B"]).reset_index(drop=True)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

    sig_counts = {alpha: int(out_df[f"sig_{alpha}"].sum()) for alpha in fdr_thresholds}

    _write_summary(
        out_path=out_summary_path,
        run_timestamp=run_timestamp,
        input_path=input_path,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        global_feature_min_cases=global_feature_min_cases,
        min_features_per_case=min_features_per_case,
        n_samples=n_samples,
        trades_burn=trades_burn,
        trades_per_sample=trades_per_sample,
        rng_seed=rng_seed,
        fdr_thresholds=fdr_thresholds,
        cases_retained=n_cases,
        features_retained=n_features,
        observed_zero_pairs=len(zero_pairs),
        sig_counts=sig_counts,
        out_csv_name=out_csv_path.name,
    )

    return {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "cases_retained": n_cases,
        "features_retained": n_features,
        "observed_zero_pairs": len(zero_pairs),
        "fdr_thresholds": list(fdr_thresholds),
        "sig_counts": sig_counts,
        "zero_overlap_csv": str(out_csv_path),
        "summary_txt": str(out_summary_path),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats into a list."""
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify significant zero-overlap case pairs under a degree-preserving null model."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Path to input .xlsx/.xls or .csv incidence matrix",
    )
    parser.add_argument(
        "--sheet-name",
        default=SHEET_NAME,
        help="Excel sheet name or index (ignored for CSV)",
    )
    parser.add_argument(
        "--case-id-column",
        type=str,
        default=CASE_ID_COLUMN,
        help="Column containing case identifiers / titles",
    )
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
        "--global-feature-min-cases",
        type=int,
        default=GLOBAL_FEATURE_MIN_CASES,
        help="Keep only features appearing in at least this many cases",
    )
    parser.add_argument(
        "--min-features-per-case",
        type=int,
        default=MIN_FEATURES_PER_CASE,
        help="Keep only cases containing at least this many retained features",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help="Number of Curveball null samples",
    )
    parser.add_argument(
        "--trades-burn",
        type=int,
        default=TRADES_BURN,
        help="Number of Curveball burn-in trades",
    )
    parser.add_argument(
        "--trades-per-sample",
        type=int,
        default=TRADES_PER_SAMPLE,
        help="Number of Curveball trades per sample",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=RNG_SEED,
        help="Random seed for null-model reproducibility",
    )
    parser.add_argument(
        "--fdr-thresholds",
        type=_parse_float_list,
        default=FDR_THRESHOLDS,
        help="Comma-separated FDR thresholds to report (e.g. 0.05,0.01)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=OUT_CSV,
        help="Output filename for zero-overlap significance CSV",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=OUT_SUMMARY,
        help="Output filename for analysis summary",
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
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        global_feature_min_cases=args.global_feature_min_cases,
        min_features_per_case=args.min_features_per_case,
        n_samples=args.n_samples,
        trades_burn=args.trades_burn,
        trades_per_sample=args.trades_per_sample,
        rng_seed=args.rng_seed,
        fdr_thresholds=args.fdr_thresholds,
        out_csv_name=args.out_csv,
        out_summary_name=args.out_summary,
    )

    print("[✓] Significant zero-overlap case analysis complete.")
    print(f"    Input: {result['input_path']}")
    print(f"    Cases retained: {result['cases_retained']}")
    print(f"    Features retained: {result['features_retained']}")
    print(f"    Observed zero-overlap pairs: {result['observed_zero_pairs']}")
    for alpha in result["fdr_thresholds"]:
        print(f"    Significant pairs @ FDR {alpha}: {result['sig_counts'][alpha]}")
    print(f"    Output CSV: {result['zero_overlap_csv']}")
    print(f"    Analysis summary: {result['summary_txt']}")


if __name__ == "__main__":
    main()