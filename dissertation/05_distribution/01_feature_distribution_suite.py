#!/usr/bin/env python3
"""
01_feature_distribution_suite.py

Analyze whether a feature's prominence in a corpus is driven by:

1) broad diffusion across producers, or
2) concentrated use within a smaller producer subset.

This script is designed for binary incidence matrices where:
- rows = cases (songs, books, documents, etc.)
- one metadata column identifies the producer (artist, author, etc.)
- feature columns contain a presence marker (default: "X")

Important structural assumption:
- All metadata columns must appear to the LEFT of the feature columns.
- The producer column must be included within those metadata columns.

For each feature, the script computes:
- case_count                 : number of cases using the feature
- producer_count             : number of unique producers using the feature
- concentration_ratio        : case_count / producer_count
- producer_diffusion_pct     : producer_count / total_producers
- top_producer               : producer with the most cases using the feature
- top_producer_count         : number of cases by the top producer using the feature
- top_producer_share         : top_producer_count / case_count
- top_3_producer_share       : share of total cases accounted for by the top 3 producers
- top_3_minus_top_1_share    : top_3_producer_share - top_producer_share

Outputs:
1) feature_distribution_metrics.csv
   Full table of all features and metrics

2) analysis_summary.txt
   Human-readable run summary, including suggested sorting workflows

This is a descriptive / distributional analysis tool. It does not build networks.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Input dataset (.xlsx or .csv)
INPUT_PATH = Path("input_incidence_matrix.csv")
SHEET_NAME = 0  # used only for Excel

# Producer column: "Artist", "Author", etc.
PRODUCER_COL = "Artist"

# Number of metadata columns before feature columns begin
# Example hip hop sheet: Artist / Album / Song / Year = 4
N_METADATA_COLS = 4

# Presence token for feature usage
PRESENCE_TOKEN = "X"

# Optional: keep only features that appear in at least this many cases
MIN_FEATURE_CASES = 1

# Output
OUTPUT_DIR = Path(".")
OUT_MAIN_CSV = "feature_distribution_metrics.csv"
OUT_SUMMARY = "analysis_summary.txt"


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def read_table(path: Path, sheet_name=0) -> pd.DataFrame:
    """Read .xlsx/.xls or .csv as strings, preserving blanks."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name, dtype=str, keep_default_na=False)
    if suffix == ".csv":
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")

    raise ValueError(f"Unsupported input format: {suffix}. Use .xlsx/.xls or .csv.")


def get_feature_columns(df: pd.DataFrame, n_metadata_cols: int) -> List[str]:
    """Assume feature columns begin after the first n_metadata_cols columns."""
    if n_metadata_cols < 0 or n_metadata_cols >= df.shape[1]:
        raise ValueError(
            f"N_METADATA_COLS={n_metadata_cols} invalid for table with {df.shape[1]} columns."
        )
    return list(df.columns[n_metadata_cols:])


def binarize_presence(df: pd.DataFrame, feature_cols: Sequence[str], token: str) -> pd.DataFrame:
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

    return df[feature_cols].apply(to_bin)


def safe_div(numerator, denominator):
    if denominator in [0, None] or pd.isna(denominator):
        return np.nan
    return numerator / denominator


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = read_table(INPUT_PATH, sheet_name=SHEET_NAME)
    df_raw.columns = df_raw.columns.map(str)

    if PRODUCER_COL not in df_raw.columns:
        raise ValueError(f"PRODUCER_COL '{PRODUCER_COL}' not found in input columns.")

    producer_index = df_raw.columns.get_loc(PRODUCER_COL)
    if producer_index >= N_METADATA_COLS:
        raise ValueError(
            f"PRODUCER_COL '{PRODUCER_COL}' is outside the first {N_METADATA_COLS} columns.\n"
            "This script assumes that all metadata columns, including the producer column, "
            "appear to the LEFT of the feature columns.\n"
            "Increase N_METADATA_COLS or rearrange the dataset so that metadata comes first."
        )

    feature_cols = get_feature_columns(df_raw, N_METADATA_COLS)
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    # Keep only producer + feature columns for analysis
    df = pd.concat([df_raw[[PRODUCER_COL]], df_raw[feature_cols]], axis=1).copy()

    # Binarize features
    bin_features = binarize_presence(df, feature_cols, PRESENCE_TOKEN)
    df_bin = pd.concat([df[[PRODUCER_COL]].copy(), bin_features], axis=1)

    total_cases = len(df_bin)
    total_producers = df_bin[PRODUCER_COL].dropna().astype(str).nunique()

    results = []

    for feature in feature_cols:
        mask = df_bin[feature] == 1
        case_count = int(mask.sum())

        if case_count < MIN_FEATURE_CASES:
            continue

        producers_using = (
            df_bin.loc[mask, PRODUCER_COL]
            .dropna()
            .astype(str)
        )

        producer_count = producers_using.nunique()
        if producer_count == 0:
            continue

        concentration_ratio = safe_div(case_count, producer_count)
        producer_diffusion_pct = safe_div(producer_count, total_producers)

        producer_case_counts = producers_using.value_counts()
        top_producer = str(producer_case_counts.index[0])
        top_producer_count = int(producer_case_counts.iloc[0])
        top_producer_share = safe_div(top_producer_count, case_count)
        top_3_producer_share = safe_div(int(producer_case_counts.head(3).sum()), case_count)
        top_3_minus_top_1_share = (
            top_3_producer_share - top_producer_share
            if pd.notna(top_3_producer_share) and pd.notna(top_producer_share)
            else np.nan
        )

        results.append({
            "feature": feature,
            "case_count": case_count,
            "producer_count": int(producer_count),
            "concentration_ratio": concentration_ratio,
            "producer_diffusion_pct": producer_diffusion_pct,
            "top_producer": top_producer,
            "top_producer_count": top_producer_count,
            "top_producer_share": top_producer_share,
            "top_3_producer_share": top_3_producer_share,
            "top_3_minus_top_1_share": top_3_minus_top_1_share,
        })

    if not results:
        print("No feature usage found after filtering.")
        return

    res_df = pd.DataFrame(results)

    # Round display values for output neatness
    for col in [
        "concentration_ratio",
        "producer_diffusion_pct",
        "top_producer_share",
        "top_3_producer_share",
        "top_3_minus_top_1_share",
    ]:
        res_df[col] = res_df[col].round(6)

    # Main output sorted by prominence first, then diffusion, then concentration
    main_df = res_df.sort_values(
        by=["case_count", "producer_count", "concentration_ratio"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    main_csv_path = OUTPUT_DIR / OUT_MAIN_CSV
    main_df.to_csv(main_csv_path, index=False, encoding="utf-8")

    # Summary
    with open(OUTPUT_DIR / OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("=== Feature Distribution / Concentration Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Input\n")
        f.write("-----\n")
        f.write(f"Dataset: {INPUT_PATH}\n")
        f.write(f"Producer column: {PRODUCER_COL}\n")
        f.write(f"Metadata columns assumed on the left: {N_METADATA_COLS}\n")
        f.write(f"Presence token: {PRESENCE_TOKEN}\n")
        f.write(f"Minimum cases per feature: {MIN_FEATURE_CASES}\n\n")

        f.write("Corpus statistics\n")
        f.write("-----------------\n")
        f.write(f"Total cases: {total_cases}\n")
        f.write(f"Total producers: {total_producers}\n")
        f.write(f"Features analyzed: {len(main_df)}\n\n")

        f.write("Interpretive dimensions\n")
        f.write("-----------------------\n")
        f.write("case_count              = prominence within the corpus\n")
        f.write("producer_count          = diffusion across producers\n")
        f.write("concentration_ratio     = prominence relative to diffusion\n")
        f.write("top_producer_share      = proportion of usage captured by the single most associated producer\n")
        f.write("top_3_producer_share    = proportion of usage captured by the three most associated producers\n")
        f.write("top_3_minus_top_1_share = concentration attributable to a small producer cluster beyond the single top producer\n\n")

        f.write("Suggested sorting workflows\n")
        f.write("---------------------------\n")
        f.write("Sort by case_count (descending)\n")
        f.write("    → identifies the most prominent features in the corpus\n\n")

        f.write("Sort by producer_diffusion_pct (descending)\n")
        f.write("    → identifies the most widely diffused features across producers\n\n")

        f.write("Sort by concentration_ratio (descending)\n")
        f.write("    → identifies features whose prominence is concentrated within a smaller producer subset\n\n")

        f.write("Sort by top_producer_share (descending)\n")
        f.write("    → identifies features strongly associated with a single producer\n\n")

        f.write("Sort by top_3_minus_top_1_share (descending)\n")
        f.write("    → identifies features concentrated within a small producer cluster or clique rather than dominated by only one producer\n\n")

        f.write("Column structure assumption\n")
        f.write("---------------------------\n")
        f.write("This script assumes that all metadata columns appear to the LEFT of the feature columns.\n")
        f.write("The producer column must be included within those metadata columns.\n")
        f.write("All columns to the right of the metadata block are treated as feature columns.\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{OUT_MAIN_CSV}\n")
        f.write(f"{OUT_SUMMARY}\n")

    # Console summary
    print("[✓] Feature distribution / concentration analysis complete.")
    print(f"    Input dataset:      {INPUT_PATH}")
    print(f"    Producer column:    {PRODUCER_COL}")
    print(f"    Metadata columns:   {N_METADATA_COLS}")
    print(f"    Total cases:        {total_cases}")
    print(f"    Total producers:    {total_producers}")
    print(f"    Features analyzed:  {len(main_df)}")
    print(f"    Main output:        {main_csv_path}")
    print(f"    Summary:            {OUTPUT_DIR / OUT_SUMMARY}")

    print("\nTop 10 by concentration_ratio:")
    preview = main_df.sort_values(
        by=["concentration_ratio", "case_count"],
        ascending=[False, False]
    ).head(10)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()