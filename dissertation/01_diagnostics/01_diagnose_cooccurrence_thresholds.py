#!/usr/bin/env python3
"""
01_diagnose_cooccurrence_thresholds.py

Projection Threshold Diagnostic
(Binary Incidence -> One-Mode Projection -> Threshold Sweep)

This script helps diagnose edge-threshold choices for one-mode projection
networks built from a binary incidence matrix.

It supports:

- feature × feature projection
- case × case projection
- or both in a single run

For each selected projection mode, the script:

1) Reads a binary incidence matrix from .xlsx/.xls or .csv
2) Ignores leading metadata columns via N_METADATA_COLS
3) Applies a node-frequency filter appropriate to the projection mode
4) Computes a weighted one-mode projection
5) Sweeps across specified edge thresholds
6) Reports, for each threshold:
   - surviving nodes
   - qualifying edges
   - possible node pairs among surviving nodes
   - density among surviving nodes
7) Exports:
   - one CSV per projection mode
   - analysis_summary.txt

This diagnostic is intended to be used *before* running a projection pipeline,
to help select interpretable edge thresholds for the dataset.

Standalone use:
    Edit the CONFIG block below, then run:
        python 01_diagnose_cooccurrence_thresholds.py

Programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG (standalone defaults)
# =============================================================================

INPUT_PATH = Path("dissertation/sample_data/first_7_books.xlsx")

# Case-label column. If None, prefer "Source Title", then "Title", then row numbers.
TITLE_COL: Optional[str] = None

# Number of metadata columns before feature columns begin.
N_METADATA_COLS = 4

PRESENCE_TOKEN = "X"

# Projection modes:
#   ["feature"]
#   ["case"]
#   ["feature", "case"]
PROJECTION_MODES = ["feature", "case"]

# Minimum node frequency filters
MIN_FEATURE_NODE_FREQ = 2
MIN_CASE_NODE_FREQ = 2

# Threshold grids to diagnose
FEATURE_EDGE_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20]
CASE_EDGE_THRESHOLDS = [1, 2, 3, 5, 10, 15]

# Output directory
OUTPUT_DIR = Path(".")

# Output names
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
    """Build binary incidence matrix (cases × features)."""
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
    """Validate and normalize projection modes."""
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


def _build_feature_projection(
    incidence: pd.DataFrame,
    *,
    min_feature_node_freq: int,
) -> tuple[pd.DataFrame, int, int]:
    """
    Build feature × feature co-occurrence matrix.

    Returns:
        cooccurrence matrix, original node count, retained node count
    """
    raw_freq = incidence.sum(axis=0)
    original_node_count = int(len(raw_freq))

    if min_feature_node_freq > 1:
        keep_cols = raw_freq[raw_freq >= min_feature_node_freq].index
        filtered = incidence.loc[:, keep_cols]
    else:
        filtered = incidence

    cooc = filtered.T.dot(filtered)
    return cooc, original_node_count, int(cooc.shape[0])


def _build_case_projection(
    incidence: pd.DataFrame,
    *,
    min_case_node_freq: int,
) -> tuple[pd.DataFrame, int, int]:
    """
    Build case × case overlap matrix.

    Returns:
        overlap matrix, original node count, retained node count
    """
    raw_case_sizes = incidence.sum(axis=1)
    original_node_count = int(len(raw_case_sizes))

    if min_case_node_freq > 1:
        keep_rows = raw_case_sizes[raw_case_sizes >= min_case_node_freq].index
        filtered = incidence.loc[keep_rows, :]
    else:
        filtered = incidence

    cooc = filtered.dot(filtered.T)
    return cooc, original_node_count, int(cooc.shape[0])


def _run_threshold_diagnostic(
    cooc: pd.DataFrame,
    thresholds: list[int],
) -> pd.DataFrame:
    """Compute surviving nodes / edges / density across thresholds."""
    n = cooc.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = cooc.to_numpy(dtype=int)

    rows = []
    for thr in thresholds:
        edges_upper = (vals >= thr) & upper_mask
        edge_count = int(edges_upper.sum())

        # Symmetrize so we can count nodes participating in at least one qualifying edge
        edges_sym = edges_upper | edges_upper.T
        node_count = int(edges_sym.any(axis=1).sum())

        possible_pairs = node_count * (node_count - 1) // 2
        density = (edge_count / possible_pairs) if possible_pairs else 0.0

        rows.append(
            {
                "threshold": int(thr),
                "nodes_surviving": node_count,
                "edges_qualifying": edge_count,
                "possible_pairs": int(possible_pairs),
                "density_surviving_nodes": float(density),
            }
        )

    return pd.DataFrame(rows)


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
    """Write plain-text diagnostic summary."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Projection Threshold Diagnostic Summary ===\n\n")
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

        f.write("Outputs\n")
        f.write("-------\n")
        for mode in projection_modes:
            mode_result = results["projections"][mode]
            f.write(f"{mode} diagnostic CSV: {Path(mode_result['diagnostic_csv']).name}\n")


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
# Entry point
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
    feature_edge_thresholds: list[int] | tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20),
    case_edge_thresholds: list[int] | tuple[int, ...] = (1, 2, 3, 5, 10, 15),
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Run threshold diagnostics and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    projection_modes = _normalize_projection_modes(list(projection_modes))
    feature_edge_thresholds = [int(x) for x in feature_edge_thresholds]
    case_edge_thresholds = [int(x) for x in case_edge_thresholds]

    df = _read_table(input_path)
    chosen_title_col = _choose_title_col(df, title_col)
    case_names = _extract_case_names(df, chosen_title_col)
    incidence = _build_incidence(df, n_metadata_cols, presence_token, case_names)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"diagnostic_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_stem = input_path.stem
    results: Dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "output_dir": str(run_dir),
        "title_col": chosen_title_col,
        "n_cases_total": int(incidence.shape[0]),
        "n_features_total": int(incidence.shape[1]),
        "projections": {},
    }

    for mode in projection_modes:
        if mode == "feature":
            cooc, original_node_count, retained_node_count = _build_feature_projection(
                incidence,
                min_feature_node_freq=min_feature_node_freq,
            )
            thresholds = feature_edge_thresholds

        elif mode == "case":
            cooc, original_node_count, retained_node_count = _build_case_projection(
                incidence,
                min_case_node_freq=min_case_node_freq,
            )
            thresholds = case_edge_thresholds

        else:
            raise ValueError(f"Unsupported projection mode: {mode}")

        diag_df = _run_threshold_diagnostic(cooc, thresholds)
        out_csv = run_dir / f"{dataset_stem}_{mode}_threshold_diagnostic.csv"
        diag_df.to_csv(out_csv, index=False, encoding="utf-8")

        results["projections"][mode] = {
            "original_node_count": original_node_count,
            "retained_node_count": retained_node_count,
            "thresholds_tested": thresholds,
            "diagnostic_csv": str(out_csv),
        }

    out_summary = run_dir / out_summary_name
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
        description="Diagnose edge-threshold choices for one-mode projection networks."
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
        help="Comma-separated thresholds for feature projection (e.g. 1,2,3,5,10)",
    )
    parser.add_argument(
        "--case-edge-thresholds",
        type=_parse_int_list,
        default=CASE_EDGE_THRESHOLDS,
        help="Comma-separated thresholds for case projection (e.g. 1,2,3,5,10)",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=OUT_SUMMARY,
        help="Output summary filename",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run the diagnostic using CONFIG defaults or CLI overrides."""
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

    print("[✓] Threshold diagnostic complete.")
    print(f"    Input:          {result['input_path']}")
    print(f"    Cases total:    {result['n_cases_total']:,}")
    print(f"    Features total: {result['n_features_total']:,}")
    print(f"    Output dir:     {result['output_dir']}")

    for mode, mode_result in result["projections"].items():
        print(f"    [{mode}]")
        print(f"      Original nodes:   {mode_result['original_node_count']:,}")
        print(f"      Retained nodes:   {mode_result['retained_node_count']:,}")
        print(f"      Thresholds:       {mode_result['thresholds_tested']}")
        print(f"      CSV:              {mode_result['diagnostic_csv']}")

    print(f"    Summary:        {result['summary_txt']}")


if __name__ == "__main__":
    main()