#!/usr/bin/env python3
"""
07_find_feature_gradients.py

Find feature gradients between zero-overlap endpoint pairs using a binary
incidence matrix and the output of:

    06_significant_zero_feature_overlap.py

A feature gradient is a chain such as:

    Feature A -> Feature B -> Feature C -> Feature D -> Feature E

such that:
- A and E never co-occur in the same case
- adjacent features do co-occur
- the chain moves gradually from A's case-distribution toward E's

This script supports:

Endpoint selection modes
------------------------
1) all
   Use all zero-overlap feature pairs from the feature zero-overlap CSV

2) significant
   Use only feature pairs passing a chosen significance column
   (e.g. sig_0.05)

3) specific
   Use one user-specified zero-overlap feature pair only

Chain length modes
------------------
1) fixed
   Search only one exact chain length

2) range
   Search across a bounded range of chain lengths

Search modes
------------
1) strict
   Requires:
   - minimum adjacent Jaccard
   - optional minimum adjacent co-occurrence count
   - strict monotone decrease in similarity to A
   - strict monotone increase in similarity to E
   - neighborhood dominance

2) ranked
   Requires only:
   - minimum adjacent Jaccard
   - optional minimum adjacent co-occurrence count

   Then scores candidate chains by:
   - adjacency strength
   - monotonicity quality
   - positional smoothness

Outputs
-------
1) feature_gradients.csv
   One row per retained chain, with endpoint pair, chain members,
   adjacent similarities/co-occurrences, and score fields.

2) analysis_summary.txt
   Human-readable record of inputs, settings, and results.

Standalone use:
    Edit the CONFIG block below, then run:
        python 07_find_feature_gradients.py

Pipeline / programmatic use:
    Import this script and call:
        run(...)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Inputs
ZERO_FEATURE_OVERLAP_CSV = Path("zero_feature_overlap_with_significance.csv")
INCIDENCE_PATH = Path("input_incidence_matrix.xlsx")   # .xlsx or .csv
SHEET_NAME = 0

# Matrix structure
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Feature frequency filter (should generally match or exceed the earlier script)
MIN_FEATURE_FREQ = 5

# Endpoint selection
ENDPOINT_MODE = "significant"     # "all", "significant", or "specific"
SIGNIFICANCE_COLUMN = "sig_0.05"  # used only if ENDPOINT_MODE = "significant"

# Used only if ENDPOINT_MODE = "specific"
SPECIFIC_FEATURE_A = ""
SPECIFIC_FEATURE_E = ""

# Chain length
CHAIN_LENGTH_MODE = "fixed"       # "fixed" or "range"
CHAIN_LENGTH = 5                  # used only if CHAIN_LENGTH_MODE = "fixed"
MIN_CHAIN_LENGTH = 4              # used only if CHAIN_LENGTH_MODE = "range"
MAX_CHAIN_LENGTH = 6              # used only if CHAIN_LENGTH_MODE = "range"

# Search mode
SEARCH_MODE = "ranked"            # "strict" or "ranked"

# Adjacency requirements for neighboring features in the chain
MIN_ADJ_JACCARD = 0.05
MIN_ADJ_COOCC = 2

# Beam search / output
BEAM_WIDTH = 20
TOP_RESULTS_PER_ENDPOINT = 10
TOP_RESULTS_TOTAL = 100

# Numerical tolerance
EPS = 1e-9

# Output
OUTPUT_DIR = Path(".")
OUT_CSV = "feature_gradients.csv"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# HELPERS: I/O and matrix prep
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


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# =============================================================================
# HELPERS: feature-space similarity
# =============================================================================

def _build_feature_similarity(X: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From case × feature matrix X, build:
    - feature × feature Jaccard matrix
    - feature × feature co-occurrence count matrix
    """
    n_features = X.shape[1]
    cooc = X.T @ X
    cooc = cooc.astype(int)

    jmat = np.zeros((n_features, n_features), dtype=float)
    counts = X.sum(axis=0).astype(int)

    for i in range(n_features):
        for j in range(i, n_features):
            inter = int(cooc[i, j])
            union = int(counts[i] + counts[j] - inter)
            score = float(inter / union) if union > 0 else 0.0
            jmat[i, j] = score
            jmat[j, i] = score

    jmat_df = pd.DataFrame(jmat, index=feature_names, columns=feature_names)
    cooc_df = pd.DataFrame(cooc, index=feature_names, columns=feature_names)
    return jmat_df, cooc_df


def _t_coord_all(feature_order: List[str], jmat_df: pd.DataFrame, a: str, e: str) -> pd.Series:
    """
    Projection coordinate:
        t(x) = (J(x,A) - J(x,E)) / (J(x,A) + J(x,E))
    """
    sA = jmat_df.loc[feature_order, a]
    sE = jmat_df.loc[feature_order, e]
    return (sA - sE) / (sA + sE + 1e-12)


# =============================================================================
# HELPERS: chain evaluation
# =============================================================================

def _adjacency_passes(
    chain: List[str],
    jmat_df: pd.DataFrame,
    cooc_df: pd.DataFrame,
    *,
    min_adj_jaccard: float,
    min_adj_coocc: int,
    eps: float,
) -> bool:
    """
    Hard adjacency requirements used in both strict and ranked modes.
    """
    for u, v in zip(chain[:-1], chain[1:]):
        if float(jmat_df.loc[u, v]) + eps < min_adj_jaccard:
            return False
        if min_adj_coocc > 0 and int(cooc_df.loc[u, v]) < min_adj_coocc:
            return False
    return True


def _strict_gradient_ok(chain: List[str], jmat_df: pd.DataFrame, *, eps: float) -> bool:
    """
    Strict mode:
    - similarity to A strictly decreases
    - similarity to E strictly increases
    - neighborhood dominance
    """
    A = chain[0]
    E = chain[-1]

    sA = [float(jmat_df.loc[x, A]) for x in chain]
    sE = [float(jmat_df.loc[x, E]) for x in chain]

    if not all(sA[k] > sA[k + 1] + eps for k in range(len(chain) - 1)):
        return False

    if not all(sE[k] + eps < sE[k + 1] for k in range(len(chain) - 1)):
        return False

    m = len(chain)
    for i in range(m):
        prev = None
        for d in range(1, m):
            j = i + d
            if j >= m:
                break
            val = float(jmat_df.loc[chain[i], chain[j]])
            if prev is not None and not (prev > val + eps):
                return False
            prev = val

    return True


def _ranked_chain_score(chain: List[str], jmat_df: pd.DataFrame, t_map: pd.Series, *, eps: float) -> Dict[str, float]:
    """
    Ranked mode score components:
    - adjacency strength
    - monotonicity penalties
    - positional smoothness penalty

    Higher total_score is better.
    """
    A = chain[0]
    E = chain[-1]
    m = len(chain)

    adj_js = [float(jmat_df.loc[u, v]) for u, v in zip(chain[:-1], chain[1:])]
    min_adj = min(adj_js)
    adj_sum = sum(adj_js)

    sA = [float(jmat_df.loc[x, A]) for x in chain]
    sE = [float(jmat_df.loc[x, E]) for x in chain]

    mono_A_viol = 0
    mono_E_viol = 0
    mono_A_mag = 0.0
    mono_E_mag = 0.0

    for k in range(m - 1):
        if not (sA[k] > sA[k + 1] + eps):
            mono_A_viol += 1
            mono_A_mag += max(0.0, sA[k + 1] - sA[k])

        if not (sE[k] + eps < sE[k + 1]):
            mono_E_viol += 1
            mono_E_mag += max(0.0, sE[k] - sE[k + 1])

    if m <= 2:
        smooth_penalty = 0.0
    else:
        targets = np.linspace(1, -1, m)[1:-1]
        actuals = np.array([float(t_map.loc[x]) for x in chain[1:-1]])
        smooth_penalty = float(np.abs(actuals - targets).sum())

    total_score = (
        (2.0 * min_adj) +
        (1.0 * adj_sum) -
        (2.0 * mono_A_viol) -
        (2.0 * mono_E_viol) -
        (5.0 * mono_A_mag) -
        (5.0 * mono_E_mag) -
        (1.0 * smooth_penalty)
    )

    return {
        "min_adj": round(min_adj, 6),
        "adj_sum": round(adj_sum, 6),
        "mono_A_viol": mono_A_viol,
        "mono_E_viol": mono_E_viol,
        "mono_A_mag": round(mono_A_mag, 6),
        "mono_E_mag": round(mono_E_mag, 6),
        "smooth_penalty": round(smooth_penalty, 6),
        "total_score": round(total_score, 6),
    }


def _chain_to_record(
    chain: List[str],
    endpoint_mode: str,
    sig_col: str,
    search_mode: str,
    jmat_df: pd.DataFrame,
    cooc_df: pd.DataFrame,
    t_map: pd.Series,
    *,
    eps: float,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "endpoint_mode": endpoint_mode,
        "search_mode": search_mode,
        "significance_column": sig_col if endpoint_mode == "significant" else "",
        "feature_A": chain[0],
        "feature_E": chain[-1],
        "chain_length": len(chain),
        "chain": " | ".join(chain),
    }

    for idx, feat in enumerate(chain, start=1):
        record[f"feature_{idx}"] = feat

    adj_pairs = list(zip(chain[:-1], chain[1:]))
    for idx, (u, v) in enumerate(adj_pairs, start=1):
        record[f"adj_{idx}_pair"] = f"{u} -> {v}"
        record[f"adj_{idx}_jaccard"] = round(float(jmat_df.loc[u, v]), 6)
        record[f"adj_{idx}_cooc"] = int(cooc_df.loc[u, v])

    sA = [round(float(jmat_df.loc[x, chain[0]]), 6) for x in chain]
    sE = [round(float(jmat_df.loc[x, chain[-1]]), 6) for x in chain]
    record["sim_to_A_seq"] = str(sA)
    record["sim_to_E_seq"] = str(sE)

    if len(chain) > 2:
        record["interior_t_seq"] = str([round(float(t_map.loc[x]), 6) for x in chain[1:-1]])
    else:
        record["interior_t_seq"] = "[]"

    record["strict_pass"] = _strict_gradient_ok(chain, jmat_df, eps=eps)
    record.update(_ranked_chain_score(chain, jmat_df, t_map, eps=eps))

    return record


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    zero_feature_overlap_csv: Path,
    incidence_path: Path,
    min_feature_freq: int,
    endpoint_mode: str,
    significance_column: str,
    specific_feature_a: str,
    specific_feature_e: str,
    endpoint_pairs: int,
    chain_length_mode: str,
    chain_length: int,
    min_chain_length: int,
    max_chain_length: int,
    search_mode: str,
    min_adj_jaccard: float,
    min_adj_coocc: int,
    beam_width: int,
    top_results_per_endpoint: int,
    top_results_total: int,
    endpoint_with_results: int,
    all_records_count: int,
    rows_written: int,
    discard_adj: int,
    discard_strict: int,
    out_csv_name: str,
    out_summary_name: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Feature Gradient Search Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Feature zero-overlap CSV: {zero_feature_overlap_csv}\n")
        f.write(f"Incidence matrix: {incidence_path}\n\n")

        f.write("Filtering\n")
        f.write("---------\n")
        f.write(f"MIN_FEATURE_FREQ: {min_feature_freq}\n\n")

        f.write("Endpoint settings\n")
        f.write("-----------------\n")
        f.write(f"ENDPOINT_MODE: {endpoint_mode}\n")
        if endpoint_mode == "significant":
            f.write(f"SIGNIFICANCE_COLUMN: {significance_column}\n")
        if endpoint_mode == "specific":
            f.write(f"SPECIFIC_FEATURE_A: {specific_feature_a}\n")
            f.write(f"SPECIFIC_FEATURE_E: {specific_feature_e}\n")
        f.write(f"Endpoint pairs searched: {endpoint_pairs}\n\n")

        f.write("Chain settings\n")
        f.write("--------------\n")
        f.write(f"CHAIN_LENGTH_MODE: {chain_length_mode}\n")
        if chain_length_mode == "fixed":
            f.write(f"CHAIN_LENGTH: {chain_length}\n")
        else:
            f.write(f"MIN_CHAIN_LENGTH: {min_chain_length}\n")
            f.write(f"MAX_CHAIN_LENGTH: {max_chain_length}\n")
        f.write(f"SEARCH_MODE: {search_mode}\n")
        f.write(f"MIN_ADJ_JACCARD: {min_adj_jaccard}\n")
        f.write(f"MIN_ADJ_COOCC: {min_adj_coocc}\n")
        f.write(f"BEAM_WIDTH: {beam_width}\n")
        f.write(f"TOP_RESULTS_PER_ENDPOINT: {top_results_per_endpoint}\n")
        f.write(f"TOP_RESULTS_TOTAL: {top_results_total}\n\n")

        f.write("Search results\n")
        f.write("--------------\n")
        f.write(f"Endpoint pairs with ≥1 retained chain: {endpoint_with_results}\n")
        f.write(f"Total chains retained before global truncation: {all_records_count}\n")
        f.write(f"Rows written: {rows_written}\n")
        f.write(f"Candidates discarded by adjacency filter: {discard_adj}\n")
        if search_mode == "strict":
            f.write(f"Candidates discarded by strict gradient rules: {discard_strict}\n")
        f.write("\n")

        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("A feature gradient is a chain of features linking two endpoint features that\n")
        f.write("never co-occur directly. Strict mode returns only chains satisfying strong\n")
        f.write("gradient constraints. Ranked mode returns plausible gradients scored by\n")
        f.write("adjacency strength, monotonicity quality, and positional smoothness.\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_csv_name}\n")
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
    min_feature_freq: int = 5,
    endpoint_mode: str = "significant",
    significance_column: str = "sig_0.05",
    specific_feature_a: str = "",
    specific_feature_e: str = "",
    chain_length_mode: str = "fixed",
    chain_length: int = 5,
    min_chain_length: int = 4,
    max_chain_length: int = 6,
    search_mode: str = "ranked",
    min_adj_jaccard: float = 0.05,
    min_adj_coocc: int = 2,
    beam_width: int = 20,
    top_results_per_endpoint: int = 10,
    top_results_total: int = 100,
    eps: float = 1e-9,
    out_csv_name: str = "feature_gradients.csv",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Find feature gradients and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if endpoint_mode not in {"all", "significant", "specific"}:
        raise ValueError("endpoint_mode must be 'all', 'significant', or 'specific'.")

    if chain_length_mode not in {"fixed", "range"}:
        raise ValueError("chain_length_mode must be 'fixed' or 'range'.")

    if search_mode not in {"strict", "ranked"}:
        raise ValueError("search_mode must be 'strict' or 'ranked'.")

    # --- Load feature zero-overlap table
    if not zero_feature_overlap_csv.exists():
        raise FileNotFoundError(f"Feature zero-overlap CSV not found: {zero_feature_overlap_csv}")

    zero_df = pd.read_csv(zero_feature_overlap_csv)

    required_cols = {"feature_a", "feature_b"}
    if not required_cols.issubset(zero_df.columns):
        raise ValueError(
            f"{zero_feature_overlap_csv} must contain columns: {sorted(required_cols)}"
        )

    zero_df[["feature_a", "feature_b"]] = zero_df[["feature_a", "feature_b"]].astype(str)

    # --- Load incidence matrix and build feature-space similarity
    df_raw = _read_table(incidence_path, sheet_name=sheet_name)

    if case_id_column not in df_raw.columns:
        raise ValueError(f"CASE_ID_COLUMN '{case_id_column}' not found in input columns.")

    case_index = df_raw.columns.get_loc(case_id_column)
    if case_index >= n_metadata_cols:
        raise ValueError(
            f"CASE_ID_COLUMN '{case_id_column}' is outside the first {n_metadata_cols} columns.\n"
            "This script assumes that all metadata columns, including the case identifier, "
            "appear to the LEFT of the feature columns."
        )

    feature_cols_all = _get_feature_columns(df_raw, n_metadata_cols)
    if not feature_cols_all:
        raise ValueError("No feature columns found. Check N_METADATA_COLS.")

    df = pd.concat([df_raw[[case_id_column]], df_raw[feature_cols_all]], axis=1).copy()
    bin_features = _binarize_presence(df, feature_cols_all, presence_token)
    df_bin = pd.concat([df[[case_id_column]].copy(), bin_features], axis=1)

    # Feature frequency filter
    feature_counts_all = df_bin.drop(columns=[case_id_column]).sum(axis=0).astype(int)
    keep_features = feature_counts_all[feature_counts_all >= min_feature_freq].index.tolist()

    if len(keep_features) < 2:
        raise ValueError(
            f"Only {len(keep_features)} features remain after MIN_FEATURE_FREQ={min_feature_freq}. "
            "Need at least 2."
        )

    X = df_bin[keep_features].to_numpy(dtype=np.uint8)
    feature_order = keep_features

    jmat_df, cooc_df = _build_feature_similarity(X, feature_order)

    # --- Endpoint selection
    if endpoint_mode == "all":
        endpoint_df = zero_df.copy()

    elif endpoint_mode == "significant":
        if significance_column not in zero_df.columns:
            raise ValueError(
                f"SIGNIFICANCE_COLUMN '{significance_column}' not found in {zero_feature_overlap_csv}"
            )
        endpoint_df = zero_df[zero_df[significance_column].astype(bool)].copy()

    else:
        if not specific_feature_a or not specific_feature_e:
            raise ValueError(
                "For endpoint_mode='specific', set both specific_feature_a and specific_feature_e."
            )

        a, e = _normalize_pair(str(specific_feature_a), str(specific_feature_e))
        pair_mask = (
            zero_df.apply(lambda r: _normalize_pair(str(r["feature_a"]), str(r["feature_b"])), axis=1) == (a, e)
        )
        endpoint_df = zero_df[pair_mask].copy()

        if endpoint_df.empty:
            raise ValueError(
                f"The specified pair ({specific_feature_a}, {specific_feature_e}) was not found "
                "in the feature zero-overlap table."
            )

    # Keep only endpoints that still exist after MIN_FEATURE_FREQ filtering
    endpoint_pairs = []
    seen_pairs = set()

    for _, row in endpoint_df.iterrows():
        a, e = _normalize_pair(str(row["feature_a"]), str(row["feature_b"]))
        if a not in feature_order or e not in feature_order:
            continue
        if int(cooc_df.loc[a, e]) != 0:
            continue
        if (a, e) not in seen_pairs:
            seen_pairs.add((a, e))
            endpoint_pairs.append((a, e))

    if not endpoint_pairs:
        raise ValueError("No valid zero-overlap feature endpoint pairs remained after filtering.")

    # --- Chain lengths
    if chain_length_mode == "fixed":
        chain_lengths = [chain_length]
    else:
        chain_lengths = list(range(min_chain_length, max_chain_length + 1))

    if min(chain_lengths) < 3:
        raise ValueError("Minimum chain length must be at least 3.")
    if max(chain_lengths) > len(feature_order):
        raise ValueError("Chain length exceeds number of available features.")

    # --- Search
    all_records: List[Dict[str, object]] = []
    discard_adj = 0
    discard_strict = 0
    endpoint_with_results = 0

    for a, e in endpoint_pairs:
        t_map = _t_coord_all(feature_order, jmat_df, a, e)
        pool_feats = [f for f in feature_order if f not in (a, e)]

        endpoint_records: List[Dict[str, object]] = []

        for current_chain_len in chain_lengths:
            n_interior = current_chain_len - 2
            if n_interior <= 0:
                continue

            targets = np.linspace(1, -1, current_chain_len)[1:-1]

            beams: List[List[str]] = []
            for tgt in targets:
                ranked_feats = sorted(
                    pool_feats,
                    key=lambda f: (
                        abs(float(t_map.loc[f]) - tgt),
                        -float(jmat_df.loc[f, a]),
                        float(jmat_df.loc[f, e]),
                    )
                )
                beams.append(ranked_feats[:beam_width])

            def recurse_build(pos: int, partial: List[str]) -> None:
                nonlocal discard_adj, discard_strict, endpoint_records

                if pos == len(beams):
                    chain = [a] + partial + [e]

                    if len(set(chain)) != len(chain):
                        return

                    if not _adjacency_passes(
                        chain,
                        jmat_df,
                        cooc_df,
                        min_adj_jaccard=min_adj_jaccard,
                        min_adj_coocc=min_adj_coocc,
                        eps=eps,
                    ):
                        discard_adj += 1
                        return

                    if search_mode == "strict":
                        if not _strict_gradient_ok(chain, jmat_df, eps=eps):
                            discard_strict += 1
                            return

                    record = _chain_to_record(
                        chain,
                        endpoint_mode=endpoint_mode,
                        sig_col=significance_column,
                        search_mode=search_mode,
                        jmat_df=jmat_df,
                        cooc_df=cooc_df,
                        t_map=t_map,
                        eps=eps,
                    )
                    endpoint_records.append(record)
                    return

                for candidate in beams[pos]:
                    if candidate in partial or candidate in (a, e):
                        continue
                    recurse_build(pos + 1, partial + [candidate])

            recurse_build(0, [])

        if endpoint_records:
            endpoint_with_results += 1
            endpoint_df_rec = pd.DataFrame(endpoint_records)

            if search_mode == "strict":
                endpoint_df_rec = endpoint_df_rec.sort_values(
                    by=["min_adj", "adj_sum"],
                    ascending=[False, False]
                )
            else:
                endpoint_df_rec = endpoint_df_rec.sort_values(
                    by=["total_score", "min_adj", "adj_sum"],
                    ascending=[False, False, False]
                )

            endpoint_df_rec = endpoint_df_rec.head(top_results_per_endpoint)
            all_records.extend(endpoint_df_rec.to_dict(orient="records"))

    # --- Output
    out_csv_path = output_dir / out_csv_name
    out_summary_path = output_dir / out_summary_name

    if not all_records:
        empty_df = pd.DataFrame(columns=[
            "endpoint_mode", "search_mode", "significance_column",
            "feature_A", "feature_E", "chain_length", "chain", "strict_pass",
            "min_adj", "adj_sum", "mono_A_viol", "mono_E_viol",
            "mono_A_mag", "mono_E_mag", "smooth_penalty", "total_score"
        ])
        empty_df.to_csv(out_csv_path, index=False, encoding="utf-8")

        with open(out_summary_path, "w", encoding="utf-8") as f:
            f.write("=== Feature Gradient Search Summary ===\n\n")
            f.write(f"Run timestamp: {run_timestamp}\n")
            f.write("No feature gradients were found under the current settings.\n")

        return {
            "run_timestamp": run_timestamp,
            "output_dir": str(output_dir),
            "rows_written": 0,
            "endpoint_pairs": len(endpoint_pairs),
            "endpoint_pairs_with_results": 0,
            "feature_gradients_csv": str(out_csv_path),
            "summary_txt": str(out_summary_path),
        }

    out_df = pd.DataFrame(all_records)

    if search_mode == "strict":
        out_df = out_df.sort_values(
            by=["min_adj", "adj_sum"],
            ascending=[False, False]
        )
    else:
        out_df = out_df.sort_values(
            by=["total_score", "min_adj", "adj_sum"],
            ascending=[False, False, False]
        )

    out_df = out_df.head(top_results_total).reset_index(drop=True)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

    _write_summary(
        out_summary_path,
        run_timestamp=run_timestamp,
        zero_feature_overlap_csv=zero_feature_overlap_csv,
        incidence_path=incidence_path,
        min_feature_freq=min_feature_freq,
        endpoint_mode=endpoint_mode,
        significance_column=significance_column,
        specific_feature_a=specific_feature_a,
        specific_feature_e=specific_feature_e,
        endpoint_pairs=len(endpoint_pairs),
        chain_length_mode=chain_length_mode,
        chain_length=chain_length,
        min_chain_length=min_chain_length,
        max_chain_length=max_chain_length,
        search_mode=search_mode,
        min_adj_jaccard=min_adj_jaccard,
        min_adj_coocc=min_adj_coocc,
        beam_width=beam_width,
        top_results_per_endpoint=top_results_per_endpoint,
        top_results_total=top_results_total,
        endpoint_with_results=endpoint_with_results,
        all_records_count=len(all_records),
        rows_written=len(out_df),
        discard_adj=discard_adj,
        discard_strict=discard_strict,
        out_csv_name=out_csv_name,
        out_summary_name=out_summary_name,
    )

    return {
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "rows_written": len(out_df),
        "endpoint_pairs": len(endpoint_pairs),
        "endpoint_pairs_with_results": endpoint_with_results,
        "feature_gradients_csv": str(out_csv_path),
        "summary_txt": str(out_summary_path),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find feature gradients between zero-overlap endpoint pairs."
    )

    parser.add_argument("--zero-feature-overlap-csv", type=Path, default=ZERO_FEATURE_OVERLAP_CSV)
    parser.add_argument("--incidence-path", type=Path, default=INCIDENCE_PATH)
    parser.add_argument("--sheet-name", default=SHEET_NAME)

    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)

    parser.add_argument("--min-feature-freq", type=int, default=MIN_FEATURE_FREQ)

    parser.add_argument("--endpoint-mode", type=str, default=ENDPOINT_MODE)
    parser.add_argument("--significance-column", type=str, default=SIGNIFICANCE_COLUMN)
    parser.add_argument("--specific-feature-a", type=str, default=SPECIFIC_FEATURE_A)
    parser.add_argument("--specific-feature-e", type=str, default=SPECIFIC_FEATURE_E)

    parser.add_argument("--chain-length-mode", type=str, default=CHAIN_LENGTH_MODE)
    parser.add_argument("--chain-length", type=int, default=CHAIN_LENGTH)
    parser.add_argument("--min-chain-length", type=int, default=MIN_CHAIN_LENGTH)
    parser.add_argument("--max-chain-length", type=int, default=MAX_CHAIN_LENGTH)

    parser.add_argument("--search-mode", type=str, default=SEARCH_MODE)
    parser.add_argument("--min-adj-jaccard", type=float, default=MIN_ADJ_JACCARD)
    parser.add_argument("--min-adj-coocc", type=int, default=MIN_ADJ_COOCC)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--top-results-per-endpoint", type=int, default=TOP_RESULTS_PER_ENDPOINT)
    parser.add_argument("--top-results-total", type=int, default=TOP_RESULTS_TOTAL)

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
        zero_feature_overlap_csv=args.zero_feature_overlap_csv,
        incidence_path=args.incidence_path,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        min_feature_freq=args.min_feature_freq,
        endpoint_mode=args.endpoint_mode,
        significance_column=args.significance_column,
        specific_feature_a=args.specific_feature_a,
        specific_feature_e=args.specific_feature_e,
        chain_length_mode=args.chain_length_mode,
        chain_length=args.chain_length,
        min_chain_length=args.min_chain_length,
        max_chain_length=args.max_chain_length,
        search_mode=args.search_mode,
        min_adj_jaccard=args.min_adj_jaccard,
        min_adj_coocc=args.min_adj_coocc,
        beam_width=args.beam_width,
        top_results_per_endpoint=args.top_results_per_endpoint,
        top_results_total=args.top_results_total,
        out_csv_name=args.out_csv,
        out_summary_name=args.out_summary,
    )

    print("[✓] Feature gradient search complete.")
    print(f"    Endpoint pairs:     {result['endpoint_pairs']}")
    print(f"    Pairs with results: {result['endpoint_pairs_with_results']}")
    print(f"    Rows written:       {result['rows_written']}")
    print(f"    Output CSV:         {result['feature_gradients_csv']}")
    print(f"    Summary:            {result['summary_txt']}")


if __name__ == "__main__":
    main()