#!/usr/bin/env python3
"""
05_find_case_gradients.py

Find discourse gradients between zero-overlap case endpoint pairs using:
- a zero-overlap table
- a Jaccard similarity matrix (existing or computed in-script)
- the original incidence matrix

A discourse gradient is a chain of cases:

    A -> ... -> E

such that:
- A and E are zero-overlap endpoints
- adjacent pairs share meaningful overlap
- the chain moves gradually from A's repertoire toward E's repertoire

Supports:

Endpoint selection modes
------------------------
1) all
   Use all zero-overlap pairs from the zero-overlap CSV

2) significant
   Use only zero-overlap pairs passing a chosen significance column
   (e.g. sig_0.05)

3) specific
   Use one user-specified zero-overlap pair only

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
   - optional minimum adjacent intersection count
   - strict monotone decrease in similarity to A
   - strict monotone increase in similarity to E
   - neighborhood dominance

2) ranked
   Requires only:
   - minimum adjacent Jaccard
   - optional minimum adjacent intersection count

   Then scores candidate chains by:
   - adjacency strength
   - monotonicity quality
   - positional smoothness

Outputs
-------
1) case_gradients.csv
   One row per retained chain, with endpoint pair, chain members, adjacent
   similarities/intersections, and score fields.

2) analysis_summary.txt
   Human-readable record of inputs, settings, and results.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

# Inputs
ZERO_OVERLAP_CSV = Path("zero_overlap_pairs_with_significance.csv")
JACCARD_MODE = "existing"   # "existing" or "compute"
JACCARD_CSV = Path("jaccard_similarity_matrix.csv")   # used only if JACCARD_MODE = "existing"
INCIDENCE_PATH = Path("input_incidence_matrix.xlsx")  # .xlsx/.xls or .csv
SHEET_NAME = 0

# Incidence matrix structure
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Optional feature filtering used only when computing Jaccard internally
GLOBAL_FEATURE_MIN_CASES = 1

# Endpoint selection
ENDPOINT_MODE = "significant"     # "all", "significant", or "specific"
SIGNIFICANCE_COLUMN = "sig_0.05"  # used only if ENDPOINT_MODE = "significant"

# Used only if ENDPOINT_MODE = "specific"
SPECIFIC_CASE_A = ""
SPECIFIC_CASE_E = ""

# Chain length
CHAIN_LENGTH_MODE = "fixed"       # "fixed" or "range"
CHAIN_LENGTH = 5                  # used only if CHAIN_LENGTH_MODE = "fixed"
MIN_CHAIN_LENGTH = 4              # used only if CHAIN_LENGTH_MODE = "range"
MAX_CHAIN_LENGTH = 6              # used only if CHAIN_LENGTH_MODE = "range"

# Search mode
SEARCH_MODE = "ranked"            # "strict" or "ranked"

# Adjacency strength requirements
MIN_ADJ = 0.20
MIN_INTERSECTION = 0

# Beam search / output
BEAM_WIDTH = 20
TOP_RESULTS_PER_ENDPOINT = 10
TOP_RESULTS_TOTAL = 100

# Numerical tolerance
EPS = 1e-9

# Output
OUTPUT_DIR = Path(".")
OUT_CSV = "case_gradients.csv"
OUT_SUMMARY = "analysis_summary.txt"


# =============================================================================
# HELPERS: reading and cleaning
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


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _prepare_incidence(
    incidence_path: Path,
    *,
    sheet_name: int | str,
    case_id_column: str,
    n_metadata_cols: int,
    presence_token: str,
    global_feature_min_cases: int,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Return:
        case_order, binary feature dataframe indexed by case_id
    """
    inc_raw = _read_table(incidence_path, sheet_name=sheet_name)
    if case_id_column not in inc_raw.columns:
        raise ValueError(f"CASE_ID_COLUMN '{case_id_column}' not found in {incidence_path}")

    producer_index = inc_raw.columns.get_loc(case_id_column)
    if producer_index >= n_metadata_cols:
        raise ValueError(
            f"CASE_ID_COLUMN '{case_id_column}' is outside the first {n_metadata_cols} columns.\n"
            "This script assumes that all metadata columns appear to the LEFT of the feature columns."
        )

    feature_cols = _get_feature_columns(inc_raw, n_metadata_cols)
    inc = pd.concat([inc_raw[[case_id_column]], inc_raw[feature_cols]], axis=1).copy()
    bin_features = _binarize_presence(inc, feature_cols, presence_token)
    bin_df = pd.concat([inc[[case_id_column]].copy(), bin_features], axis=1)
    bin_df[case_id_column] = bin_df[case_id_column].astype(str)

    if global_feature_min_cases > 1:
        feat_freq = bin_df.drop(columns=[case_id_column]).sum(axis=0)
        keep_features = feat_freq[feat_freq >= global_feature_min_cases].index.tolist()
        if not keep_features:
            raise ValueError("No features remain after GLOBAL_FEATURE_MIN_CASES filter.")
        bin_df = pd.concat([bin_df[[case_id_column]], bin_df[keep_features]], axis=1)

    case_order = bin_df[case_id_column].astype(str).tolist()
    bin_df = bin_df.set_index(case_id_column)

    return case_order, bin_df


def _compute_jaccard_matrix(case_order: List[str], X: np.ndarray) -> pd.DataFrame:
    """Case × case Jaccard matrix."""
    n = len(case_order)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        ai = X[i]
        for j in range(i, n):
            inter = int(np.bitwise_and(ai, X[j]).sum())
            union = int(np.bitwise_or(ai, X[j]).sum())
            jacc = inter / union if union > 0 else 0.0
            mat[i, j] = jacc
            mat[j, i] = jacc

    return pd.DataFrame(mat, index=case_order, columns=case_order)


def _build_intersection_matrix(case_order: List[str], X: np.ndarray) -> pd.DataFrame:
    """Case × case raw intersection counts."""
    n = len(case_order)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        ai = X[i]
        for j in range(i, n):
            inter = int(np.bitwise_and(ai, X[j]).sum())
            mat[i, j] = inter
            mat[j, i] = inter
    return pd.DataFrame(mat, index=case_order, columns=case_order)


# =============================================================================
# HELPERS: chain evaluation
# =============================================================================

def _t_coord_all(case_order: List[str], jmat_df: pd.DataFrame, a: str, e: str) -> pd.Series:
    """
    Projection coordinate:
        t(x) = (J(x,A) - J(x,E)) / (J(x,A) + J(x,E))
    """
    sA = jmat_df.loc[case_order, a]
    sE = jmat_df.loc[case_order, e]
    return (sA - sE) / (sA + sE + 1e-12)


def _adjacency_passes(
    chain: List[str],
    jmat_df: pd.DataFrame,
    imat_df: pd.DataFrame,
    *,
    min_adj: float,
    min_intersection: int,
    eps: float,
) -> bool:
    """Hard adjacency requirements used in both strict and ranked modes."""
    for u, v in zip(chain[:-1], chain[1:]):
        if float(jmat_df.loc[u, v]) + eps < min_adj:
            return False
        if min_intersection > 0 and int(imat_df.loc[u, v]) < min_intersection:
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
    *,
    endpoint_mode: str,
    sig_col: str,
    search_mode: str,
    jmat_df: pd.DataFrame,
    imat_df: pd.DataFrame,
    t_map: pd.Series,
    eps: float,
) -> Dict[str, object]:
    """Create one output row for a chain."""
    record: Dict[str, object] = {
        "endpoint_mode": endpoint_mode,
        "search_mode": search_mode,
        "significance_column": sig_col if endpoint_mode == "significant" else "",
        "case_A": chain[0],
        "case_E": chain[-1],
        "chain_length": len(chain),
        "chain": " | ".join(chain),
    }

    for idx, case in enumerate(chain, start=1):
        record[f"case_{idx}"] = case

    adj_pairs = list(zip(chain[:-1], chain[1:]))
    for idx, (u, v) in enumerate(adj_pairs, start=1):
        record[f"adj_{idx}_pair"] = f"{u} -> {v}"
        record[f"adj_{idx}_jaccard"] = round(float(jmat_df.loc[u, v]), 6)
        record[f"adj_{idx}_intersection"] = int(imat_df.loc[u, v])

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


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run(
    *,
    zero_overlap_csv: Path,
    incidence_path: Path,
    output_dir: Path,
    jaccard_mode: str = "existing",            # "existing" or "compute"
    jaccard_csv: Optional[Path] = None,        # used only if jaccard_mode="existing"
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    global_feature_min_cases: int = 1,
    endpoint_mode: str = "significant",        # "all", "significant", or "specific"
    significance_column: str = "sig_0.05",
    specific_case_a: str = "",
    specific_case_e: str = "",
    chain_length_mode: str = "fixed",          # "fixed" or "range"
    chain_length: int = 5,
    min_chain_length: int = 4,
    max_chain_length: int = 6,
    search_mode: str = "ranked",               # "strict" or "ranked"
    min_adj: float = 0.20,
    min_intersection: int = 0,
    beam_width: int = 20,
    top_results_per_endpoint: int = 10,
    top_results_total: int = 100,
    eps: float = 1e-9,
    out_csv_name: str = "case_gradients.csv",
    out_summary_name: str = "analysis_summary.txt",
) -> Dict[str, Any]:
    """
    Run case gradient search and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if endpoint_mode not in {"all", "significant", "specific"}:
        raise ValueError("endpoint_mode must be 'all', 'significant', or 'specific'.")

    if chain_length_mode not in {"fixed", "range"}:
        raise ValueError("chain_length_mode must be 'fixed' or 'range'.")

    if search_mode not in {"strict", "ranked"}:
        raise ValueError("search_mode must be 'strict' or 'ranked'.")

    if jaccard_mode not in {"existing", "compute"}:
        raise ValueError("jaccard_mode must be 'existing' or 'compute'.")

    if jaccard_mode == "existing" and (jaccard_csv is None or not jaccard_csv.exists()):
        raise ValueError("For jaccard_mode='existing', a valid jaccard_csv must be supplied.")

    # --- Read zero-overlap table
    if not zero_overlap_csv.exists():
        raise FileNotFoundError(f"Zero-overlap CSV not found: {zero_overlap_csv}")

    zero_df = pd.read_csv(zero_overlap_csv)
    required_zero_cols = {"case_A", "case_B"}
    if not required_zero_cols.issubset(zero_df.columns):
        raise ValueError(
            f"{zero_overlap_csv} must contain columns: {sorted(required_zero_cols)}"
        )
    zero_df[["case_A", "case_B"]] = zero_df[["case_A", "case_B"]].astype(str)

    # --- Prepare incidence once
    case_order_inc, bin_df = _prepare_incidence(
        incidence_path,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        global_feature_min_cases=global_feature_min_cases,
    )

    # --- Jaccard handling
    computed_jaccard_csv_path: Optional[Path] = None

    if jaccard_mode == "existing":
        jmat_df = pd.read_csv(jaccard_csv, index_col=0)  # type: ignore[arg-type]
        jmat_df.index = jmat_df.index.map(str)
        jmat_df.columns = jmat_df.columns.map(str)

        if list(jmat_df.index) != list(jmat_df.columns):
            raise ValueError("JACCARD_CSV must be a square matrix with matching row/column labels.")

        case_order = list(jmat_df.index)

        # Keep only cases that appear in the Jaccard matrix
        bin_df = bin_df[bin_df.index.isin(case_order)].copy()
        bin_df = bin_df.reindex(case_order)
        if bin_df.isna().any().any():
            missing_cases = bin_df.index[bin_df.isna().any(axis=1)].tolist()
            raise ValueError(
                "Some cases in the Jaccard matrix are missing from the incidence matrix after alignment: "
                f"{missing_cases[:10]}"
            )

    else:
        case_order = case_order_inc
        X_tmp = bin_df.values.astype(np.uint8)
        jmat_df = _compute_jaccard_matrix(case_order, X_tmp)
        computed_jaccard_csv_path = output_dir / "jaccard_similarity_matrix.csv"
        jmat_df.to_csv(computed_jaccard_csv_path, encoding="utf-8")

    # --- Build intersections aligned to case_order
    bin_df = bin_df.reindex(case_order)
    X = bin_df.values.astype(np.uint8)
    imat_df = _build_intersection_matrix(case_order, X)

    # --- Endpoint selection
    if endpoint_mode == "all":
        endpoint_df = zero_df.copy()

    elif endpoint_mode == "significant":
        if significance_column not in zero_df.columns:
            raise ValueError(
                f"significance_column '{significance_column}' not found in {zero_overlap_csv}"
            )
        endpoint_df = zero_df[zero_df[significance_column].astype(bool)].copy()

    else:
        if not specific_case_a or not specific_case_e:
            raise ValueError(
                "For endpoint_mode='specific', set both specific_case_a and specific_case_e."
            )

        a, e = _normalize_pair(str(specific_case_a), str(specific_case_e))
        pair_mask = (
            zero_df.apply(lambda r: _normalize_pair(str(r["case_A"]), str(r["case_B"])), axis=1) == (a, e)
        )
        endpoint_df = zero_df[pair_mask].copy()

        if endpoint_df.empty:
            raise ValueError(
                f"The specified pair ({specific_case_a}, {specific_case_e}) was not found "
                "in the zero-overlap table."
            )

    endpoint_pairs = []
    seen_pairs = set()

    for _, row in endpoint_df.iterrows():
        a, e = _normalize_pair(str(row["case_A"]), str(row["case_B"]))
        if a not in case_order or e not in case_order:
            continue
        if float(jmat_df.loc[a, e]) > eps:
            continue
        if (a, e) not in seen_pairs:
            seen_pairs.add((a, e))
            endpoint_pairs.append((a, e))

    if not endpoint_pairs:
        raise ValueError("No valid zero-overlap endpoint pairs remained after filtering.")

    # --- Chain lengths
    if chain_length_mode == "fixed":
        chain_lengths = [chain_length]
    else:
        chain_lengths = list(range(min_chain_length, max_chain_length + 1))

    if min(chain_lengths) < 3:
        raise ValueError("Minimum chain length must be at least 3.")
    if max(chain_lengths) > len(case_order):
        raise ValueError("Chain length exceeds number of available cases.")

    # --- Search
    all_records: List[Dict[str, object]] = []
    discard_adj = 0
    discard_strict = 0
    endpoint_with_results = 0

    for a, e in endpoint_pairs:
        t_map = _t_coord_all(case_order, jmat_df, a, e)
        pool_idx = [k for k, c in enumerate(case_order) if c not in (a, e)]

        endpoint_records: List[Dict[str, object]] = []

        for current_chain_len in chain_lengths:
            n_interior = current_chain_len - 2
            if n_interior <= 0:
                continue

            targets = np.linspace(1, -1, current_chain_len)[1:-1]

            beams: List[List[str]] = []
            for tgt in targets:
                ranked_cases = sorted(
                    [case_order[k] for k in pool_idx],
                    key=lambda c: (
                        abs(float(t_map.loc[c]) - tgt),
                        -float(jmat_df.loc[c, a]),
                        float(jmat_df.loc[c, e]),
                    ),
                )
                beams.append(ranked_cases[:beam_width])

            def recurse_build(pos: int, partial: List[str]) -> None:
                nonlocal discard_adj, discard_strict, endpoint_records

                if pos == len(beams):
                    chain = [a] + partial + [e]

                    if len(set(chain)) != len(chain):
                        return

                    if not _adjacency_passes(
                        chain,
                        jmat_df,
                        imat_df,
                        min_adj=min_adj,
                        min_intersection=min_intersection,
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
                        imat_df=imat_df,
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
                    ascending=[False, False],
                )
            else:
                endpoint_df_rec = endpoint_df_rec.sort_values(
                    by=["total_score", "min_adj", "adj_sum"],
                    ascending=[False, False, False],
                )

            endpoint_df_rec = endpoint_df_rec.head(top_results_per_endpoint)
            all_records.extend(endpoint_df_rec.to_dict(orient="records"))

    # --- Final output
    out_csv_path = output_dir / out_csv_name
    out_summary_path = output_dir / out_summary_name

    if not all_records:
        empty_df = pd.DataFrame(columns=[
            "endpoint_mode", "search_mode", "significance_column",
            "case_A", "case_E", "chain_length", "chain", "strict_pass",
            "min_adj", "adj_sum", "mono_A_viol", "mono_E_viol",
            "mono_A_mag", "mono_E_mag", "smooth_penalty", "total_score"
        ])
        empty_df.to_csv(out_csv_path, index=False, encoding="utf-8")

        with open(out_summary_path, "w", encoding="utf-8") as f:
            f.write("=== Case Gradient Search Summary ===\n\n")
            f.write(f"Run timestamp: {run_timestamp}\n")
            f.write("No case gradients were found under the current settings.\n\n")
            f.write("Inputs\n")
            f.write("------\n")
            f.write(f"Zero-overlap CSV: {zero_overlap_csv}\n")
            f.write(f"Jaccard mode: {jaccard_mode}\n")
            if jaccard_mode == "existing":
                f.write(f"Jaccard CSV: {jaccard_csv}\n")
            else:
                f.write(f"Jaccard CSV (computed): {computed_jaccard_csv_path}\n")
            f.write(f"Incidence matrix: {incidence_path}\n")

        return {
            "run_timestamp": run_timestamp,
            "output_dir": str(output_dir),
            "gradients_csv": str(out_csv_path),
            "summary_txt": str(out_summary_path),
            "rows_written": 0,
            "endpoint_pairs": len(endpoint_pairs),
            "endpoint_pairs_with_results": 0,
            "computed_jaccard_csv": str(computed_jaccard_csv_path) if computed_jaccard_csv_path else None,
        }

    out_df = pd.DataFrame(all_records)

    if search_mode == "strict":
        out_df = out_df.sort_values(
            by=["min_adj", "adj_sum"],
            ascending=[False, False],
        )
    else:
        out_df = out_df.sort_values(
            by=["total_score", "min_adj", "adj_sum"],
            ascending=[False, False, False],
        )

    out_df = out_df.head(top_results_total).reset_index(drop=True)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

    with open(out_summary_path, "w", encoding="utf-8") as f:
        f.write("=== Case Gradient Search Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Zero-overlap CSV: {zero_overlap_csv}\n")
        f.write(f"Jaccard mode: {jaccard_mode}\n")
        if jaccard_mode == "existing":
            f.write(f"Jaccard CSV: {jaccard_csv}\n")
        else:
            f.write(f"Jaccard CSV (computed): {computed_jaccard_csv_path}\n")
        f.write(f"Incidence matrix: {incidence_path}\n\n")

        f.write("Endpoint settings\n")
        f.write("-----------------\n")
        f.write(f"ENDPOINT_MODE: {endpoint_mode}\n")
        if endpoint_mode == "significant":
            f.write(f"SIGNIFICANCE_COLUMN: {significance_column}\n")
        if endpoint_mode == "specific":
            f.write(f"SPECIFIC_CASE_A: {specific_case_a}\n")
            f.write(f"SPECIFIC_CASE_E: {specific_case_e}\n")
        f.write(f"Endpoint pairs searched: {len(endpoint_pairs)}\n\n")

        f.write("Chain settings\n")
        f.write("--------------\n")
        f.write(f"CHAIN_LENGTH_MODE: {chain_length_mode}\n")
        if chain_length_mode == "fixed":
            f.write(f"CHAIN_LENGTH: {chain_length}\n")
        else:
            f.write(f"MIN_CHAIN_LENGTH: {min_chain_length}\n")
            f.write(f"MAX_CHAIN_LENGTH: {max_chain_length}\n")
        f.write(f"SEARCH_MODE: {search_mode}\n")
        f.write(f"MIN_ADJ: {min_adj}\n")
        f.write(f"MIN_INTERSECTION: {min_intersection}\n")
        f.write(f"BEAM_WIDTH: {beam_width}\n")
        f.write(f"TOP_RESULTS_PER_ENDPOINT: {top_results_per_endpoint}\n")
        f.write(f"TOP_RESULTS_TOTAL: {top_results_total}\n\n")

        f.write("Search results\n")
        f.write("--------------\n")
        f.write(f"Endpoint pairs with ≥1 retained chain: {endpoint_with_results}\n")
        f.write(f"Total chains retained before global truncation: {len(all_records)}\n")
        f.write(f"Rows written: {len(out_df)}\n")
        f.write(f"Candidates discarded by adjacency filter: {discard_adj}\n")
        if search_mode == "strict":
            f.write(f"Candidates discarded by strict gradient rules: {discard_strict}\n")
        f.write("\n")

        f.write("Interpretation\n")
        f.write("--------------\n")
        f.write("Strict mode returns only chains that satisfy strong gradient constraints,\n")
        f.write("including strict monotonicity and neighborhood dominance.\n")
        f.write("Ranked mode returns plausible gradients scored by adjacency strength,\n")
        f.write("monotonicity quality, and positional smoothness, without requiring\n")
        f.write("neighborhood dominance.\n\n")

        f.write("Output files\n")
        f.write("------------\n")
        f.write(f"{out_csv_name}\n")
        f.write(f"{out_summary_name}\n")

    return {
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "gradients_csv": str(out_csv_path),
        "summary_txt": str(out_summary_path),
        "rows_written": len(out_df),
        "endpoint_pairs": len(endpoint_pairs),
        "endpoint_pairs_with_results": endpoint_with_results,
        "computed_jaccard_csv": str(computed_jaccard_csv_path) if computed_jaccard_csv_path else None,
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find discourse gradients between zero-overlap case endpoint pairs."
    )

    parser.add_argument("--zero-overlap-csv", type=Path, default=ZERO_OVERLAP_CSV)
    parser.add_argument("--jaccard-mode", type=str, default=JACCARD_MODE)
    parser.add_argument("--jaccard-csv", type=Path, default=JACCARD_CSV)
    parser.add_argument("--incidence-path", type=Path, default=INCIDENCE_PATH)
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--case-id-column", type=str, default=CASE_ID_COLUMN)
    parser.add_argument("--n-metadata-cols", type=int, default=N_METADATA_COLS)
    parser.add_argument("--presence-token", type=str, default=PRESENCE_TOKEN)
    parser.add_argument("--global-feature-min-cases", type=int, default=GLOBAL_FEATURE_MIN_CASES)

    parser.add_argument("--endpoint-mode", type=str, default=ENDPOINT_MODE)
    parser.add_argument("--significance-column", type=str, default=SIGNIFICANCE_COLUMN)
    parser.add_argument("--specific-case-a", type=str, default=SPECIFIC_CASE_A)
    parser.add_argument("--specific-case-e", type=str, default=SPECIFIC_CASE_E)

    parser.add_argument("--chain-length-mode", type=str, default=CHAIN_LENGTH_MODE)
    parser.add_argument("--chain-length", type=int, default=CHAIN_LENGTH)
    parser.add_argument("--min-chain-length", type=int, default=MIN_CHAIN_LENGTH)
    parser.add_argument("--max-chain-length", type=int, default=MAX_CHAIN_LENGTH)

    parser.add_argument("--search-mode", type=str, default=SEARCH_MODE)
    parser.add_argument("--min-adj", type=float, default=MIN_ADJ)
    parser.add_argument("--min-intersection", type=int, default=MIN_INTERSECTION)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--top-results-per-endpoint", type=int, default=TOP_RESULTS_PER_ENDPOINT)
    parser.add_argument("--top-results-total", type=int, default=TOP_RESULTS_TOTAL)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out-csv", type=str, default=OUT_CSV)
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    result = run(
        zero_overlap_csv=args.zero_overlap_csv,
        incidence_path=args.incidence_path,
        output_dir=args.output_dir,
        jaccard_mode=args.jaccard_mode,
        jaccard_csv=args.jaccard_csv,
        sheet_name=args.sheet_name,
        case_id_column=args.case_id_column,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        global_feature_min_cases=args.global_feature_min_cases,
        endpoint_mode=args.endpoint_mode,
        significance_column=args.significance_column,
        specific_case_a=args.specific_case_a,
        specific_case_e=args.specific_case_e,
        chain_length_mode=args.chain_length_mode,
        chain_length=args.chain_length,
        min_chain_length=args.min_chain_length,
        max_chain_length=args.max_chain_length,
        search_mode=args.search_mode,
        min_adj=args.min_adj,
        min_intersection=args.min_intersection,
        beam_width=args.beam_width,
        top_results_per_endpoint=args.top_results_per_endpoint,
        top_results_total=args.top_results_total,
        out_csv_name=args.out_csv,
        out_summary_name=args.out_summary,
    )

    print("[✓] Case gradient search complete.")
    print(f"    Endpoint pairs:     {result['endpoint_pairs']}")
    print(f"    Pairs with results: {result['endpoint_pairs_with_results']}")
    print(f"    Rows written:       {result['rows_written']}")
    if result["computed_jaccard_csv"] is not None:
        print(f"    Jaccard CSV:        {result['computed_jaccard_csv']}")
    print(f"    Output CSV:         {result['gradients_csv']}")
    print(f"    Summary:            {result['summary_txt']}")


if __name__ == "__main__":
    main()