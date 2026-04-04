#!/usr/bin/env python3
"""
02_cluster_from_jaccard.py

Cluster cases from a square similarity matrix (case × case) by:

1) converting similarity to distance
2) performing hierarchical clustering
3) assigning each case to a cluster
4) writing cluster outputs and an optional dendrogram

Outputs:

1) cluster summary CSV (average intra-cluster similarity)
2) per-case mean similarity-to-cluster CSV
3) dendrogram PNG (optional)
4) analysis_summary.txt

This script supports TWO clustering modes. Use ONLY ONE at a time:

1) Fixed-k mode:
   - Set N_CLUSTERS to an integer, such as 3
   - Set DISTANCE_CUTOFF = None

   Example:
       N_CLUSTERS = 3
       DISTANCE_CUTOFF = None

   Meaning:
       Force the cases into exactly 3 clusters.

2) Distance-cutoff mode:
   - Set DISTANCE_CUTOFF to a number, such as 0.65
   - Set N_CLUSTERS = None

   Example:
       N_CLUSTERS = None
       DISTANCE_CUTOFF = 0.65

   Meaning:
       Cut the dendrogram at distance 0.65 and let the number of clusters emerge.

For Jaccard-based similarity matrices:
    distance = 1 - similarity

So, for example:
    DISTANCE_CUTOFF = 0.65
means:
    similarity >= 0.35 is required for cases to remain grouped together.

Standalone use:
    Edit the CONFIG block below, then run:
        python 02_cluster_from_jaccard.py

Pipeline / programmatic use:
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# =============================================================================
# CONFIG (standalone defaults)
# Edit these values for normal standalone use.
# Pipeline runners can override them via run(...) or CLI arguments.
# =============================================================================

# Input similarity matrix CSV (square; same labels on rows/cols)
INPUT_SIM_CSV = Path("7book_jaccard_threshold2.csv")

# Similarity → distance transform
# For Jaccard similarity, distance = 1 - similarity.
DISTANCE_MODE = "one_minus"

# Linkage method.
# For precomputed distances like 1-Jaccard, "average" or "complete" are usually
# easier to justify than "ward".
LINKAGE_METHOD = "average"  # options: "average", "complete", "single", "ward"

# ---------------------------------------------------------------------------
# Choose EXACTLY ONE clustering mode by setting one parameter and setting the
# other to None.
#
# Fixed-k mode:
#     N_CLUSTERS = 3
#     DISTANCE_CUTOFF = None
#
# Distance-cutoff mode:
#     N_CLUSTERS = None
#     DISTANCE_CUTOFF = 0.65
#
# Do NOT set both to a value at the same time.
# ---------------------------------------------------------------------------

# Fixed number of clusters (use an integer, or set to None)
N_CLUSTERS: Optional[int] = 3

# Distance threshold for cutting the dendrogram (use a float, or set to None)
DISTANCE_CUTOFF: Optional[float] = None

# Output directory
OUTPUT_DIR = Path(".")

# Output filenames
# If these are set to None, default names will be generated automatically
# based on the clustering rule, e.g.:
#   cluster_summary_k3.csv
#   case_similarity_to_cluster_k3.csv
#   dendrogram_k3.png
#
# or, for cutoff mode:
#   cluster_summary_d0_65.csv
#   case_similarity_to_cluster_d0_65.csv
#   dendrogram_d0_65.png
OUT_CLUSTER_SUMMARY: Optional[str] = None
OUT_CASE_TO_CLUSTER: Optional[str] = None
OUT_DENDROGRAM_PNG: Optional[str] = None
OUT_SUMMARY = "analysis_summary.txt"

# Dendrogram options
WRITE_DENDROGRAM = True
DENDROGRAM_FIGSIZE = (20, 10)
DPI = 300

# Dendrogram readability options
LEAF_ROTATION = 60
LEAF_FONTSIZE = 8
MAX_LABEL_LEN: Optional[int] = 38  # None disables abbreviation


# =============================================================================
# Helpers
# =============================================================================

def _read_similarity_matrix(path: Path) -> pd.DataFrame:
    """Read a square similarity matrix CSV with row labels in column 0."""
    if not path.exists():
        raise FileNotFoundError(f"Input similarity CSV not found: {path}")

    try:
        df = pd.read_csv(path, index_col=0)
    except UnicodeDecodeError:
        df = pd.read_csv(path, index_col=0, encoding="latin-1")

    df.index = df.index.map(str)
    df.columns = df.columns.map(str)
    return df


def _validate_square_matrix(df: pd.DataFrame) -> None:
    """Validate that the input is a square similarity matrix."""
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Matrix must be square. Got shape {df.shape}.")

    if list(df.index) != list(df.columns):
        raise ValueError(
            "Row labels and column labels do not match. "
            "Similarity matrix should have identical row/column labels in the same order."
        )

    vals = df.to_numpy(dtype=float)

    if np.any(vals < 0) or np.any(vals > 1):
        raise ValueError("Similarity values should be in [0, 1].")

    diag = np.diag(vals)
    if not np.allclose(diag, 1.0, atol=1e-6):
        raise ValueError("Diagonal should be 1.0 for a similarity matrix.")


def _similarity_to_distance(sim: np.ndarray, mode: str) -> np.ndarray:
    """Convert similarity matrix to distance matrix."""
    if mode == "one_minus":
        dist = 1.0 - sim
    else:
        raise ValueError(f"Unsupported DISTANCE_MODE: {mode}")

    np.fill_diagonal(dist, 0.0)
    return dist


def _choose_cluster_rule(
    n_clusters: Optional[int],
    distance_cutoff: Optional[float],
) -> tuple[str, float, str, str]:
    """
    Validate and resolve the cluster-assignment rule.

    Returns:
        criterion, threshold, human-readable description, filename suffix
    """
    if n_clusters is not None and distance_cutoff is not None:
        raise ValueError(
            "Set only one clustering mode:\n"
            "- For fixed-k mode, set N_CLUSTERS to an integer and DISTANCE_CUTOFF = None\n"
            "- For distance-cutoff mode, set DISTANCE_CUTOFF to a float and N_CLUSTERS = None"
        )

    if n_clusters is None and distance_cutoff is None:
        raise ValueError(
            "No clustering mode was selected.\n"
            "Set one of the following:\n"
            "- N_CLUSTERS = 3 and DISTANCE_CUTOFF = None\n"
            "- N_CLUSTERS = None and DISTANCE_CUTOFF = 0.65"
        )

    if distance_cutoff is not None:
        suffix = f"d{distance_cutoff:.2f}".replace(".", "_")
        return "distance", float(distance_cutoff), f"distance cutoff ≤ {distance_cutoff}", suffix

    k = int(n_clusters)
    if k < 1:
        raise ValueError("N_CLUSTERS must be at least 1.")

    suffix = f"k{k}"
    return "maxclust", float(k), f"maxclust = {k}", suffix


def _resolve_output_names(
    suffix: str,
    out_cluster_summary_name: Optional[str],
    out_case_to_cluster_name: Optional[str],
    out_dendrogram_name: Optional[str],
) -> tuple[str, str, str]:
    """Resolve default output filenames based on the clustering rule."""
    if out_cluster_summary_name is None:
        out_cluster_summary_name = f"cluster_summary_{suffix}.csv"

    if out_case_to_cluster_name is None:
        out_case_to_cluster_name = f"case_similarity_to_cluster_{suffix}.csv"

    if out_dendrogram_name is None:
        out_dendrogram_name = f"dendrogram_{suffix}.png"

    return (
        out_cluster_summary_name,
        out_case_to_cluster_name,
        out_dendrogram_name,
    )


def _avg_intra_cluster_similarity(sim_df: pd.DataFrame, members: list[str]) -> float:
    """Average pairwise similarity among cases within a cluster."""
    sub = sim_df.loc[members, members].to_numpy(dtype=float)

    if sub.shape[0] <= 1:
        return float("nan")

    mask = ~np.eye(sub.shape[0], dtype=bool)
    return float(sub[mask].mean())


def _case_mean_similarity_to_cluster(sim_df: pd.DataFrame, case: str, members: list[str]) -> float:
    """Mean similarity of one case to the other cases in its cluster."""
    others = [m for m in members if m != case]
    if not others:
        return float("nan")
    return float(sim_df.loc[case, others].mean())


def _abbreviate_label(label: str, max_len: Optional[int]) -> str:
    """Abbreviate long plot labels for readability."""
    if max_len is None or len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + "…"


def _write_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str],
    out_png: Path,
    *,
    linkage_method: str,
    figsize: tuple[float, float],
    dpi: int,
    leaf_rotation: float,
    leaf_font_size: float,
    max_label_len: Optional[int],
) -> None:
    """Write a dendrogram PNG."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    plot_labels = [_abbreviate_label(label, max_label_len) for label in labels]

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=plot_labels,
        leaf_rotation=0,  # disable internal rotation
        leaf_font_size=leaf_font_size,
        ax=ax,
    )
    
    # Explicit control over label positioning
    import matplotlib.pyplot as plt
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method})")
    ax.set_xlabel("")
    ax.set_ylabel("Distance")

    fig.subplots_adjust(bottom=0.40, left=0.08, right=0.98, top=0.92)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_sim_csv: Path,
    distance_mode: str,
    linkage_method: str,
    rule_desc: str,
    write_dendrogram: bool,
    n_cases: int,
    n_clusters_found: int,
    out_cluster_summary: Path,
    out_case_to_cluster: Path,
    out_dendrogram: Optional[Path],
) -> None:
    """Write a plain-text summary of the run."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Cluster from Similarity Matrix Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Input similarity CSV: {input_sim_csv}\n\n")

        f.write("Settings\n")
        f.write("--------\n")
        f.write(f"DISTANCE_MODE: {distance_mode}\n")
        f.write(f"LINKAGE_METHOD: {linkage_method}\n")
        f.write(f"Cluster rule: {rule_desc}\n")
        f.write(f"WRITE_DENDROGRAM: {write_dendrogram}\n\n")

        f.write("Dataset summary\n")
        f.write("---------------\n")
        f.write(f"Cases: {n_cases}\n")
        f.write(f"Clusters found: {n_clusters_found}\n\n")

        f.write("Outputs\n")
        f.write("-------\n")
        f.write(f"Cluster summary CSV: {out_cluster_summary.name}\n")
        f.write(f"Case similarity-to-cluster CSV: {out_case_to_cluster.name}\n")
        if out_dendrogram is not None:
            f.write(f"Dendrogram PNG: {out_dendrogram.name}\n")
        else:
            f.write("Dendrogram PNG: (not generated)\n")


# =============================================================================
# Pipeline-ready entry point
# =============================================================================

def run(
    *,
    input_sim_csv: Path,
    output_dir: Path,
    distance_mode: str = "one_minus",
    linkage_method: str = "average",
    n_clusters: Optional[int] = 3,
    distance_cutoff: Optional[float] = None,
    out_cluster_summary_name: Optional[str] = None,
    out_case_to_cluster_name: Optional[str] = None,
    out_dendrogram_name: Optional[str] = None,
    out_summary_name: str = "analysis_summary.txt",
    write_dendrogram: bool = True,
    dendrogram_figsize: tuple[float, float] = (20, 10),
    dpi: int = 300,
    leaf_rotation: float = 60,
    leaf_font_size: float = 8,
    max_label_len: Optional[int] = 38,
) -> Dict[str, Any]:
    """
    Run hierarchical clustering from a similarity matrix and return a structured
    result dictionary.

    This is the entry point pipeline runners should call.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sim_df = _read_similarity_matrix(input_sim_csv)
    _validate_square_matrix(sim_df)

    labels = list(sim_df.index)
    sim = sim_df.to_numpy(dtype=float)

    dist_sq = _similarity_to_distance(sim, distance_mode)
    condensed = squareform(dist_sq, checks=True)
    link = linkage(condensed, method=linkage_method)

    criterion, threshold, rule_desc, suffix = _choose_cluster_rule(n_clusters, distance_cutoff)

    (
        resolved_out_cluster_summary_name,
        resolved_out_case_to_cluster_name,
        resolved_out_dendrogram_name,
    ) = _resolve_output_names(
        suffix=suffix,
        out_cluster_summary_name=out_cluster_summary_name,
        out_case_to_cluster_name=out_case_to_cluster_name,
        out_dendrogram_name=out_dendrogram_name,
    )

    clusters = fcluster(link, t=threshold, criterion=criterion)
    cluster_df = pd.DataFrame({"Case": labels, "Cluster": clusters})

    output_dir.mkdir(parents=True, exist_ok=True)

    out_cluster_summary = output_dir / resolved_out_cluster_summary_name
    out_case_to_cluster = output_dir / resolved_out_case_to_cluster_name
    out_dendrogram = output_dir / resolved_out_dendrogram_name
    out_summary = output_dir / out_summary_name

    cluster_summaries: list[dict[str, object]] = []
    case_fit_rows: list[dict[str, object]] = []

    for cid in sorted(cluster_df["Cluster"].unique()):
        members = cluster_df.loc[cluster_df["Cluster"] == cid, "Case"].tolist()
        avg_sim = _avg_intra_cluster_similarity(sim_df, members)

        cluster_summaries.append(
            {
                "Cluster": cid,
                "n_cases": len(members),
                "avg_intra_similarity": avg_sim,
            }
        )

        for case in members:
            mean_to_cluster = _case_mean_similarity_to_cluster(sim_df, case, members)
            case_fit_rows.append(
                {
                    "Case": case,
                    "Cluster": cid,
                    "avg_similarity_to_cluster": mean_to_cluster,
                }
            )

    cluster_summary_df = pd.DataFrame(cluster_summaries)
    case_fit_df = pd.DataFrame(case_fit_rows)

    cluster_summary_df.to_csv(out_cluster_summary, index=False, encoding="utf-8")
    case_fit_df.to_csv(out_case_to_cluster, index=False, encoding="utf-8")

    dendrogram_path: Optional[Path] = None
    if write_dendrogram:
        _write_dendrogram(
            link,
            labels,
            out_dendrogram,
            linkage_method=linkage_method,
            figsize=dendrogram_figsize,
            dpi=dpi,
            leaf_rotation=leaf_rotation,
            leaf_font_size=leaf_font_size,
            max_label_len=max_label_len,
        )
        dendrogram_path = out_dendrogram

    _write_summary(
        out_path=out_summary,
        run_timestamp=run_timestamp,
        input_sim_csv=input_sim_csv,
        distance_mode=distance_mode,
        linkage_method=linkage_method,
        rule_desc=rule_desc,
        write_dendrogram=write_dendrogram,
        n_cases=len(labels),
        n_clusters_found=int(cluster_df["Cluster"].nunique()),
        out_cluster_summary=out_cluster_summary,
        out_case_to_cluster=out_case_to_cluster,
        out_dendrogram=dendrogram_path,
    )

    return {
        "run_timestamp": run_timestamp,
        "input_sim_csv": str(input_sim_csv),
        "output_dir": str(output_dir),
        "distance_mode": distance_mode,
        "linkage_method": linkage_method,
        "cluster_rule": rule_desc,
        "n_cases": len(labels),
        "n_clusters_found": int(cluster_df["Cluster"].nunique()),
        "cluster_summary_csv": str(out_cluster_summary),
        "case_similarity_to_cluster_csv": str(out_case_to_cluster),
        "dendrogram_png": str(out_dendrogram) if dendrogram_path is not None else None,
        "summary_txt": str(out_summary),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster cases from a square similarity matrix."
    )

    parser.add_argument(
        "--input-sim-csv",
        type=Path,
        default=INPUT_SIM_CSV,
        help="Path to square similarity matrix CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs",
    )
    parser.add_argument(
        "--distance-mode",
        type=str,
        default=DISTANCE_MODE,
        help='Similarity-to-distance transform (currently only "one_minus")',
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default=LINKAGE_METHOD,
        help='Hierarchical clustering linkage method (e.g. "average", "complete", "single", "ward")',
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=N_CLUSTERS,
        help=(
            "Fixed number of clusters. "
            "Use this mode by setting --n-clusters to an integer and leaving --distance-cutoff unset."
        ),
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=DISTANCE_CUTOFF,
        help=(
            "Distance threshold for cutting the dendrogram. "
            "Use this mode by setting --distance-cutoff to a float and leaving --n-clusters unset."
        ),
    )

    parser.add_argument(
        "--out-cluster-summary",
        type=str,
        default=OUT_CLUSTER_SUMMARY,
        help="Output filename for cluster summary CSV (default: auto-generated from clustering rule)",
    )
    parser.add_argument(
        "--out-case-to-cluster",
        type=str,
        default=OUT_CASE_TO_CLUSTER,
        help="Output filename for per-case similarity-to-cluster CSV (default: auto-generated from clustering rule)",
    )
    parser.add_argument(
        "--out-dendrogram",
        type=str,
        default=OUT_DENDROGRAM_PNG,
        help="Output filename for dendrogram PNG (default: auto-generated from clustering rule)",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=OUT_SUMMARY,
        help="Output filename for analysis summary",
    )

    parser.add_argument(
        "--no-dendrogram",
        action="store_true",
        help="Disable dendrogram generation",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=DENDROGRAM_FIGSIZE[0],
        help="Dendrogram figure width",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=DENDROGRAM_FIGSIZE[1],
        help="Dendrogram figure height",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="Dendrogram DPI",
    )
    parser.add_argument(
        "--leaf-rotation",
        type=float,
        default=LEAF_ROTATION,
        help="Leaf label rotation in degrees",
    )
    parser.add_argument(
        "--leaf-font-size",
        type=float,
        default=LEAF_FONTSIZE,
        help="Leaf label font size",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=(MAX_LABEL_LEN if MAX_LABEL_LEN is not None else 0),
        help="Maximum label length for plotting (0 disables abbreviation)",
    )

    return parser.parse_args()


# =============================================================================
# Main (standalone execution)
# =============================================================================

def main() -> None:
    """Run the script using CONFIG defaults or CLI overrides."""
    args = _parse_args()

    max_label_len: Optional[int]
    if args.max_label_len == 0:
        max_label_len = None
    else:
        max_label_len = args.max_label_len

    result = run(
        input_sim_csv=args.input_sim_csv,
        output_dir=args.output_dir,
        distance_mode=args.distance_mode,
        linkage_method=args.linkage_method,
        n_clusters=args.n_clusters,
        distance_cutoff=args.distance_cutoff,
        out_cluster_summary_name=args.out_cluster_summary,
        out_case_to_cluster_name=args.out_case_to_cluster,
        out_dendrogram_name=args.out_dendrogram,
        out_summary_name=args.out_summary,
        write_dendrogram=not args.no_dendrogram,
        dendrogram_figsize=(args.fig_width, args.fig_height),
        dpi=args.dpi,
        leaf_rotation=args.leaf_rotation,
        leaf_font_size=args.leaf_font_size,
        max_label_len=max_label_len,
    )

    print("[✓] Clustering complete.")
    print(f"    Input similarity CSV:   {result['input_sim_csv']}")
    print(f"    Linkage method:         {result['linkage_method']}")
    print(f"    Cluster rule:           {result['cluster_rule']}")
    print(f"    Clusters found:         {result['n_clusters_found']}")
    print(f"    Cluster summary CSV:    {result['cluster_summary_csv']}")
    print(f"    Case-fit CSV:           {result['case_similarity_to_cluster_csv']}")
    if result["dendrogram_png"] is not None:
        print(f"    Dendrogram PNG:         {result['dendrogram_png']}")
    else:
        print("    Dendrogram PNG:         (not generated)")
    print(f"    Summary:                {result['summary_txt']}")


if __name__ == "__main__":
    main()