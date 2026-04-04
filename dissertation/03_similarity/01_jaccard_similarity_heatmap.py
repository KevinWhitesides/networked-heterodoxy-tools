#!/usr/bin/env python3
"""
01_jaccard_similarity_heatmap.py

Compute a case × case Jaccard similarity matrix from a binary incidence matrix
(case × feature/trope), optionally filter features by minimum frequency, and
export:

1) Jaccard similarity matrix as CSV
2) Heatmap as PNG (optional)
3) analysis_summary.txt

Standalone use:
    Edit the CONFIG block below, then run:
        python 01_jaccard_similarity_heatmap.py

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
from sklearn.metrics import jaccard_score


# =============================================================================
# CONFIG (standalone defaults)
# Edit these values for normal standalone use.
# Pipeline runners can override them via run(...) or CLI arguments.
# =============================================================================

INPUT_PATH = Path("first_7_books.xlsx")

# Column containing case names (book titles, song IDs, etc.).
# If None, the script will prefer "Source Title", then "Title", then row numbers.
TITLE_COL: Optional[str] = None

# Number of metadata columns before feature columns begin.
# Example: if features begin at column E, then N_METADATA_COLS = 4.
N_METADATA_COLS = 4

# Presence token marking feature presence in the matrix
PRESENCE_TOKEN = "X"

# Keep only features appearing in at least this many cases
MIN_FEATURE_FREQ = 2

# Output directory
OUTPUT_DIR = Path(".")

# Output filenames
# If OUT_CSV or OUT_PNG is set to None, the script will generate a default name
# using MIN_FEATURE_FREQ, e.g. jaccard_similarity_thr2.csv
OUT_CSV: Optional[str] = None
OUT_PNG: Optional[str] = None
OUT_SUMMARY = "analysis_summary.txt"

# Plot options
MAKE_PLOT = True
FIGSIZE = (18, 15)
DPI = 300
ANNOTATE_CELLS = False
ANNOT_FORMAT = ".2f"

# Plot readability options
CLUSTER_FOR_PLOT = True
X_LABEL_ROTATION = 45
X_LABEL_FONTSIZE = 8
Y_LABEL_FONTSIZE = 8
MAX_LABEL_LEN: Optional[int] = 38  # None disables abbreviation for plotting


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


def _build_incidence(df: pd.DataFrame, n_metadata_cols: int, token: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a binary incidence matrix (cases × features).

    Metadata columns are allowed; the first n_metadata_cols columns are ignored
    when constructing the feature matrix.
    """
    if n_metadata_cols < 0 or n_metadata_cols >= df.shape[1]:
        raise ValueError(
            f"N_METADATA_COLS={n_metadata_cols} is invalid for a table with {df.shape[1]} columns."
        )

    feature_cols = list(df.columns[n_metadata_cols:])
    if not feature_cols:
        raise ValueError("No feature columns found. Check N_METADATA_COLS and your input file.")

    incidence = df[feature_cols].eq(token).astype(int)
    return incidence, feature_cols


def _filter_features_by_freq(incidence: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """Keep only features appearing in at least min_freq cases."""
    if min_freq <= 1:
        return incidence

    freq = incidence.sum(axis=0)
    return incidence.loc[:, freq >= min_freq]


def _resolve_output_names(
    min_feature_freq: int,
    out_csv_name: Optional[str],
    out_png_name: Optional[str],
) -> tuple[str, str]:
    """Resolve default output filenames, including threshold when not explicitly provided."""
    if out_csv_name is None:
        out_csv_name = f"jaccard_similarity_thr{min_feature_freq}.csv"
    if out_png_name is None:
        out_png_name = f"jaccard_similarity_thr{min_feature_freq}.png"
    return out_csv_name, out_png_name


def _compute_jaccard_matrix(incidence: pd.DataFrame, case_names: list[str]) -> pd.DataFrame:
    """Compute a symmetric case × case Jaccard similarity matrix."""
    n_cases = incidence.shape[0]
    if n_cases != len(case_names):
        raise ValueError("case_names length does not match number of rows in incidence matrix.")

    mat = np.zeros((n_cases, n_cases), dtype=float)

    row_sums = incidence.sum(axis=1).to_numpy()
    all_zero = row_sums == 0
    X = incidence.to_numpy(dtype=int)

    for i in range(n_cases):
        mat[i, i] = 0.0 if all_zero[i] else 1.0

        for j in range(i + 1, n_cases):
            if all_zero[i] and all_zero[j]:
                sim = 0.0
            else:
                sim = float(jaccard_score(X[i], X[j]))
            mat[i, j] = sim
            mat[j, i] = sim

    return pd.DataFrame(mat, index=case_names, columns=case_names)


def _abbreviate_label(label: str, max_len: Optional[int]) -> str:
    """Abbreviate long plot labels for readability."""
    if max_len is None or len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + "…"


def _reorder_for_plot(jaccard_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder the similarity matrix for plotting using hierarchical clustering.
    Reordering affects the PNG heatmap only, not the CSV export.
    """
    if jaccard_df.shape[0] <= 2:
        return jaccard_df

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except Exception:
        return jaccard_df

    data = jaccard_df.to_numpy(dtype=float)

    # Convert similarity to distance; clip for numerical safety
    dist = 1.0 - data
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    # Condensed distance for linkage
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    return jaccard_df.iloc[order, order]


def _plot_heatmap(
    jaccard_df: pd.DataFrame,
    out_png: Path,
    *,
    min_feature_freq: int,
    figsize: tuple[float, float],
    dpi: int,
    annotate_cells: bool,
    annot_format: str,
    cluster_for_plot: bool,
    x_label_rotation: float,
    x_label_fontsize: float,
    y_label_fontsize: float,
    max_label_len: Optional[int],
) -> None:
    """Plot and save the heatmap."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
        use_seaborn = True
    except Exception:
        use_seaborn = False

    plot_df = _reorder_for_plot(jaccard_df) if cluster_for_plot else jaccard_df
    plot_labels_x = [_abbreviate_label(x, max_label_len) for x in plot_df.columns.tolist()]
    plot_labels_y = [_abbreviate_label(y, max_label_len) for y in plot_df.index.tolist()]
    data = plot_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if use_seaborn:
        sns.heatmap(
            plot_df.astype(float),
            annot=annotate_cells,
            fmt=annot_format,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.85},
            xticklabels=plot_labels_x,
            yticklabels=plot_labels_y,
        )
    else:
        im = ax.imshow(data, vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if annotate_cells:
            for (i, j), val in np.ndenumerate(data):
                ax.text(j, i, format(val, annot_format), ha="center", va="center")

        ax.set_xticks(range(plot_df.shape[1]))
        ax.set_yticks(range(plot_df.shape[0]))
        ax.set_xticklabels(plot_labels_x)
        ax.set_yticklabels(plot_labels_y)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_label_rotation, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(x_label_fontsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(y_label_fontsize)

    ax.set_title(f"Jaccard Similarity (features with freq ≥ {min_feature_freq})")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_path: Path,
    title_col: Optional[str],
    n_metadata_cols: int,
    presence_token: str,
    min_feature_freq: int,
    make_plot: bool,
    n_cases: int,
    n_features_total: int,
    n_features_kept: int,
    out_csv: Path,
    out_png: Optional[Path],
) -> None:
    """Write a plain-text summary of the run."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Jaccard Similarity Heatmap Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Inputs\n")
        f.write("------\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Title column: {title_col if title_col is not None else '(auto / fallback)'}\n")
        f.write(f"N_METADATA_COLS: {n_metadata_cols}\n")
        f.write(f"Presence token: {presence_token}\n\n")

        f.write("Settings\n")
        f.write("--------\n")
        f.write(f"MIN_FEATURE_FREQ: {min_feature_freq}\n")
        f.write(f"MAKE_PLOT: {make_plot}\n\n")

        f.write("Dataset summary\n")
        f.write("---------------\n")
        f.write(f"Cases (rows): {n_cases}\n")
        f.write(f"Features before filtering: {n_features_total}\n")
        f.write(f"Features after filtering: {n_features_kept}\n\n")

        f.write("Outputs\n")
        f.write("-------\n")
        f.write(f"Jaccard matrix CSV: {out_csv.name}\n")
        if out_png is not None:
            f.write(f"Heatmap PNG: {out_png.name}\n")
        else:
            f.write("Heatmap PNG: (not generated)\n")


# =============================================================================
# Pipeline-ready entry point
# =============================================================================

def run(
    *,
    input_path: Path,
    output_dir: Path,
    title_col: Optional[str] = None,
    n_metadata_cols: int = 0,
    presence_token: str = "X",
    min_feature_freq: int = 2,
    out_csv_name: Optional[str] = None,
    out_png_name: Optional[str] = None,
    out_summary_name: str = "analysis_summary.txt",
    make_plot: bool = True,
    figsize: tuple[float, float] = (18, 15),
    dpi: int = 300,
    annotate_cells: bool = False,
    annot_format: str = ".2f",
    cluster_for_plot: bool = True,
    x_label_rotation: float = 45,
    x_label_fontsize: float = 8,
    y_label_fontsize: float = 8,
    max_label_len: Optional[int] = 38,
) -> Dict[str, Any]:
    """
    Run the analysis and return a structured result dictionary.

    This is the entry point pipeline runners should call.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = _read_table(input_path)
    chosen_title_col = _choose_title_col(df, title_col)
    case_names = _extract_case_names(df, chosen_title_col)

    incidence, feature_cols = _build_incidence(df, n_metadata_cols, presence_token)
    n_features_total = len(feature_cols)

    filtered = _filter_features_by_freq(incidence, min_feature_freq)
    n_features_kept = filtered.shape[1]

    jaccard_df = _compute_jaccard_matrix(filtered, case_names)

    resolved_out_csv_name, resolved_out_png_name = _resolve_output_names(
        min_feature_freq=min_feature_freq,
        out_csv_name=out_csv_name,
        out_png_name=out_png_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / resolved_out_csv_name
    out_png = output_dir / resolved_out_png_name
    out_summary = output_dir / out_summary_name

    # CSV stays in original case order
    jaccard_df.to_csv(out_csv, encoding="utf-8")
    clustered_df = _reorder_for_plot(jaccard_df)
    clustered_df.to_csv(out_csv.with_name(out_csv.stem + "_clustered.csv"), encoding="utf-8")

    png_path: Optional[Path] = None
    if make_plot:
        _plot_heatmap(
            jaccard_df,
            out_png,
            min_feature_freq=min_feature_freq,
            figsize=figsize,
            dpi=dpi,
            annotate_cells=annotate_cells,
            annot_format=annot_format,
            cluster_for_plot=cluster_for_plot,
            x_label_rotation=x_label_rotation,
            x_label_fontsize=x_label_fontsize,
            y_label_fontsize=y_label_fontsize,
            max_label_len=max_label_len,
        )
        png_path = out_png

    _write_summary(
        out_path=out_summary,
        run_timestamp=run_timestamp,
        input_path=input_path,
        title_col=chosen_title_col,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        min_feature_freq=min_feature_freq,
        make_plot=make_plot,
        n_cases=len(case_names),
        n_features_total=n_features_total,
        n_features_kept=n_features_kept,
        out_csv=out_csv,
        out_png=png_path,
    )

    return {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "title_col": chosen_title_col,
        "n_cases": len(case_names),
        "n_features_total": n_features_total,
        "n_features_kept": n_features_kept,
        "jaccard_csv": str(out_csv),
        "heatmap_png": str(out_png) if png_path is not None else None,
        "summary_txt": str(out_summary),
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute a case × case Jaccard similarity matrix from a binary incidence matrix."
    )

    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Path to input .xlsx/.xls or .csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--title-col", type=str, default=TITLE_COL, help="Column containing case names")
    parser.add_argument(
        "--n-metadata-cols",
        type=int,
        default=N_METADATA_COLS,
        help="Number of metadata columns before features begin",
    )
    parser.add_argument(
        "--presence-token",
        type=str,
        default=PRESENCE_TOKEN,
        help="Presence token in the matrix",
    )
    parser.add_argument(
        "--min-feature-freq",
        type=int,
        default=MIN_FEATURE_FREQ,
        help="Keep only features appearing in at least this many cases",
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default=OUT_CSV,
        help="Output CSV filename (default: auto-generated from MIN_FEATURE_FREQ)",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=OUT_PNG,
        help="Output PNG filename (default: auto-generated from MIN_FEATURE_FREQ)",
    )
    parser.add_argument("--out-summary", type=str, default=OUT_SUMMARY, help="Output summary filename")

    parser.add_argument("--no-plot", action="store_true", help="Disable heatmap generation")
    parser.add_argument("--fig-width", type=float, default=FIGSIZE[0], help="Figure width")
    parser.add_argument("--fig-height", type=float, default=FIGSIZE[1], help="Figure height")
    parser.add_argument("--dpi", type=int, default=DPI, help="Heatmap DPI")
    parser.add_argument("--annotate", action="store_true", help="Enable cell annotations")
    parser.add_argument("--annot-format", type=str, default=ANNOT_FORMAT, help="Annotation format string")
    parser.add_argument("--no-cluster-for-plot", action="store_true", help="Do not reorder heatmap for plotting")
    parser.add_argument("--x-label-rotation", type=float, default=X_LABEL_ROTATION, help="X-axis label rotation")
    parser.add_argument("--x-label-fontsize", type=float, default=X_LABEL_FONTSIZE, help="X-axis label font size")
    parser.add_argument("--y-label-fontsize", type=float, default=Y_LABEL_FONTSIZE, help="Y-axis label font size")
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
        input_path=args.input,
        output_dir=args.output_dir,
        title_col=args.title_col,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        min_feature_freq=args.min_feature_freq,
        out_csv_name=args.out_csv,
        out_png_name=args.out_png,
        out_summary_name=args.out_summary,
        make_plot=not args.no_plot,
        figsize=(args.fig_width, args.fig_height),
        dpi=args.dpi,
        annotate_cells=args.annotate,
        annot_format=args.annot_format,
        cluster_for_plot=not args.no_cluster_for_plot,
        x_label_rotation=args.x_label_rotation,
        x_label_fontsize=args.x_label_fontsize,
        y_label_fontsize=args.y_label_fontsize,
        max_label_len=max_label_len,
    )

    print("[✓] Jaccard similarity analysis complete.")
    print(f"    Input:               {result['input_path']}")
    print(f"    Cases (rows):        {result['n_cases']:,}")
    print(f"    Features (total):    {result['n_features_total']:,}")
    print(f"    Features kept:       {result['n_features_kept']:,}")
    print(f"    Matrix CSV:          {result['jaccard_csv']}")
    if result["heatmap_png"] is not None:
        print(f"    Heatmap PNG:         {result['heatmap_png']}")
    else:
        print("    Heatmap PNG:         (not generated)")
    print(f"    Summary:             {result['summary_txt']}")


if __name__ == "__main__":
    main()