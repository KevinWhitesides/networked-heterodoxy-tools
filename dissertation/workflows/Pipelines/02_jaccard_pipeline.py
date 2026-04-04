#!/usr/bin/env python3
"""
02_jaccard_pipeline.py

Run the Jaccard → clustering pipeline:

1) Compute a case × case Jaccard similarity matrix from a binary incidence matrix
2) Cluster cases from that similarity matrix

This pipeline is designed for datasets such as books × tropes, songs × tropes,
or other binary incidence matrices where metadata columns may appear before
feature columns.

Pipeline stages:
    Stage 1: 03_similarity/01_jaccard_similarity_heatmap.py
    Stage 2: 03_similarity/02_cluster_from_jaccard.py

Standalone use:
    Edit the CONFIG block below, then run:
        python 02_run_jaccard_clustering_pipeline.py

This pipeline writes:
    - stage-specific outputs in separate subfolders
    - a top-level pipeline_summary.txt with a timestamp and output inventory
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# CONFIG (standalone defaults)
# Edit these values for normal standalone use.
# =============================================================================

# Input binary incidence matrix
INPUT_PATH = Path("sample_files/full_workflow_sample_data.xlsx")

# Shared matrix settings
TITLE_COL: Optional[str] = None
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Pipeline output root.
# If PIPELINE_OUTPUT_DIR is ".", the pipeline creates a timestamped subfolder
# in the current working directory.
PIPELINE_OUTPUT_DIR = Path(".")

# Stage 1: Jaccard settings
MIN_FEATURE_FREQ = 2
MAKE_HEATMAP = True
HEATMAP_FIGSIZE = (18, 15)
HEATMAP_DPI = 300
ANNOTATE_CELLS = False
ANNOT_FORMAT = ".2f"

# Stage 1: heatmap readability options
CLUSTER_FOR_PLOT = True
X_LABEL_ROTATION = 45
X_LABEL_FONTSIZE = 8
Y_LABEL_FONTSIZE = 8
HEATMAP_MAX_LABEL_LEN: Optional[int] = 38

# Stage 2: Clustering settings
DISTANCE_MODE = "one_minus"
LINKAGE_METHOD = "average"

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
N_CLUSTERS: Optional[int] = None
DISTANCE_CUTOFF: Optional[float] = 0.90

WRITE_DENDROGRAM = True
DENDROGRAM_FIGSIZE = (20, 10)
DENDROGRAM_DPI = 300

# Stage 2: dendrogram readability options
LEAF_ROTATION = 60
LEAF_FONTSIZE = 8
DENDROGRAM_MAX_LABEL_LEN: Optional[int] = 38

# Pipeline summary filename
PIPELINE_SUMMARY_NAME = "pipeline_summary.txt"


# =============================================================================
# Helpers
# =============================================================================

def _load_module(module_path: Path, module_name: str):
    """Dynamically load a Python module from a file path."""
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_pipeline_output_dir(base_dir: Path) -> Path:
    """Create a timestamped default pipeline output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"02_jaccard_clustering_pipeline_{timestamp}"


def _resolve_pipeline_output_dir(output_dir: Path) -> Path:
    """
    Resolve the pipeline output directory.

    If output_dir == ".", create a timestamped subfolder in the current directory.
    Otherwise, use the provided directory directly.
    """
    if output_dir == Path("."):
        return _default_pipeline_output_dir(Path("."))
    return output_dir


def _write_pipeline_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_path: Path,
    pipeline_output_dir: Path,
    title_col: Optional[str],
    n_metadata_cols: int,
    presence_token: str,
    min_feature_freq: int,
    make_heatmap: bool,
    distance_mode: str,
    linkage_method: str,
    n_clusters: Optional[int],
    distance_cutoff: Optional[float],
    write_dendrogram: bool,
    stage1_result: Dict[str, Any],
    stage2_result: Dict[str, Any],
) -> None:
    """Write a top-level pipeline summary."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Jaccard → Clustering Pipeline Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Pipeline inputs\n")
        f.write("---------------\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Pipeline output directory: {pipeline_output_dir}\n\n")

        f.write("Shared matrix settings\n")
        f.write("----------------------\n")
        f.write(f"TITLE_COL: {title_col if title_col is not None else '(auto / fallback)'}\n")
        f.write(f"N_METADATA_COLS: {n_metadata_cols}\n")
        f.write(f"PRESENCE_TOKEN: {presence_token}\n\n")

        f.write("Stage 1: Jaccard settings\n")
        f.write("-------------------------\n")
        f.write(f"MIN_FEATURE_FREQ: {min_feature_freq}\n")
        f.write(f"MAKE_HEATMAP: {make_heatmap}\n\n")

        f.write("Stage 2: Clustering settings\n")
        f.write("----------------------------\n")
        f.write(f"DISTANCE_MODE: {distance_mode}\n")
        f.write(f"LINKAGE_METHOD: {linkage_method}\n")
        f.write(f"N_CLUSTERS: {n_clusters}\n")
        f.write(f"DISTANCE_CUTOFF: {distance_cutoff}\n")
        f.write(f"WRITE_DENDROGRAM: {write_dendrogram}\n\n")

        f.write("Stage 1 outputs\n")
        f.write("---------------\n")
        f.write(f"Jaccard CSV: {stage1_result['jaccard_csv']}\n")
        f.write(f"Heatmap PNG: {stage1_result['heatmap_png']}\n")
        f.write(f"Summary TXT: {stage1_result['summary_txt']}\n\n")

        f.write("Stage 2 outputs\n")
        f.write("---------------\n")
        f.write(f"Cluster summary CSV: {stage2_result['cluster_summary_csv']}\n")
        f.write(f"Case similarity-to-cluster CSV: {stage2_result['case_similarity_to_cluster_csv']}\n")
        f.write(f"Dendrogram PNG: {stage2_result['dendrogram_png']}\n")
        f.write(f"Summary TXT: {stage2_result['summary_txt']}\n\n")

        f.write("Run summary\n")
        f.write("-----------\n")
        f.write(f"Cases: {stage1_result['n_cases']}\n")
        f.write(f"Features before filtering: {stage1_result['n_features_total']}\n")
        f.write(f"Features after filtering: {stage1_result['n_features_kept']}\n")
        f.write(f"Clusters found: {stage2_result['n_clusters_found']}\n")


# =============================================================================
# Pipeline entry point
# =============================================================================

def run(
    *,
    input_path: Path,
    pipeline_output_dir: Path,
    title_col: Optional[str] = None,
    n_metadata_cols: int = 0,
    presence_token: str = "X",
    min_feature_freq: int = 2,
    make_heatmap: bool = True,
    heatmap_figsize: tuple[float, float] = (18, 15),
    heatmap_dpi: int = 300,
    annotate_cells: bool = False,
    annot_format: str = ".2f",
    cluster_for_plot: bool = True,
    x_label_rotation: float = 45,
    x_label_fontsize: float = 8,
    y_label_fontsize: float = 8,
    heatmap_max_label_len: Optional[int] = 38,
    distance_mode: str = "one_minus",
    linkage_method: str = "average",
    n_clusters: Optional[int] = None,
    distance_cutoff: Optional[float] = 0.90,
    write_dendrogram: bool = True,
    dendrogram_figsize: tuple[float, float] = (20, 10),
    dendrogram_dpi: int = 300,
    leaf_rotation: float = 60,
    leaf_font_size: float = 8,
    dendrogram_max_label_len: Optional[int] = 38,
    pipeline_summary_name: str = "pipeline_summary.txt",
) -> Dict[str, Any]:
    """
    Run the full Jaccard → clustering pipeline and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    resolved_pipeline_output_dir = _resolve_pipeline_output_dir(pipeline_output_dir)
    resolved_pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = resolved_pipeline_output_dir / "01_jaccard"
    stage2_dir = resolved_pipeline_output_dir / "02_clustering"

    # Load component scripts
    repo_root = Path(__file__).resolve().parents[2]
    jaccard_module = _load_module(
        repo_root / "03_similarity" / "01_jaccard_similarity_heatmap.py",
        "jaccard_similarity_heatmap",
    )
    clustering_module = _load_module(
        repo_root / "03_similarity" / "02_cluster_from_jaccard.py",
        "cluster_from_jaccard",
    )

    # Stage 1: Jaccard similarity
    stage1_result = jaccard_module.run(
        input_path=input_path,
        output_dir=stage1_dir,
        title_col=title_col,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        min_feature_freq=min_feature_freq,
        out_csv_name=None,
        out_png_name=None,
        out_summary_name="analysis_summary.txt",
        make_plot=make_heatmap,
        figsize=heatmap_figsize,
        dpi=heatmap_dpi,
        annotate_cells=annotate_cells,
        annot_format=annot_format,
        cluster_for_plot=cluster_for_plot,
        x_label_rotation=x_label_rotation,
        x_label_fontsize=x_label_fontsize,
        y_label_fontsize=y_label_fontsize,
        max_label_len=heatmap_max_label_len,
    )

    # Stage 2: clustering
    stage2_result = clustering_module.run(
        input_sim_csv=Path(stage1_result["jaccard_csv"]),
        output_dir=stage2_dir,
        distance_mode=distance_mode,
        linkage_method=linkage_method,
        n_clusters=n_clusters,
        distance_cutoff=distance_cutoff,
        out_cluster_summary_name=None,
        out_case_to_cluster_name=None,
        out_dendrogram_name=None,
        out_summary_name="analysis_summary.txt",
        write_dendrogram=write_dendrogram,
        dendrogram_figsize=dendrogram_figsize,
        dpi=dendrogram_dpi,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        max_label_len=dendrogram_max_label_len,
    )

    pipeline_summary_path = resolved_pipeline_output_dir / pipeline_summary_name
    _write_pipeline_summary(
        out_path=pipeline_summary_path,
        run_timestamp=run_timestamp,
        input_path=input_path,
        pipeline_output_dir=resolved_pipeline_output_dir,
        title_col=title_col,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        min_feature_freq=min_feature_freq,
        make_heatmap=make_heatmap,
        distance_mode=distance_mode,
        linkage_method=linkage_method,
        n_clusters=n_clusters,
        distance_cutoff=distance_cutoff,
        write_dendrogram=write_dendrogram,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
    )

    return {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "pipeline_output_dir": str(resolved_pipeline_output_dir),
        "stage1_output_dir": str(stage1_dir),
        "stage2_output_dir": str(stage2_dir),
        "pipeline_summary_txt": str(pipeline_summary_path),
        "jaccard_csv": stage1_result["jaccard_csv"],
        "heatmap_png": stage1_result["heatmap_png"],
        "cluster_summary_csv": stage2_result["cluster_summary_csv"],
        "case_similarity_to_cluster_csv": stage2_result["case_similarity_to_cluster_csv"],
        "dendrogram_png": stage2_result["dendrogram_png"],
        "n_cases": stage1_result["n_cases"],
        "n_features_total": stage1_result["n_features_total"],
        "n_features_kept": stage1_result["n_features_kept"],
        "n_clusters_found": stage2_result["n_clusters_found"],
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Jaccard → clustering pipeline."
    )

    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Path to input .xlsx/.xls or .csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PIPELINE_OUTPUT_DIR,
        help=(
            "Pipeline output directory. If set to '.', the pipeline creates a "
            "timestamped subfolder in the current directory."
        ),
    )

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
        help="Presence token in the matrix",
    )

    parser.add_argument(
        "--min-feature-freq",
        type=int,
        default=MIN_FEATURE_FREQ,
        help="Keep only features appearing in at least this many cases",
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Disable Jaccard heatmap generation")
    parser.add_argument("--heatmap-width", type=float, default=HEATMAP_FIGSIZE[0], help="Heatmap figure width")
    parser.add_argument("--heatmap-height", type=float, default=HEATMAP_FIGSIZE[1], help="Heatmap figure height")
    parser.add_argument("--heatmap-dpi", type=int, default=HEATMAP_DPI, help="Heatmap DPI")
    parser.add_argument("--annotate", action="store_true", help="Enable heatmap cell annotations")
    parser.add_argument("--annot-format", type=str, default=ANNOT_FORMAT, help="Heatmap annotation format string")
    parser.add_argument("--no-cluster-for-plot", action="store_true", help="Do not reorder heatmap for plotting")
    parser.add_argument("--x-label-rotation", type=float, default=X_LABEL_ROTATION, help="Heatmap x-label rotation")
    parser.add_argument("--x-label-fontsize", type=float, default=X_LABEL_FONTSIZE, help="Heatmap x-label font size")
    parser.add_argument("--y-label-fontsize", type=float, default=Y_LABEL_FONTSIZE, help="Heatmap y-label font size")
    parser.add_argument(
        "--heatmap-max-label-len",
        type=int,
        default=(HEATMAP_MAX_LABEL_LEN if HEATMAP_MAX_LABEL_LEN is not None else 0),
        help="Maximum heatmap label length for plotting (0 disables abbreviation)",
    )

    parser.add_argument("--distance-mode", type=str, default=DISTANCE_MODE, help="Similarity-to-distance transform")
    parser.add_argument("--linkage-method", type=str, default=LINKAGE_METHOD, help="Hierarchical clustering linkage")

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=N_CLUSTERS,
        help=(
            "Fixed number of clusters. Use this mode by setting --n-clusters to an integer "
            "and leaving --distance-cutoff unset."
        ),
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=DISTANCE_CUTOFF,
        help=(
            "Distance threshold for cutting the dendrogram. Use this mode by setting "
            "--distance-cutoff to a float and leaving --n-clusters unset."
        ),
    )

    parser.add_argument("--no-dendrogram", action="store_true", help="Disable dendrogram generation")
    parser.add_argument("--dendrogram-width", type=float, default=DENDROGRAM_FIGSIZE[0], help="Dendrogram figure width")
    parser.add_argument("--dendrogram-height", type=float, default=DENDROGRAM_FIGSIZE[1], help="Dendrogram figure height")
    parser.add_argument("--dendrogram-dpi", type=int, default=DENDROGRAM_DPI, help="Dendrogram DPI")
    parser.add_argument("--leaf-rotation", type=float, default=LEAF_ROTATION, help="Dendrogram leaf rotation")
    parser.add_argument("--leaf-font-size", type=float, default=LEAF_FONTSIZE, help="Dendrogram leaf font size")
    parser.add_argument(
        "--dendrogram-max-label-len",
        type=int,
        default=(DENDROGRAM_MAX_LABEL_LEN if DENDROGRAM_MAX_LABEL_LEN is not None else 0),
        help="Maximum dendrogram label length for plotting (0 disables abbreviation)",
    )

    return parser.parse_args()


# =============================================================================
# Main (standalone execution)
# =============================================================================

def main() -> None:
    """Run the pipeline using CONFIG defaults or CLI overrides."""
    args = _parse_args()

    heatmap_max_label_len: Optional[int]
    if args.heatmap_max_label_len == 0:
        heatmap_max_label_len = None
    else:
        heatmap_max_label_len = args.heatmap_max_label_len

    dendrogram_max_label_len: Optional[int]
    if args.dendrogram_max_label_len == 0:
        dendrogram_max_label_len = None
    else:
        dendrogram_max_label_len = args.dendrogram_max_label_len

    result = run(
        input_path=args.input,
        pipeline_output_dir=args.output_dir,
        title_col=args.title_col,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        min_feature_freq=args.min_feature_freq,
        make_heatmap=not args.no_heatmap,
        heatmap_figsize=(args.heatmap_width, args.heatmap_height),
        heatmap_dpi=args.heatmap_dpi,
        annotate_cells=args.annotate,
        annot_format=args.annot_format,
        cluster_for_plot=not args.no_cluster_for_plot,
        x_label_rotation=args.x_label_rotation,
        x_label_fontsize=args.x_label_fontsize,
        y_label_fontsize=args.y_label_fontsize,
        heatmap_max_label_len=heatmap_max_label_len,
        distance_mode=args.distance_mode,
        linkage_method=args.linkage_method,
        n_clusters=args.n_clusters,
        distance_cutoff=args.distance_cutoff,
        write_dendrogram=not args.no_dendrogram,
        dendrogram_figsize=(args.dendrogram_width, args.dendrogram_height),
        dendrogram_dpi=args.dendrogram_dpi,
        leaf_rotation=args.leaf_rotation,
        leaf_font_size=args.leaf_font_size,
        dendrogram_max_label_len=dendrogram_max_label_len,
        pipeline_summary_name=PIPELINE_SUMMARY_NAME,
    )

    print("[✓] Jaccard → clustering pipeline complete.")
    print(f"    Input:                    {result['input_path']}")
    print(f"    Pipeline output dir:      {result['pipeline_output_dir']}")
    print(f"    Jaccard CSV:              {result['jaccard_csv']}")
    print(f"    Heatmap PNG:              {result['heatmap_png']}")
    print(f"    Cluster summary CSV:      {result['cluster_summary_csv']}")
    print(f"    Case-fit CSV:             {result['case_similarity_to_cluster_csv']}")
    print(f"    Dendrogram PNG:           {result['dendrogram_png']}")
    print(f"    Pipeline summary:         {result['pipeline_summary_txt']}")


if __name__ == "__main__":
    main()