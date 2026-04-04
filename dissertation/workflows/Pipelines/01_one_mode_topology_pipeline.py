#!/usr/bin/env python3
"""
01_one_mode_topology_pipeline.py

Run the One-Mode + Topology pipeline:

1) Build one-mode networks from a binary incidence matrix
2) For each generated thresholded network:
   - compute k-components
   - compute Burt brokerage metrics

This pipeline is designed for datasets such as books × tropes, songs × tropes,
or other binary incidence matrices where metadata columns may appear before
feature columns.

Pipeline stages:
    Stage 1: 02_networks/01_build_one_mode_projection.py
    Stage 2: 04_topology/01_k_components_from_gexf.py
    Stage 3: 04_topology/02_burt_brokerage_metrics.py

Standalone use:
    Edit the CONFIG block below, then run:
        python workflows/pipelines/01_one_mode_topology_pipeline.py

This pipeline writes:
    - threshold-specific outputs in separate subfolders
    - a top-level pipeline_summary.txt with a timestamp and output inventory
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
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

# Projection modes:
#   ["feature"]
#   ["case"]
#   ["feature", "case"]
PROJECTION_MODES = ["feature", "case"]

# Projection-stage node filters
MIN_FEATURE_NODE_FREQ = 2
MIN_CASE_NODE_FREQ = 2

# Projection-stage edge thresholds
FEATURE_EDGE_THRESHOLDS = [10]
CASE_EDGE_THRESHOLDS = [10]

# Topology-stage settings
EXPORT_ONLY_K: Optional[list[int]] = None
EXPORT_CORE_NUMBERS = True
EXPORT_NODE_SUMMARY = True
WEIGHT_ATTR = "weight"
COMPONENT_PREFIX = "kcomp"

# Burt settings
PROGRESS_EVERY = 200

# Pipeline output root.
# If PIPELINE_OUTPUT_DIR is ".", the pipeline creates a timestamped subfolder
# in the current working directory.
PIPELINE_OUTPUT_DIR = Path(".")

# Pipeline summary filename
PIPELINE_SUMMARY_NAME = "pipeline_summary.txt"


# =============================================================================
# Helpers
# =============================================================================

def _timestamp() -> str:
    """Return a short timestamp string for console messages."""
    return datetime.now().strftime("%H:%M:%S")


def _print_stage_start(message: str) -> None:
    """Print a standardized stage start message."""
    print(f"[{_timestamp()}] [→] {message}")


def _print_stage_done(message: str) -> None:
    """Print a standardized stage completion message."""
    print(f"[{_timestamp()}] [✓] {message}")


def _print_info(message: str) -> None:
    """Print a standardized informational message."""
    print(f"[{_timestamp()}] [i] {message}")


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
    return base_dir / f"01_one_mode_topology_pipeline_{timestamp}"


def _resolve_pipeline_output_dir(output_dir: Path) -> Path:
    """
    Resolve the pipeline output directory.

    If output_dir == ".", create a timestamped subfolder in the current directory.
    Otherwise, use the provided directory directly.
    """
    if output_dir == Path("."):
        return _default_pipeline_output_dir(Path("."))
    return output_dir


def _normalize_projection_modes(modes: list[str]) -> list[str]:
    """Validate and normalize projection mode list."""
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


def _format_optional_list(value: Optional[list[int]]) -> str:
    """Pretty-print optional integer list."""
    return "all" if value is None else str(value)


def _write_pipeline_summary(
    out_path: Path,
    *,
    run_timestamp: str,
    input_path: Path,
    pipeline_output_dir: Path,
    title_col: Optional[str],
    n_metadata_cols: int,
    presence_token: str,
    projection_modes: list[str],
    min_feature_node_freq: int,
    min_case_node_freq: int,
    feature_edge_thresholds: list[int],
    case_edge_thresholds: list[int],
    export_only_k: Optional[list[int]],
    export_core_numbers: bool,
    export_node_summary: bool,
    weight_attr: str,
    progress_every: int,
    projection_result: Dict[str, Any],
    network_runs: list[Dict[str, Any]],
) -> None:
    """Write a top-level pipeline summary."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== One-Mode + Topology Pipeline Summary ===\n\n")
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

        f.write("Projection settings\n")
        f.write("-------------------\n")
        f.write(f"PROJECTION_MODES: {projection_modes}\n")
        f.write(f"MIN_FEATURE_NODE_FREQ: {min_feature_node_freq}\n")
        f.write(f"MIN_CASE_NODE_FREQ: {min_case_node_freq}\n")
        f.write(f"FEATURE_EDGE_THRESHOLDS: {feature_edge_thresholds}\n")
        f.write(f"CASE_EDGE_THRESHOLDS: {case_edge_thresholds}\n\n")

        f.write("Topology settings\n")
        f.write("-----------------\n")
        f.write(f"EXPORT_ONLY_K: {_format_optional_list(export_only_k)}\n")
        f.write(f"EXPORT_CORE_NUMBERS: {export_core_numbers}\n")
        f.write(f"EXPORT_NODE_SUMMARY: {export_node_summary}\n")
        f.write(f"WEIGHT_ATTR: {weight_attr}\n")
        f.write(f"PROGRESS_EVERY: {progress_every}\n\n")

        f.write("Stage 1 output\n")
        f.write("--------------\n")
        f.write(f"Projection summary TXT: {projection_result['summary_txt']}\n\n")

        f.write("Per-network runs\n")
        f.write("----------------\n")
        for item in network_runs:
            f.write(f"[{item['projection_mode']} | thr{item['threshold']}]\n")
            f.write(f"  One-mode network dir: {item['one_mode_network_dir']}\n")
            f.write(f"  GEXF: {item['gexf']}\n")
            f.write(f"  k-components dir: {item['k_components_dir']}\n")
            f.write(f"  k-components summary: {item['k_components_summary_csv']}\n")
            f.write(f"  Burt dir: {item['burt_dir']}\n")
            f.write(f"  Burt CSV: {item['burt_metrics_csv']}\n")
            f.write(f"  Burt GEXF: {item['annotated_gexf']}\n")
            f.write("\n")

        f.write("Run summary\n")
        f.write("-----------\n")
        f.write(f"Cases total: {projection_result['n_cases_total']}\n")
        f.write(f"Features total: {projection_result['n_features_total']}\n")
        f.write(f"Networks analyzed: {len(network_runs)}\n")


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
    projection_modes: list[str] | tuple[str, ...] = ("feature", "case"),
    min_feature_node_freq: int = 2,
    min_case_node_freq: int = 2,
    feature_edge_thresholds: list[int] | tuple[int, ...] = (3, 5),
    case_edge_thresholds: list[int] | tuple[int, ...] = (10, 15),
    export_only_k: Optional[list[int]] = None,
    export_core_numbers: bool = True,
    export_node_summary: bool = True,
    weight_attr: str = "weight",
    component_prefix: str = "kcomp",
    progress_every: int = 200,
    pipeline_summary_name: str = "pipeline_summary.txt",
) -> Dict[str, Any]:
    """
    Run the full One-Mode + Topology pipeline and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    projection_modes = _normalize_projection_modes(list(projection_modes))
    feature_edge_thresholds = [int(x) for x in feature_edge_thresholds]
    case_edge_thresholds = [int(x) for x in case_edge_thresholds]

    resolved_pipeline_output_dir = _resolve_pipeline_output_dir(pipeline_output_dir)
    resolved_pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    _print_info(f"Input file: {input_path}")
    _print_info(f"Pipeline output directory: {resolved_pipeline_output_dir}")
    _print_info(f"Projection modes: {projection_modes}")
    _print_info(f"Feature thresholds: {feature_edge_thresholds}")
    _print_info(f"Case thresholds: {case_edge_thresholds}")

    # Load component scripts
    repo_root = Path(__file__).resolve().parents[2]
    _print_stage_start("Loading pipeline modules")
    projection_module = _load_module(
        repo_root / "02_networks" / "01_build_one_mode_projection.py",
        "build_one_mode_projection",
    )
    kcomp_module = _load_module(
        repo_root / "04_topology" / "01_k_components_from_gexf.py",
        "k_components_from_gexf",
    )
    burt_module = _load_module(
        repo_root / "04_topology" / "02_burt_brokerage_metrics.py",
        "burt_brokerage_metrics",
    )
    _print_stage_done("Pipeline modules loaded")

    # Stage 1: build one-mode networks
    stage1_dir = resolved_pipeline_output_dir / "00_projection_build"
    _print_stage_start("Stage 1/3: Building one-mode projections")
    projection_result = projection_module.run(
        input_path=input_path,
        output_dir=stage1_dir,
        title_col=title_col,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        projection_modes=projection_modes,
        min_feature_node_freq=min_feature_node_freq,
        min_case_node_freq=min_case_node_freq,
        feature_edge_thresholds=feature_edge_thresholds,
        case_edge_thresholds=case_edge_thresholds,
        out_summary_name="analysis_summary.txt",
    )
    _print_stage_done("Stage 1/3 complete: One-mode projections built")

    network_runs: list[Dict[str, Any]] = []

    # Stages 2 + 3: iterate over each generated GEXF
    for mode in projection_modes:
        mode_info = projection_result["projections"][mode]
        _print_info(f"Preparing {mode} projection outputs")

        for thr, thr_info in mode_info["thresholds"].items():
            threshold_root = resolved_pipeline_output_dir / mode / f"thr{thr}"
            one_mode_dir = threshold_root / "01_one_mode_network"
            kcomp_dir = threshold_root / "02_k_components"
            burt_dir = threshold_root / "03_burt"

            _print_stage_start(f"{mode} | thr{thr}: Preparing threshold-specific workspace")

            one_mode_dir.mkdir(parents=True, exist_ok=True)
            gexf_path = Path(thr_info["gexf"])
            edge_csv_path = Path(thr_info["edge_csv"])

            copied_gexf = one_mode_dir / gexf_path.name
            copied_edge_csv = one_mode_dir / edge_csv_path.name

            shutil.copy2(gexf_path, copied_gexf)
            shutil.copy2(edge_csv_path, copied_edge_csv)

            _print_info(f"{mode} | thr{thr}: GEXF copied to {copied_gexf}")
            _print_info(f"{mode} | thr{thr}: Edge CSV copied to {copied_edge_csv}")
            _print_stage_done(f"{mode} | thr{thr}: Workspace ready")

            # Stage 2: k-components
            _print_stage_start(f"{mode} | thr{thr}: Starting k-components")
            kcomp_result = kcomp_module.run(
                input_gexf=copied_gexf,
                output_dir=kcomp_dir,
                export_only_k=export_only_k,
                export_core_numbers=export_core_numbers,
                core_numbers_csv="node_core_numbers.csv",
                export_node_summary=export_node_summary,
                node_summary_csv="node_summary.csv",
                weight_attr=weight_attr,
                component_prefix=component_prefix,
                out_summary_name="analysis_summary.txt",
            )
            _print_stage_done(f"{mode} | thr{thr}: k-components complete")

            # Stage 3: Burt metrics
            _print_stage_start(f"{mode} | thr{thr}: Starting Burt brokerage metrics")
            burt_result = burt_module.run(
                input_gexf=copied_gexf,
                output_dir=burt_dir,
                weight_attr=weight_attr,
                out_csv_name="burt_metrics.csv",
                out_gexf_name="network_with_burt.gexf",
                out_summary_name="analysis_summary.txt",
                progress_every=progress_every,
            )
            _print_stage_done(f"{mode} | thr{thr}: Burt brokerage metrics complete")

            network_runs.append(
                {
                    "projection_mode": mode,
                    "threshold": int(thr),
                    "one_mode_network_dir": str(one_mode_dir),
                    "edge_csv": str(copied_edge_csv),
                    "gexf": str(copied_gexf),
                    "k_components_dir": str(kcomp_dir),
                    "k_components_summary_csv": kcomp_result["component_summary_csv"],
                    "k_components_summary_txt": kcomp_result["summary_txt"],
                    "burt_dir": str(burt_dir),
                    "burt_metrics_csv": burt_result["burt_metrics_csv"],
                    "annotated_gexf": burt_result["annotated_gexf"],
                    "burt_summary_txt": burt_result["summary_txt"],
                }
            )

            _print_stage_done(f"{mode} | thr{thr}: Threshold run complete")

    # Cleanup stage 1 temp output
    if stage1_dir.exists():
        _print_stage_start("Cleaning up temporary projection-build folder")
        shutil.rmtree(stage1_dir)
        _print_stage_done("Temporary projection-build folder removed")

    pipeline_summary_path = resolved_pipeline_output_dir / pipeline_summary_name
    _print_stage_start("Writing top-level pipeline summary")
    _write_pipeline_summary(
        out_path=pipeline_summary_path,
        run_timestamp=run_timestamp,
        input_path=input_path,
        pipeline_output_dir=resolved_pipeline_output_dir,
        title_col=title_col,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        projection_modes=projection_modes,
        min_feature_node_freq=min_feature_node_freq,
        min_case_node_freq=min_case_node_freq,
        feature_edge_thresholds=feature_edge_thresholds,
        case_edge_thresholds=case_edge_thresholds,
        export_only_k=export_only_k,
        export_core_numbers=export_core_numbers,
        export_node_summary=export_node_summary,
        weight_attr=weight_attr,
        progress_every=progress_every,
        projection_result=projection_result,
        network_runs=network_runs,
    )
    _print_stage_done("Top-level pipeline summary written")

    return {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "pipeline_output_dir": str(resolved_pipeline_output_dir),
        "projection_summary_txt": projection_result["summary_txt"],
        "pipeline_summary_txt": str(pipeline_summary_path),
        "n_cases_total": projection_result["n_cases_total"],
        "n_features_total": projection_result["n_features_total"],
        "network_runs": network_runs,
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the One-Mode + Topology pipeline."
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
        help="Comma-separated edge thresholds for feature projection (e.g. 3,5)",
    )
    parser.add_argument(
        "--case-edge-thresholds",
        type=_parse_int_list,
        default=CASE_EDGE_THRESHOLDS,
        help="Comma-separated edge thresholds for case projection (e.g. 10,15)",
    )
    parser.add_argument(
        "--export-only-k",
        type=_parse_int_list,
        default=EXPORT_ONLY_K,
        help="Comma-separated list of k-levels to export (e.g. 2,3,4). Default: export all",
    )
    parser.add_argument(
        "--no-core-numbers",
        action="store_true",
        help="Disable export of node core numbers CSV",
    )
    parser.add_argument(
        "--no-node-summary",
        action="store_true",
        help="Disable export of node degree / weighted degree CSV",
    )
    parser.add_argument(
        "--weight-attr",
        type=str,
        default=WEIGHT_ATTR,
        help='Edge weight attribute name (default: "weight")',
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help="Print progress every N nodes (0 disables progress messages)",
    )

    return parser.parse_args()


# =============================================================================
# Main (standalone execution)
# =============================================================================

def main() -> None:
    """Run the pipeline using CONFIG defaults or CLI overrides."""
    args = _parse_args()

    result = run(
        input_path=args.input,
        pipeline_output_dir=args.output_dir,
        title_col=args.title_col,
        n_metadata_cols=args.n_metadata_cols,
        presence_token=args.presence_token,
        projection_modes=args.projection_modes,
        min_feature_node_freq=args.min_feature_node_freq,
        min_case_node_freq=args.min_case_node_freq,
        feature_edge_thresholds=args.feature_edge_thresholds,
        case_edge_thresholds=args.case_edge_thresholds,
        export_only_k=args.export_only_k,
        export_core_numbers=not args.no_core_numbers,
        export_node_summary=not args.no_node_summary,
        weight_attr=args.weight_attr,
        component_prefix=COMPONENT_PREFIX,
        progress_every=args.progress_every,
        pipeline_summary_name=PIPELINE_SUMMARY_NAME,
    )

    print("[✓] One-Mode + Topology pipeline complete.")
    print(f"    Input:                {result['input_path']}")
    print(f"    Pipeline output dir:  {result['pipeline_output_dir']}")
    print(f"    Projection summary:   {result['projection_summary_txt']}")
    print(f"    Pipeline summary:     {result['pipeline_summary_txt']}")
    print(f"    Networks analyzed:    {len(result['network_runs'])}")


if __name__ == "__main__":
    main()