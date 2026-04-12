#!/usr/bin/env python3
"""
04_feature_zero_overlap_pipeline.py

Run the Feature Zero-Overlap Suite pipeline:

1) Significant zero-overlap feature analysis
2) Feature absence network construction
3) Feature gradient search
4) Optional feature gradient-network construction
5) Optional gradient recurrence analysis

This pipeline is designed for binary incidence matrices (case × feature) where
metadata columns may appear before feature columns.

Pipeline stages:
    Stage 1: 03_similarity/06_significant_zero_feature_overlap.py
    Stage 2: 02_networks/05_build_feature_absence_network.py
    Stage 3: 03_similarity/07_find_feature_gradients.py
    Stage 4: 02_networks/06_build_feature_gradient_networks.py    (optional)
    Stage 5: 03_similarity/08_gradient_recurrence_analyzer.py     (optional)

Standalone use:
    Edit the CONFIG block below, then run:
        python workflows/pipelines/04_feature_zero_overlap_pipeline.py

Outputs:
    - structured stage subfolders
    - optional gradient-network subfolder
    - optional gradient-recurrence subfolder
    - top-level pipeline_summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# CONFIG (standalone defaults)
# =============================================================================

# Input incidence matrix
INPUT_PATH = Path("input_incidence_matrix.xlsx")
SHEET_NAME = 0

# Shared matrix settings
CASE_ID_COLUMN = "Source Title"
N_METADATA_COLS = 4
PRESENCE_TOKEN = "X"

# Stage 1: significant zero-overlap feature settings
MIN_FEATURE_FREQ = 5
N_SAMPLES = 250
TRADES_BURN = 20000
TRADES_PER_SAMPLE = 5000
RNG_SEED = 42
FDR_THRESHOLDS = [0.05, 0.01]

# Stage 2: feature absence network settings
SIGNIFICANCE_COLUMN = "sig_0.05"
MIN_ZERO_NEIGHBORS = 2
MIN_FEATURES_PER_CASE = 1
MIN_CASES_PER_FEATURE = 2
SHORTEN_CASE_LABELS = True
TITLE_MAX_LEN = 36
APPEND_ID_FOR_UNIQUENESS = True

# Stage 3: feature gradient search settings
ENDPOINT_MODE = "significant"     # "all", "significant", or "specific"
SPECIFIC_FEATURE_A = ""
SPECIFIC_FEATURE_E = ""

CHAIN_LENGTH_MODE = "fixed"       # "fixed" or "range"
CHAIN_LENGTH = 5
MIN_CHAIN_LENGTH = 4
MAX_CHAIN_LENGTH = 6

SEARCH_MODE = "ranked"            # "strict" or "ranked"
MIN_ADJ_JACCARD = 0.05
MIN_ADJ_COOCC = 2
BEAM_WIDTH = 20
TOP_RESULTS_PER_ENDPOINT = 10
TOP_RESULTS_TOTAL = 100

# Stage 4: optional feature gradient-network settings
RUN_GRADIENT_NETWORK = True
GRADIENT_NETWORK_MODE = "top_row"    # "skip", "top_row", "specific_endpoints", "specific_chain"
GRADIENT_SELECTED_ROW = 0
GRADIENT_MIN_FEATURES_PER_CASE = 2

# Used only if GRADIENT_NETWORK_MODE = "specific_chain"
EXACT_CHAIN_STRING = ""

# Stage 5: optional gradient recurrence analysis
RUN_GRADIENT_RECURRENCE = True
RECURRENCE_MIN_WITHIN_GRADIENT_SUPPORT = 2
RECURRENCE_MIN_GRADIENTS_FOR_META = 2
RECURRENCE_WEIGHT_BY_SCORE = True
RECURRENCE_SCORE_COLUMN = "total_score"

# Pipeline output root
PIPELINE_OUTPUT_DIR = Path(".")

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
    return base_dir / f"04_feature_zero_overlap_pipeline_{timestamp}"


def _resolve_pipeline_output_dir(output_dir: Path) -> Path:
    """
    Resolve pipeline output directory.

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
    sheet_name: int | str,
    case_id_column: str,
    n_metadata_cols: int,
    presence_token: str,
    min_feature_freq: int,
    n_samples: int,
    trades_burn: int,
    trades_per_sample: int,
    rng_seed: int,
    fdr_thresholds: list[float],
    significance_column: str,
    min_zero_neighbors: int,
    min_features_per_case: int,
    min_cases_per_feature: int,
    shorten_case_labels: bool,
    title_max_len: int,
    append_id_for_uniqueness: bool,
    endpoint_mode: str,
    specific_feature_a: str,
    specific_feature_e: str,
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
    run_gradient_network: bool,
    gradient_network_mode: str,
    gradient_selected_row: int,
    gradient_min_features_per_case: int,
    exact_chain_string: str,
    run_gradient_recurrence: bool,
    recurrence_min_within_gradient_support: int,
    recurrence_min_gradients_for_meta: int,
    recurrence_weight_by_score: bool,
    recurrence_score_column: str,
    stage1_result: Dict[str, Any],
    stage2_result: Dict[str, Any],
    stage3_result: Dict[str, Any],
    stage4_result: Optional[Dict[str, Any]],
    stage5_result: Optional[Dict[str, Any]],
) -> None:
    """Write a top-level pipeline summary."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Feature Zero-Overlap Pipeline Summary ===\n\n")
        f.write(f"Run timestamp: {run_timestamp}\n\n")

        f.write("Pipeline inputs\n")
        f.write("---------------\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Sheet name: {sheet_name}\n")
        f.write(f"Pipeline output directory: {pipeline_output_dir}\n\n")

        f.write("Shared matrix settings\n")
        f.write("----------------------\n")
        f.write(f"CASE_ID_COLUMN: {case_id_column}\n")
        f.write(f"N_METADATA_COLS: {n_metadata_cols}\n")
        f.write(f"PRESENCE_TOKEN: {presence_token}\n\n")

        f.write("Stage 1 settings: significant zero-overlap features\n")
        f.write("---------------------------------------------------\n")
        f.write(f"MIN_FEATURE_FREQ: {min_feature_freq}\n")
        f.write(f"N_SAMPLES: {n_samples}\n")
        f.write(f"TRADES_BURN: {trades_burn}\n")
        f.write(f"TRADES_PER_SAMPLE: {trades_per_sample}\n")
        f.write(f"RNG_SEED: {rng_seed}\n")
        f.write(f"FDR_THRESHOLDS: {fdr_thresholds}\n\n")

        f.write("Stage 2 settings: feature absence networks\n")
        f.write("------------------------------------------\n")
        f.write(f"SIGNIFICANCE_COLUMN: {significance_column}\n")
        f.write(f"MIN_ZERO_NEIGHBORS: {min_zero_neighbors}\n")
        f.write(f"MIN_FEATURES_PER_CASE: {min_features_per_case}\n")
        f.write(f"MIN_CASES_PER_FEATURE: {min_cases_per_feature}\n")
        f.write(f"SHORTEN_CASE_LABELS: {shorten_case_labels}\n")
        f.write(f"TITLE_MAX_LEN: {title_max_len}\n")
        f.write(f"APPEND_ID_FOR_UNIQUENESS: {append_id_for_uniqueness}\n\n")

        f.write("Stage 3 settings: feature gradients\n")
        f.write("-----------------------------------\n")
        f.write(f"ENDPOINT_MODE: {endpoint_mode}\n")
        if endpoint_mode == "significant":
            f.write(f"SIGNIFICANCE_COLUMN: {significance_column}\n")
        if endpoint_mode == "specific":
            f.write(f"SPECIFIC_FEATURE_A: {specific_feature_a}\n")
            f.write(f"SPECIFIC_FEATURE_E: {specific_feature_e}\n")
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

        f.write("Stage 4 settings: feature gradient network\n")
        f.write("------------------------------------------\n")
        f.write(f"RUN_GRADIENT_NETWORK: {run_gradient_network}\n")
        f.write(f"GRADIENT_NETWORK_MODE: {gradient_network_mode}\n")
        f.write(f"GRADIENT_SELECTED_ROW: {gradient_selected_row}\n")
        f.write(f"GRADIENT_MIN_FEATURES_PER_CASE: {gradient_min_features_per_case}\n")
        if gradient_network_mode == "specific_chain":
            f.write(f"EXACT_CHAIN_STRING: {exact_chain_string}\n")
        f.write("\n")

        f.write("Stage 5 settings: gradient recurrence\n")
        f.write("-------------------------------------\n")
        f.write(f"RUN_GRADIENT_RECURRENCE: {run_gradient_recurrence}\n")
        f.write(f"RECURRENCE_MIN_WITHIN_GRADIENT_SUPPORT: {recurrence_min_within_gradient_support}\n")
        f.write(f"RECURRENCE_MIN_GRADIENTS_FOR_META: {recurrence_min_gradients_for_meta}\n")
        f.write(f"RECURRENCE_WEIGHT_BY_SCORE: {recurrence_weight_by_score}\n")
        f.write(f"RECURRENCE_SCORE_COLUMN: {recurrence_score_column}\n\n")

        f.write("Stage outputs\n")
        f.write("-------------\n")
        f.write(f"Stage 1 summary: {stage1_result['summary_txt']}\n")
        f.write(f"Stage 2 summary: {stage2_result['summary_txt']}\n")
        f.write(f"Stage 3 summary: {stage3_result['summary_txt']}\n")
        if stage4_result is not None:
            f.write(f"Stage 4 summary: {stage4_result['summary_txt']}\n")
        else:
            f.write("Stage 4 summary: (not run)\n")
        if stage5_result is not None:
            f.write(f"Stage 5 summary: {stage5_result['summary_txt']}\n")
        else:
            f.write("Stage 5 summary: (not run)\n")
        f.write("\n")

        f.write("Run summary\n")
        f.write("-----------\n")
        f.write(f"Total cases: {stage1_result['total_cases']}\n")
        f.write(f"Features before filtering: {stage1_result['features_before_filtering']}\n")
        f.write(f"Features after filtering: {stage1_result['features_after_filtering']}\n")
        f.write(f"Observed zero-overlap feature pairs: {stage1_result['observed_zero_pairs']}\n")
        for alpha in stage1_result["fdr_thresholds"]:
            f.write(f"Significant pairs @ FDR {alpha}: {stage1_result['sig_counts'][alpha]}\n")
        f.write(f"Absence graph nodes: {stage2_result['absence_graph_nodes']}\n")
        f.write(f"Absence graph edges: {stage2_result['absence_graph_edges']}\n")
        f.write(f"Gradient rows written: {stage3_result['rows_written']}\n")
        if stage4_result is not None:
            f.write(f"Gradient network nodes: {stage4_result['bipartite_nodes']}\n")
            f.write(f"Gradient network edges: {stage4_result['bipartite_edges']}\n")
        if stage5_result is not None:
            f.write(f"Recurring items written: {stage5_result['recurring_items_written']}\n")
            f.write(f"Co-recurrence graph nodes: {stage5_result['corecurrence_graph_nodes']}\n")
            f.write(f"Co-recurrence graph edges: {stage5_result['corecurrence_graph_edges']}\n")


# =============================================================================
# Pipeline entry point
# =============================================================================

def run(
    *,
    input_path: Path,
    pipeline_output_dir: Path,
    sheet_name: int | str = 0,
    case_id_column: str = "Source Title",
    n_metadata_cols: int = 4,
    presence_token: str = "X",
    min_feature_freq: int = 5,
    n_samples: int = 250,
    trades_burn: int = 20000,
    trades_per_sample: int = 5000,
    rng_seed: int = 42,
    fdr_thresholds: list[float] | tuple[float, ...] = (0.05, 0.01),
    significance_column: str = "sig_0.05",
    min_zero_neighbors: int = 2,
    min_features_per_case: int = 1,
    min_cases_per_feature: int = 2,
    shorten_case_labels: bool = True,
    title_max_len: int = 36,
    append_id_for_uniqueness: bool = True,
    endpoint_mode: str = "significant",
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
    run_gradient_network: bool = True,
    gradient_network_mode: str = "top_row",   # "skip", "top_row", "specific_endpoints", "specific_chain"
    gradient_selected_row: int = 0,
    gradient_min_features_per_case: int = 2,
    exact_chain_string: str = "",
    run_gradient_recurrence: bool = True,
    recurrence_min_within_gradient_support: int = 2,
    recurrence_min_gradients_for_meta: int = 2,
    recurrence_weight_by_score: bool = True,
    recurrence_score_column: str = "total_score",
    pipeline_summary_name: str = "pipeline_summary.txt",
) -> Dict[str, Any]:
    """
    Run the full Feature Zero-Overlap Suite pipeline and return a structured result dictionary.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if gradient_network_mode not in {"skip", "top_row", "specific_endpoints", "specific_chain"}:
        raise ValueError(
            "gradient_network_mode must be 'skip', 'top_row', 'specific_endpoints', or 'specific_chain'."
        )

    resolved_pipeline_output_dir = _resolve_pipeline_output_dir(pipeline_output_dir)
    resolved_pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    # Load component scripts
    repo_root = Path(__file__).resolve().parents[2]

    stage1_module = _load_module(
        repo_root / "03_similarity" / "06_significant_zero_feature_overlap.py",
        "significant_zero_feature_overlap",
    )
    stage2_module = _load_module(
        repo_root / "02_networks" / "05_build_feature_absence_network.py",
        "build_feature_absence_network",
    )
    stage3_module = _load_module(
        repo_root / "03_similarity" / "07_find_feature_gradients.py",
        "find_feature_gradients",
    )
    stage4_module = _load_module(
        repo_root / "02_networks" / "06_build_feature_gradient_networks.py",
        "build_feature_gradient_networks",
    )
    stage5_module = _load_module(
        repo_root / "03_similarity" / "08_gradient_recurrence_analyzer.py",
        "gradient_recurrence_analyzer",
    )

    # -----------------------------------------------------------------
    # Stage 1: significant zero-overlap feature analysis
    # -----------------------------------------------------------------
    stage1_dir = resolved_pipeline_output_dir / "01_significant_zero_overlap"
    stage1_result = stage1_module.run(
        input_path=input_path,
        output_dir=stage1_dir,
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
        out_csv_name="zero_feature_overlap_with_significance.csv",
        out_summary_name="analysis_summary.txt",
    )

    # -----------------------------------------------------------------
    # Stage 2: feature absence networks
    # -----------------------------------------------------------------
    stage2_dir = resolved_pipeline_output_dir / "02_absence_networks"
    stage2_result = stage2_module.run(
        zero_feature_overlap_csv=Path(stage1_result["zero_feature_overlap_csv"]),
        incidence_path=input_path,
        output_dir=stage2_dir,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        significance_column=significance_column,
        min_zero_neighbors=min_zero_neighbors,
        min_features_per_case=min_features_per_case,
        min_cases_per_feature=min_cases_per_feature,
        shorten_case_labels=shorten_case_labels,
        title_max_len=title_max_len,
        append_id_for_uniqueness=append_id_for_uniqueness,
        out_abs_graph_name="feature_absence_graph_sig.gexf",
        out_bip_graph_name="feature_absence_bipartite.gexf",
        out_summary_name="analysis_summary.txt",
    )

    # -----------------------------------------------------------------
    # Stage 3: feature gradient search
    # -----------------------------------------------------------------
    stage3_dir = resolved_pipeline_output_dir / "03_feature_gradients"
    stage3_result = stage3_module.run(
        zero_feature_overlap_csv=Path(stage1_result["zero_feature_overlap_csv"]),
        incidence_path=input_path,
        output_dir=stage3_dir,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        min_feature_freq=min_feature_freq,
        endpoint_mode=endpoint_mode,
        significance_column=significance_column,
        specific_feature_a=specific_feature_a,
        specific_feature_e=specific_feature_e,
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
        out_csv_name="feature_gradients.csv",
        out_summary_name="analysis_summary.txt",
    )

    # -----------------------------------------------------------------
    # Stage 4: optional feature gradient-network construction
    # -----------------------------------------------------------------
    stage4_result: Optional[Dict[str, Any]] = None

    should_run_stage4 = (
        run_gradient_network
        and gradient_network_mode != "skip"
        and stage3_result["rows_written"] > 0
    )

    if should_run_stage4:
        stage4_dir = resolved_pipeline_output_dir / "04_gradient_network"

        if gradient_network_mode == "top_row":
            stage4_result = stage4_module.run(
                gradients_csv=Path(stage3_result["feature_gradients_csv"]),
                incidence_path=input_path,
                output_dir=stage4_dir,
                sheet_name=sheet_name,
                case_id_column=case_id_column,
                n_metadata_cols=n_metadata_cols,
                presence_token=presence_token,
                selection_mode="row",
                selected_row=gradient_selected_row,
                feature_a="",
                feature_e="",
                exact_chain_string="",
                min_gradient_features_per_case=gradient_min_features_per_case,
                shorten_case_labels=shorten_case_labels,
                title_max_len=title_max_len,
                append_id_for_uniqueness=append_id_for_uniqueness,
                out_bip_gexf_name="feature_gradient_bipartite.gexf",
                out_jaccard_csv_name="feature_gradient_jaccard_subset.csv",
                out_jaccard_pairs_name="feature_gradient_jaccard_pairs_ranked.csv",
                out_jaccard_png_name="feature_gradient_jaccard_heatmap.png",
                out_summary_name="analysis_summary.txt",
            )

        elif gradient_network_mode == "specific_endpoints":
            if not specific_feature_a or not specific_feature_e:
                raise ValueError(
                    "For gradient_network_mode='specific_endpoints', set specific_feature_a and specific_feature_e."
                )

            stage4_result = stage4_module.run(
                gradients_csv=Path(stage3_result["feature_gradients_csv"]),
                incidence_path=input_path,
                output_dir=stage4_dir,
                sheet_name=sheet_name,
                case_id_column=case_id_column,
                n_metadata_cols=n_metadata_cols,
                presence_token=presence_token,
                selection_mode="endpoint",
                selected_row=0,
                feature_a=specific_feature_a,
                feature_e=specific_feature_e,
                exact_chain_string="",
                min_gradient_features_per_case=gradient_min_features_per_case,
                shorten_case_labels=shorten_case_labels,
                title_max_len=title_max_len,
                append_id_for_uniqueness=append_id_for_uniqueness,
                out_bip_gexf_name="feature_gradient_bipartite.gexf",
                out_jaccard_csv_name="feature_gradient_jaccard_subset.csv",
                out_jaccard_pairs_name="feature_gradient_jaccard_pairs_ranked.csv",
                out_jaccard_png_name="feature_gradient_jaccard_heatmap.png",
                out_summary_name="analysis_summary.txt",
            )

        elif gradient_network_mode == "specific_chain":
            if not exact_chain_string:
                raise ValueError(
                    "For gradient_network_mode='specific_chain', set exact_chain_string."
                )

            stage4_result = stage4_module.run(
                gradients_csv=Path(stage3_result["feature_gradients_csv"]),
                incidence_path=input_path,
                output_dir=stage4_dir,
                sheet_name=sheet_name,
                case_id_column=case_id_column,
                n_metadata_cols=n_metadata_cols,
                presence_token=presence_token,
                selection_mode="chain",
                selected_row=0,
                feature_a="",
                feature_e="",
                exact_chain_string=exact_chain_string,
                min_gradient_features_per_case=gradient_min_features_per_case,
                shorten_case_labels=shorten_case_labels,
                title_max_len=title_max_len,
                append_id_for_uniqueness=append_id_for_uniqueness,
                out_bip_gexf_name="feature_gradient_bipartite.gexf",
                out_jaccard_csv_name="feature_gradient_jaccard_subset.csv",
                out_jaccard_pairs_name="feature_gradient_jaccard_pairs_ranked.csv",
                out_jaccard_png_name="feature_gradient_jaccard_heatmap.png",
                out_summary_name="analysis_summary.txt",
            )

    # -----------------------------------------------------------------
    # Stage 5: optional gradient recurrence analysis
    # -----------------------------------------------------------------
    stage5_result: Optional[Dict[str, Any]] = None

    should_run_stage5 = (
        run_gradient_recurrence
        and stage3_result["rows_written"] > 0
    )

    if should_run_stage5:
        stage5_dir = resolved_pipeline_output_dir / "05_gradient_recurrence"

        stage5_result = stage5_module.run(
            gradients_csv=Path(stage3_result["feature_gradients_csv"]),
            incidence_path=input_path,
            output_dir=stage5_dir,
            gradient_kind="feature",
            sheet_name=sheet_name,
            case_id_column=case_id_column,
            n_metadata_cols=n_metadata_cols,
            presence_token=presence_token,
            producer_col=None,
            min_within_gradient_support=recurrence_min_within_gradient_support,
            min_gradients_for_meta=recurrence_min_gradients_for_meta,
            weight_by_score=recurrence_weight_by_score,
            score_column=recurrence_score_column,
            out_summary_csv="gradient_recurrence_summary.csv",
            out_membership_long_csv="gradient_recurrence_membership_long.csv",
            out_network_gexf="gradient_recurrence_network.gexf",
            out_summary_name="analysis_summary.txt",
        )

    # -----------------------------------------------------------------
    # Pipeline summary
    # -----------------------------------------------------------------
    pipeline_summary_path = resolved_pipeline_output_dir / pipeline_summary_name
    _write_pipeline_summary(
        out_path=pipeline_summary_path,
        run_timestamp=run_timestamp,
        input_path=input_path,
        pipeline_output_dir=resolved_pipeline_output_dir,
        sheet_name=sheet_name,
        case_id_column=case_id_column,
        n_metadata_cols=n_metadata_cols,
        presence_token=presence_token,
        min_feature_freq=min_feature_freq,
        n_samples=n_samples,
        trades_burn=trades_burn,
        trades_per_sample=trades_per_sample,
        rng_seed=rng_seed,
        fdr_thresholds=[float(x) for x in fdr_thresholds],
        significance_column=significance_column,
        min_zero_neighbors=min_zero_neighbors,
        min_features_per_case=min_features_per_case,
        min_cases_per_feature=min_cases_per_feature,
        shorten_case_labels=shorten_case_labels,
        title_max_len=title_max_len,
        append_id_for_uniqueness=append_id_for_uniqueness,
        endpoint_mode=endpoint_mode,
        specific_feature_a=specific_feature_a,
        specific_feature_e=specific_feature_e,
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
        run_gradient_network=run_gradient_network,
        gradient_network_mode=gradient_network_mode,
        gradient_selected_row=gradient_selected_row,
        gradient_min_features_per_case=gradient_min_features_per_case,
        exact_chain_string=exact_chain_string,
        run_gradient_recurrence=run_gradient_recurrence,
        recurrence_min_within_gradient_support=recurrence_min_within_gradient_support,
        recurrence_min_gradients_for_meta=recurrence_min_gradients_for_meta,
        recurrence_weight_by_score=recurrence_weight_by_score,
        recurrence_score_column=recurrence_score_column,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
        stage3_result=stage3_result,
        stage4_result=stage4_result,
        stage5_result=stage5_result,
    )

    return {
        "run_timestamp": run_timestamp,
        "input_path": str(input_path),
        "pipeline_output_dir": str(resolved_pipeline_output_dir),
        "pipeline_summary_txt": str(pipeline_summary_path),
        "stage1_result": stage1_result,
        "stage2_result": stage2_result,
        "stage3_result": stage3_result,
        "stage4_result": stage4_result,
        "stage5_result": stage5_result,
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
        description="Run the Feature Zero-Overlap Suite pipeline."
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

    parser.add_argument("--significance-column", type=str, default=SIGNIFICANCE_COLUMN)
    parser.add_argument("--min-zero-neighbors", type=int, default=MIN_ZERO_NEIGHBORS)
    parser.add_argument("--min-features-per-case", type=int, default=MIN_FEATURES_PER_CASE)
    parser.add_argument("--min-cases-per-feature", type=int, default=MIN_CASES_PER_FEATURE)

    parser.add_argument("--endpoint-mode", type=str, default=ENDPOINT_MODE)
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

    parser.add_argument("--run-gradient-network", action="store_true", default=RUN_GRADIENT_NETWORK)
    parser.add_argument("--gradient-network-mode", type=str, default=GRADIENT_NETWORK_MODE)
    parser.add_argument("--gradient-selected-row", type=int, default=GRADIENT_SELECTED_ROW)
    parser.add_argument("--gradient-min-features-per-case", type=int, default=GRADIENT_MIN_FEATURES_PER_CASE)
    parser.add_argument("--exact-chain-string", type=str, default=EXACT_CHAIN_STRING)

    parser.add_argument("--run-gradient-recurrence", action="store_true", default=RUN_GRADIENT_RECURRENCE)
    parser.add_argument("--recurrence-min-within-gradient-support", type=int, default=RECURRENCE_MIN_WITHIN_GRADIENT_SUPPORT)
    parser.add_argument("--recurrence-min-gradients-for-meta", type=int, default=RECURRENCE_MIN_GRADIENTS_FOR_META)
    parser.add_argument("--no-recurrence-weight-by-score", action="store_true")
    parser.add_argument("--recurrence-score-column", type=str, default=RECURRENCE_SCORE_COLUMN)

    parser.add_argument("--output-dir", type=Path, default=PIPELINE_OUTPUT_DIR)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    result = run(
        input_path=args.input,
        pipeline_output_dir=args.output_dir,
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
        significance_column=args.significance_column,
        min_zero_neighbors=args.min_zero_neighbors,
        min_features_per_case=args.min_features_per_case,
        min_cases_per_feature=args.min_cases_per_feature,
        shorten_case_labels=SHORTEN_CASE_LABELS,
        title_max_len=TITLE_MAX_LEN,
        append_id_for_uniqueness=APPEND_ID_FOR_UNIQUENESS,
        endpoint_mode=args.endpoint_mode,
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
        run_gradient_network=args.run_gradient_network,
        gradient_network_mode=args.gradient_network_mode,
        gradient_selected_row=args.gradient_selected_row,
        gradient_min_features_per_case=args.gradient_min_features_per_case,
        exact_chain_string=args.exact_chain_string,
        run_gradient_recurrence=args.run_gradient_recurrence,
        recurrence_min_within_gradient_support=args.recurrence_min_within_gradient_support,
        recurrence_min_gradients_for_meta=args.recurrence_min_gradients_for_meta,
        recurrence_weight_by_score=not args.no_recurrence_weight_by_score,
        recurrence_score_column=args.recurrence_score_column,
        pipeline_summary_name=PIPELINE_SUMMARY_NAME,
    )

    print("[✓] Feature Zero-Overlap pipeline complete.")
    print(f"    Input:                    {result['input_path']}")
    print(f"    Output dir:               {result['pipeline_output_dir']}")
    print(f"    Pipeline summary:         {result['pipeline_summary_txt']}")
    print(f"    Stage 1 pairs:            {result['stage1_result']['observed_zero_pairs']}")
    print(f"    Stage 3 rows:             {result['stage3_result']['rows_written']}")
    if result["stage4_result"] is not None:
        print(
            f"    Stage 4 graph:            "
            f"{result['stage4_result']['bipartite_nodes']} nodes | {result['stage4_result']['bipartite_edges']} edges"
        )
    if result["stage5_result"] is not None:
        print(
            f"    Stage 5 recurrence:       "
            f"{result['stage5_result']['recurring_entities_written']} entities | "
            f"{result['stage5_result']['corecurrence_edges_written']} co-recurrence edges"
        )


if __name__ == "__main__":
    main()