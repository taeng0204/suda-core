"""Forgetting Granularity Benchmark: Sample-Level vs Tree-Level Forgetting.

This benchmark compares SUDA's sample-level exact forgetting against
ARF baselines with class balancing, testing the "Forgetting Granularity"
framing of the paper.

Key comparisons:
  1. SUDA-Full vs ARF+ClassWeight-70x   (fair comparison with matched oversampling)
  2. SUDA-Full vs SUDA-AK-Only          (budget+forget added value)
  3. SUDA-Full vs SlidingWindow-ARF      (exact vs approximate forgetting)
  4. SUDA-Full vs SUDA-NoForgetBudget    (forest.forget() necessity)

Configs:
  - SUDA-Full:            Adaptive-k + Budget(3000) + Exact Forgetting
  - SUDA-AK-Only:         Adaptive-k only (no budget, no forgetting)
  - SUDA-NoForgetBudget:  Budget eviction without forest.forget (HOW ablation)
  - ARF:                  Bare ARF
  - ARF+Oversample:       ARF + batch minority oversampling (30%)
  - ARF+ClassWeight-10x:  ARF + minority repeated 10x
  - ARF+ClassWeight-70x:  ARF + minority repeated 70x (matches SUDA k_max=70)
  - SlidingWindow-ARF:    ARF with 3000-sample sliding window + periodic retrain

Scenarios (intermediate drift):
  1. moderate_sudden:      1% → 20%
  2. stepwise:             1% → 10% → 20% → 10% → 5%
  3. gradual_ramp:         1% → 5% → 10% → 15% → 20%
  4. asymmetric_recovery:  1% → 20% → 3%

ANoShift natural drift (separate):
  - AK-Only, Budget-FIFO, Trigger-Conservative

Usage:
    # Smoke test
    uv run python -m src.experiments.granularity_benchmark --smoke

    # Full benchmark
    uv run python -m src.experiments.granularity_benchmark

    # ANoShift only
    uv run python -m src.experiments.granularity_benchmark --anoshift

    # Specific datasets
    uv run python -m src.experiments.granularity_benchmark --datasets nslkdd unswnb15
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src.data.nids import (
    make_asymmetric_recovery_stream,
    make_gradual_ramp_stream,
    make_moderate_sudden_drift_stream,
    make_stepwise_drift_stream,
    make_anoshift_temporal_stream,
)
from src.models.suda import SUDA
from src.baselines.river_models import (
    ARFModel,
    ARFWithOversampling,
    ARFWithClassWeight,
)
from src.baselines.limited_arf import LimitedMemoryARF

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from src.experiments.utils import NumpyEncoder, compute_gmean, compute_f1, compute_phase_metrics  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

DATASETS = ["nslkdd", "unswnb15", "cicids2018"]
DATASET_FEATURES = {"nslkdd": 41, "unswnb15": 42, "cicids2018": 78}
DEFAULT_SEEDS = [42, 123, 456]
BATCH_SIZE = 200
WARMUP_RATIO = 0.3


@dataclass
class ModelConfig:
    name: str
    model_type: str  # "suda", "arf", "arf_oversample", "arf_classweight", "sliding_window"
    params: dict


CONFIGS = [
    # --- SUDA Variants ---
    ModelConfig("SUDA-Full", "suda", {
        "adaptive_k": True, "budget": True, "budget_max_samples": 3000,
        "unlearning": False, "influence_tracking": False,
    }),
    ModelConfig("SUDA-AK-Only", "suda", {
        "adaptive_k": True, "budget": False, "unlearning": False,
    }),
    ModelConfig("SUDA-NoForgetBudget", "suda", {
        "adaptive_k": True, "budget": True, "budget_max_samples": 3000,
        "budget_skip_forest_forget": True, "unlearning": False,
    }),
    # --- ARF Baselines ---
    ModelConfig("ARF", "arf", {}),
    # --- Sliding Window ---
    ModelConfig("SlidingWindow-ARF", "sliding_window", {
        "max_samples": 3000, "retrain_frequency": 50, "enable_retrain": True,
    }),
]

# Extended configs (includes slow ARF+CB variants for thorough comparison)
CONFIGS_EXTENDED = CONFIGS + [
    ModelConfig("ARF+Oversample", "arf_oversample", {"target_ratio": 0.3}),
    ModelConfig("ARF+ClassWeight-10x", "arf_classweight", {"minority_weight": 10}),
]

SCENARIOS = {
    "moderate_sudden": {
        "fn": make_moderate_sudden_drift_stream,
        "description": "1% → 20% sudden drift",
    },
    "stepwise": {
        "fn": make_stepwise_drift_stream,
        "description": "1% → 10% → 20% → 10% → 5%",
    },
    "gradual_ramp": {
        "fn": make_gradual_ramp_stream,
        "description": "1% → 5% → 10% → 15% → 20%",
    },
    "asymmetric_recovery": {
        "fn": make_asymmetric_recovery_stream,
        "description": "1% → 20% → 3%",
    },
}


# =============================================================================
# Model Creation
# =============================================================================


def create_model(config: ModelConfig, num_features: int, seed: int) -> Any:
    if config.model_type == "suda":
        p = config.params
        return SUDA(
            num_features=num_features,
            num_trees=50,
            k=10,
            max_depth=15,
            seed=seed,
            warmup_samples=1000,
            metrics_window=1000,
            adaptive_k_enabled=p.get("adaptive_k", True),
            k_min=1,
            k_max=70,
            unlearning_enabled=p.get("unlearning", False),
            selection_strategy="oob_influence",
            proactive_enabled=p.get("unlearning", False),
            drift_type_detection_enabled=p.get("unlearning", False),
            smart_cooldown_enabled=p.get("unlearning", False),
            budget_enabled=p.get("budget", False),
            budget_max_samples=p.get("budget_max_samples", 10000),
            budget_eviction_batch=100,
            budget_minority_protection=0.1,
            influence_tracking=p.get("influence_tracking", False),
            budget_skip_forest_forget=p.get("budget_skip_forest_forget", False),
        )
    elif config.model_type == "arf":
        return ARFModel(n_models=50, seed=seed)
    elif config.model_type == "arf_oversample":
        return ARFWithOversampling(
            n_models=50, seed=seed,
            target_ratio=config.params.get("target_ratio", 0.3),
        )
    elif config.model_type == "arf_classweight":
        return ARFWithClassWeight(
            n_models=50, seed=seed,
            minority_weight=config.params.get("minority_weight", 10),
        )
    elif config.model_type == "sliding_window":
        return LimitedMemoryARF(
            n_models=50, seed=seed,
            max_samples=config.params.get("max_samples", 3000),
            retrain_frequency=config.params.get("retrain_frequency", 5),
            enable_retrain=config.params.get("enable_retrain", True),
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


# =============================================================================
# Run Single Experiment
# =============================================================================


def run_single(
    dataset: str,
    scenario_name: str,
    config: ModelConfig,
    seed: int,
) -> dict:
    """Run a single experiment with any model type."""
    num_features = DATASET_FEATURES[dataset]

    # Create data stream
    scenario_fn = SCENARIOS[scenario_name]["fn"]
    X_stream, y_stream, metadata = scenario_fn(dataset, seed=seed)

    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)

    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    model = create_model(config, num_features, seed)

    # Warmup
    if config.model_type == "suda":
        model.fit(X_warmup, y_warmup.astype(bool))
    else:
        # ARF baselines: batch warmup via partial_fit
        model.partial_fit(X_warmup, y_warmup.astype(np.int64))

    # Streaming (test-then-train)
    all_preds = []
    all_labels = []
    n_unlearning = 0
    registry_sizes = []

    n_batches = len(X_test) // BATCH_SIZE
    start_time = time.time()

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        X_batch = X_test[start:end]
        y_batch_raw = y_test[start:end]

        if config.model_type == "suda":
            y_batch = y_batch_raw.astype(bool)
            result = model.partial_fit(X_batch, y_batch, record_history=False)
            preds = result.predictions.astype(int)
            registry_sizes.append(result.registry_size)
        else:
            # ARF baselines: predict first, then learn
            preds = model.predict(X_batch)
            model.partial_fit(X_batch, y_batch_raw.astype(np.int64))

        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch_raw.astype(int).tolist())

    elapsed = time.time() - start_time

    # Overall metrics
    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    overall_gmean = compute_gmean(y_true, y_pred)
    overall_f1 = compute_f1(y_true, y_pred)
    overall_bacc = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    attack_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    benign_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Phase metrics
    phase_boundaries = metadata.get("phase_boundaries", [])
    adjusted_boundaries = [max(0, b - warmup_n) for b in phase_boundaries if b > warmup_n]
    phase_metrics = compute_phase_metrics(y_true, y_pred, adjusted_boundaries)

    # Budget stats (SUDA only)
    total_budget_evicted = 0
    budget_stats = {}
    if config.model_type == "suda" and config.params.get("budget", False):
        total_budget_evicted = model.total_budget_evicted
        budget_stats = model.get_budget_eviction_stats()

    return {
        "dataset": dataset,
        "scenario": scenario_name,
        "config": config.name,
        "seed": seed,
        "gmean": overall_gmean,
        "f1": overall_f1,
        "balanced_accuracy": overall_bacc,
        "attack_recall": attack_recall,
        "benign_recall": benign_recall,
        "n_unlearning_events": n_unlearning,
        "total_budget_evicted": total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
        "phase_metrics": phase_metrics,
        "budget_stats": budget_stats,
    }


# =============================================================================
# ANoShift Experiment
# =============================================================================

ANOSHIFT_CONFIGS = [
    ModelConfig("AK-Only", "suda", {
        "adaptive_k": True, "budget": False, "unlearning": False,
    }),
    ModelConfig("Budget-FIFO", "suda", {
        "adaptive_k": True, "budget": True, "budget_max_samples": 3000,
        "unlearning": False, "influence_tracking": False,
    }),
    ModelConfig("Trigger-Conservative", "suda", {
        "adaptive_k": True, "budget": False, "unlearning": True,
    }),
]


def run_anoshift_single(
    config: ModelConfig,
    seed: int,
) -> dict:
    """Run a single ANoShift experiment."""
    X_stream, y_stream, metadata = make_anoshift_temporal_stream(
        samples_per_year=5000, seed=seed,
    )

    # Determine feature count from data
    num_features = X_stream.shape[1]

    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)

    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    model = create_model(config, num_features, seed)

    # Warmup
    model.fit(X_warmup, y_warmup.astype(bool))

    # Streaming
    all_preds = []
    all_labels = []
    n_unlearning = 0
    registry_sizes = []

    n_batches = len(X_test) // BATCH_SIZE
    start_time = time.time()

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx].astype(bool)

        result = model.partial_fit(X_batch, y_batch, record_history=False)
        all_preds.extend(result.predictions.astype(int).tolist())
        all_labels.extend(y_batch.astype(int).tolist())

        registry_sizes.append(result.registry_size)

    elapsed = time.time() - start_time

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    total_budget_evicted = 0
    budget_stats = {}
    if config.params.get("budget", False):
        total_budget_evicted = model.total_budget_evicted
        budget_stats = model.get_budget_eviction_stats()

    return {
        "dataset": "anoshift",
        "scenario": "temporal_10year",
        "config": config.name,
        "seed": seed,
        "gmean": compute_gmean(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "benign_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "n_unlearning_events": n_unlearning,
        "total_budget_evicted": total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
        "budget_stats": budget_stats,
        "metadata": metadata,
    }


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    results: list[dict],
    datasets: list[str],
    scenarios: list[str],
    configs: list[ModelConfig],
    seeds: list[int],
    anoshift_results: list[dict] | None = None,
) -> str:
    lines = [
        "# Forgetting Granularity Benchmark Results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {seeds}",
        f"Configs: {[c.name for c in configs]}",
        "",
    ]

    # Per-scenario tables
    for scenario_name in scenarios:
        desc = SCENARIOS[scenario_name]["description"]
        lines.append(f"## Scenario: {scenario_name} ({desc})")
        lines.append("")
        lines.append("| Dataset | Config | G-mean (mean±std) | F1 (mean±std) | Atk Recall | Time(s) |")
        lines.append("|---------|--------|------------------|---------------|------------|---------|")

        for dataset in datasets:
            for config in configs:
                matching = [
                    r for r in results
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config.name
                    and "error" not in r
                ]
                if not matching:
                    lines.append(f"| {dataset} | {config.name} | - | - | - | - |")
                    continue

                gmeans = [r["gmean"] for r in matching]
                f1s = [r["f1"] for r in matching]
                atk = [r["attack_recall"] for r in matching]
                times = [r["elapsed_seconds"] for r in matching]

                lines.append(
                    f"| {dataset} | {config.name} | "
                    f"{np.mean(gmeans):.4f}±{np.std(gmeans):.4f} | "
                    f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} | "
                    f"{np.mean(atk):.4f} | "
                    f"{np.mean(times):.1f} |"
                )
        lines.append("")

    # Key comparisons with statistics
    comparison_pairs = [
        ("SUDA-Full vs ARF", "SUDA-Full", "ARF"),
        ("SUDA-Full vs SUDA-AK-Only", "SUDA-Full", "SUDA-AK-Only"),
        ("SUDA-Full vs SlidingWindow-ARF", "SUDA-Full", "SlidingWindow-ARF"),
        ("SUDA-Full vs SUDA-NoForgetBudget", "SUDA-Full", "SUDA-NoForgetBudget"),
    ]

    for pair_name, config_a, config_b in comparison_pairs:
        lines.append(f"## {pair_name}")
        lines.append("")
        lines.append("| Dataset | Scenario | A G-mean | B G-mean | Δ G-mean | p-value (Holm) | Cohen's d |")
        lines.append("|---------|----------|----------|----------|----------|----------------|-----------|")

        raw_p_values = []
        comparison_keys = []
        comparison_rows = []

        for dataset in datasets:
            for scenario_name in scenarios:
                a_results = [
                    r for r in results
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config_a
                    and "error" not in r
                ]
                b_results = [
                    r for r in results
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config_b
                    and "error" not in r
                ]

                if not a_results or not b_results:
                    continue

                a_gmeans = [r["gmean"] for r in a_results]
                b_gmeans = [r["gmean"] for r in b_results]
                n = min(len(a_gmeans), len(b_gmeans))
                delta = np.mean(a_gmeans[:n]) - np.mean(b_gmeans[:n])

                p_str = "N/A"
                d_str = "N/A"
                if n >= 5:
                    try:
                        from scipy.stats import wilcoxon
                        stat, p_val = wilcoxon(
                            a_gmeans[:n], b_gmeans[:n], alternative="greater"
                        )
                        raw_p_values.append(p_val)
                        comparison_keys.append((dataset, scenario_name))
                        diff = np.array(a_gmeans[:n]) - np.array(b_gmeans[:n])
                        pooled_sd = np.sqrt(
                            (np.std(a_gmeans[:n], ddof=1)**2
                             + np.std(b_gmeans[:n], ddof=1)**2) / 2
                        )
                        cohens_d = np.mean(diff) / pooled_sd if pooled_sd > 0 else 0.0
                        p_str = f"{p_val:.4f}"
                        d_str = f"{cohens_d:.2f}"
                    except Exception:
                        pass

                sign = "+" if delta > 0 else ""
                comparison_rows.append(
                    (dataset, scenario_name,
                     np.mean(a_gmeans[:n]), np.mean(b_gmeans[:n]),
                     delta, sign, p_str, d_str)
                )

        # Holm-Bonferroni correction
        corrected_p_strs = {}
        if raw_p_values:
            n_tests = len(raw_p_values)
            sorted_indices = np.argsort(raw_p_values)
            for rank, idx in enumerate(sorted_indices):
                corrected_p = min(raw_p_values[idx] * (n_tests - rank), 1.0)
                key = comparison_keys[idx]
                sig = "*" if corrected_p < 0.05 else ""
                corrected_p_strs[key] = f"{corrected_p:.4f}{sig}"

        for row in comparison_rows:
            dataset, scenario_name, a_mean, b_mean, delta, sign, p_str, d_str = row
            key = (dataset, scenario_name)
            holm_p = corrected_p_strs.get(key, p_str)
            lines.append(
                f"| {dataset} | {scenario_name} | "
                f"{a_mean:.4f} | "
                f"{b_mean:.4f} | "
                f"{sign}{delta:.4f} | "
                f"{holm_p} | "
                f"{d_str} |"
            )
        lines.append("")

    # Speed comparison
    lines.append("## Speed Comparison (SUDA vs ARF)")
    lines.append("")
    lines.append("| Dataset | Scenario | SUDA-Full (s) | ARF (s) | Speedup |")
    lines.append("|---------|----------|---------------|---------|---------|")

    for dataset in datasets:
        for scenario_name in scenarios:
            suda_times = [
                r["elapsed_seconds"] for r in results
                if r.get("dataset") == dataset
                and r.get("scenario") == scenario_name
                and r.get("config") == "SUDA-Full"
                and "error" not in r
            ]
            arf_times = [
                r["elapsed_seconds"] for r in results
                if r.get("dataset") == dataset
                and r.get("scenario") == scenario_name
                and r.get("config") == "ARF"
                and "error" not in r
            ]
            if suda_times and arf_times:
                suda_t = np.mean(suda_times)
                arf_t = np.mean(arf_times)
                speedup = arf_t / suda_t if suda_t > 0 else 0
                lines.append(
                    f"| {dataset} | {scenario_name} | "
                    f"{suda_t:.2f} | {arf_t:.2f} | {speedup:.0f}x |"
                )
    lines.append("")

    # ANoShift results
    if anoshift_results:
        lines.append("## ANoShift Natural Drift (10-Year Temporal)")
        lines.append("")
        lines.append("| Config | G-mean (mean±std) | F1 (mean±std) | Atk Recall | Triggers | Budget Evicted |")
        lines.append("|--------|------------------|---------------|------------|----------|----------------|")

        for ac in ANOSHIFT_CONFIGS:
            matching = [
                r for r in anoshift_results
                if r.get("config") == ac.name and "error" not in r
            ]
            if not matching:
                continue
            gmeans = [r["gmean"] for r in matching]
            f1s = [r["f1"] for r in matching]
            atk = [r["attack_recall"] for r in matching]
            triggers = [r["n_unlearning_events"] for r in matching]
            budgets = [r["total_budget_evicted"] for r in matching]

            lines.append(
                f"| {ac.name} | "
                f"{np.mean(gmeans):.4f}±{np.std(gmeans):.4f} | "
                f"{np.mean(f1s):.4f}±{np.std(f1s):.4f} | "
                f"{np.mean(atk):.4f} | "
                f"{np.mean(triggers):.1f} | "
                f"{int(np.mean(budgets))} |"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(
    datasets: list[str],
    scenarios: list[str],
    configs: list[ModelConfig],
    seeds: list[int],
    output_dir: Path,
    run_anoshift: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(datasets) * len(scenarios) * len(configs) * len(seeds)
    logger.info(f"Starting Granularity Benchmark: {total_runs} runs")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Scenarios: {scenarios}")
    logger.info(f"  Configs: {[c.name for c in configs]}")
    logger.info(f"  Seeds: {seeds}")

    all_results = []
    run_count = 0

    for dataset in datasets:
        for scenario_name in scenarios:
            for config in configs:
                for seed in seeds:
                    run_count += 1
                    logger.info(
                        f"[{run_count}/{total_runs}] "
                        f"{dataset}/{scenario_name}/{config.name}/seed={seed}"
                    )
                    try:
                        result = run_single(dataset, scenario_name, config, seed)
                        all_results.append(result)
                        logger.info(
                            f"  → G-mean={result['gmean']:.4f}, "
                            f"F1={result['f1']:.4f}, "
                            f"Time={result['elapsed_seconds']:.1f}s"
                        )
                    except Exception as e:
                        logger.error(f"  → FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            "dataset": dataset,
                            "scenario": scenario_name,
                            "config": config.name,
                            "seed": seed,
                            "error": str(e),
                        })

    # ANoShift
    anoshift_results = None
    if run_anoshift:
        anoshift_results = []
        anoshift_seeds = seeds[:3]  # 3 seeds for ANoShift
        total_anoshift = len(ANOSHIFT_CONFIGS) * len(anoshift_seeds)
        logger.info(f"\nStarting ANoShift Benchmark: {total_anoshift} runs")

        ano_count = 0
        for config in ANOSHIFT_CONFIGS:
            for seed in anoshift_seeds:
                ano_count += 1
                logger.info(
                    f"[ANoShift {ano_count}/{total_anoshift}] "
                    f"{config.name}/seed={seed}"
                )
                try:
                    result = run_anoshift_single(config, seed)
                    anoshift_results.append(result)
                    logger.info(
                        f"  → G-mean={result['gmean']:.4f}, "
                        f"Triggers={result['n_unlearning_events']}, "
                        f"Budget={result['total_budget_evicted']}"
                    )
                except Exception as e:
                    logger.error(f"  → FAILED: {e}")
                    anoshift_results.append({
                        "dataset": "anoshift",
                        "config": config.name,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save raw results
    results_path = output_dir / "granularity_benchmark_results.json"
    save_data = {
        "main_results": all_results,
        "anoshift_results": anoshift_results,
        "metadata": {
            "datasets": datasets,
            "scenarios": scenarios,
            "configs": [c.name for c in configs],
            "seeds": seeds,
            "timestamp": datetime.now().isoformat(),
        },
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved to {results_path}")

    # Generate summary
    summary = generate_summary(
        all_results, datasets, scenarios, configs, seeds,
        anoshift_results=anoshift_results,
    )

    summary_path = output_dir / "granularity_benchmark_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"Summary saved to {summary_path}")

    return {"results": all_results, "anoshift_results": anoshift_results, "summary_path": str(summary_path)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Forgetting Granularity Benchmark")
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Datasets to test",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIOS.keys()),
        help="Drift scenarios to test",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/granularity_benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test (1 seed, 1 dataset, 1 scenario)",
    )
    parser.add_argument(
        "--anoshift", action="store_true",
        help="Include ANoShift natural drift experiment",
    )

    args = parser.parse_args()

    if args.smoke:
        datasets = ["nslkdd"]
        scenarios = ["moderate_sudden"]
        seeds = [42]
    else:
        datasets = args.datasets
        scenarios = args.scenarios
        seeds = args.seeds

    output_dir = Path(args.output_dir)

    run_benchmark(
        datasets, scenarios, CONFIGS, seeds, output_dir,
        run_anoshift=args.anoshift or not args.smoke,
    )


if __name__ == "__main__":
    main()
