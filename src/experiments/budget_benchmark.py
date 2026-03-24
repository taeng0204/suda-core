"""Budget Management Benchmark: Continuous Sample Quality Maintenance.

This benchmark tests whether sample budget management (continuous eviction)
provides value beyond Adaptive-k alone.

Hypothesis:
  Full SUDA > AdaptiveK-Only when budget management continuously removes
  stale/harmful samples, especially in intermediate drift scenarios where
  reactive triggers don't fire.

Configurations:
  1. AdaptiveK-Only:   Adaptive-k ON, Trigger OFF, Budget OFF
  2. Trigger-Only:     Adaptive-k ON, Trigger ON,  Budget OFF
  3. Budget-Only:      Adaptive-k ON, Trigger OFF, Budget ON
  4. Full-v2:          Adaptive-k ON, Trigger ON,  Budget ON, Influence Track ON

Scenarios (intermediate drift - between mild and extreme):
  1. Moderate Sudden:   1% → 20%
  2. Step-wise:         1% → 10% → 20% → 10% → 5%
  3. Gradual Ramp:      1% → 5% → 10% → 15% → 20%
  4. Asymmetric Recovery: 1% → 20% → 3%

Usage:
    # Quick smoke test (1 seed, 1 dataset)
    uv run python -m src.experiments.budget_benchmark --smoke

    # Full benchmark (5 seeds, 3 datasets, 4 scenarios)
    uv run python -m src.experiments.budget_benchmark

    # Specific dataset
    uv run python -m src.experiments.budget_benchmark --datasets nslkdd
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from src.data.nids import (
    create_imbalanced_dataset,
    get_dataset_info,
    make_asymmetric_recovery_stream,
    make_gradual_ramp_stream,
    make_moderate_sudden_drift_stream,
    make_stepwise_drift_stream,
)
from src.models.suda import SUDA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from src.experiments.utils import NumpyEncoder, compute_gmean, compute_phase_metrics  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

DATASETS = ["nslkdd", "unswnb15", "cicids2018"]
DATASET_FEATURES = {"nslkdd": 41, "unswnb15": 42, "cicids2018": 78}
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
BATCH_SIZE = 200
WARMUP_RATIO = 0.3  # 30% for warmup fit


@dataclass
class ModelConfig:
    name: str
    adaptive_k: bool = True
    unlearning: bool = False
    budget: bool = False
    influence_tracking: bool = False
    budget_max_samples: int = 10000
    budget_skip_forest_forget: bool = False


CONFIGS = [
    ModelConfig("AdaptiveK-Only", adaptive_k=True, unlearning=False, budget=False),
    ModelConfig("Trigger-Only", adaptive_k=True, unlearning=True, budget=False),
    ModelConfig("Budget-Only", adaptive_k=True, unlearning=False, budget=True, influence_tracking=False, budget_max_samples=3000),
    ModelConfig("Full-v2", adaptive_k=True, unlearning=True, budget=True, influence_tracking=True, budget_max_samples=3000),
]

# Additional configs for "HOW > WHAT" ablation (No-Forest-Forget experiment)
CONFIGS_HOW_VS_WHAT = [
    ModelConfig("AdaptiveK-Only", adaptive_k=True, unlearning=False, budget=False),
    ModelConfig("No-Forest-Forget", adaptive_k=True, unlearning=False, budget=True, influence_tracking=False, budget_max_samples=3000, budget_skip_forest_forget=True),
    ModelConfig("FIFO-Forget", adaptive_k=True, unlearning=False, budget=True, influence_tracking=False, budget_max_samples=3000),
    ModelConfig("Full-v2", adaptive_k=True, unlearning=True, budget=True, influence_tracking=True, budget_max_samples=3000),
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
# Run Single Experiment
# =============================================================================


def create_model(
    config: ModelConfig,
    num_features: int,
    seed: int,
) -> SUDA:
    return SUDA(
        num_features=num_features,
        num_trees=50,
        k=10,
        max_depth=15,
        seed=seed,
        warmup_samples=1000,
        metrics_window=1000,
        adaptive_k_enabled=config.adaptive_k,
        k_min=1,
        k_max=70,
        unlearning_enabled=config.unlearning,
        selection_strategy="oob_influence",
        proactive_enabled=config.unlearning,
        drift_type_detection_enabled=config.unlearning,
        smart_cooldown_enabled=config.unlearning,
        budget_enabled=config.budget,
        budget_max_samples=config.budget_max_samples,
        budget_eviction_batch=100,
        budget_minority_protection=0.1,
        influence_tracking=config.influence_tracking,
        budget_skip_forest_forget=config.budget_skip_forest_forget,
    )


def run_single(
    dataset: str,
    scenario_name: str,
    config: ModelConfig,
    seed: int,
) -> dict:
    """Run a single experiment."""
    num_features = DATASET_FEATURES[dataset]

    # Create data stream
    scenario_fn = SCENARIOS[scenario_name]["fn"]
    X_stream, y_stream, metadata = scenario_fn(dataset, seed=seed)

    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)

    # Split into warmup and streaming
    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    # Create model
    model = create_model(config, num_features, seed)

    # Warmup fit
    model.fit(X_warmup, y_warmup.astype(bool))

    # Streaming
    all_preds = []
    all_labels = []
    n_unlearning = 0
    total_budget_evicted = 0
    registry_sizes = []

    n_batches = len(X_test) // BATCH_SIZE
    start_time = time.time()

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        X_batch = X_test[start:end]
        y_batch = y_test[start:end].astype(bool)

        result = model.partial_fit(X_batch, y_batch, record_history=False)

        all_preds.extend(result.predictions.tolist())
        all_labels.extend(y_batch.tolist())

        registry_sizes.append(result.registry_size)

    elapsed = time.time() - start_time
    total_budget_evicted = model.total_budget_evicted if config.budget else 0

    # Compute overall metrics
    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    overall_gmean = compute_gmean(y_true, y_pred)
    overall_bacc = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    attack_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    benign_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Compute phase metrics (adjust boundaries for warmup offset)
    phase_boundaries = metadata.get("phase_boundaries", [])
    adjusted_boundaries = [max(0, b - warmup_n) for b in phase_boundaries if b > warmup_n]
    phase_metrics = compute_phase_metrics(y_true, y_pred, adjusted_boundaries)

    # Budget stats
    budget_stats = {}
    if config.budget:
        budget_stats = model.get_budget_eviction_stats()

    return {
        "dataset": dataset,
        "scenario": scenario_name,
        "config": config.name,
        "seed": seed,
        "gmean": overall_gmean,
        "balanced_accuracy": overall_bacc,
        "attack_recall": attack_recall,
        "benign_recall": benign_recall,
        "n_unlearning_events": n_unlearning,
        "total_budget_evicted": total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "max_registry_size": max(registry_sizes) if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
        "phase_metrics": phase_metrics,
        "budget_stats": budget_stats,
        "metadata": metadata,
    }


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(
    datasets: list[str],
    scenarios: list[str],
    configs: list[ModelConfig],
    seeds: list[int],
    output_dir: Path,
) -> dict:
    """Run full benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(datasets) * len(scenarios) * len(configs) * len(seeds)
    logger.info(f"Starting benchmark: {total_runs} runs")
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
                            f"Unlearning={result['n_unlearning_events']}, "
                            f"Budget evicted={result['total_budget_evicted']}, "
                            f"Registry={result['final_registry_size']}"
                        )
                    except Exception as e:
                        logger.error(f"  → FAILED: {e}")
                        all_results.append({
                            "dataset": dataset,
                            "scenario": scenario_name,
                            "config": config.name,
                            "seed": seed,
                            "error": str(e),
                        })

    # Save raw results
    results_path = output_dir / "budget_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved to {results_path}")

    # Generate summary
    summary = generate_summary(all_results, datasets, scenarios, configs, seeds)

    summary_path = output_dir / "budget_benchmark_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"Summary saved to {summary_path}")

    return {"results": all_results, "summary_path": str(summary_path)}


def generate_summary(
    results: list[dict],
    datasets: list[str],
    scenarios: list[str],
    configs: list[ModelConfig],
    seeds: list[int],
) -> str:
    """Generate markdown summary table."""
    lines = [
        f"# Budget Management Benchmark Results",
        f"",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {seeds}",
        f"",
    ]

    # Per-scenario comparison table
    for scenario_name in scenarios:
        desc = SCENARIOS[scenario_name]["description"]
        lines.append(f"## Scenario: {scenario_name} ({desc})")
        lines.append("")
        lines.append("| Dataset | Config | G-mean (mean±std) | Atk Recall | Registry | Budget Evicted | Triggers |")
        lines.append("|---------|--------|------------------|------------|----------|----------------|----------|")

        for dataset in datasets:
            for config in configs:
                # Filter results
                matching = [
                    r for r in results
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config.name
                    and "error" not in r
                ]

                if not matching:
                    lines.append(f"| {dataset} | {config.name} | - | - | - | - | - |")
                    continue

                gmeans = [r["gmean"] for r in matching]
                atk_recalls = [r["attack_recall"] for r in matching]
                registries = [r["final_registry_size"] for r in matching]
                budgets = [r["total_budget_evicted"] for r in matching]
                triggers = [r["n_unlearning_events"] for r in matching]

                lines.append(
                    f"| {dataset} | {config.name} | "
                    f"{np.mean(gmeans):.4f}±{np.std(gmeans):.4f} | "
                    f"{np.mean(atk_recalls):.4f} | "
                    f"{int(np.mean(registries))} | "
                    f"{int(np.mean(budgets))} | "
                    f"{np.mean(triggers):.1f} |"
                )

        lines.append("")

    # Key comparison: Full-v2 vs AdaptiveK-Only
    lines.append("## Key Comparison: Full-v2 vs AdaptiveK-Only")
    lines.append("")
    lines.append("| Dataset | Scenario | AdaptiveK G-mean | Full-v2 G-mean | Δ G-mean | p-value (Holm) | Cohen's d |")
    lines.append("|---------|----------|-----------------|----------------|----------|----------------|-----------|")

    raw_p_values = []
    comparison_keys = []
    comparison_rows = []

    for dataset in datasets:
        for scenario_name in scenarios:
            ak_results = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("scenario") == scenario_name
                and r.get("config") == "AdaptiveK-Only"
                and "error" not in r
            ]
            fv2_results = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("scenario") == scenario_name
                and r.get("config") == "Full-v2"
                and "error" not in r
            ]

            if not ak_results or not fv2_results:
                continue

            ak_gmeans = [r["gmean"] for r in ak_results]
            fv2_gmeans = [r["gmean"] for r in fv2_results]

            delta = np.mean(fv2_gmeans) - np.mean(ak_gmeans)

            # Statistical test: Wilcoxon signed-rank (paired)
            if len(ak_gmeans) >= 5 and len(fv2_gmeans) >= 5:
                try:
                    from scipy.stats import wilcoxon
                    # Pair by seed order (same seed index)
                    n_pairs = min(len(ak_gmeans), len(fv2_gmeans))
                    stat, p_val = wilcoxon(
                        fv2_gmeans[:n_pairs], ak_gmeans[:n_pairs],
                        alternative="greater"
                    )
                    raw_p_values.append(p_val)
                    comparison_keys.append((dataset, scenario_name))
                    # Cohen's d (pooled SD)
                    diff = np.array(fv2_gmeans[:n_pairs]) - np.array(ak_gmeans[:n_pairs])
                    pooled_sd = np.sqrt(
                        (np.std(ak_gmeans[:n_pairs], ddof=1)**2
                         + np.std(fv2_gmeans[:n_pairs], ddof=1)**2) / 2
                    )
                    cohens_d = np.mean(diff) / pooled_sd if pooled_sd > 0 else 0.0
                    p_str = f"{p_val:.4f}"
                    d_str = f"{cohens_d:.2f}"
                except Exception:
                    p_str = "N/A"
                    d_str = "N/A"
            else:
                p_str = "N/A"
                d_str = "N/A"

            sign = "+" if delta > 0 else ""
            comparison_rows.append(
                (dataset, scenario_name,
                 np.mean(ak_gmeans), np.mean(fv2_gmeans),
                 delta, sign, p_str, d_str)
            )

    # Holm-Bonferroni correction for multiple comparisons
    corrected_p_strs = {}
    if raw_p_values:
        n_tests = len(raw_p_values)
        # Sort by p-value
        sorted_indices = np.argsort(raw_p_values)
        for rank, idx in enumerate(sorted_indices):
            corrected_p = min(raw_p_values[idx] * (n_tests - rank), 1.0)
            key = comparison_keys[idx]
            sig = "*" if corrected_p < 0.05 else ""
            corrected_p_strs[key] = f"{corrected_p:.4f}{sig}"

    for row in comparison_rows:
        dataset, scenario_name, ak_mean, fv2_mean, delta, sign, p_str, d_str = row
        key = (dataset, scenario_name)
        holm_p = corrected_p_strs.get(key, p_str)
        lines.append(
            f"| {dataset} | {scenario_name} | "
            f"{ak_mean:.4f} | "
            f"{fv2_mean:.4f} | "
            f"{sign}{delta:.4f} | "
            f"{holm_p} | "
            f"{d_str} |"
        )

    lines.append("")

    # Budget eviction analysis
    lines.append("## Budget Eviction Analysis")
    lines.append("")

    for dataset in datasets:
        budget_results = [
            r for r in results
            if r.get("dataset") == dataset
            and r.get("config") == "Full-v2"
            and "error" not in r
            and r.get("budget_stats")
        ]
        if not budget_results:
            continue

        lines.append(f"### {dataset}")
        total_evicted = np.mean([r["total_budget_evicted"] for r in budget_results])
        benign_evicted = np.mean([r["budget_stats"].get("benign", 0) for r in budget_results])
        attack_evicted = np.mean([r["budget_stats"].get("attack", 0) for r in budget_results])
        degraded_evicted = np.mean([r["budget_stats"].get("degraded", 0) for r in budget_results])

        lines.append(f"- Total evicted: {total_evicted:.0f}")
        lines.append(f"- Benign evicted: {benign_evicted:.0f}")
        lines.append(f"- Attack evicted: {attack_evicted:.0f}")
        lines.append(f"- Influence-degraded evicted: {degraded_evicted:.0f}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Budget Management Benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to test",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        help="Drift scenarios to test",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/budget_benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (1 seed, 1 dataset, 1 scenario)",
    )
    parser.add_argument(
        "--budget_sizes",
        nargs="+",
        type=int,
        default=None,
        help="Budget sizes to test (overrides default 10000)",
    )
    parser.add_argument(
        "--how_vs_what",
        action="store_true",
        help="Run HOW vs WHAT ablation (No-Forest-Forget, FIFO-Forget, Full-v2)",
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

    configs = CONFIGS_HOW_VS_WHAT if args.how_vs_what else CONFIGS

    # If testing different budget sizes
    if args.budget_sizes:
        configs = [CONFIGS[0]]  # AdaptiveK-Only as baseline
        for bs in args.budget_sizes:
            configs.append(
                ModelConfig(
                    f"Budget-{bs}",
                    adaptive_k=True,
                    unlearning=True,
                    budget=True,
                    influence_tracking=True,
                    budget_max_samples=bs,
                )
            )

    output_dir = Path(args.output_dir)

    run_benchmark(datasets, scenarios, configs, seeds, output_dir)


if __name__ == "__main__":
    main()
