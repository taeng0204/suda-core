"""Feature Drift × Selection Strategy Benchmark.

Tests whether influence-based selective unlearning provides value over FIFO
in **feature drift** scenarios (attack type shift, label ratio fixed).

Key hypothesis:
  - Label shift: FIFO ≈ Selective (all old samples equally harmful) — confirmed
  - Feature drift: Selective > FIFO (only specific feature-region samples harmful)

Configurations:
  1. Pure-FIFO:       age_weight=1.0, influence_weight=0.0, class_weight=0.0
  2. Pure-Influence:  age_weight=0.0, influence_weight=0.8, class_weight=0.2
  3. Balanced:        age_weight=0.4, influence_weight=0.4, class_weight=0.2

Drift Types:
  1. Label shift:   make_moderate_sudden_drift_stream (1% → 20%) — control
  2. Feature drift: make_attack_type_shift_stream (DoS→Probe→R2L, ratio fixed 1%)

Design: 2 drift types × 3 configs × 3 datasets × 5 seeds = 90 runs

Usage:
    # Quick smoke test
    uv run python -m src.experiments.feature_drift_benchmark --smoke

    # Full benchmark
    uv run python -m src.experiments.feature_drift_benchmark

    # Specific dataset
    uv run python -m src.experiments.feature_drift_benchmark --datasets nslkdd
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from src.data.nids import (
    make_attack_type_shift_stream,
    make_moderate_sudden_drift_stream,
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
WARMUP_RATIO = 0.3


@dataclass
class SelectionConfig:
    """Selection strategy configuration for budget eviction."""
    name: str
    budget_age_weight: float
    budget_influence_weight: float
    budget_class_weight: float
    influence_tracking: bool
    influence_update_interval: int = 10
    influence_sample_count: int = 200


SELECTION_CONFIGS = [
    SelectionConfig(
        name="Pure-FIFO",
        budget_age_weight=1.0,
        budget_influence_weight=0.0,
        budget_class_weight=0.0,
        influence_tracking=False,
    ),
    SelectionConfig(
        name="Pure-Influence",
        budget_age_weight=0.0,
        budget_influence_weight=0.8,
        budget_class_weight=0.2,
        influence_tracking=True,
        influence_update_interval=5,
        influence_sample_count=500,
    ),
    SelectionConfig(
        name="Balanced",
        budget_age_weight=0.4,
        budget_influence_weight=0.4,
        budget_class_weight=0.2,
        influence_tracking=True,
    ),
]


DRIFT_SCENARIOS = {
    "label_shift": {
        "fn": lambda name, seed: make_moderate_sudden_drift_stream(name, seed=seed),
        "description": "Label shift: 1% → 20% attack ratio",
    },
    "feature_drift": {
        "fn": lambda name, seed: make_attack_type_shift_stream(name, seed=seed),
        "description": "Feature drift: attack type shift (ratio fixed 1%)",
    },
}


# =============================================================================
# Model Creation & Single Run
# =============================================================================

def create_model(
    selection_config: SelectionConfig,
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
        # Adaptive-k always on
        adaptive_k_enabled=True,
        k_min=1,
        k_max=70,
        # No reactive triggers — budget only
        unlearning_enabled=False,
        proactive_enabled=False,
        drift_type_detection_enabled=False,
        smart_cooldown_enabled=False,
        # Budget management ON
        budget_enabled=True,
        budget_max_samples=3000,
        budget_eviction_batch=100,
        budget_minority_protection=0.1,
        # Selection strategy weights
        budget_age_weight=selection_config.budget_age_weight,
        budget_influence_weight=selection_config.budget_influence_weight,
        budget_class_weight=selection_config.budget_class_weight,
        # Influence tracking
        influence_tracking=selection_config.influence_tracking,
        influence_update_interval=selection_config.influence_update_interval,
        influence_sample_count=selection_config.influence_sample_count,
    )


def run_single(
    dataset: str,
    drift_type: str,
    selection_config: SelectionConfig,
    seed: int,
) -> dict:
    """Run a single experiment."""
    num_features = DATASET_FEATURES[dataset]

    # Create data stream
    scenario_fn = DRIFT_SCENARIOS[drift_type]["fn"]
    X_stream, y_stream, metadata = scenario_fn(dataset, seed)

    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)

    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    model = create_model(selection_config, num_features, seed)
    model.fit(X_warmup, y_warmup.astype(bool))

    # Streaming
    all_preds = []
    all_labels = []
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
    total_budget_evicted = model.total_budget_evicted

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    overall_gmean = compute_gmean(y_true, y_pred)
    overall_bacc = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    attack_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    benign_recall = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))

    phase_boundaries = metadata.get("phase_boundaries", [])
    adjusted_boundaries = [max(0, b - warmup_n) for b in phase_boundaries if b > warmup_n]
    phase_metrics = compute_phase_metrics(y_true, y_pred, adjusted_boundaries)

    budget_stats = model.get_budget_eviction_stats()

    return {
        "dataset": dataset,
        "drift_type": drift_type,
        "config": selection_config.name,
        "seed": seed,
        "gmean": overall_gmean,
        "balanced_accuracy": overall_bacc,
        "attack_recall": attack_recall,
        "benign_recall": benign_recall,
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
# Benchmark Runner
# =============================================================================

def run_benchmark(
    datasets: list[str],
    drift_types: list[str],
    configs: list[SelectionConfig],
    seeds: list[int],
    output_dir: Path,
) -> list[dict]:
    """Run full benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    total_runs = len(datasets) * len(drift_types) * len(configs) * len(seeds)
    current = 0

    for dataset in datasets:
        for drift_type in drift_types:
            for config in configs:
                for seed in seeds:
                    current += 1
                    logger.info(
                        f"[{current}/{total_runs}] {dataset} / {drift_type} / "
                        f"{config.name} / seed={seed}"
                    )

                    try:
                        result = run_single(dataset, drift_type, config, seed)
                        results.append(result)
                        logger.info(
                            f"  G-mean={result['gmean']:.4f}, "
                            f"Atk Recall={result['attack_recall']:.4f}, "
                            f"Evicted={result['total_budget_evicted']}"
                        )
                    except Exception as e:
                        logger.error(f"  FAILED: {e}")
                        results.append({
                            "dataset": dataset,
                            "drift_type": drift_type,
                            "config": config.name,
                            "seed": seed,
                            "error": str(e),
                        })

    # Save results
    results_file = output_dir / "feature_drift_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved: {results_file}")

    # Generate summary
    summary = generate_summary(results, datasets, drift_types, configs)
    summary_file = output_dir / "feature_drift_summary.md"
    with open(summary_file, "w") as f:
        f.write(summary)
    logger.info(f"Summary saved: {summary_file}")

    return results


# =============================================================================
# Summary & Statistical Analysis
# =============================================================================

def generate_summary(
    results: list[dict],
    datasets: list[str],
    drift_types: list[str],
    configs: list[SelectionConfig],
) -> str:
    lines = []
    lines.append("# Feature Drift × Selection Strategy Benchmark Results")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nTotal runs: {len(results)}")
    lines.append("")

    # Per drift-type × dataset comparison
    for drift_type in drift_types:
        desc = DRIFT_SCENARIOS[drift_type]["description"]
        lines.append(f"## {drift_type}: {desc}")
        lines.append("")
        lines.append("| Dataset | Config | G-mean (mean±std) | Atk Recall | Benign Recall | Evicted |")
        lines.append("|---------|--------|-------------------|------------|---------------|---------|")

        for dataset in datasets:
            for config in configs:
                cfg_results = [
                    r for r in results
                    if r.get("dataset") == dataset
                    and r.get("drift_type") == drift_type
                    and r.get("config") == config.name
                    and "error" not in r
                ]
                if not cfg_results:
                    continue

                gmeans = [r["gmean"] for r in cfg_results]
                atk_recalls = [r["attack_recall"] for r in cfg_results]
                ben_recalls = [r["benign_recall"] for r in cfg_results]
                evicted = [r["total_budget_evicted"] for r in cfg_results]

                lines.append(
                    f"| {dataset} | {config.name} | "
                    f"{np.mean(gmeans):.4f}±{np.std(gmeans):.4f} | "
                    f"{np.mean(atk_recalls):.4f} | "
                    f"{np.mean(ben_recalls):.4f} | "
                    f"{int(np.mean(evicted))} |"
                )

        lines.append("")

    # Key comparison: Pure-Influence vs Pure-FIFO
    lines.append("## Key Comparison: Pure-Influence vs Pure-FIFO")
    lines.append("")
    lines.append("| Drift Type | Dataset | FIFO G-mean | Influence G-mean | Δ G-mean | p-value (Holm) | Cohen's d |")
    lines.append("|------------|---------|-------------|------------------|----------|----------------|-----------|")

    raw_p_values = []
    comparison_keys = []
    comparison_rows = []

    for drift_type in drift_types:
        for dataset in datasets:
            fifo_results = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("drift_type") == drift_type
                and r.get("config") == "Pure-FIFO"
                and "error" not in r
            ]
            inf_results = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("drift_type") == drift_type
                and r.get("config") == "Pure-Influence"
                and "error" not in r
            ]

            if len(fifo_results) < 3 or len(inf_results) < 3:
                continue

            # Sort by seed for pairing
            fifo_results.sort(key=lambda r: r.get("seed", 0))
            inf_results.sort(key=lambda r: r.get("seed", 0))

            fifo_g = [r["gmean"] for r in fifo_results]
            inf_g = [r["gmean"] for r in inf_results]
            n = min(len(fifo_g), len(inf_g))

            delta = np.mean(inf_g[:n]) - np.mean(fifo_g[:n])

            # Wilcoxon signed-rank (paired)
            p_str = "N/A"
            d_str = "N/A"
            if n >= 5:
                try:
                    from scipy.stats import wilcoxon
                    stat, p_val = wilcoxon(
                        inf_g[:n], fifo_g[:n], alternative="greater"
                    )
                    raw_p_values.append(p_val)
                    comparison_keys.append((drift_type, dataset))

                    diff = np.array(inf_g[:n]) - np.array(fifo_g[:n])
                    pooled_sd = np.sqrt(
                        (np.std(fifo_g[:n], ddof=1)**2
                         + np.std(inf_g[:n], ddof=1)**2) / 2
                    )
                    cohens_d = np.mean(diff) / pooled_sd if pooled_sd > 0 else 0.0
                    p_str = f"{p_val:.4f}"
                    d_str = f"{cohens_d:.2f}"
                except Exception:
                    pass

            sign = "+" if delta > 0 else ""
            comparison_rows.append(
                (drift_type, dataset,
                 np.mean(fifo_g[:n]), np.mean(inf_g[:n]),
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
        drift_type, dataset, fifo_mean, inf_mean, delta, sign, p_str, d_str = row
        key = (drift_type, dataset)
        holm_p = corrected_p_strs.get(key, p_str)
        lines.append(
            f"| {drift_type} | {dataset} | "
            f"{fifo_mean:.4f} | "
            f"{inf_mean:.4f} | "
            f"{sign}{delta:.4f} | "
            f"{holm_p} | "
            f"{d_str} |"
        )

    lines.append("")

    # Interaction effect analysis
    lines.append("## Interaction Effect: Drift Type × Selection Strategy")
    lines.append("")

    for dataset in datasets:
        lines.append(f"### {dataset}")
        lines.append("")

        # Get deltas for each drift type
        for drift_type in drift_types:
            fifo_r = sorted(
                [r for r in results
                 if r.get("dataset") == dataset
                 and r.get("drift_type") == drift_type
                 and r.get("config") == "Pure-FIFO"
                 and "error" not in r],
                key=lambda r: r.get("seed", 0)
            )
            inf_r = sorted(
                [r for r in results
                 if r.get("dataset") == dataset
                 and r.get("drift_type") == drift_type
                 and r.get("config") == "Pure-Influence"
                 and "error" not in r],
                key=lambda r: r.get("seed", 0)
            )

            if fifo_r and inf_r:
                n = min(len(fifo_r), len(inf_r))
                deltas = [inf_r[i]["gmean"] - fifo_r[i]["gmean"] for i in range(n)]
                lines.append(
                    f"- **{drift_type}**: Influence - FIFO = "
                    f"{np.mean(deltas):+.4f} ± {np.std(deltas):.4f} "
                    f"(per-seed: {[f'{d:+.4f}' for d in deltas]})"
                )

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature Drift × Selection Strategy Benchmark"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS, help="Datasets to test"
    )
    parser.add_argument(
        "--drift_types", nargs="+", default=list(DRIFT_SCENARIOS.keys()),
        help="Drift types to test"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/feature_drift_benchmark",
        help="Output directory"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test (1 seed, 1 dataset)"
    )

    args = parser.parse_args()

    if args.smoke:
        datasets = ["nslkdd"]
        drift_types = ["feature_drift"]
        seeds = [42]
        configs = [SELECTION_CONFIGS[0], SELECTION_CONFIGS[1]]  # FIFO + Influence
    else:
        datasets = args.datasets
        drift_types = args.drift_types
        seeds = args.seeds
        configs = SELECTION_CONFIGS

    output_dir = Path(args.output_dir)

    total = len(datasets) * len(drift_types) * len(configs) * len(seeds)
    logger.info(f"Feature Drift Benchmark: {total} runs")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Drift types: {drift_types}")
    logger.info(f"  Configs: {[c.name for c in configs]}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Output: {output_dir}")

    run_benchmark(datasets, drift_types, configs, seeds, output_dir)


if __name__ == "__main__":
    main()
