"""Selective vs FIFO Benchmark: Influence-Aware Eviction Differentiation.

핵심 연구 질문:
  "Influence-aware selective eviction이 age-based FIFO eviction보다
   concept drift 적응에서 우수한가?"

Configurations:
  1. AdaptiveK-Only:   Budget OFF (upper bound without eviction)
  2. FIFO-Window:      Budget ON, age_weight=1.0, influence=0, class=0, protect=0
  3. FIFO-Protected:   Budget ON, age_weight=1.0, influence=0, class=0, protect=0.1
  4. Selective-Budget:  Budget ON, age=0.4, influence=0.4, class=0.2, protect=0.1, influence recompute ON
  5. Full-Selective:    Budget ON + influence recompute ON + reactive triggers ON

Scenarios (6):
  - moderate_sudden (1% → 20%)          : FIFO ≈ Selective
  - stepwise (1→10→20→10→5%)            : FIFO ≈ Selective
  - asymmetric_recovery (1→20→3%)        : Selective slight advantage
  - attack_decrease (20→5→1%)            : Selective >> FIFO (key scenario)
  - noisy_burst (1%+noise burst)         : Selective >> FIFO
  - class_oscillation (2↔20% x3)         : Selective > FIFO

Statistical test: Wilcoxon signed-rank (paired), not Mann-Whitney (independent).

Usage:
    # Quick smoke test
    uv run python -m src.experiments.selective_vs_fifo_benchmark --smoke

    # Full benchmark (5 seeds, 3 datasets, 6 scenarios = 450 runs)
    uv run python -m src.experiments.selective_vs_fifo_benchmark

    # Specific dataset
    uv run python -m src.experiments.selective_vs_fifo_benchmark --datasets nslkdd
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
    make_asymmetric_recovery_stream,
    make_attack_decrease_stream,
    make_class_oscillation_stream,
    make_moderate_sudden_drift_stream,
    make_noisy_burst_stream,
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
WARMUP_SIZE = 1000  # Fixed 1000 samples (not ratio-based)
BUDGET_SIZE = 3000


@dataclass
class ModelConfig:
    name: str
    adaptive_k: bool = True
    unlearning: bool = False
    budget: bool = False
    budget_max_samples: int = BUDGET_SIZE
    budget_age_weight: float = 0.4
    budget_influence_weight: float = 0.4
    budget_class_weight: float = 0.2
    budget_minority_protection: float = 0.1
    influence_tracking: bool = False
    influence_update_interval: int = 10
    influence_sample_count: int = 200


CONFIGS = [
    ModelConfig(
        "AdaptiveK-Only",
        budget=False,
        unlearning=False,
    ),
    ModelConfig(
        "FIFO-Window",
        budget=True,
        budget_age_weight=1.0,
        budget_influence_weight=0.0,
        budget_class_weight=0.0,
        budget_minority_protection=0.0,
        influence_tracking=False,
        influence_update_interval=0,
    ),
    ModelConfig(
        "FIFO-Protected",
        budget=True,
        budget_age_weight=1.0,
        budget_influence_weight=0.0,
        budget_class_weight=0.0,
        budget_minority_protection=0.1,
        influence_tracking=False,
        influence_update_interval=0,
    ),
    ModelConfig(
        "Selective-Budget",
        budget=True,
        budget_age_weight=0.4,
        budget_influence_weight=0.4,
        budget_class_weight=0.2,
        budget_minority_protection=0.1,
        influence_tracking=True,
        influence_update_interval=10,
        influence_sample_count=200,
        unlearning=False,  # Budget only, no reactive trigger
    ),
    ModelConfig(
        "Full-Selective",
        budget=True,
        budget_age_weight=0.4,
        budget_influence_weight=0.4,
        budget_class_weight=0.2,
        budget_minority_protection=0.1,
        influence_tracking=True,
        influence_update_interval=10,
        influence_sample_count=200,
        unlearning=True,  # Budget + reactive triggers
    ),
]


SCENARIOS = {
    "moderate_sudden": {
        "fn": make_moderate_sudden_drift_stream,
        "description": "1% → 20% sudden drift",
        "expected": "Tie (FIFO ≈ Selective)",
    },
    "stepwise": {
        "fn": make_stepwise_drift_stream,
        "description": "1% → 10% → 20% → 10% → 5%",
        "expected": "Tie (FIFO ≈ Selective)",
    },
    "asymmetric_recovery": {
        "fn": make_asymmetric_recovery_stream,
        "description": "1% → 20% → 3%",
        "expected": "Selective slight advantage",
    },
    "attack_decrease": {
        "fn": make_attack_decrease_stream,
        "description": "20% → 5% → 1%",
        "expected": "Selective >> FIFO (KEY)",
    },
    "noisy_burst": {
        "fn": make_noisy_burst_stream,
        "description": "1% + 10% label noise burst",
        "expected": "Selective >> FIFO",
    },
    "class_oscillation": {
        "fn": make_class_oscillation_stream,
        "description": "2% ↔ 20% × 3 cycles",
        "expected": "Selective > FIFO",
    },
}


# =============================================================================
# Run Single Experiment
# =============================================================================


def create_model(config: ModelConfig, num_features: int, seed: int) -> SUDA:
    return SUDA(
        num_features=num_features,
        num_trees=50,
        k=10,
        max_depth=15,
        seed=seed,
        warmup_samples=WARMUP_SIZE,
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
        budget_minority_protection=config.budget_minority_protection,
        budget_age_weight=config.budget_age_weight,
        budget_influence_weight=config.budget_influence_weight,
        budget_class_weight=config.budget_class_weight,
        influence_tracking=config.influence_tracking,
        influence_update_interval=config.influence_update_interval,
        influence_sample_count=config.influence_sample_count,
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

    # Fixed warmup size (not ratio-based)
    warmup_n = min(WARMUP_SIZE, total // 2)

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

    # Budget & influence diagnostics
    budget_stats = {}
    influence_cov = (0, 0)
    if config.budget:
        budget_stats = model.get_budget_eviction_stats()
        try:
            budget_stats_ext = model.get_budget_eviction_stats_extended()
            budget_stats.update(budget_stats_ext)
        except Exception:
            pass
        try:
            influence_cov = model.get_influence_coverage()
        except Exception:
            pass

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
        "total_budget_evicted": model.total_budget_evicted if config.budget else 0,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
        "phase_metrics": phase_metrics,
        "budget_stats": budget_stats,
        "influence_coverage": influence_cov,
        "metadata": {k: v for k, v in metadata.items() if k != "fn"},
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

                        inf_cov = result.get("influence_coverage", (0, 0))
                        inf_pct = (
                            f"{inf_cov[0]}/{inf_cov[1]} ({100*inf_cov[0]/max(inf_cov[1],1):.0f}%)"
                            if inf_cov[1] > 0 else "N/A"
                        )
                        logger.info(
                            f"  → G-mean={result['gmean']:.4f}, "
                            f"AtkRecall={result['attack_recall']:.4f}, "
                            f"Evicted={result['total_budget_evicted']}, "
                            f"InfCov={inf_pct}, "
                            f"Time={result['elapsed_seconds']:.1f}s"
                        )
                    except Exception as e:
                        logger.error(f"  → FAILED: {e}", exc_info=True)
                        all_results.append({
                            "dataset": dataset,
                            "scenario": scenario_name,
                            "config": config.name,
                            "seed": seed,
                            "error": str(e),
                        })

    # Save raw results
    results_path = output_dir / "selective_vs_fifo_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved to {results_path}")

    # Generate summary
    summary = generate_summary(all_results, datasets, scenarios, configs, seeds)

    summary_path = output_dir / "selective_vs_fifo_summary.md"
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
    """Generate markdown summary with Wilcoxon signed-rank tests."""
    lines = [
        "# Selective vs FIFO Benchmark Results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {seeds}",
        f"Warmup: {WARMUP_SIZE} (fixed)",
        f"Budget: {BUDGET_SIZE}",
        "",
    ]

    # Per-scenario comparison table
    for scenario_name in scenarios:
        desc = SCENARIOS[scenario_name]["description"]
        expected = SCENARIOS[scenario_name]["expected"]
        lines.append(f"## {scenario_name}: {desc}")
        lines.append(f"Expected: {expected}")
        lines.append("")
        lines.append("| Dataset | Config | G-mean (mean±std) | Atk Recall | Benign Recall | Inf Coverage | Evicted |")
        lines.append("|---------|--------|------------------|------------|---------------|-------------|---------|")

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
                    lines.append(f"| {dataset} | {config.name} | - | - | - | - | - |")
                    continue

                gmeans = [r["gmean"] for r in matching]
                atk_recalls = [r["attack_recall"] for r in matching]
                ben_recalls = [r["benign_recall"] for r in matching]
                evicted = [r["total_budget_evicted"] for r in matching]

                # Influence coverage
                inf_covs = [r.get("influence_coverage", (0, 0)) for r in matching]
                if any(c[1] > 0 for c in inf_covs):
                    avg_pct = np.mean([c[0]/max(c[1],1)*100 for c in inf_covs])
                    inf_str = f"{avg_pct:.0f}%"
                else:
                    inf_str = "N/A"

                lines.append(
                    f"| {dataset} | {config.name} | "
                    f"{np.mean(gmeans):.4f}±{np.std(gmeans):.4f} | "
                    f"{np.mean(atk_recalls):.4f} | "
                    f"{np.mean(ben_recalls):.4f} | "
                    f"{inf_str} | "
                    f"{int(np.mean(evicted))} |"
                )

        lines.append("")

    # KEY: FIFO-Protected vs Selective-Budget (paired Wilcoxon)
    lines.append("## KEY COMPARISON: FIFO-Protected vs Selective-Budget (Wilcoxon signed-rank)")
    lines.append("")
    lines.append("| Dataset | Scenario | FIFO-Prot G-mean | Selective G-mean | Δ G-mean | p-value | Winner |")
    lines.append("|---------|----------|-----------------|-----------------|----------|---------|--------|")

    win_counts = {"Selective-Budget": 0, "FIFO-Protected": 0, "Tie": 0}

    for dataset in datasets:
        for scenario_name in scenarios:
            fifo_results = sorted(
                [r for r in results
                 if r.get("dataset") == dataset
                 and r.get("scenario") == scenario_name
                 and r.get("config") == "FIFO-Protected"
                 and "error" not in r],
                key=lambda r: r["seed"],
            )
            sel_results = sorted(
                [r for r in results
                 if r.get("dataset") == dataset
                 and r.get("scenario") == scenario_name
                 and r.get("config") == "Selective-Budget"
                 and "error" not in r],
                key=lambda r: r["seed"],
            )

            if len(fifo_results) < 3 or len(sel_results) < 3:
                continue

            # Pair by seed
            fifo_by_seed = {r["seed"]: r["gmean"] for r in fifo_results}
            sel_by_seed = {r["seed"]: r["gmean"] for r in sel_results}
            common_seeds = sorted(set(fifo_by_seed) & set(sel_by_seed))

            if len(common_seeds) < 3:
                continue

            fifo_gmeans = [fifo_by_seed[s] for s in common_seeds]
            sel_gmeans = [sel_by_seed[s] for s in common_seeds]

            delta = np.mean(sel_gmeans) - np.mean(fifo_gmeans)

            # Wilcoxon signed-rank test (paired)
            try:
                from scipy.stats import wilcoxon
                diffs = [s - f for s, f in zip(sel_gmeans, fifo_gmeans)]
                if all(d == 0 for d in diffs):
                    p_str = "1.0"
                    winner = "Tie"
                else:
                    stat, p_val = wilcoxon(sel_gmeans, fifo_gmeans)
                    p_str = f"{p_val:.4f}" + ("*" if p_val < 0.05 else "")
                    if p_val < 0.05:
                        winner = "Selective" if delta > 0 else "FIFO"
                    else:
                        winner = "Tie"
            except Exception:
                p_str = "N/A"
                winner = "?"

            if winner == "Selective":
                win_counts["Selective-Budget"] += 1
            elif winner == "FIFO":
                win_counts["FIFO-Protected"] += 1
            else:
                win_counts["Tie"] += 1

            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {dataset} | {scenario_name} | "
                f"{np.mean(fifo_gmeans):.4f} | "
                f"{np.mean(sel_gmeans):.4f} | "
                f"{sign}{delta:.4f} | "
                f"{p_str} | "
                f"**{winner}** |"
            )

    lines.append("")
    lines.append(f"### Win Count: Selective={win_counts['Selective-Budget']}, "
                 f"FIFO={win_counts['FIFO-Protected']}, Tie={win_counts['Tie']}")
    lines.append("")

    # Influence coverage analysis
    lines.append("## Influence Coverage Analysis")
    lines.append("")
    lines.append("Selective-Budget config에서 influence recomputation이 실제 작동하는지 확인:")
    lines.append("")

    for dataset in datasets:
        sel_results = [
            r for r in results
            if r.get("dataset") == dataset
            and r.get("config") == "Selective-Budget"
            and "error" not in r
        ]
        if not sel_results:
            continue

        inf_covs = [r.get("influence_coverage", (0, 0)) for r in sel_results]
        pcts = [c[0]/max(c[1],1)*100 for c in inf_covs]
        lines.append(f"- **{dataset}**: {np.mean(pcts):.1f}% ± {np.std(pcts):.1f}% "
                     f"(mean {int(np.mean([c[0] for c in inf_covs]))} / {int(np.mean([c[1] for c in inf_covs]))})")

    lines.append("")

    # FIFO vs FIFO-Protected (value of minority protection)
    lines.append("## Ablation: FIFO vs FIFO-Protected (minority protection value)")
    lines.append("")
    lines.append("| Dataset | Scenario | FIFO G-mean | FIFO-Prot G-mean | Δ |")
    lines.append("|---------|----------|------------|-----------------|---|")

    for dataset in datasets:
        for scenario_name in scenarios:
            fifo = [r["gmean"] for r in results
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == "FIFO-Window"
                    and "error" not in r]
            fifo_prot = [r["gmean"] for r in results
                         if r.get("dataset") == dataset
                         and r.get("scenario") == scenario_name
                         and r.get("config") == "FIFO-Protected"
                         and "error" not in r]

            if not fifo or not fifo_prot:
                continue

            delta = np.mean(fifo_prot) - np.mean(fifo)
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {dataset} | {scenario_name} | "
                f"{np.mean(fifo):.4f} | "
                f"{np.mean(fifo_prot):.4f} | "
                f"{sign}{delta:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Selective vs FIFO Benchmark")
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
        default="results/selective_vs_fifo",
        help="Output directory",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (1 seed, 1 dataset, 2 scenarios)",
    )

    args = parser.parse_args()

    if args.smoke:
        logger.info("=== SMOKE TEST MODE ===")
        datasets = ["nslkdd"]
        scenarios = ["attack_decrease", "moderate_sudden"]
        seeds = [42]
        configs = [
            CONFIGS[0],  # AdaptiveK-Only
            CONFIGS[2],  # FIFO-Protected
            CONFIGS[3],  # Selective-Budget
        ]
    else:
        datasets = args.datasets
        scenarios = args.scenarios
        seeds = args.seeds
        configs = CONFIGS

    output_dir = Path(args.output_dir)

    result = run_benchmark(datasets, scenarios, configs, seeds, output_dir)
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
