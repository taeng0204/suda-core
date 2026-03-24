"""Paper V2 Rerun: Critical experiments addressing reviewer feedback.

R1: 4-Strategy Eviction Benchmark (240 runs)
    Tests 4 eviction strategies to determine which selection method works best:
    - PureFIFO: Age-only, no minority protection
    - FIFO+ClassProt: Age-only + minority protection (= current SUDA-Full)
    - FeatDist: Feature-distance based eviction (influence_weight=0.7)
    - FeatDist+ClassProt: Feature-distance + class protection

    NOTE: ClassAware removed (consistently worst in initial run).
    NOTE: FeatDist fixed — influence_weight must be >0 for feature distance
          to affect eviction scoring (stored in cached_influence field).
    Total: 240 runs (4 configs × 4 scenarios × 3 datasets × 5 seeds)

R2: Fair Baseline Comparison (240 runs)
    SRP and LeveragingBagging with n_models=50 (same as SUDA/ARF),
    fixing the unfair n_models=10 comparison in original Experiment E.
    Total: 240 runs (2 configs × 4 scenarios × 3 datasets × 10 seeds)

Usage:
    # Run R1 only
    uv run python -m src.experiments.paper_v2_rerun --experiment R1

    # Run R2 only
    uv run python -m src.experiments.paper_v2_rerun --experiment R2

    # Run both
    uv run python -m src.experiments.paper_v2_rerun --experiment all

    # Resume from checkpoint
    uv run python -m src.experiments.paper_v2_rerun --experiment R1 --resume

    # Sanity check (1 seed, 1 dataset, 1 scenario)
    uv run python -m src.experiments.paper_v2_rerun --sanity
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
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from src.data.nids import (
    make_asymmetric_recovery_stream,
    make_gradual_ramp_stream,
    make_moderate_sudden_drift_stream,
    make_stepwise_drift_stream,
)
from src.models.suda import SUDA
from src.baselines.river_models import (
    ARFModel,
    SRPModel,
    LeveragingBaggingModel,
)
from src.experiments.utils import NumpyEncoder, compute_gmean

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Common Settings
# =============================================================================

SEEDS_10 = [42, 123, 456, 789, 2026, 314, 628, 999, 1234, 5678]
SEEDS_5 = [42, 123, 456, 789, 2026]

DATASETS = ["nslkdd", "unswnb15", "cicids2018"]
DATASET_FEATURES = {"nslkdd": 41, "unswnb15": 42, "cicids2018": 78}
BATCH_SIZE = 200
WARMUP_RATIO = 0.3

SCENARIOS = {
    "moderate_sudden": make_moderate_sudden_drift_stream,
    "stepwise": make_stepwise_drift_stream,
    "gradual_ramp": make_gradual_ramp_stream,
    "asymmetric_recovery": make_asymmetric_recovery_stream,
}


# =============================================================================
# Model Creation (compatible with current SUDAConfig)
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    model_type: str  # "suda", "arf", "srp", "leveraging_bagging"
    params: dict


def create_model(config: ModelConfig, num_features: int, seed: int):
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
            adaptive_k_enabled=p.get("adaptive_k_enabled", True),
            k_min=p.get("k_min", 1),
            k_max=p.get("k_max", 70),
            # Budget management (key mechanism)
            budget_enabled=p.get("budget_enabled", False),
            budget_max_samples=p.get("budget_max_samples", 3000),
            budget_eviction_batch=p.get("budget_eviction_batch", 100),
            budget_age_weight=p.get("budget_age_weight", 1.0),
            budget_influence_weight=p.get("budget_influence_weight", 0.0),
            budget_class_weight=p.get("budget_class_weight", 0.0),
            budget_minority_protection=p.get("budget_minority_protection", 0.1),
            budget_skip_forest_forget=p.get("budget_skip_forest_forget", False),
            budget_use_feature_distance=p.get("budget_use_feature_distance", False),
            # Influence tracking (needed for influence-based strategies)
            influence_tracking=p.get("influence_tracking", False),
            influence_update_interval=p.get("influence_update_interval", 10),
            influence_sample_count=p.get("influence_sample_count", 200),
            influence_strategy=p.get("influence_strategy", "none"),
            feat_dist_update_interval=p.get("feat_dist_update_interval", 2000),
        )
    elif config.model_type == "arf":
        return ARFModel(n_models=config.params.get("n_models", 50), seed=seed)
    elif config.model_type == "srp":
        return SRPModel(n_models=config.params.get("n_models", 50), seed=seed)
    elif config.model_type == "leveraging_bagging":
        return LeveragingBaggingModel(
            n_models=config.params.get("n_models", 50), seed=seed,
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
    num_features = DATASET_FEATURES[dataset]
    scenario_fn = SCENARIOS[scenario_name]
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
        model.partial_fit(X_warmup, y_warmup.astype(np.int64))

    # Streaming (test-then-train)
    all_preds, all_labels = [], []
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
            preds = model.predict(X_batch)
            model.partial_fit(X_batch, y_batch_raw.astype(np.int64))

        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch_raw.astype(int).tolist())

    elapsed = time.time() - start_time

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    total_budget_evicted = 0
    if config.model_type == "suda" and config.params.get("budget_enabled", False):
        total_budget_evicted = model.total_budget_evicted

    return {
        "dataset": dataset,
        "scenario": scenario_name,
        "config": config.name,
        "seed": seed,
        "gmean": compute_gmean(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
            if len(np.unique(y_true)) > 1 else 0.0,
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "benign_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "total_budget_evicted": total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
    }


# =============================================================================
# R1: 5-Strategy Eviction Benchmark
# =============================================================================

CONFIGS_R1 = [
    # Strategy 1: PureFIFO — age-only, no class protection
    ModelConfig("PureFIFO", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_eviction_batch": 100,
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,  # NO protection
        "budget_skip_forest_forget": False,
    }),
    # Strategy 2: FIFO+ClassProt — age + minority protection (= current SUDA-Full)
    ModelConfig("FIFO+ClassProt", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_eviction_batch": 100,
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.1,
        "budget_skip_forest_forget": False,
    }),
    # Strategy 3: FeatDist — feature-distance dominant eviction
    # NOTE: feature distance is stored in cached_influence, so influence_weight
    # must be > 0 for it to affect eviction scoring.
    ModelConfig("FeatDist", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_eviction_batch": 100,
        "budget_age_weight": 0.3,
        "budget_influence_weight": 0.7,  # feature distance contribution
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        "budget_skip_forest_forget": False,
        "budget_use_feature_distance": True,
    }),
    # Strategy 4: FeatDist+ClassProt — feature-distance + class protection
    ModelConfig("FeatDist+ClassProt", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_eviction_batch": 100,
        "budget_age_weight": 0.2,
        "budget_influence_weight": 0.5,  # feature distance contribution
        "budget_class_weight": 0.3,
        "budget_minority_protection": 0.1,
        "budget_skip_forest_forget": False,
        "budget_use_feature_distance": True,
    }),
    # NOTE: ARF loaded from Experiment A results (no re-run needed)
]


# =============================================================================
# R2: Fair Baseline Comparison (n_models=50)
# =============================================================================

CONFIGS_R2 = [
    # Fair comparison: same n_models=50 as SUDA and ARF
    ModelConfig("SRP-50", "srp", {"n_models": 50}),
    ModelConfig("LB-50", "leveraging_bagging", {"n_models": 50}),
    # NOTE: SUDA-Full and ARF loaded from Experiment A results (no re-run needed)
]


# =============================================================================
# Statistical Tests
# =============================================================================

def wilcoxon_test(a: list[float], b: list[float]) -> tuple[float, float]:
    from scipy.stats import wilcoxon
    n = min(len(a), len(b))
    if n < 5:
        return 0.0, 1.0
    try:
        stat, p_val = wilcoxon(a[:n], b[:n], alternative="greater")
        return float(stat), float(p_val)
    except Exception:
        return 0.0, 1.0


def cohens_d_z(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    diff = np.array(a[:n]) - np.array(b[:n])
    sd_diff = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0


def bh_fdr_correction(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    corrected = [0.0] * n
    for rank_idx, orig_idx in enumerate(sorted_indices):
        rank = rank_idx + 1
        corrected[orig_idx] = p_values[orig_idx] * n / rank
    for i in range(n - 2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        corrected[idx] = min(corrected[idx], corrected[next_idx])
    return [(min(p, 1.0), min(p, 1.0) < alpha) for p in corrected]


# =============================================================================
# Checkpoint / Resume
# =============================================================================

def save_checkpoint(results: list[dict], output_path: Path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


def load_checkpoint(output_path: Path) -> list[dict]:
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return []


def is_run_completed(results: list[dict], dataset: str, scenario: str,
                     config_name: str, seed: int) -> bool:
    return any(
        r.get("dataset") == dataset
        and r.get("scenario") == scenario
        and r.get("config") == config_name
        and r.get("seed") == seed
        and "error" not in r
        for r in results
    )


# =============================================================================
# Experiment Loop
# =============================================================================

def run_experiment_loop(
    configs: list[ModelConfig],
    datasets: list[str],
    scenarios: list[str],
    seeds: list[int],
    output_dir: Path,
    experiment_name: str,
    resume: bool = False,
) -> list[dict]:
    checkpoint_path = output_dir / f"experiment_{experiment_name}_raw.json"
    results = load_checkpoint(checkpoint_path) if resume else []

    if resume and results:
        logger.info(f"Resuming from checkpoint: {len(results)} completed runs")

    total_runs = len(datasets) * len(scenarios) * len(configs) * len(seeds)
    logger.info(f"Experiment {experiment_name}: {total_runs} total runs")
    run_count = 0

    for dataset in datasets:
        for scenario_name in scenarios:
            for config in configs:
                for seed in seeds:
                    run_count += 1
                    if resume and is_run_completed(
                        results, dataset, scenario_name, config.name, seed
                    ):
                        logger.info(
                            f"[{run_count}/{total_runs}] SKIP (cached) "
                            f"{dataset}/{scenario_name}/{config.name}/seed={seed}"
                        )
                        continue
                    logger.info(
                        f"[{run_count}/{total_runs}] "
                        f"{dataset}/{scenario_name}/{config.name}/seed={seed}"
                    )
                    try:
                        result = run_single(dataset, scenario_name, config, seed)
                        results.append(result)
                        logger.info(
                            f"  -> G-mean={result['gmean']:.4f}, "
                            f"F1={result['f1']:.4f}, "
                            f"Time={result['elapsed_seconds']:.1f}s"
                        )
                    except Exception as e:
                        logger.error(f"  -> FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            "dataset": dataset,
                            "scenario": scenario_name,
                            "config": config.name,
                            "seed": seed,
                            "error": str(e),
                        })
                    # Checkpoint every run
                    save_checkpoint(results, checkpoint_path)

    save_checkpoint(results, checkpoint_path)
    return results


# =============================================================================
# Summary Helpers
# =============================================================================

def get_gmeans_for(results: list[dict], dataset: str, scenario: str,
                   config_name: str) -> list[float]:
    return [
        r["gmean"] for r in results
        if r.get("dataset") == dataset
        and r.get("scenario") == scenario
        and r.get("config") == config_name
        and "error" not in r
    ]


def get_metric_for(results: list[dict], dataset: str, scenario: str,
                   config_name: str, metric: str) -> list[float]:
    return [
        r[metric] for r in results
        if r.get("dataset") == dataset
        and r.get("scenario") == scenario
        and r.get("config") == config_name
        and "error" not in r
        and metric in r
    ]


# =============================================================================
# R1 Summary
# =============================================================================

def generate_summary_R1(results: list[dict], output_dir: Path):
    scenarios = list(SCENARIOS.keys())
    config_names = [c.name for c in CONFIGS_R1]

    lines = [
        "# R1: 5-Strategy Eviction Benchmark",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_5}",
        f"Configs: {config_names}",
        f"Budget: B=3000 for all SUDA variants",
        "",
        "## Research Question",
        "Which eviction strategy works best for budget-based exact forgetting?",
        "Does feature-distance or class-aware selection improve over simple FIFO?",
        "",
    ]

    # Table 1: Per-scenario results
    for scenario_name in scenarios:
        lines.append(f"\n### Scenario: {scenario_name}")
        lines.append(
            "| Dataset | Config | G-mean (mean+-std) | Atk Recall | "
            "Ben Recall | Evicted | Time(s) |"
        )
        lines.append(
            "|---------|--------|--------------------|------------|"
            "-----------|---------|---------|"
        )

        for dataset in DATASETS:
            for config_name in config_names:
                matching = [
                    r for r in results
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config_name
                    and "error" not in r
                ]
                if not matching:
                    lines.append(
                        f"| {dataset} | {config_name} | - | - | - | - | - |"
                    )
                    continue
                gmeans = [r["gmean"] for r in matching]
                atk = [r["attack_recall"] for r in matching]
                ben = [r["benign_recall"] for r in matching]
                evicted = [r["total_budget_evicted"] for r in matching]
                times = [r["elapsed_seconds"] for r in matching]
                lines.append(
                    f"| {dataset} | {config_name} | "
                    f"{np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                    f"{np.mean(atk):.4f} | {np.mean(ben):.4f} | "
                    f"{int(np.mean(evicted))} | {np.mean(times):.1f} |"
                )

    # Pairwise: each strategy vs FIFO+ClassProt (= SUDA-Full baseline)
    baseline = "FIFO+ClassProt"
    other_strategies = [c for c in config_names if c != baseline and c != "ARF"]

    lines.append(f"\n## Pairwise: Each Strategy vs {baseline} (Wilcoxon, BH-FDR)")
    for strategy in other_strategies:
        lines.append(f"\n### {strategy} vs {baseline}")
        lines.append(
            "| Dataset | Scenario | Strategy G-mean | Baseline G-mean | "
            "Delta | p-value (BH) | Cohen's d_z |"
        )
        lines.append(
            "|---------|----------|-----------------|-----------------|"
            "-------|--------------|-------------|"
        )

        all_p_values = []
        all_rows = []

        for dataset in DATASETS:
            for scenario_name in scenarios:
                a_gmeans = get_gmeans_for(
                    results, dataset, scenario_name, strategy
                )
                b_gmeans = get_gmeans_for(
                    results, dataset, scenario_name, baseline
                )
                if not a_gmeans or not b_gmeans:
                    continue
                n = min(len(a_gmeans), len(b_gmeans))
                delta = np.mean(a_gmeans[:n]) - np.mean(b_gmeans[:n])
                _, p_val = wilcoxon_test(a_gmeans[:n], b_gmeans[:n])
                d_z = cohens_d_z(a_gmeans[:n], b_gmeans[:n])
                all_p_values.append(p_val)
                all_rows.append((
                    dataset, scenario_name,
                    np.mean(a_gmeans[:n]), np.mean(b_gmeans[:n]),
                    delta, d_z,
                ))

        corrected = bh_fdr_correction(all_p_values)
        for i, (ds, sc, a_m, b_m, delta, d_z) in enumerate(all_rows):
            p_corr, sig = corrected[i] if i < len(corrected) else (1.0, False)
            sig_str = "*" if sig else ""
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {ds} | {sc} | {a_m:.4f} | {b_m:.4f} | "
                f"{sign}{delta:.4f} | {p_corr:.4f}{sig_str} | {d_z:.2f} |"
            )

    # All strategies vs ARF
    lines.append("\n## All Strategies vs ARF")
    lines.append(
        "| Dataset | Scenario | Config | SUDA G-mean | ARF G-mean | "
        "Delta | Wins? |"
    )
    lines.append(
        "|---------|----------|--------|-------------|------------|"
        "-------|-------|"
    )
    suda_configs = [c for c in config_names if c != "ARF"]
    for dataset in DATASETS:
        for scenario_name in scenarios:
            arf_gmeans = get_gmeans_for(results, dataset, scenario_name, "ARF")
            if not arf_gmeans:
                continue
            arf_mean = np.mean(arf_gmeans)
            for cfg in suda_configs:
                s_gmeans = get_gmeans_for(results, dataset, scenario_name, cfg)
                if not s_gmeans:
                    continue
                s_mean = np.mean(s_gmeans)
                delta = s_mean - arf_mean
                win = "Y" if delta > 0 else "N"
                lines.append(
                    f"| {dataset} | {scenario_name} | {cfg} | "
                    f"{s_mean:.4f} | {arf_mean:.4f} | "
                    f"{delta:+.4f} | {win} |"
                )

    # Best strategy per dataset/scenario
    lines.append("\n## Best Strategy per Dataset/Scenario")
    for dataset in DATASETS:
        for scenario_name in scenarios:
            best_name, best_gmean = "", -1.0
            for cfg in suda_configs:
                gmeans = get_gmeans_for(results, dataset, scenario_name, cfg)
                if gmeans and np.mean(gmeans) > best_gmean:
                    best_gmean = np.mean(gmeans)
                    best_name = cfg
            if best_name:
                lines.append(
                    f"- {dataset}/{scenario_name}: **{best_name}** "
                    f"(G-mean={best_gmean:.4f})"
                )

    # Win count summary
    lines.append("\n## Win Count Summary (across all dataset-scenario pairs)")
    lines.append("| Strategy | Wins (best G-mean) | Avg Rank |")
    lines.append("|----------|--------------------|----------|")

    strategy_wins = {c: 0 for c in suda_configs}
    strategy_ranks = {c: [] for c in suda_configs}
    for dataset in DATASETS:
        for scenario_name in scenarios:
            gmean_map = {}
            for cfg in suda_configs:
                gmeans = get_gmeans_for(results, dataset, scenario_name, cfg)
                if gmeans:
                    gmean_map[cfg] = np.mean(gmeans)
            if not gmean_map:
                continue
            sorted_cfgs = sorted(gmean_map.keys(), key=lambda c: gmean_map[c], reverse=True)
            for rank, cfg in enumerate(sorted_cfgs, 1):
                strategy_ranks[cfg].append(rank)
            strategy_wins[sorted_cfgs[0]] += 1

    for cfg in suda_configs:
        wins = strategy_wins[cfg]
        avg_rank = np.mean(strategy_ranks[cfg]) if strategy_ranks[cfg] else 0
        lines.append(f"| {cfg} | {wins} | {avg_rank:.2f} |")

    summary_path = output_dir / "experiment_R1_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary R1 saved to {summary_path}")


# =============================================================================
# R2 Summary
# =============================================================================

def generate_summary_R2(results: list[dict], output_dir: Path):
    """Generate summary for R2: Fair Baseline Comparison (n_models=50).

    Loads SUDA-Full and ARF results from Experiment A (no re-run needed).
    """
    scenarios = list(SCENARIOS.keys())
    config_names = [c.name for c in CONFIGS_R2]

    # Load SUDA-Full and ARF from Experiment A
    exp_a_results = []
    for exp_a_dir in [output_dir.parent / "paper_v2", output_dir.parent / "paper_v2_rerun"]:
        exp_a_path = exp_a_dir / "experiment_A_raw.json"
        if exp_a_path.exists():
            with open(exp_a_path) as f:
                exp_a_all = json.load(f)
            exp_a_results = [
                r for r in exp_a_all
                if r.get("config") in ("SUDA-Full", "ARF") and "error" not in r
            ]
            logger.info(f"Loaded {len(exp_a_results)} SUDA-Full/ARF results from {exp_a_path}")
            break

    # Also load R1 results for ARF if Exp A not available
    if not exp_a_results:
        r1_path = output_dir / "experiment_R1_raw.json"
        if r1_path.exists():
            with open(r1_path) as f:
                r1_all = json.load(f)
            # Use FIFO+ClassProt as SUDA-Full equivalent, and ARF
            for r in r1_all:
                if r.get("config") == "FIFO+ClassProt" and "error" not in r:
                    r_copy = dict(r)
                    r_copy["config"] = "SUDA-Full"
                    exp_a_results.append(r_copy)
                elif r.get("config") == "ARF" and "error" not in r:
                    exp_a_results.append(r)
            logger.info(f"Loaded {len(exp_a_results)} SUDA-Full/ARF results from R1")

    # Try to load Exp E results (n_models=10) for comparison
    exp_e_path = output_dir.parent / "paper_v2" / "experiment_E_raw.json"
    exp_e_results = []
    if exp_e_path.exists():
        with open(exp_e_path) as f:
            exp_e_all = json.load(f)
        exp_e_results = [
            r for r in exp_e_all
            if r.get("config") in ("SRP", "LeveragingBagging") and "error" not in r
        ]
        # Rename for clarity
        for r in exp_e_results:
            if r["config"] == "SRP":
                r["config"] = "SRP-10"
            elif r["config"] == "LeveragingBagging":
                r["config"] = "LB-10"
        logger.info(f"Loaded {len(exp_e_results)} SRP-10/LB-10 results from Experiment E")

    merged = results + exp_a_results + exp_e_results

    lines = [
        "# R2: Fair Baseline Comparison (n_models=50)",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_10}",
        "",
        "## Motivation",
        "Original Experiment E used SRP/LB with n_models=10 (River default),",
        "while SUDA and ARF used n_models=50. This creates an unfair comparison.",
        "R2 reruns SRP and LB with n_models=50 for fair apples-to-apples comparison.",
        "",
    ]

    # Per-scenario results
    all_config_names = config_names + (["SRP-10", "LB-10"] if exp_e_results else [])
    for scenario_name in scenarios:
        lines.append(f"\n### Scenario: {scenario_name}")
        lines.append(
            "| Dataset | Config | n_models | G-mean (mean+-std) | "
            "Atk Recall | Ben Recall | Time(s) |"
        )
        lines.append(
            "|---------|--------|----------|--------------------|-"
            "-----------|-----------|---------|"
        )

        for dataset in DATASETS:
            for config_name in all_config_names:
                matching = [
                    r for r in merged
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config_name
                    and "error" not in r
                ]
                if not matching:
                    continue
                gmeans = [r["gmean"] for r in matching]
                atk = [r["attack_recall"] for r in matching]
                ben = [r["benign_recall"] for r in matching]
                times = [r["elapsed_seconds"] for r in matching]
                n_models = "50" if "50" in config_name or config_name in ("SUDA-Full", "ARF") else "10"
                lines.append(
                    f"| {dataset} | {config_name} | {n_models} | "
                    f"{np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                    f"{np.mean(atk):.4f} | {np.mean(ben):.4f} | "
                    f"{np.mean(times):.1f} |"
                )

    # Pairwise: SUDA-Full vs each fair baseline
    for baseline in ["SRP-50", "LB-50", "ARF"]:
        lines.append(f"\n## SUDA-Full vs {baseline}")
        lines.append(
            "| Dataset | Scenario | SUDA G-mean | Baseline G-mean | "
            "Delta | p-value (BH) | Cohen's d_z |"
        )
        lines.append(
            "|---------|----------|-------------|-----------------|"
            "-------|--------------|-------------|"
        )

        all_p_values = []
        all_rows = []

        for dataset in DATASETS:
            for scenario_name in scenarios:
                a_gmeans = get_gmeans_for(merged, dataset, scenario_name, "SUDA-Full")
                b_gmeans = get_gmeans_for(merged, dataset, scenario_name, baseline)
                if not a_gmeans or not b_gmeans:
                    continue
                n = min(len(a_gmeans), len(b_gmeans))
                delta = np.mean(a_gmeans[:n]) - np.mean(b_gmeans[:n])
                _, p_val = wilcoxon_test(a_gmeans[:n], b_gmeans[:n])
                d_z = cohens_d_z(a_gmeans[:n], b_gmeans[:n])
                all_p_values.append(p_val)
                all_rows.append((
                    dataset, scenario_name,
                    np.mean(a_gmeans[:n]), np.mean(b_gmeans[:n]),
                    delta, d_z,
                ))

        corrected = bh_fdr_correction(all_p_values)
        for i, (ds, sc, a_m, b_m, delta, d_z) in enumerate(all_rows):
            p_corr, sig = corrected[i] if i < len(corrected) else (1.0, False)
            sig_str = "*" if sig else ""
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {ds} | {sc} | {a_m:.4f} | {b_m:.4f} | "
                f"{sign}{delta:.4f} | {p_corr:.4f}{sig_str} | {d_z:.2f} |"
            )

    # n_models=10 vs n_models=50 comparison (if Exp E data available)
    if exp_e_results:
        lines.append("\n## Impact of n_models: 10 vs 50")
        lines.append(
            "| Baseline | Dataset | Scenario | n=10 G-mean | n=50 G-mean | "
            "Delta | Improvement |"
        )
        lines.append(
            "|----------|---------|----------|-------------|-------------|"
            "-------|-------------|"
        )
        for base_name, n10_name, n50_name in [
            ("SRP", "SRP-10", "SRP-50"),
            ("LB", "LB-10", "LB-50"),
        ]:
            for dataset in DATASETS:
                for scenario_name in scenarios:
                    g10 = get_gmeans_for(merged, dataset, scenario_name, n10_name)
                    g50 = get_gmeans_for(merged, dataset, scenario_name, n50_name)
                    if not g10 or not g50:
                        continue
                    m10 = np.mean(g10)
                    m50 = np.mean(g50)
                    delta = m50 - m10
                    pct = delta / m10 * 100 if m10 > 0 else 0
                    lines.append(
                        f"| {base_name} | {dataset} | {scenario_name} | "
                        f"{m10:.4f} | {m50:.4f} | {delta:+.4f} | "
                        f"{pct:+.1f}% |"
                    )

    # Win/Loss summary
    lines.append("\n## Win/Loss Summary (SUDA-Full vs fair baselines, n=50)")
    lines.append("| Baseline | Wins | Losses | Win Rate |")
    lines.append("|----------|------|--------|----------|")
    for baseline in ["SRP-50", "LB-50", "ARF"]:
        wins, losses = 0, 0
        for dataset in DATASETS:
            for scenario_name in scenarios:
                a_g = get_gmeans_for(merged, dataset, scenario_name, "SUDA-Full")
                b_g = get_gmeans_for(merged, dataset, scenario_name, baseline)
                if a_g and b_g:
                    if np.mean(a_g) > np.mean(b_g):
                        wins += 1
                    else:
                        losses += 1
        total = wins + losses
        rate = wins / total * 100 if total > 0 else 0
        lines.append(f"| {baseline} | {wins} | {losses} | {rate:.0f}% |")

    summary_path = output_dir / "experiment_R2_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary R2 saved to {summary_path}")


# =============================================================================
# Experiment Runners
# =============================================================================

def run_experiment_R1(output_dir: Path, resume: bool = False) -> list[dict]:
    """R1: 5-Strategy Eviction Benchmark."""
    logger.info("=" * 80)
    logger.info("R1: 5-Strategy Eviction Benchmark")
    logger.info("  5 configs x 4 scenarios x 3 datasets x 5 seeds = 300 runs")
    logger.info("  Strategies: PureFIFO, FIFO+ClassProt, ClassAware, FeatDist, FeatDist+ClassProt")
    logger.info("  (ARF loaded from Experiment A)")
    logger.info("=" * 80)

    return run_experiment_loop(
        CONFIGS_R1, DATASETS, list(SCENARIOS.keys()), SEEDS_5,
        output_dir, "R1", resume=resume,
    )


def run_experiment_R2(output_dir: Path, resume: bool = False) -> list[dict]:
    """R2: Fair Baseline Comparison (n_models=50)."""
    logger.info("=" * 80)
    logger.info("R2: Fair Baseline Comparison (n_models=50)")
    logger.info("  2 configs x 4 scenarios x 3 datasets x 10 seeds = 240 runs")
    logger.info("  SRP-50, LB-50 (SUDA-Full & ARF loaded from Exp A)")
    logger.info("=" * 80)

    return run_experiment_loop(
        CONFIGS_R2, DATASETS, list(SCENARIOS.keys()), SEEDS_10,
        output_dir, "R2", resume=resume,
    )


# =============================================================================
# Sanity Check
# =============================================================================

def run_sanity_check(output_dir: Path):
    """Quick sanity: nslkdd / moderate_sudden / seed=42 / all R1+R2 configs."""
    logger.info("=" * 80)
    logger.info("SANITY CHECK: nslkdd / moderate_sudden / seed=42")
    logger.info("=" * 80)

    all_configs = CONFIGS_R1 + [c for c in CONFIGS_R2 if c.name not in
                                 {c2.name for c2 in CONFIGS_R1}]
    results = []
    for config in all_configs:
        logger.info(f"  Testing {config.name}...")
        try:
            result = run_single("nslkdd", "moderate_sudden", config, 42)
            results.append(result)
            logger.info(
                f"  -> G-mean={result['gmean']:.4f}, "
                f"Time={result['elapsed_seconds']:.1f}s"
            )
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config": config.name, "error": str(e)})

    print("\n" + "=" * 70)
    print("SANITY CHECK RESULTS")
    print("=" * 70)
    print(f"{'Config':<25} {'G-mean':>10} {'Time(s)':>10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['config']:<25} {'ERROR':>10}")
        else:
            print(f"{r['config']:<25} {r['gmean']:>10.4f} {r['elapsed_seconds']:>10.1f}")

    save_checkpoint(results, output_dir / "sanity_check.json")
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SUDA Paper V2 Rerun: R1 (Strategy) + R2 (Fair Baseline)"
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "R1", "R2", "sanity"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/paper_v2_rerun",
        help="Output directory",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Run sanity check only",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sanity or args.experiment == "sanity":
        run_sanity_check(output_dir)
        return

    experiments = (
        ["R1", "R2"] if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments:
        if exp == "R1":
            results = run_experiment_R1(output_dir, resume=args.resume)
            generate_summary_R1(results, output_dir)
        elif exp == "R2":
            results = run_experiment_R2(output_dir, resume=args.resume)
            generate_summary_R2(results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("All requested experiments complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
