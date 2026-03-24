"""Paper V2 Benchmark: Unified experiment script for SUDA paper submission.

Implements all experiments from the v2 plan (260219_experiment_plan_v2.md):
  A: Main Comparison + HOW Ablation (720 runs)
  B: ANoShift Natural Drift (30 runs)
  C: Budget Size Sensitivity (90 runs: budget_sizes=[500,1000,2000,3000,5000,10000])
  D: Class Balancing Comparison (120 runs)
  E: Additional Baselines - SRP & LeveragingBagging (240 runs)
  F: Hyperparameter Sensitivity - k_min/k_max (60 runs)

Total: 1230 runs across 5-10 seeds with BH-FDR, Friedman, Cohen's d_z.

Usage:
    # Sanity check (1 dataset x 1 scenario x 1 seed x all configs)
    uv run python -m src.experiments.paper_v2_benchmark --sanity

    # Run specific experiment
    uv run python -m src.experiments.paper_v2_benchmark --experiment A
    uv run python -m src.experiments.paper_v2_benchmark --experiment B
    uv run python -m src.experiments.paper_v2_benchmark --experiment C
    uv run python -m src.experiments.paper_v2_benchmark --experiment D
    uv run python -m src.experiments.paper_v2_benchmark --experiment E
    uv run python -m src.experiments.paper_v2_benchmark --experiment F

    # Run all experiments
    uv run python -m src.experiments.paper_v2_benchmark --experiment all

    # Resume from checkpoint
    uv run python -m src.experiments.paper_v2_benchmark --experiment A --resume
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src.data.nids import (
    make_anoshift_temporal_stream,
    make_asymmetric_recovery_stream,
    make_gradual_ramp_stream,
    make_moderate_sudden_drift_stream,
    make_stepwise_drift_stream,
)
from src.models.suda import SUDA
from src.baselines.river_models import (
    ARFModel,
    ARFWithClassWeight,
    ARFWithOversampling,
    SRPModel,
    LeveragingBaggingModel,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from src.experiments.utils import NumpyEncoder, compute_gmean  # noqa: E402


# =============================================================================
# Common Settings (from experiment plan v2)
# =============================================================================

SEEDS_10 = [42, 123, 456, 789, 2026, 314, 628, 999, 1234, 5678]
SEEDS_5 = [42, 123, 456, 789, 2026]

DATASETS = ["nslkdd", "unswnb15", "cicids2018"]
DATASET_FEATURES = {"nslkdd": 41, "unswnb15": 42, "cicids2018": 78}
BATCH_SIZE = 200
WARMUP_RATIO = 0.3

SUDA_DEFAULTS = {
    "num_trees": 50,
    "k": 10,
    "max_depth": 15,
    "warmup_samples": 1000,
    "metrics_window": 1000,
    "adaptive_k_enabled": True,
    "k_min": 1,
    "k_max": 70,
    "gmean_drop_threshold": 0.05,
    "recall_drop_threshold": 0.10,
}

SCENARIOS = {
    "moderate_sudden": make_moderate_sudden_drift_stream,
    "stepwise": make_stepwise_drift_stream,
    "gradual_ramp": make_gradual_ramp_stream,
    "asymmetric_recovery": make_asymmetric_recovery_stream,
}


# =============================================================================
# Model Creation
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    model_type: str  # "suda", "arf", "arf_classweight", "arf_oversample"
    params: dict


def create_model(config: ModelConfig, num_features: int, seed: int):
    if config.model_type == "suda":
        p = config.params
        # k_min/k_max: config.params에 _k_min/_k_max가 있으면 사용, 없으면 SUDA_DEFAULTS
        k_min = p.get("_k_min", SUDA_DEFAULTS["k_min"])
        k_max = p.get("_k_max", SUDA_DEFAULTS["k_max"])
        return SUDA(
            num_features=num_features,
            num_trees=SUDA_DEFAULTS["num_trees"],
            k=SUDA_DEFAULTS["k"],
            max_depth=SUDA_DEFAULTS["max_depth"],
            seed=seed,
            warmup_samples=SUDA_DEFAULTS["warmup_samples"],
            metrics_window=SUDA_DEFAULTS["metrics_window"],
            adaptive_k_enabled=p.get("adaptive_k_enabled", SUDA_DEFAULTS["adaptive_k_enabled"]),
            k_min=k_min,
            k_max=k_max,
            gmean_drop_threshold=SUDA_DEFAULTS["gmean_drop_threshold"],
            recall_drop_threshold=SUDA_DEFAULTS["recall_drop_threshold"],
            unlearning_enabled=p.get("unlearning_enabled", False),
            selection_strategy="oob_influence",
            proactive_enabled=False,
            drift_type_detection_enabled=False,
            smart_cooldown_enabled=False,
            budget_enabled=p.get("budget_enabled", False),
            budget_max_samples=p.get("budget_max_samples", 10000),
            budget_eviction_batch=p.get("budget_eviction_batch", 100),
            budget_age_weight=p.get("budget_age_weight", 1.0),
            budget_influence_weight=p.get("budget_influence_weight", 0.0),
            budget_class_weight=p.get("budget_class_weight", 0.0),
            budget_minority_protection=p.get("budget_minority_protection", 0.1),
            budget_skip_forest_forget=p.get("budget_skip_forest_forget", False),
            influence_tracking=p.get("influence_tracking", False),
        )
    elif config.model_type == "arf":
        return ARFModel(n_models=50, seed=seed)
    elif config.model_type == "arf_classweight":
        return ARFWithClassWeight(
            n_models=50, seed=seed,
            minority_weight=config.params.get("minority_weight", 10),
        )
    elif config.model_type == "arf_oversample":
        return ARFWithOversampling(
            n_models=50, seed=seed,
            target_ratio=config.params.get("target_ratio", 0.3),
        )
    elif config.model_type == "srp":
        return SRPModel(
            n_models=config.params.get("n_models", 50), seed=seed,
        )
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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "benign_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "n_unlearning_events": n_unlearning,
        "total_budget_evicted": total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
    }


def run_anoshift_single(config: ModelConfig, seed: int) -> dict:
    X_stream, y_stream, metadata = make_anoshift_temporal_stream(
        samples_per_year=5000, seed=seed,
    )
    num_features = X_stream.shape[1]
    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)
    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    model = create_model(config, num_features, seed)
    model.fit(X_warmup, y_warmup.astype(bool))

    all_preds, all_labels = [], []
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
    if config.params.get("budget_enabled", False):
        total_budget_evicted = model.total_budget_evicted

    return {
        "dataset": "anoshift",
        "scenario": "temporal_10year",
        "config": config.name,
        "seed": seed,
        "gmean": compute_gmean(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "benign_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "n_unlearning_events": n_unlearning,
        "total_budget_evicted": total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
    }


# =============================================================================
# Experiment Configs
# =============================================================================

# --- Experiment A: Main Comparison + HOW Ablation ---
CONFIGS_A = [
    ModelConfig("SUDA-Full", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_eviction_batch": 100, "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0, "budget_class_weight": 0.0,
        "budget_minority_protection": 0.1, "budget_skip_forest_forget": False,
        "unlearning_enabled": True,
    }),
    ModelConfig("SUDA-AK-Only", "suda", {
        "budget_enabled": False, "unlearning_enabled": False,
    }),
    ModelConfig("SUDA-NoForgetBudget", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_skip_forest_forget": True, "unlearning_enabled": True,
    }),
    ModelConfig("SUDA-NoAdaptiveK", "suda", {
        "adaptive_k_enabled": False,
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_skip_forest_forget": False, "unlearning_enabled": True,
    }),
    ModelConfig("ARF", "arf", {}),
    ModelConfig("ARF+CW10x", "arf_classweight", {"minority_weight": 10}),
]

# --- Experiment B: ANoShift Natural Drift ---
CONFIGS_B = [
    ModelConfig("Budget-FIFO", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_skip_forest_forget": False, "unlearning_enabled": True,
    }),
    ModelConfig("AK-Only", "suda", {
        "budget_enabled": False, "unlearning_enabled": False,
    }),
    ModelConfig("Trigger-Conservative", "suda", {
        "budget_enabled": False, "unlearning_enabled": True,
    }),
]

# --- Experiment E: Additional Baselines (new runs only) ---
CONFIGS_E_RUN = [
    ModelConfig("SRP", "srp", {"n_models": 10}),
    ModelConfig("LeveragingBagging", "leveraging_bagging", {"n_models": 10}),
]
# Full config list for summary (includes SUDA-Full and ARF from Exp A)
CONFIGS_E_ALL_NAMES = ["SUDA-Full", "ARF", "SRP", "LeveragingBagging"]

# --- Experiment F: k_min/k_max Hyperparameter Sensitivity ---
# k_min/k_max 조합: (k_min, k_max) 6가지 설정
K_CONFIGS_F = [(1, 30), (1, 50), (1, 70), (3, 50), (3, 70), (5, 70)]

# --- Experiment D: Class Balancing Comparison ---
CONFIGS_D = [
    ModelConfig("SUDA-Full", "suda", {
        "budget_enabled": True, "budget_max_samples": 3000,
        "budget_skip_forest_forget": False, "unlearning_enabled": True,
    }),
    ModelConfig("ARF+CW10x", "arf_classweight", {"minority_weight": 10}),
    ModelConfig("ARF+CW70x", "arf_classweight", {"minority_weight": 70}),
    ModelConfig("ARF+Oversample30", "arf_oversample", {"target_ratio": 0.3}),
]


# =============================================================================
# Statistical Tests
# =============================================================================

def wilcoxon_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test (paired by seed). Returns (stat, p_value)."""
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
    """Cohen's d_z for paired samples."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    diff = np.array(a[:n]) - np.array(b[:n])
    sd_diff = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0


def bh_fdr_correction(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction. Returns [(corrected_p, significant), ...]."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    corrected = [0.0] * n
    for rank_idx, orig_idx in enumerate(sorted_indices):
        rank = rank_idx + 1
        corrected[orig_idx] = p_values[orig_idx] * n / rank
    # Enforce monotonicity (step-up)
    for i in range(n - 2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        corrected[idx] = min(corrected[idx], corrected[next_idx])
    return [(min(p, 1.0), min(p, 1.0) < alpha) for p in corrected]


def friedman_test(data_matrix: np.ndarray) -> tuple[float, float]:
    """Friedman test for cross-dataset comparison.
    data_matrix: (n_datasets, n_methods) mean values."""
    from scipy.stats import friedmanchisquare
    if data_matrix.shape[0] < 3 or data_matrix.shape[1] < 2:
        return 0.0, 1.0
    try:
        stat, p_val = friedmanchisquare(*[data_matrix[:, i] for i in range(data_matrix.shape[1])])
        return float(stat), float(p_val)
    except Exception:
        return 0.0, 1.0


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


def is_run_completed(results: list[dict], dataset: str, scenario: str, config_name: str, seed: int) -> bool:
    return any(
        r.get("dataset") == dataset
        and r.get("scenario") == scenario
        and r.get("config") == config_name
        and r.get("seed") == seed
        and "error" not in r
        for r in results
    )


# =============================================================================
# Experiment Runners
# =============================================================================

def run_experiment_loop(
    configs: list[ModelConfig],
    datasets: list[str],
    scenarios: list[str],
    seeds: list[int],
    output_dir: Path,
    experiment_name: str,
    is_anoshift: bool = False,
    resume: bool = False,
) -> list[dict]:
    checkpoint_path = output_dir / f"experiment_{experiment_name}_raw.json"
    results = load_checkpoint(checkpoint_path) if resume else []

    if resume and results:
        logger.info(f"Resuming from checkpoint: {len(results)} completed runs")

    if is_anoshift:
        total_runs = len(configs) * len(seeds)
    else:
        total_runs = len(datasets) * len(scenarios) * len(configs) * len(seeds)

    logger.info(f"Experiment {experiment_name}: {total_runs} total runs")
    run_count = 0

    if is_anoshift:
        for config in configs:
            for seed in seeds:
                run_count += 1
                if resume and is_run_completed(results, "anoshift", "temporal_10year", config.name, seed):
                    logger.info(f"[{run_count}/{total_runs}] SKIP (cached) anoshift/{config.name}/seed={seed}")
                    continue
                logger.info(f"[{run_count}/{total_runs}] anoshift/{config.name}/seed={seed}")
                try:
                    result = run_anoshift_single(config, seed)
                    results.append(result)
                    logger.info(f"  -> G-mean={result['gmean']:.4f}, Time={result['elapsed_seconds']:.1f}s")
                except Exception as e:
                    logger.error(f"  -> FAILED: {e}")
                    results.append({
                        "dataset": "anoshift", "scenario": "temporal_10year",
                        "config": config.name, "seed": seed, "error": str(e),
                    })
                save_checkpoint(results, checkpoint_path)
    else:
        for dataset in datasets:
            for scenario_name in scenarios:
                for config in configs:
                    for seed in seeds:
                        run_count += 1
                        if resume and is_run_completed(results, dataset, scenario_name, config.name, seed):
                            logger.info(f"[{run_count}/{total_runs}] SKIP (cached) {dataset}/{scenario_name}/{config.name}/seed={seed}")
                            continue
                        logger.info(f"[{run_count}/{total_runs}] {dataset}/{scenario_name}/{config.name}/seed={seed}")
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
                            results.append({
                                "dataset": dataset, "scenario": scenario_name,
                                "config": config.name, "seed": seed, "error": str(e),
                            })
                        # Checkpoint every run
                        if run_count % 10 == 0:
                            save_checkpoint(results, checkpoint_path)

    save_checkpoint(results, checkpoint_path)
    return results


def run_experiment_A(output_dir: Path, resume: bool = False) -> list[dict]:
    """Experiment A: Main Comparison + HOW Ablation (720 runs)."""
    logger.info("=" * 80)
    logger.info("Experiment A: Main Comparison + HOW Ablation")
    logger.info("  6 configs x 4 scenarios x 3 datasets x 10 seeds = 720 runs")
    logger.info("=" * 80)

    return run_experiment_loop(
        CONFIGS_A, DATASETS, list(SCENARIOS.keys()), SEEDS_10,
        output_dir, "A", resume=resume,
    )


def run_experiment_B(output_dir: Path, resume: bool = False) -> list[dict]:
    """Experiment B: ANoShift Natural Drift (30 runs)."""
    logger.info("=" * 80)
    logger.info("Experiment B: ANoShift Natural Drift")
    logger.info("  3 configs x 1 dataset x 10 seeds = 30 runs")
    logger.info("=" * 80)

    return run_experiment_loop(
        CONFIGS_B, ["anoshift"], ["temporal_10year"], SEEDS_10,
        output_dir, "B", is_anoshift=True, resume=resume,
    )


def run_experiment_C(output_dir: Path, resume: bool = False) -> list[dict]:
    """Experiment C: Budget Size Sensitivity (90 runs)."""
    logger.info("=" * 80)
    logger.info("Experiment C: Budget Size Sensitivity")
    logger.info("  6 sizes x 1 scenario x 3 datasets x 5 seeds = 90 runs")
    logger.info("=" * 80)

    # 500 추가: 매우 작은 budget에서 성능 하락 여부 확인
    budget_sizes = [500, 1000, 2000, 3000, 5000, 10000]
    configs_c = []
    for bs in budget_sizes:
        configs_c.append(ModelConfig(f"Budget-{bs}", "suda", {
            "budget_enabled": True, "budget_max_samples": bs,
            "budget_skip_forest_forget": False, "unlearning_enabled": True,
            "budget_eviction_batch": 100,
        }))

    return run_experiment_loop(
        configs_c, DATASETS, ["moderate_sudden"], SEEDS_5,
        output_dir, "C", resume=resume,
    )


def run_experiment_D(output_dir: Path, resume: bool = False) -> list[dict]:
    """Experiment D: Class Balancing Comparison (120 runs)."""
    logger.info("=" * 80)
    logger.info("Experiment D: Class Balancing Comparison")
    logger.info("  4 configs x 1 scenario x 3 datasets x 10 seeds = 120 runs")
    logger.info("=" * 80)

    return run_experiment_loop(
        CONFIGS_D, DATASETS, ["moderate_sudden"], SEEDS_10,
        output_dir, "D", resume=resume,
    )


def run_experiment_E(output_dir: Path, resume: bool = False) -> list[dict]:
    """Experiment E: Additional Baselines - SRP & LeveragingBagging (240 runs).

    Runs SRP and LeveragingBagging across 4 scenarios x 3 datasets x 10 seeds.
    SUDA-Full and ARF results are loaded from Experiment A for comparison.
    """
    logger.info("=" * 80)
    logger.info("Experiment E: Additional Baselines (SRP, LeveragingBagging)")
    logger.info("  2 new configs x 4 scenarios x 3 datasets x 10 seeds = 240 runs")
    logger.info("  (SUDA-Full & ARF reused from Experiment A)")
    logger.info("=" * 80)

    return run_experiment_loop(
        CONFIGS_E_RUN, DATASETS, list(SCENARIOS.keys()), SEEDS_10,
        output_dir, "E", resume=resume,
    )


def run_experiment_F(output_dir: Path, resume: bool = False) -> list[dict]:
    """Experiment F: k_min/k_max Hyperparameter Sensitivity (60 runs).

    6 k_configs x 1 scenario x 2 datasets x 5 seeds = 60 runs.
    moderate_sudden 시나리오만 사용, balanced(nslkdd)와 extreme imbalance(cicids2018) 비교.
    """
    logger.info("=" * 80)
    logger.info("Experiment F: k_min/k_max Hyperparameter Sensitivity")
    logger.info("  6 k_configs x 1 scenario x 2 datasets x 5 seeds = 60 runs")
    logger.info("=" * 80)

    # k_min/k_max 조합별 ModelConfig 생성 (SUDA_DEFAULTS + budget_enabled=True, budget_max_samples=3000)
    configs_f = []
    for k_min, k_max in K_CONFIGS_F:
        configs_f.append(ModelConfig(f"kmin{k_min}_kmax{k_max}", "suda", {
            "budget_enabled": True,
            "budget_max_samples": 3000,
            "budget_skip_forest_forget": False,
            "unlearning_enabled": True,
            "budget_eviction_batch": 100,
            # k_min/k_max는 create_model에서 덮어써야 하므로 params에 저장
            "_k_min": k_min,
            "_k_max": k_max,
        }))

    return run_experiment_loop(
        configs_f,
        ["nslkdd", "cicids2018"],  # balanced vs extreme imbalance
        ["moderate_sudden"],
        SEEDS_5,
        output_dir, "F", resume=resume,
    )


# =============================================================================
# Summary Generation
# =============================================================================

def get_gmeans_for(results: list[dict], dataset: str, scenario: str, config_name: str) -> list[float]:
    return [
        r["gmean"] for r in results
        if r.get("dataset") == dataset
        and r.get("scenario") == scenario
        and r.get("config") == config_name
        and "error" not in r
    ]


def generate_summary_A(results: list[dict], output_dir: Path):
    lines = [
        "# Experiment A: Main Comparison + HOW Ablation",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_10}", "",
    ]

    scenarios = list(SCENARIOS.keys())
    config_names = [c.name for c in CONFIGS_A]

    # Table 1: Per-scenario results
    for scenario_name in scenarios:
        lines.append(f"\n## Scenario: {scenario_name}")
        lines.append("| Dataset | Config | G-mean (mean+-std) | Atk Recall | Time(s) |")
        lines.append("|---------|--------|--------------------|------------|---------|")

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
                    lines.append(f"| {dataset} | {config_name} | - | - | - |")
                    continue
                gmeans = [r["gmean"] for r in matching]
                atk = [r["attack_recall"] for r in matching]
                times = [r["elapsed_seconds"] for r in matching]
                lines.append(
                    f"| {dataset} | {config_name} | "
                    f"{np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                    f"{np.mean(atk):.4f} | {np.mean(times):.1f} |"
                )

    # Statistical comparisons
    comparison_pairs = [
        ("SUDA-Full vs ARF", "SUDA-Full", "ARF"),
        ("SUDA-Full vs ARF+CW10x", "SUDA-Full", "ARF+CW10x"),
        ("SUDA-Full vs SUDA-AK-Only (Budget value)", "SUDA-Full", "SUDA-AK-Only"),
        ("SUDA-Full vs SUDA-NoForgetBudget (HOW ablation)", "SUDA-Full", "SUDA-NoForgetBudget"),
        ("SUDA-Full vs SUDA-NoAdaptiveK (AK ablation)", "SUDA-Full", "SUDA-NoAdaptiveK"),
    ]

    for pair_name, config_a, config_b in comparison_pairs:
        lines.append(f"\n## {pair_name}")
        lines.append("| Dataset | Scenario | A G-mean | B G-mean | Delta | p-value (BH) | Cohen's d_z |")
        lines.append("|---------|----------|----------|----------|-------|--------------|-------------|")

        all_p_values = []
        all_rows = []

        for dataset in DATASETS:
            for scenario_name in scenarios:
                a_gmeans = get_gmeans_for(results, dataset, scenario_name, config_a)
                b_gmeans = get_gmeans_for(results, dataset, scenario_name, config_b)
                if not a_gmeans or not b_gmeans:
                    continue

                n = min(len(a_gmeans), len(b_gmeans))
                delta = np.mean(a_gmeans[:n]) - np.mean(b_gmeans[:n])
                _, p_val = wilcoxon_test(a_gmeans[:n], b_gmeans[:n])
                d_z = cohens_d_z(a_gmeans[:n], b_gmeans[:n])
                all_p_values.append(p_val)
                all_rows.append((dataset, scenario_name, np.mean(a_gmeans[:n]),
                                np.mean(b_gmeans[:n]), delta, d_z))

        # BH-FDR correction
        corrected = bh_fdr_correction(all_p_values)
        for i, (dataset, scenario_name, a_mean, b_mean, delta, d_z) in enumerate(all_rows):
            p_corr, sig = corrected[i] if i < len(corrected) else (1.0, False)
            sig_str = "*" if sig else ""
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {dataset} | {scenario_name} | "
                f"{a_mean:.4f} | {b_mean:.4f} | {sign}{delta:.4f} | "
                f"{p_corr:.4f}{sig_str} | {d_z:.2f} |"
            )

    # Cross-dataset Friedman test
    lines.append("\n## Cross-Dataset Friedman Test")
    for scenario_name in scenarios:
        method_names = ["SUDA-Full", "ARF", "ARF+CW10x"]
        matrix = []
        for dataset in DATASETS:
            row = []
            for method in method_names:
                gmeans = get_gmeans_for(results, dataset, scenario_name, method)
                row.append(np.mean(gmeans) if gmeans else 0.0)
            matrix.append(row)
        data_matrix = np.array(matrix)
        if data_matrix.shape[0] >= 3:
            stat, p_val = friedman_test(data_matrix)
            lines.append(f"- {scenario_name}: chi2={stat:.2f}, p={p_val:.4f}")

    # Speed comparison
    lines.append("\n## Speed Comparison")
    lines.append("| Dataset | Scenario | SUDA-Full(s) | ARF(s) | Speedup |")
    lines.append("|---------|----------|--------------|--------|---------|")
    for dataset in DATASETS:
        for scenario_name in scenarios:
            suda_times = [r["elapsed_seconds"] for r in results
                         if r.get("dataset") == dataset and r.get("scenario") == scenario_name
                         and r.get("config") == "SUDA-Full" and "error" not in r]
            arf_times = [r["elapsed_seconds"] for r in results
                        if r.get("dataset") == dataset and r.get("scenario") == scenario_name
                        and r.get("config") == "ARF" and "error" not in r]
            if suda_times and arf_times:
                st = np.mean(suda_times)
                at = np.mean(arf_times)
                speedup = at / st if st > 0 else 0
                lines.append(f"| {dataset} | {scenario_name} | {st:.2f} | {at:.2f} | {speedup:.0f}x |")

    summary_path = output_dir / "experiment_A_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary A saved to {summary_path}")


def generate_summary_B(results: list[dict], output_dir: Path):
    lines = [
        "# Experiment B: ANoShift Natural Drift",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_10}", "",
        "| Config | G-mean (mean+-std) | F1 (mean+-std) | Atk Recall | Triggers | Budget Evicted |",
        "|--------|--------------------|--------------------|------------|----------|----------------|",
    ]

    for config in CONFIGS_B:
        matching = [r for r in results if r.get("config") == config.name and "error" not in r]
        if not matching:
            continue
        gmeans = [r["gmean"] for r in matching]
        f1s = [r["f1"] for r in matching]
        atk = [r["attack_recall"] for r in matching]
        triggers = [r["n_unlearning_events"] for r in matching]
        budgets = [r["total_budget_evicted"] for r in matching]
        lines.append(
            f"| {config.name} | {np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
            f"{np.mean(f1s):.4f}+-{np.std(f1s):.4f} | {np.mean(atk):.4f} | "
            f"{np.mean(triggers):.1f} | {int(np.mean(budgets))} |"
        )

    # Pairwise Wilcoxon
    lines.append("\n## Pairwise Comparisons (Wilcoxon)")
    pairs = [("Budget-FIFO", "AK-Only"), ("Budget-FIFO", "Trigger-Conservative")]
    for ca, cb in pairs:
        a_g = [r["gmean"] for r in results if r.get("config") == ca and "error" not in r]
        b_g = [r["gmean"] for r in results if r.get("config") == cb and "error" not in r]
        if a_g and b_g:
            _, p = wilcoxon_test(a_g, b_g)
            d = cohens_d_z(a_g, b_g)
            delta = np.mean(a_g) - np.mean(b_g)
            lines.append(f"- {ca} vs {cb}: Delta={delta:+.4f}, p={p:.4f}, d_z={d:.2f}")

    summary_path = output_dir / "experiment_B_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary B saved to {summary_path}")


def generate_summary_C(results: list[dict], output_dir: Path):
    budget_sizes = [500, 1000, 2000, 3000, 5000, 10000]
    lines = [
        "# Experiment C: Budget Size Sensitivity",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_5}", "",
        "| Dataset | Budget | G-mean (mean+-std) | Atk Recall | Evicted |",
        "|---------|--------|--------------------|------------|---------|",
    ]

    for dataset in DATASETS:
        for bs in budget_sizes:
            config_name = f"Budget-{bs}"
            matching = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("config") == config_name
                and "error" not in r
            ]
            if not matching:
                continue
            gmeans = [r["gmean"] for r in matching]
            atk = [r["attack_recall"] for r in matching]
            evicted = [r["total_budget_evicted"] for r in matching]
            lines.append(
                f"| {dataset} | {bs} | {np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                f"{np.mean(atk):.4f} | {int(np.mean(evicted))} |"
            )

    # Highlight optimal
    lines.append("\n## Optimal Budget per Dataset")
    for dataset in DATASETS:
        best_bs, best_gmean = 0, 0.0
        for bs in budget_sizes:
            config_name = f"Budget-{bs}"
            matching = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("config") == config_name
                and "error" not in r
            ]
            if matching:
                mean_g = np.mean([r["gmean"] for r in matching])
                if mean_g > best_gmean:
                    best_gmean = mean_g
                    best_bs = bs
        lines.append(f"- {dataset}: B={best_bs} (G-mean={best_gmean:.4f})")

    summary_path = output_dir / "experiment_C_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary C saved to {summary_path}")


def generate_summary_D(results: list[dict], output_dir: Path):
    lines = [
        "# Experiment D: Class Balancing Comparison",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_10}", "",
        "| Dataset | Config | G-mean (mean+-std) | Atk Recall | Benign Recall |",
        "|---------|--------|--------------------|------------|---------------|",
    ]

    config_names = [c.name for c in CONFIGS_D]
    for dataset in DATASETS:
        for config_name in config_names:
            matching = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("config") == config_name
                and "error" not in r
            ]
            if not matching:
                lines.append(f"| {dataset} | {config_name} | - | - | - |")
                continue
            gmeans = [r["gmean"] for r in matching]
            atk = [r["attack_recall"] for r in matching]
            ben = [r["benign_recall"] for r in matching]
            lines.append(
                f"| {dataset} | {config_name} | "
                f"{np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                f"{np.mean(atk):.4f} | {np.mean(ben):.4f} |"
            )

    # Pairwise vs SUDA-Full
    lines.append("\n## Pairwise: SUDA-Full vs Others (Wilcoxon, BH-FDR)")
    all_p_values = []
    all_labels = []
    for dataset in DATASETS:
        for config_name in config_names[1:]:  # skip SUDA-Full
            a_g = get_gmeans_for(results, dataset, "moderate_sudden", "SUDA-Full")
            b_g = get_gmeans_for(results, dataset, "moderate_sudden", config_name)
            if a_g and b_g:
                _, p = wilcoxon_test(a_g, b_g)
                d = cohens_d_z(a_g, b_g)
                all_p_values.append(p)
                all_labels.append((dataset, config_name, np.mean(a_g) - np.mean(b_g), d))

    corrected = bh_fdr_correction(all_p_values)
    for i, (dataset, config_name, delta, d_z) in enumerate(all_labels):
        p_corr, sig = corrected[i] if i < len(corrected) else (1.0, False)
        sig_str = "*" if sig else ""
        lines.append(f"- {dataset}: SUDA-Full vs {config_name}: Delta={delta:+.4f}, p={p_corr:.4f}{sig_str}, d_z={d_z:.2f}")

    summary_path = output_dir / "experiment_D_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary D saved to {summary_path}")


def generate_summary_E(results: list[dict], output_dir: Path):
    """Generate summary for Experiment E: Additional Baselines.

    Merges SUDA-Full and ARF results from Experiment A for comparison.
    """
    # Load Experiment A results for SUDA-Full and ARF
    exp_a_path = output_dir / "experiment_A_raw.json"
    exp_a_results = []
    if exp_a_path.exists():
        with open(exp_a_path) as f:
            exp_a_all = json.load(f)
        exp_a_results = [
            r for r in exp_a_all
            if r.get("config") in ("SUDA-Full", "ARF") and "error" not in r
        ]
        logger.info(f"Loaded {len(exp_a_results)} SUDA-Full/ARF results from Experiment A")

    # Merge: Experiment E new results + Experiment A SUDA-Full/ARF
    merged = results + exp_a_results

    lines = [
        "# Experiment E: Additional Baselines (SRP, LeveragingBagging)",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_10}",
        f"\nNote: SUDA-Full and ARF results from Experiment A (n_models=50).",
        f"SRP and LeveragingBagging use n_models=10 (River default).", "",
    ]

    scenarios = list(SCENARIOS.keys())
    config_names = CONFIGS_E_ALL_NAMES

    # Per-scenario results table
    for scenario_name in scenarios:
        lines.append(f"\n## Scenario: {scenario_name}")
        lines.append("| Dataset | Config | G-mean (mean+-std) | Atk Recall | Benign Recall | Time(s) |")
        lines.append("|---------|--------|--------------------|------------|---------------|---------|")

        for dataset in DATASETS:
            for config_name in config_names:
                matching = [
                    r for r in merged
                    if r.get("dataset") == dataset
                    and r.get("scenario") == scenario_name
                    and r.get("config") == config_name
                    and "error" not in r
                ]
                if not matching:
                    lines.append(f"| {dataset} | {config_name} | - | - | - | - |")
                    continue
                gmeans = [r["gmean"] for r in matching]
                atk = [r["attack_recall"] for r in matching]
                ben = [r["benign_recall"] for r in matching]
                times = [r["elapsed_seconds"] for r in matching]
                lines.append(
                    f"| {dataset} | {config_name} | "
                    f"{np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                    f"{np.mean(atk):.4f} | {np.mean(ben):.4f} | {np.mean(times):.1f} |"
                )

    # Pairwise: SUDA-Full vs each baseline
    baseline_names = ["ARF", "SRP", "LeveragingBagging"]
    for baseline in baseline_names:
        lines.append(f"\n## SUDA-Full vs {baseline}")
        lines.append("| Dataset | Scenario | SUDA G-mean | Baseline G-mean | Delta | p-value (BH) | Cohen's d_z |")
        lines.append("|---------|----------|-------------|-----------------|-------|--------------|-------------|")

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
                all_rows.append((dataset, scenario_name, np.mean(a_gmeans[:n]),
                                np.mean(b_gmeans[:n]), delta, d_z))

        corrected = bh_fdr_correction(all_p_values)
        for i, (dataset, scenario_name, a_mean, b_mean, delta, d_z) in enumerate(all_rows):
            p_corr, sig = corrected[i] if i < len(corrected) else (1.0, False)
            sig_str = "*" if sig else ""
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {dataset} | {scenario_name} | "
                f"{a_mean:.4f} | {b_mean:.4f} | {sign}{delta:.4f} | "
                f"{p_corr:.4f}{sig_str} | {d_z:.2f} |"
            )

    # Win/Loss summary
    lines.append("\n## Win/Loss Summary (SUDA-Full vs each baseline)")
    lines.append("| Baseline | Wins | Losses | Ties | Win Rate |")
    lines.append("|----------|------|--------|------|----------|")

    for baseline in baseline_names:
        wins, losses, ties = 0, 0, 0
        all_p_values = []
        all_deltas = []
        for dataset in DATASETS:
            for scenario_name in scenarios:
                a_g = get_gmeans_for(merged, dataset, scenario_name, "SUDA-Full")
                b_g = get_gmeans_for(merged, dataset, scenario_name, baseline)
                if a_g and b_g:
                    n = min(len(a_g), len(b_g))
                    _, p = wilcoxon_test(a_g[:n], b_g[:n])
                    all_p_values.append(p)
                    all_deltas.append(np.mean(a_g[:n]) - np.mean(b_g[:n]))

        corrected = bh_fdr_correction(all_p_values)
        for i, (p_corr, sig) in enumerate(corrected):
            if sig:
                if all_deltas[i] > 0:
                    wins += 1
                else:
                    losses += 1
            else:
                ties += 1

        total = wins + losses + ties
        wr = f"{wins}/{total}" if total > 0 else "-"
        lines.append(f"| {baseline} | {wins} | {losses} | {ties} | {wr} |")

    # Speed comparison
    lines.append("\n## Speed Comparison (average across all scenarios)")
    lines.append("| Dataset | SUDA-Full(s) | ARF(s) | SRP(s) | LB(s) |")
    lines.append("|---------|--------------|--------|--------|-------|")
    for dataset in DATASETS:
        times_by_config = {}
        for config_name in config_names:
            t = [r["elapsed_seconds"] for r in merged
                 if r.get("dataset") == dataset and r.get("config") == config_name
                 and "error" not in r]
            times_by_config[config_name] = np.mean(t) if t else 0.0
        lines.append(
            f"| {dataset} | "
            f"{times_by_config.get('SUDA-Full', 0):.1f} | "
            f"{times_by_config.get('ARF', 0):.1f} | "
            f"{times_by_config.get('SRP', 0):.1f} | "
            f"{times_by_config.get('LeveragingBagging', 0):.1f} |"
        )

    # Cross-dataset Friedman test (all 4 methods)
    lines.append("\n## Cross-Dataset Friedman Test (SUDA-Full vs ARF vs SRP vs LB)")
    for scenario_name in scenarios:
        method_names = config_names
        matrix = []
        for dataset in DATASETS:
            row = []
            for method in method_names:
                gmeans = get_gmeans_for(merged, dataset, scenario_name, method)
                row.append(np.mean(gmeans) if gmeans else 0.0)
            matrix.append(row)
        data_matrix = np.array(matrix)
        if data_matrix.shape[0] >= 3:
            stat, p_val = friedman_test(data_matrix)
            lines.append(f"- {scenario_name}: chi2={stat:.2f}, p={p_val:.4f}")

    summary_path = output_dir / "experiment_E_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary E saved to {summary_path}")


def generate_summary_F(results: list[dict], output_dir: Path):
    """Experiment F: k_min/k_max Hyperparameter Sensitivity 결과 요약."""
    datasets_f = ["nslkdd", "cicids2018"]
    lines = [
        "# Experiment F: k_min/k_max Hyperparameter Sensitivity",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seeds: {SEEDS_5}",
        f"Scenario: moderate_sudden",
        f"Datasets: {datasets_f}", "",
        "| Dataset | k_min | k_max | G-mean (mean+-std) | Atk Recall | Benign Recall |",
        "|---------|-------|-------|--------------------|------------|---------------|",
    ]

    for dataset in datasets_f:
        for k_min, k_max in K_CONFIGS_F:
            config_name = f"kmin{k_min}_kmax{k_max}"
            matching = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("config") == config_name
                and "error" not in r
            ]
            if not matching:
                lines.append(f"| {dataset} | {k_min} | {k_max} | - | - | - |")
                continue
            gmeans = [r["gmean"] for r in matching]
            atk = [r["attack_recall"] for r in matching]
            ben = [r["benign_recall"] for r in matching]
            lines.append(
                f"| {dataset} | {k_min} | {k_max} | "
                f"{np.mean(gmeans):.4f}+-{np.std(gmeans):.4f} | "
                f"{np.mean(atk):.4f} | {np.mean(ben):.4f} |"
            )

    # 데이터셋별 최적 k_min/k_max
    lines.append("\n## 데이터셋별 최적 k_min/k_max (G-mean 기준)")
    for dataset in datasets_f:
        best_config, best_gmean = None, -1.0
        for k_min, k_max in K_CONFIGS_F:
            config_name = f"kmin{k_min}_kmax{k_max}"
            matching = [
                r for r in results
                if r.get("dataset") == dataset
                and r.get("config") == config_name
                and "error" not in r
            ]
            if matching:
                mean_g = np.mean([r["gmean"] for r in matching])
                if mean_g > best_gmean:
                    best_gmean = mean_g
                    best_config = (k_min, k_max)
        if best_config:
            lines.append(
                f"- {dataset}: k_min={best_config[0]}, k_max={best_config[1]} "
                f"(G-mean={best_gmean:.4f})"
            )

    # SUDA 기본값 (1, 70) 대비 각 설정 비교
    lines.append("\n## 기본값 (k_min=1, k_max=70) 대비 비교 (Wilcoxon, BH-FDR)")
    baseline_config = "kmin1_kmax70"
    all_p_values = []
    all_rows = []
    for dataset in datasets_f:
        for k_min, k_max in K_CONFIGS_F:
            config_name = f"kmin{k_min}_kmax{k_max}"
            if config_name == baseline_config:
                continue
            a_gmeans = get_gmeans_for(results, dataset, "moderate_sudden", baseline_config)
            b_gmeans = get_gmeans_for(results, dataset, "moderate_sudden", config_name)
            if not a_gmeans or not b_gmeans:
                continue
            n = min(len(a_gmeans), len(b_gmeans))
            delta = np.mean(a_gmeans[:n]) - np.mean(b_gmeans[:n])
            _, p_val = wilcoxon_test(a_gmeans[:n], b_gmeans[:n])
            d_z = cohens_d_z(a_gmeans[:n], b_gmeans[:n])
            all_p_values.append(p_val)
            all_rows.append((dataset, config_name, np.mean(a_gmeans[:n]), np.mean(b_gmeans[:n]), delta, d_z))

    lines.append("| Dataset | Config | Baseline G-mean | Config G-mean | Delta | p-value (BH) | Cohen's d_z |")
    lines.append("|---------|--------|-----------------|---------------|-------|--------------|-------------|")
    corrected = bh_fdr_correction(all_p_values)
    for i, (dataset, config_name, a_mean, b_mean, delta, d_z) in enumerate(all_rows):
        p_corr, sig = corrected[i] if i < len(corrected) else (1.0, False)
        sig_str = "*" if sig else ""
        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {dataset} | {config_name} | "
            f"{a_mean:.4f} | {b_mean:.4f} | {sign}{delta:.4f} | "
            f"{p_corr:.4f}{sig_str} | {d_z:.2f} |"
        )

    summary_path = output_dir / "experiment_F_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary F saved to {summary_path}")


# =============================================================================
# Sanity Check
# =============================================================================

def run_sanity_check(output_dir: Path):
    """Quick sanity check: 1 dataset x 1 scenario x 1 seed x all experiment A configs."""
    logger.info("=" * 80)
    logger.info("SANITY CHECK: nslkdd / moderate_sudden / seed=42 / all configs")
    logger.info("=" * 80)

    results = []
    for config in CONFIGS_A:
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

    # Print summary
    print("\n" + "=" * 60)
    print("SANITY CHECK RESULTS")
    print("=" * 60)
    print(f"{'Config':<25} {'G-mean':>10} {'Time(s)':>10}")
    print("-" * 60)
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
    parser = argparse.ArgumentParser(description="SUDA Paper V2 Benchmark")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "A", "B", "C", "D", "E", "F", "sanity"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/paper_v2",
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

    experiments = ["C", "F", "B", "D", "A", "E"] if args.experiment == "all" else [args.experiment]

    for exp in experiments:
        if exp == "A":
            results = run_experiment_A(output_dir, resume=args.resume)
            generate_summary_A(results, output_dir)
        elif exp == "B":
            results = run_experiment_B(output_dir, resume=args.resume)
            generate_summary_B(results, output_dir)
        elif exp == "C":
            results = run_experiment_C(output_dir, resume=args.resume)
            generate_summary_C(results, output_dir)
        elif exp == "D":
            results = run_experiment_D(output_dir, resume=args.resume)
            generate_summary_D(results, output_dir)
        elif exp == "E":
            results = run_experiment_E(output_dir, resume=args.resume)
            generate_summary_E(results, output_dir)
        elif exp == "F":
            results = run_experiment_F(output_dir, resume=args.resume)
            generate_summary_F(results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("All requested experiments complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
