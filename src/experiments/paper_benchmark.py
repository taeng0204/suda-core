"""Paper Benchmark: Comprehensive Experiments for SUDA Publication.

This script runs all experiments needed for the SUDA paper:

Table 1: Main Benchmark (1% attack streaming)
  - Datasets: NSL-KDD, UNSW-NB15, CIC-IDS2018
  - Models: SUDA, ARF, HAT, SRP, LeveragingBagging, EFDT, HoeffdingTree, ARF+SMOTE
  - Metrics: G-mean, Balanced Acc, Attack Recall, Benign Recall, Processing Time

Table 2: Ablation Study
  - SUDA Full vs AdaptiveK-Only vs Baseline
  - Dataset: NSL-KDD (representative)

Table 3: Extreme Drift (1%→50%→1%)
  - SUDA vs ARF vs SRP
  - Dataset: NSL-KDD
  - With/Without Unlearning comparison

Usage:
    # Run all tables with 5 seeds (default)
    uv run python -m src.experiments.paper_benchmark --table all

    # Run all tables with 10 seeds (full validation)
    uv run python -m src.experiments.paper_benchmark --table all --full

    # Run specific table
    uv run python -m src.experiments.paper_benchmark --table 1
    uv run python -m src.experiments.paper_benchmark --table 2
    uv run python -m src.experiments.paper_benchmark --table 3

    # Custom output directory
    uv run python -m src.experiments.paper_benchmark --table all --output_dir results/paper

Author: Claude Sonnet 4.5
Date: 2026-02-08
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)
from tqdm import tqdm

from src.baselines.river_models import (
    ARFModel,
    EFDTModel,
    HATModel,
    HoeffdingTreeModel,
    LeveragingBaggingModel,
    SRPModel,
)

from src.data.nids import (
    create_imbalanced_dataset,
    get_dataset_info,
    load_dataset,
    make_realistic_drift_stream,
)
from src.models.suda import SUDA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from src.experiments.utils import NumpyEncoder, compute_gmean  # noqa: E402


def compute_attack_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute attack recall (TPR)."""
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def compute_benign_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute benign recall (TNR)."""
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy."""
    return float(balanced_accuracy_score(y_true, y_pred))


# =============================================================================
# Table 1: Main Benchmark (1% Attack Streaming)
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""

    name: str
    model_class: type
    init_kwargs: dict


def get_table1_models(num_features: int, seed: int) -> list[ModelConfig]:
    """Get models for Table 1 benchmark."""
    models = [
        ModelConfig(
            name="SUDA",
            model_class=SUDA,
            init_kwargs={
                "num_features": num_features,
                "num_trees": 50,
                "k": 10,
                "max_depth": 15,
                "adaptive_k_enabled": True,
                "k_min": 1,
                "k_max": 70,
                "gmean_drop_threshold": 0.05,
                "recall_drop_threshold": 0.10,
                "drift_type_detection_enabled": True,
                "smart_cooldown_enabled": True,
                "benign_forget_ratio": 0.10,
                "harm_threshold": -0.1,
                "seed": seed,
            },
        ),
        ModelConfig(
            name="ARF",
            model_class=ARFModel,
            init_kwargs={"n_models": 50, "seed": seed},
        ),
        ModelConfig(
            name="HAT",
            model_class=HATModel,
            init_kwargs={"seed": seed},
        ),
        ModelConfig(
            name="SRP",
            model_class=SRPModel,
            init_kwargs={"n_models": 10, "seed": seed},
        ),
        ModelConfig(
            name="LeveragingBagging",
            model_class=LeveragingBaggingModel,
            init_kwargs={"n_models": 10, "seed": seed},
        ),
        ModelConfig(
            name="EFDT",
            model_class=EFDTModel,
            init_kwargs={"seed": seed},
        ),
        ModelConfig(
            name="HoeffdingTree",
            model_class=HoeffdingTreeModel,
            init_kwargs={"seed": seed},
        ),
    ]

    return models


def run_single_experiment_table1(
    model_config: ModelConfig,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 200,
    metrics_window: int = 1000,
    eval_interval: int = 1000,
    warmup_samples: int = 1000,
) -> dict:
    """Run a single experiment for Table 1.

    Uses sliding-window prequential evaluation consistent with streaming ML
    methodology. G-mean is computed over a rolling window and then averaged,
    rather than computed cumulatively over all predictions.

    For SUDA models, the first `warmup_samples` are used for batch pre-training
    via fit(), then the rest are streamed with partial_fit().
    """
    # Create model
    model = model_config.model_class(**model_config.init_kwargs)
    is_suda = model_config.name == "SUDA"

    # SUDA warmup: batch pre-train on first N samples
    stream_start = 0
    total_time_ms = 0
    if is_suda and warmup_samples > 0 and len(y) > warmup_samples:
        start_time = time.perf_counter()
        model.fit(X[:warmup_samples], y[:warmup_samples])
        end_time = time.perf_counter()
        total_time_ms += (end_time - start_time) * 1000
        stream_start = warmup_samples

    X_stream = X[stream_start:]
    y_stream = y[stream_start:]

    # Prequential evaluation (test-then-train)
    all_y_true = []
    all_y_pred = []

    # Sliding window metrics
    gmean_series = []
    attack_recall_series = []
    balanced_acc_series = []
    benign_recall_series = []

    n_batches = len(y_stream) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_stream[start_idx:end_idx]
        y_batch = y_stream[start_idx:end_idx]

        # Test (predict)
        start_time = time.perf_counter()
        if is_suda:
            # SUDA: partial_fit does predict + train
            result = model.partial_fit(X_batch, y_batch)
            preds = result.predictions.astype(int)
        else:
            # Other models: predict then train separately
            preds = model.predict(X_batch)
            model.partial_fit(X_batch, y_batch)
        end_time = time.perf_counter()

        total_time_ms += (end_time - start_time) * 1000

        all_y_true.extend(y_batch)
        all_y_pred.extend(preds)

        # Compute window metrics periodically
        total_processed = stream_start + end_idx
        if total_processed % eval_interval == 0 or end_idx >= len(y_stream):
            window_start = max(0, len(all_y_true) - metrics_window)
            window_true = np.array(all_y_true[window_start:], dtype=np.int64)
            window_pred = np.array(all_y_pred[window_start:], dtype=np.int64)

            gmean_series.append(compute_gmean(window_true, window_pred))
            attack_recall_series.append(compute_attack_recall(window_true, window_pred))
            balanced_acc_series.append(compute_balanced_accuracy(window_true, window_pred))
            benign_recall_series.append(compute_benign_recall(window_true, window_pred))

    return {
        "gmean": float(np.mean(gmean_series)) if gmean_series else 0.0,
        "gmean_std": float(np.std(gmean_series)) if gmean_series else 0.0,
        "balanced_accuracy": float(np.mean(balanced_acc_series)) if balanced_acc_series else 0.0,
        "attack_recall": float(np.mean(attack_recall_series)) if attack_recall_series else 0.0,
        "benign_recall": float(np.mean(benign_recall_series)) if benign_recall_series else 0.0,
        "time_ms": total_time_ms,
        "samples": len(all_y_true),
    }


def run_table1_benchmark(
    datasets: list[str],
    seeds: list[int],
    output_dir: Path,
) -> dict:
    """Run Table 1: Main Benchmark on 1% attack streaming."""
    logger.info("=" * 80)
    logger.info("Table 1: Main Benchmark (1% Attack Streaming)")
    logger.info("=" * 80)

    results = {}

    for dataset_name in datasets:
        logger.info(f"\nDataset: {dataset_name.upper()}")
        info = get_dataset_info(dataset_name)
        num_features = info.n_features

        dataset_results = {}

        for seed in tqdm(seeds, desc=f"{dataset_name}"):
            # Create 1% attack streaming data with realistic drift (1% → 5%)
            X, y = make_realistic_drift_stream(
                dataset_name,
                pre_attack_ratio=0.01,
                post_attack_ratio=0.05,
                total_samples=50000,
                drift_point=25000,
                seed=seed,
            )

            models = get_table1_models(num_features, seed)

            for model_config in models:
                logger.info(f"  Running {model_config.name} (seed={seed})")
                try:
                    result = run_single_experiment_table1(model_config, X, y)

                    if model_config.name not in dataset_results:
                        dataset_results[model_config.name] = []
                    dataset_results[model_config.name].append(result)

                except Exception as e:
                    logger.error(f"  Error with {model_config.name}: {e}")
                    continue

        results[dataset_name] = dataset_results

    # Save raw results
    output_file = output_dir / "table1_raw_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nSaved raw results to {output_file}")

    # Compute summary statistics
    summary = compute_table1_summary(results)
    summary_file = output_dir / "table1_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved summary to {summary_file}")

    # Print summary table
    print_table1_summary(summary)

    return results


def compute_table1_summary(results: dict) -> dict:
    """Compute summary statistics for Table 1."""
    summary = {}

    for dataset_name, dataset_results in results.items():
        summary[dataset_name] = {}

        for model_name, runs in dataset_results.items():
            if not runs:
                continue

            metrics = {}
            for key in ["gmean", "balanced_accuracy", "attack_recall", "benign_recall", "time_ms"]:
                values = [r[key] for r in runs]
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

            summary[dataset_name][model_name] = metrics

    return summary


def print_table1_summary(summary: dict):
    """Print Table 1 summary in table format."""
    print("\n" + "=" * 100)
    print("Table 1: Main Benchmark Results (mean ± std)")
    print("=" * 100)

    for dataset_name, dataset_results in summary.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 100)
        print(
            f"{'Model':<20} {'G-mean':>15} {'Balanced Acc':>15} {'Attack Recall':>15} {'Benign Recall':>15} {'Time (ms)':>15}"
        )
        print("-" * 100)

        for model_name, metrics in dataset_results.items():
            gmean = metrics["gmean"]
            bal_acc = metrics["balanced_accuracy"]
            atk_rec = metrics["attack_recall"]
            ben_rec = metrics["benign_recall"]
            time_ms = metrics["time_ms"]

            print(
                f"{model_name:<20} "
                f"{gmean['mean']:>6.4f}±{gmean['std']:<6.4f} "
                f"{bal_acc['mean']:>6.4f}±{bal_acc['std']:<6.4f} "
                f"{atk_rec['mean']:>6.4f}±{atk_rec['std']:<6.4f} "
                f"{ben_rec['mean']:>6.4f}±{ben_rec['std']:<6.4f} "
                f"{time_ms['mean']:>8.1f}±{time_ms['std']:<6.1f}"
            )


# =============================================================================
# Table 2: Ablation Study
# =============================================================================


@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    name: str
    adaptive_k_enabled: bool
    unlearning_enabled: bool
    k_min: int
    k_max: int
    description: str


def get_table2_configs() -> list[AblationConfig]:
    """Get ablation configurations for Table 2."""
    return [
        AblationConfig(
            name="SUDA-Full",
            adaptive_k_enabled=True,
            unlearning_enabled=True,
            k_min=1,
            k_max=70,
            description="Full SUDA with Adaptive-k + Unlearning",
        ),
        AblationConfig(
            name="AdaptiveK-Only",
            adaptive_k_enabled=True,
            unlearning_enabled=False,
            k_min=1,
            k_max=70,
            description="Adaptive-k without Unlearning",
        ),
        AblationConfig(
            name="Baseline",
            adaptive_k_enabled=False,
            unlearning_enabled=False,
            k_min=10,
            k_max=10,
            description="No Adaptive-k, No Unlearning",
        ),
    ]


def run_single_experiment_table2(
    config: AblationConfig,
    X: np.ndarray,
    y: np.ndarray,
    num_features: int,
    seed: int,
    batch_size: int = 200,
    metrics_window: int = 1000,
    eval_interval: int = 1000,
    warmup_samples: int = 1000,
) -> dict:
    """Run a single ablation experiment for Table 2.

    Uses sliding-window prequential evaluation with batch pre-training warmup.
    """
    model = SUDA(
        num_features=num_features,
        num_trees=50,
        k=10,
        max_depth=15,
        adaptive_k_enabled=config.adaptive_k_enabled,
        unlearning_enabled=config.unlearning_enabled,
        k_min=config.k_min,
        k_max=config.k_max,
        gmean_drop_threshold=0.05,
        recall_drop_threshold=0.10,
        seed=seed,
    )

    # Batch pre-train on first N samples
    stream_start = 0
    if warmup_samples > 0 and len(y) > warmup_samples:
        model.fit(X[:warmup_samples], y[:warmup_samples])
        stream_start = warmup_samples

    X_stream = X[stream_start:]
    y_stream = y[stream_start:]

    all_y_true = []
    all_y_pred = []
    gmean_series = []
    attack_recall_series = []
    balanced_acc_series = []
    benign_recall_series = []

    n_batches = len(y_stream) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_stream[start_idx:end_idx]
        y_batch = y_stream[start_idx:end_idx]

        result = model.partial_fit(X_batch, y_batch)
        all_y_true.extend(y_batch)
        all_y_pred.extend(result.predictions.astype(int))

        # Compute window metrics periodically
        total_processed = stream_start + end_idx
        if total_processed % eval_interval == 0 or end_idx >= len(y_stream):
            window_start = max(0, len(all_y_true) - metrics_window)
            window_true = np.array(all_y_true[window_start:], dtype=np.int64)
            window_pred = np.array(all_y_pred[window_start:], dtype=np.int64)

            gmean_series.append(compute_gmean(window_true, window_pred))
            attack_recall_series.append(compute_attack_recall(window_true, window_pred))
            balanced_acc_series.append(compute_balanced_accuracy(window_true, window_pred))
            benign_recall_series.append(compute_benign_recall(window_true, window_pred))

    return {
        "gmean": float(np.mean(gmean_series)) if gmean_series else 0.0,
        "balanced_accuracy": float(np.mean(balanced_acc_series)) if balanced_acc_series else 0.0,
        "attack_recall": float(np.mean(attack_recall_series)) if attack_recall_series else 0.0,
        "benign_recall": float(np.mean(benign_recall_series)) if benign_recall_series else 0.0,
    }


def run_table2_benchmark(seeds: list[int], output_dir: Path) -> dict:
    """Run Table 2: Ablation Study on NSL-KDD."""
    logger.info("=" * 80)
    logger.info("Table 2: Ablation Study (NSL-KDD)")
    logger.info("=" * 80)

    dataset_name = "nslkdd"
    info = get_dataset_info(dataset_name)
    num_features = info.n_features

    configs = get_table2_configs()
    results = {config.name: [] for config in configs}

    for seed in tqdm(seeds, desc="Ablation"):
        X, y = make_realistic_drift_stream(
            dataset_name,
            pre_attack_ratio=0.01,
            post_attack_ratio=0.05,
            total_samples=50000,
            drift_point=25000,
            seed=seed,
        )

        for config in configs:
            logger.info(f"  Running {config.name} (seed={seed})")
            try:
                result = run_single_experiment_table2(config, X, y, num_features, seed)
                results[config.name].append(result)
            except Exception as e:
                logger.error(f"  Error with {config.name}: {e}")
                continue

    # Save results
    output_file = output_dir / "table2_raw_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nSaved raw results to {output_file}")

    # Compute summary
    summary = compute_table2_summary(results)
    summary_file = output_dir / "table2_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved summary to {summary_file}")

    # Print summary
    print_table2_summary(summary)

    return results


def compute_table2_summary(results: dict) -> dict:
    """Compute summary statistics for Table 2."""
    summary = {}

    for config_name, runs in results.items():
        if not runs:
            continue

        metrics = {}
        for key in ["gmean", "balanced_accuracy", "attack_recall", "benign_recall"]:
            values = [r[key] for r in runs]
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        summary[config_name] = metrics

    return summary


def print_table2_summary(summary: dict):
    """Print Table 2 summary."""
    print("\n" + "=" * 100)
    print("Table 2: Ablation Study Results (mean ± std)")
    print("=" * 100)
    print(
        f"{'Configuration':<20} {'G-mean':>15} {'Balanced Acc':>15} {'Attack Recall':>15} {'Benign Recall':>15}"
    )
    print("-" * 100)

    for config_name, metrics in summary.items():
        gmean = metrics["gmean"]
        bal_acc = metrics["balanced_accuracy"]
        atk_rec = metrics["attack_recall"]
        ben_rec = metrics["benign_recall"]

        print(
            f"{config_name:<20} "
            f"{gmean['mean']:>6.4f}±{gmean['std']:<6.4f} "
            f"{bal_acc['mean']:>6.4f}±{bal_acc['std']:<6.4f} "
            f"{atk_rec['mean']:>6.4f}±{atk_rec['std']:<6.4f} "
            f"{ben_rec['mean']:>6.4f}±{ben_rec['std']:<6.4f}"
        )


# =============================================================================
# Table 3: Extreme Drift (1%→50%→1%)
# =============================================================================


def make_extreme_drift_stream(
    name: str,
    phases: list[tuple[int, float]],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a multi-phase extreme drift stream.

    Args:
        name: Dataset name
        phases: List of (n_samples, attack_ratio) tuples for each phase
        seed: Random seed

    Returns:
        Tuple of (X, y) concatenated stream
    """
    rng = np.random.default_rng(seed)
    X_raw, y_raw = load_dataset(name)

    benign_mask = y_raw == 0
    attack_mask = y_raw == 1
    X_benign, y_benign = X_raw[benign_mask], y_raw[benign_mask]
    X_attack, y_attack = X_raw[attack_mask], y_raw[attack_mask]

    X_parts, y_parts = [], []
    for n_samples, attack_ratio in phases:
        n_attack = int(n_samples * attack_ratio)
        n_benign = n_samples - n_attack

        b_idx = rng.choice(len(X_benign), size=n_benign, replace=True)
        a_idx = rng.choice(len(X_attack), size=n_attack, replace=True)

        X_phase = np.vstack([X_benign[b_idx], X_attack[a_idx]])
        y_phase = np.hstack([y_benign[b_idx], y_attack[a_idx]])

        perm = rng.permutation(len(y_phase))
        X_parts.append(X_phase[perm])
        y_parts.append(y_phase[perm])

    return np.vstack(X_parts), np.hstack(y_parts)


def run_single_experiment_table3(
    model_name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    phase_boundaries: list[int],
    batch_size: int = 200,
    metrics_window: int = 1000,
    warmup_samples: int = 1000,
) -> dict:
    """Run a single extreme drift experiment with window-based metrics."""
    is_suda = "SUDA" in model_name

    # SUDA warmup: batch pre-train on first N samples
    stream_start = 0
    if is_suda and warmup_samples > 0 and len(y) > warmup_samples:
        model.fit(X[:warmup_samples], y[:warmup_samples])
        stream_start = warmup_samples

    X_stream = X[stream_start:]
    y_stream = y[stream_start:]
    # Adjust phase boundaries for the warmup offset
    adjusted_boundaries = [max(0, b - stream_start) for b in phase_boundaries]

    all_y_true = []
    all_y_pred = []

    n_batches = len(y_stream) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_stream[start_idx:end_idx]
        y_batch = y_stream[start_idx:end_idx]

        if is_suda:
            result = model.partial_fit(X_batch, y_batch)
            preds = result.predictions.astype(int)
        else:
            preds = model.predict(X_batch)
            model.partial_fit(X_batch, y_batch)

        all_y_true.extend(y_batch)
        all_y_pred.extend(preds)

    y_true_arr = np.array(all_y_true, dtype=np.int64)
    y_pred_arr = np.array(all_y_pred, dtype=np.int64)

    # Compute per-phase window-based metrics
    phase_metrics = {}
    boundaries = [0] + adjusted_boundaries + [len(y_true_arr)]

    for phase_idx in range(len(boundaries) - 1):
        start = boundaries[phase_idx]
        end = min(boundaries[phase_idx + 1], len(y_true_arr))

        if end <= start:
            continue

        phase_y_true = y_true_arr[start:end]
        phase_y_pred = y_pred_arr[start:end]

        # Window-based G-mean for this phase
        phase_gmeans = []
        for w_start in range(0, len(phase_y_true) - metrics_window + 1, metrics_window // 2):
            w_end = w_start + metrics_window
            wt = phase_y_true[w_start:w_end]
            wp = phase_y_pred[w_start:w_end]
            phase_gmeans.append(compute_gmean(wt, wp))

        if not phase_gmeans:
            phase_gmeans = [compute_gmean(phase_y_true, phase_y_pred)]

        phase_metrics[f"phase_{phase_idx + 1}"] = {
            "gmean": float(np.mean(phase_gmeans)),
            "attack_recall": compute_attack_recall(phase_y_true, phase_y_pred),
        }

    return {
        "overall_gmean": float(np.mean([pm["gmean"] for pm in phase_metrics.values()])),
        "phase_metrics": phase_metrics,
        "phase3_gmean": phase_metrics.get("phase_3", {}).get("gmean", 0.0),
    }


def run_table3_benchmark(seeds: list[int], output_dir: Path) -> dict:
    """Run Table 3: Extreme Drift (1%→50%→1%)."""
    logger.info("=" * 80)
    logger.info("Table 3: Extreme Drift (1%→50%→1%)")
    logger.info("=" * 80)

    dataset_name = "nslkdd"
    info = get_dataset_info(dataset_name)
    num_features = info.n_features

    results = {
        "SUDA": [],
        "SUDA-NoUnlearn": [],
        "ARF": [],
        "SRP": [],
    }

    for seed in tqdm(seeds, desc="Extreme Drift"):
        # Create 3-phase extreme drift stream: 1% → 50% → 1%
        X, y = make_extreme_drift_stream(
            dataset_name,
            phases=[
                (20000, 0.01),  # Phase 1: Normal (1%)
                (20000, 0.50),  # Phase 2: Spike (50%)
                (20000, 0.01),  # Phase 3: Recovery (1%)
            ],
            seed=seed,
        )
        phase_boundaries = [20000, 40000]

        # SUDA with unlearning
        logger.info(f"  Running SUDA (seed={seed})")
        try:
            model = SUDA(
                num_features=num_features,
                num_trees=50,
                k=10,
                max_depth=15,
                adaptive_k_enabled=True,
                unlearning_enabled=True,
                k_min=1,
                k_max=70,
                seed=seed,
            )
            result = run_single_experiment_table3(
                "SUDA", model, X, y, phase_boundaries
            )
            results["SUDA"].append(result)
        except Exception as e:
            logger.error(f"  Error with SUDA: {e}")

        # SUDA without unlearning
        logger.info(f"  Running SUDA-NoUnlearn (seed={seed})")
        try:
            model = SUDA(
                num_features=num_features,
                num_trees=50,
                k=10,
                max_depth=15,
                adaptive_k_enabled=True,
                unlearning_enabled=False,
                k_min=1,
                k_max=70,
                seed=seed,
            )
            result = run_single_experiment_table3(
                "SUDA", model, X, y, phase_boundaries
            )
            results["SUDA-NoUnlearn"].append(result)
        except Exception as e:
            logger.error(f"  Error with SUDA-NoUnlearn: {e}")

        # ARF
        logger.info(f"  Running ARF (seed={seed})")
        try:
            model = ARFModel(n_models=50, seed=seed)
            result = run_single_experiment_table3("ARF", model, X, y, phase_boundaries)
            results["ARF"].append(result)
        except Exception as e:
            logger.error(f"  Error with ARF: {e}")

        # SRP
        logger.info(f"  Running SRP (seed={seed})")
        try:
            model = SRPModel(n_models=10, seed=seed)
            result = run_single_experiment_table3("SRP", model, X, y, phase_boundaries)
            results["SRP"].append(result)
        except Exception as e:
            logger.error(f"  Error with SRP: {e}")

    # Save results
    output_file = output_dir / "table3_raw_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nSaved raw results to {output_file}")

    # Compute summary
    summary = compute_table3_summary(results)
    summary_file = output_dir / "table3_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved summary to {summary_file}")

    # Print summary
    print_table3_summary(summary)

    return results


def compute_table3_summary(results: dict) -> dict:
    """Compute summary statistics for Table 3."""
    summary = {}

    for model_name, runs in results.items():
        if not runs:
            continue

        overall_gmeans = [r["overall_gmean"] for r in runs]
        phase3_gmeans = [r["phase3_gmean"] for r in runs]

        summary[model_name] = {
            "overall_gmean": {
                "mean": float(np.mean(overall_gmeans)),
                "std": float(np.std(overall_gmeans)),
            },
            "phase3_gmean": {
                "mean": float(np.mean(phase3_gmeans)),
                "std": float(np.std(phase3_gmeans)),
            },
        }

    return summary


def print_table3_summary(summary: dict):
    """Print Table 3 summary."""
    print("\n" + "=" * 80)
    print("Table 3: Extreme Drift Results (mean ± std)")
    print("=" * 80)
    print(f"{'Model':<20} {'Overall G-mean':>20} {'Phase 3 G-mean (Recovery)':>30}")
    print("-" * 80)

    for model_name, metrics in summary.items():
        overall = metrics["overall_gmean"]
        phase3 = metrics["phase3_gmean"]

        print(
            f"{model_name:<20} "
            f"{overall['mean']:>8.4f}±{overall['std']:<8.4f} "
            f"{phase3['mean']:>12.4f}±{phase3['std']:<12.4f}"
        )


# =============================================================================
# Figure Generation
# =============================================================================


def generate_figures(results: dict, output_dir: Path):
    """Generate comparison figures."""
    logger.info("\nGenerating figures...")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Figure 1: G-mean comparison across datasets (Table 1)
    if "table1_summary" in results:
        generate_gmean_comparison_figure(results["table1_summary"], figures_dir)

    # Figure 2: Ablation study (Table 2)
    if "table2_summary" in results:
        generate_ablation_figure(results["table2_summary"], figures_dir)

    # Figure 3: Extreme drift recovery (Table 3)
    if "table3_summary" in results:
        generate_extreme_drift_figure(results["table3_summary"], figures_dir)

    logger.info(f"Figures saved to {figures_dir}")


def generate_gmean_comparison_figure(summary: dict, figures_dir: Path):
    """Generate G-mean comparison bar chart for Table 1."""
    datasets = list(summary.keys())
    models = list(next(iter(summary.values())).keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.1
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        means = [summary[ds][model]["gmean"]["mean"] for ds in datasets]
        stds = [summary[ds][model]["gmean"]["std"] for ds in datasets]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=model,
            color=colors[i],
            capsize=3,
        )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("G-mean", fontsize=12)
    ax.set_title("Table 1: G-mean Comparison Across Datasets", fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([ds.upper() for ds in datasets])
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "table1_gmean_comparison.png", dpi=300)
    plt.close()


def generate_ablation_figure(summary: dict, figures_dir: Path):
    """Generate ablation study bar chart for Table 2."""
    configs = list(summary.keys())
    metrics = ["gmean", "balanced_accuracy", "attack_recall", "benign_recall"]
    metric_labels = ["G-mean", "Balanced Acc", "Attack Recall", "Benign Recall"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, config in enumerate(configs):
        means = [summary[config][metric]["mean"] for metric in metrics]
        stds = [summary[config][metric]["std"] for metric in metrics]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=config,
            color=colors[i],
            capsize=3,
        )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Table 2: Ablation Study Results", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "table2_ablation.png", dpi=300)
    plt.close()


def generate_extreme_drift_figure(summary: dict, figures_dir: Path):
    """Generate extreme drift comparison for Table 3."""
    models = list(summary.keys())

    phase3_means = [summary[m]["phase3_gmean"]["mean"] for m in models]
    phase3_stds = [summary[m]["phase3_gmean"]["std"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars = ax.bar(x, phase3_means, color=colors[: len(models)], capsize=5)
    ax.errorbar(x, phase3_means, yerr=phase3_stds, fmt="none", color="black", capsize=5)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Phase 3 G-mean (Recovery)", fontsize=12)
    ax.set_title("Table 3: Extreme Drift Recovery Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "table3_extreme_drift.png", dpi=300)
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SUDA Paper Benchmark - Comprehensive Experiments"
    )
    parser.add_argument(
        "--table",
        type=str,
        default="all",
        choices=["all", "1", "2", "3"],
        help="Which table to run (default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use 10 seeds instead of 5 (full validation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="note/260208/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["nslkdd", "unswnb15", "cicids2018"],
        help="Datasets for Table 1 (default: nslkdd unswnb15 cicids2018)",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seeds
    seeds = (
        [42, 123, 456, 789, 2026, 314, 628, 999, 1234, 5678]
        if args.full
        else [42, 123, 456, 789, 2026]
    )

    logger.info(f"Running paper benchmark with {len(seeds)} seeds")
    logger.info(f"Output directory: {output_dir}")

    all_results = {}

    # Run requested tables
    if args.table in ["all", "1"]:
        table1_results = run_table1_benchmark(args.datasets, seeds, output_dir)
        all_results["table1"] = table1_results
        all_results["table1_summary"] = compute_table1_summary(table1_results)

    if args.table in ["all", "2"]:
        table2_results = run_table2_benchmark(seeds, output_dir)
        all_results["table2"] = table2_results
        all_results["table2_summary"] = compute_table2_summary(table2_results)

    if args.table in ["all", "3"]:
        table3_results = run_table3_benchmark(seeds, output_dir)
        all_results["table3"] = table3_results
        all_results["table3_summary"] = compute_table3_summary(table3_results)

    # Generate figures
    generate_figures(all_results, output_dir)

    # Save complete results
    complete_file = output_dir / "paper_benchmark_complete.json"
    with open(complete_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nComplete results saved to {complete_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Paper Benchmark Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
