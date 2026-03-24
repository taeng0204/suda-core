"""R3: Influence Strategy Exploration Experiment.

Tests 6 different influence computation strategies for budget eviction:
  1. FIFO-Baseline: No influence (age-only)
  2. OOB-Influence: Standard OOB influence
  3. Loss-Based: Cross-entropy loss difference
  4. Confidence-Redundant: Redundancy-based (easy samples evicted first)
  5. Cumulative-OOB: EMA of OOB influence
  6. FeatDist-Frequent: Feature-distance with frequent updates

Phase A: 6 configs × 13 combos × 3 seeds = 234 runs (NIDS + ANoShift)
Phase B: Weight sweep on top strategies (post-analysis)
Phase C: ANoShift-only subset (included in Phase A)

Usage:
    # Run Phase A (all)
    uv run python -m src.experiments.r3_influence_strategies --phase A --seeds 42 123 456

    # Run Phase A NIDS-only (faster)
    uv run python -m src.experiments.r3_influence_strategies --phase A --no-anoshift

    # Sanity check (1 seed, 1 dataset, 1 scenario)
    uv run python -m src.experiments.r3_influence_strategies --sanity

    # Resume from checkpoint
    uv run python -m src.experiments.r3_influence_strategies --phase A --resume
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
    make_anoshift_temporal_stream,
)
from src.models.suda import SUDA
from src.experiments.utils import NumpyEncoder, compute_gmean

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Common Settings
# =============================================================================

SEEDS_3 = [42, 123, 456]

NIDS_DATASETS = ["nslkdd", "unswnb15", "cicids2018"]
DATASET_FEATURES = {"nslkdd": 41, "unswnb15": 42, "cicids2018": 78, "anoshift": 6}
BATCH_SIZE = 200
WARMUP_RATIO = 0.3

NIDS_SCENARIOS = {
    "moderate_sudden": make_moderate_sudden_drift_stream,
    "stepwise": make_stepwise_drift_stream,
    "gradual_ramp": make_gradual_ramp_stream,
    "asymmetric_recovery": make_asymmetric_recovery_stream,
}


# =============================================================================
# Model Config
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    params: dict


def create_model(config: ModelConfig, num_features: int, seed: int) -> SUDA:
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
        budget_enabled=True,
        budget_max_samples=p.get("budget_max_samples", 3000),
        budget_eviction_batch=100,
        budget_age_weight=p.get("budget_age_weight", 1.0),
        budget_influence_weight=p.get("budget_influence_weight", 0.0),
        budget_class_weight=p.get("budget_class_weight", 0.0),
        budget_minority_protection=p.get("budget_minority_protection", 0.1),
        budget_skip_forest_forget=False,
        budget_use_feature_distance=p.get("budget_use_feature_distance", False),
        influence_tracking=p.get("influence_tracking", False),
        influence_update_interval=p.get("influence_update_interval", 10),
        influence_sample_count=p.get("influence_sample_count", 200),
        influence_strategy=p.get("influence_strategy", "none"),
        feat_dist_update_interval=p.get("feat_dist_update_interval", 2000),
    )


# =============================================================================
# Phase A: 6 Influence Strategies
# =============================================================================

CONFIGS_PHASE_A = [
    # 1. FIFO-Baseline: age-only, no influence computation
    ModelConfig("FIFO-Baseline", {
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.1,
        "influence_strategy": "none",
    }),
    # 2. OOB-Influence: standard OOB with more frequent updates
    ModelConfig("OOB-Influence", {
        "budget_age_weight": 0.3,
        "budget_influence_weight": 0.5,
        "budget_class_weight": 0.2,
        "budget_minority_protection": 0.1,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 500,
        "influence_strategy": "oob",
    }),
    # 3. Loss-Based: cross-entropy loss difference
    ModelConfig("Loss-Based", {
        "budget_age_weight": 0.3,
        "budget_influence_weight": 0.5,
        "budget_class_weight": 0.2,
        "budget_minority_protection": 0.1,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 500,
        "influence_strategy": "loss",
    }),
    # 4. Confidence-Redundant: easy/redundant samples evicted first
    ModelConfig("Confidence-Redundant", {
        "budget_age_weight": 0.3,
        "budget_influence_weight": 0.5,
        "budget_class_weight": 0.2,
        "budget_minority_protection": 0.1,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 500,
        "influence_strategy": "confidence",
    }),
    # 5. Cumulative-OOB: EMA of OOB for stability
    ModelConfig("Cumulative-OOB", {
        "budget_age_weight": 0.3,
        "budget_influence_weight": 0.5,
        "budget_class_weight": 0.2,
        "budget_minority_protection": 0.1,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 500,
        "influence_strategy": "cumulative_oob",
    }),
    # 6. FeatDist-Frequent: feature-distance with 500-sample update interval
    ModelConfig("FeatDist-Frequent", {
        "budget_age_weight": 0.3,
        "budget_influence_weight": 0.5,
        "budget_class_weight": 0.2,
        "budget_minority_protection": 0.1,
        "budget_use_feature_distance": True,
        "influence_strategy": "feature_distance",
        "feat_dist_update_interval": 500,
    }),
]


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

    # Load data
    if dataset == "anoshift":
        X_stream, y_stream, metadata = make_anoshift_temporal_stream(seed=seed)
    else:
        scenario_fn = NIDS_SCENARIOS[scenario_name]
        X_stream, y_stream, metadata = scenario_fn(dataset, seed=seed)

    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)
    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    model = create_model(config, num_features, seed)
    model.fit(X_warmup, y_warmup.astype(bool))

    # Streaming (test-then-train)
    all_preds, all_labels = [], []
    registry_sizes = []
    n_batches = len(X_test) // BATCH_SIZE
    start_time = time.time()

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        X_batch = X_test[start:end]
        y_batch = y_test[start:end].astype(bool)

        result = model.partial_fit(X_batch, y_batch, record_history=False)
        preds = result.predictions.astype(int)
        registry_sizes.append(result.registry_size)

        all_preds.extend(preds.tolist())
        all_labels.extend(y_test[start:end].astype(int).tolist())

    elapsed = time.time() - start_time

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

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
        "total_budget_evicted": model.total_budget_evicted,
        "final_registry_size": registry_sizes[-1] if registry_sizes else 0,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
    }


# =============================================================================
# Checkpoint / Resume
# =============================================================================

def save_checkpoint(results: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    run_specs: list[tuple[str, str]],  # (dataset, scenario) pairs
    seeds: list[int],
    output_dir: Path,
    experiment_name: str,
    resume: bool = False,
) -> list[dict]:
    checkpoint_path = output_dir / f"{experiment_name}_raw.json"
    results = load_checkpoint(checkpoint_path) if resume else []

    if resume and results:
        logger.info(f"Resuming from checkpoint: {len(results)} completed runs")

    total_runs = len(run_specs) * len(configs) * len(seeds)
    logger.info(f"Experiment {experiment_name}: {total_runs} total runs")
    run_count = 0

    for dataset, scenario_name in run_specs:
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
                save_checkpoint(results, checkpoint_path)

    save_checkpoint(results, checkpoint_path)
    return results


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(results: list[dict], output_dir: Path, experiment_name: str):
    summary_path = output_dir / f"{experiment_name}_summary.md"

    # Group by (dataset, scenario)
    combos = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["dataset"], r["scenario"])
        if key not in combos:
            combos[key] = {}
        config_name = r["config"]
        if config_name not in combos[key]:
            combos[key][config_name] = []
        combos[key][config_name].append(r["gmean"])

    lines = [
        f"# R3 Influence Strategy Results — {experiment_name}",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## G-mean Summary (mean ± std)",
        "",
    ]

    config_names = [c.name for c in CONFIGS_PHASE_A]

    # Header
    header = "| Dataset | Scenario | " + " | ".join(config_names) + " |"
    sep = "|---------|----------|" + "|".join(["----------"] * len(config_names)) + "|"
    lines.extend([header, sep])

    for (dataset, scenario), configs_data in sorted(combos.items()):
        row = f"| {dataset} | {scenario} |"
        best_mean = -1.0
        means = {}
        for cn in config_names:
            vals = configs_data.get(cn, [])
            if vals:
                m = np.mean(vals)
                means[cn] = m
                if m > best_mean:
                    best_mean = m
            else:
                means[cn] = None

        for cn in config_names:
            vals = configs_data.get(cn, [])
            if vals:
                m = np.mean(vals)
                s = np.std(vals)
                cell = f" {m:.4f}±{s:.4f}"
                if m == best_mean:
                    cell = f" **{m:.4f}**±{s:.4f}"
                row += cell + " |"
            else:
                row += " — |"
        lines.append(row)

    # Cross-dataset averages
    lines.extend(["", "## Cross-Dataset Averages", ""])
    avg_header = "| Config | Mean G-mean | Mean Time(s) |"
    avg_sep = "|--------|-------------|--------------|"
    lines.extend([avg_header, avg_sep])

    for cn in config_names:
        gmeans = [r["gmean"] for r in results if r.get("config") == cn and "error" not in r]
        times = [r["elapsed_seconds"] for r in results if r.get("config") == cn and "error" not in r]
        if gmeans:
            lines.append(f"| {cn} | {np.mean(gmeans):.4f}±{np.std(gmeans):.4f} | {np.mean(times):.1f} |")

    # Passthrough verification
    lines.extend(["", "## Passthrough Verification", ""])
    fifo_gmeans = [r["gmean"] for r in results if r.get("config") == "FIFO-Baseline" and "error" not in r]
    oob_gmeans = [r["gmean"] for r in results if r.get("config") == "OOB-Influence" and "error" not in r]
    if fifo_gmeans and oob_gmeans:
        fifo_m = np.mean(fifo_gmeans)
        oob_m = np.mean(oob_gmeans)
        diff = abs(fifo_m - oob_m)
        if diff < 0.0001:
            lines.append(f"**WARNING**: FIFO ({fifo_m:.4f}) ≈ OOB ({oob_m:.4f}) — possible passthrough bug!")
        else:
            lines.append(f"OK: FIFO ({fifo_m:.4f}) ≠ OOB ({oob_m:.4f}), diff={diff:.4f}")

    lines.append("")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written to {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="R3: Influence Strategy Exploration")
    parser.add_argument("--phase", choices=["A", "B", "C"], default="A")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sanity", action="store_true", help="Quick sanity check (1 seed, 1 dataset)")
    parser.add_argument("--no-anoshift", action="store_true", help="Skip ANoShift dataset")
    parser.add_argument("--configs", nargs="+", type=str, default=None, help="Filter configs by name")
    parser.add_argument("--output-dir", type=str, default="results/paper_v2_rerun")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter configs if --configs specified
    configs = CONFIGS_PHASE_A
    if args.configs:
        configs = [c for c in CONFIGS_PHASE_A if c.name in args.configs]
        if not configs:
            logger.error(f"No matching configs for: {args.configs}")
            logger.info(f"Available: {[c.name for c in CONFIGS_PHASE_A]}")
            return
        logger.info(f"Filtered configs: {[c.name for c in configs]}")

    if args.sanity:
        logger.info("=== SANITY CHECK MODE ===")
        run_specs = [("nslkdd", "moderate_sudden")]
        results = run_experiment_loop(
            CONFIGS_PHASE_A, run_specs, [42], output_dir, "r3_sanity", resume=False,
        )
        generate_summary(results, output_dir, "r3_sanity")
        return

    if args.phase == "A":
        # Build run specs: 3 NIDS × 4 scenarios + optional ANoShift × 1
        run_specs = []
        for ds in NIDS_DATASETS:
            for sc in NIDS_SCENARIOS:
                run_specs.append((ds, sc))
        if not args.no_anoshift:
            run_specs.append(("anoshift", "temporal_10year"))

        logger.info(f"Phase A: {len(configs)} configs × {len(run_specs)} combos × {len(args.seeds)} seeds")
        results = run_experiment_loop(
            configs, run_specs, args.seeds, output_dir, "r3_phase_a", resume=args.resume,
        )
        generate_summary(results, output_dir, "r3_phase_a")

    elif args.phase == "C":
        # ANoShift-only
        run_specs = [("anoshift", "temporal_10year")]
        results = run_experiment_loop(
            configs, run_specs, args.seeds, output_dir, "r3_phase_c", resume=args.resume,
        )
        generate_summary(results, output_dir, "r3_phase_c")

    elif args.phase == "B":
        logger.info("Phase B: Run after Phase A analysis. Implement weight sweep based on top strategies.")
        logger.info("Not yet implemented — analyze Phase A results first.")


if __name__ == "__main__":
    main()
