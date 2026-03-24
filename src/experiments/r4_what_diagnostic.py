"""R4: "WHAT Matters" Diagnostic Experiment.

Phase 1: Pure Confidence vs FIFO (78 runs)
  - Remove age component entirely (age_w=0.0, inf_w=1.0)
  - 100% influence coverage (influence_sample_count=3000)
  - No minority protection (mp=0.0) to isolate influence effect

Phase 2: Confidence ↔ Minority Protection separation (36 runs)
  - Test if Confidence effect ≈ minority protection
  - 4 configs: FIFO-Only, FIFO+MP, Pure-Confidence, Confidence+MP
  - moderate_sudden only (Confidence won 3/3 in R3)

Phase 3: Registry diagnostics (2 instrumented runs)
  - Age-Influence Spearman correlation over time
  - nslkdd + cicids2018, moderate_sudden, seed=42

Usage:
    # Phase 1
    uv run python -m src.experiments.r4_what_diagnostic --phase 1 --seeds 42 123 456

    # Phase 2
    uv run python -m src.experiments.r4_what_diagnostic --phase 2 --seeds 42 123 456

    # Phase 3 (instrumented, 2 runs)
    uv run python -m src.experiments.r4_what_diagnostic --phase 3 --seeds 42

    # Sanity check
    uv run python -m src.experiments.r4_what_diagnostic --sanity

    # Resume from checkpoint
    uv run python -m src.experiments.r4_what_diagnostic --phase 1 --resume
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
        budget_minority_protection=p.get("budget_minority_protection", 0.0),
        budget_skip_forest_forget=False,
        budget_use_feature_distance=False,
        influence_tracking=p.get("influence_tracking", False),
        influence_update_interval=p.get("influence_update_interval", 10),
        influence_sample_count=p.get("influence_sample_count", 200),
        influence_strategy=p.get("influence_strategy", "none"),
    )


# =============================================================================
# Phase 1: Pure Confidence vs FIFO
# =============================================================================

CONFIGS_PHASE_1 = [
    # FIFO-Baseline: pure age-based, no influence, no MP
    ModelConfig("FIFO-Baseline", {
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        "influence_strategy": "none",
    }),
    # Pure-Confidence: 100% influence, 100% coverage, no age, no MP
    ModelConfig("Pure-Confidence", {
        "budget_age_weight": 0.0,
        "budget_influence_weight": 1.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 3000,  # 100% coverage (budget=3000)
        "influence_strategy": "confidence",
    }),
]


# =============================================================================
# Phase 2: Confidence ↔ Minority Protection Separation
# =============================================================================

CONFIGS_PHASE_2 = [
    # FIFO-Only: pure age, no MP
    ModelConfig("FIFO-Only", {
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        "influence_strategy": "none",
    }),
    # FIFO+MP: age + minority protection
    ModelConfig("FIFO+MP", {
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.1,
        "influence_strategy": "none",
    }),
    # Pure-Confidence: 100% influence, no MP
    ModelConfig("Pure-Confidence", {
        "budget_age_weight": 0.0,
        "budget_influence_weight": 1.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 3000,
        "influence_strategy": "confidence",
    }),
    # Confidence+MP: confidence + minority protection
    ModelConfig("Confidence+MP", {
        "budget_age_weight": 0.0,
        "budget_influence_weight": 1.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.1,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 3000,
        "influence_strategy": "confidence",
    }),
]


# =============================================================================
# Phase 3: Instrumented configs (same as Phase 1 but with diagnostics)
# =============================================================================

CONFIGS_PHASE_3 = [
    ModelConfig("FIFO-Diag", {
        "budget_age_weight": 1.0,
        "budget_influence_weight": 0.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        # Enable influence tracking for diagnostics even though eviction is FIFO
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 3000,
        "influence_strategy": "confidence",
    }),
    ModelConfig("Confidence-Diag", {
        "budget_age_weight": 0.0,
        "budget_influence_weight": 1.0,
        "budget_class_weight": 0.0,
        "budget_minority_protection": 0.0,
        "influence_tracking": True,
        "influence_update_interval": 1,
        "influence_sample_count": 3000,
        "influence_strategy": "confidence",
    }),
]


# =============================================================================
# Run Functions
# =============================================================================

def run_single(
    dataset: str,
    scenario_name: str,
    config: ModelConfig,
    seed: int,
) -> dict:
    num_features = DATASET_FEATURES[dataset]

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

    all_preds, all_labels = [], []
    n_batches = len(X_test) // BATCH_SIZE
    start_time = time.time()

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        X_batch = X_test[start:end]
        y_batch = y_test[start:end].astype(bool)

        result = model.partial_fit(X_batch, y_batch, record_history=False)
        preds = result.predictions.astype(int)
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
        "final_registry_size": model.registry_size,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
    }


def run_single_instrumented(
    dataset: str,
    scenario_name: str,
    config: ModelConfig,
    seed: int,
    diag_interval: int = 100,
) -> dict:
    """Run with periodic registry diagnostics collection."""
    num_features = DATASET_FEATURES[dataset]

    scenario_fn = NIDS_SCENARIOS[scenario_name]
    X_stream, y_stream, metadata = scenario_fn(dataset, seed=seed)

    total = len(y_stream)
    warmup_n = int(total * WARMUP_RATIO)
    X_warmup, y_warmup = X_stream[:warmup_n], y_stream[:warmup_n]
    X_test, y_test = X_stream[warmup_n:], y_stream[warmup_n:]

    model = create_model(config, num_features, seed)
    model.fit(X_warmup, y_warmup.astype(bool))

    all_preds, all_labels = [], []
    diagnostics_snapshots = []
    n_batches = len(X_test) // BATCH_SIZE
    start_time = time.time()

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        X_batch = X_test[start:end]
        y_batch = y_test[start:end].astype(bool)

        result = model.partial_fit(X_batch, y_batch, record_history=False)
        preds = result.predictions.astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_test[start:end].astype(int).tolist())

        # Collect diagnostics every diag_interval batches
        if (i + 1) % diag_interval == 0 or i == n_batches - 1:
            diag = model.get_registry_diagnostics()
            diag["batch"] = i + 1
            diag["total_batches"] = n_batches

            # Compute running metrics
            y_true_so_far = np.array(all_labels, dtype=int)
            y_pred_so_far = np.array(all_preds, dtype=int)
            diag["running_gmean"] = compute_gmean(y_true_so_far, y_pred_so_far)
            diag["running_ar"] = float(recall_score(
                y_true_so_far, y_pred_so_far, pos_label=1, zero_division=0))
            diag["running_br"] = float(recall_score(
                y_true_so_far, y_pred_so_far, pos_label=0, zero_division=0))

            diagnostics_snapshots.append(diag)
            logger.info(
                f"  [Diag batch={i+1}] spearman={diag.get('age_influence_spearman', 'N/A'):.4f}, "
                f"coverage={diag.get('influence_coverage', 0):.2%}, "
                f"gmean={diag['running_gmean']:.4f}"
            )

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
        "final_registry_size": model.registry_size,
        "elapsed_seconds": elapsed,
        "total_samples": len(y_true),
        "diagnostics": diagnostics_snapshots,
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
    run_specs: list[tuple[str, str]],
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
                        f"AR={result['attack_recall']:.4f}, "
                        f"BR={result['benign_recall']:.4f}, "
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


def run_phase_3(
    output_dir: Path,
    seed: int = 42,
) -> list[dict]:
    """Phase 3: Instrumented runs for registry diagnostics."""
    checkpoint_path = output_dir / "r4_phase_3_raw.json"
    results = []

    phase3_specs = [
        ("nslkdd", "moderate_sudden"),
        ("cicids2018", "moderate_sudden"),
    ]

    total = len(phase3_specs) * len(CONFIGS_PHASE_3)
    run_count = 0

    for dataset, scenario in phase3_specs:
        for config in CONFIGS_PHASE_3:
            run_count += 1
            logger.info(
                f"[Phase 3: {run_count}/{total}] "
                f"{dataset}/{scenario}/{config.name}/seed={seed}"
            )
            try:
                result = run_single_instrumented(
                    dataset, scenario, config, seed, diag_interval=100,
                )
                results.append(result)
                logger.info(
                    f"  -> G-mean={result['gmean']:.4f}, "
                    f"Snapshots={len(result['diagnostics'])}"
                )
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "config": config.name,
                    "seed": seed,
                    "error": str(e),
                })

    save_checkpoint(results, checkpoint_path)
    return results


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(
    results: list[dict],
    output_dir: Path,
    experiment_name: str,
    config_names: list[str],
):
    summary_path = output_dir / f"{experiment_name}_summary.md"

    # Group by (dataset, scenario)
    combos = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["dataset"], r["scenario"])
        if key not in combos:
            combos[key] = {}
        cn = r["config"]
        if cn not in combos[key]:
            combos[key][cn] = {"gmean": [], "ar": [], "br": []}
        combos[key][cn]["gmean"].append(r["gmean"])
        combos[key][cn]["ar"].append(r["attack_recall"])
        combos[key][cn]["br"].append(r["benign_recall"])

    lines = [
        f"# R4 WHAT Diagnostic Results — {experiment_name}",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## G-mean Summary (mean ± std)",
        "",
    ]

    # Header
    header = "| Dataset | Scenario | " + " | ".join(config_names) + " |"
    sep = "|---------|----------|" + "|".join(["----------"] * len(config_names)) + "|"
    lines.extend([header, sep])

    wins = {cn: 0 for cn in config_names}

    for (dataset, scenario), configs_data in sorted(combos.items()):
        row = f"| {dataset} | {scenario} |"
        best_mean = -1.0
        means = {}
        for cn in config_names:
            vals = configs_data.get(cn, {}).get("gmean", [])
            if vals:
                m = np.mean(vals)
                means[cn] = m
                if m > best_mean:
                    best_mean = m

        best_cn = None
        for cn in config_names:
            if means.get(cn) == best_mean:
                best_cn = cn
                break

        if best_cn:
            wins[best_cn] = wins.get(best_cn, 0) + 1

        for cn in config_names:
            vals = configs_data.get(cn, {}).get("gmean", [])
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

    # AR/BR summary
    lines.extend(["", "## Attack Recall / Benign Recall", ""])
    header2 = "| Dataset | Scenario | " + " | ".join(
        [f"{cn} AR/BR" for cn in config_names]) + " |"
    sep2 = "|---------|----------|" + "|".join(["----------"] * len(config_names)) + "|"
    lines.extend([header2, sep2])

    for (dataset, scenario), configs_data in sorted(combos.items()):
        row = f"| {dataset} | {scenario} |"
        for cn in config_names:
            ar_vals = configs_data.get(cn, {}).get("ar", [])
            br_vals = configs_data.get(cn, {}).get("br", [])
            if ar_vals and br_vals:
                row += f" {np.mean(ar_vals):.4f}/{np.mean(br_vals):.4f} |"
            else:
                row += " — |"
        lines.append(row)

    # Win count
    lines.extend(["", "## Win Count (by G-mean)", ""])
    for cn in config_names:
        lines.append(f"- **{cn}**: {wins.get(cn, 0)} wins")

    # Cross-dataset averages
    lines.extend(["", "## Cross-Dataset Averages", ""])
    for cn in config_names:
        gmeans = [r["gmean"] for r in results if r.get("config") == cn and "error" not in r]
        times = [r["elapsed_seconds"] for r in results if r.get("config") == cn and "error" not in r]
        if gmeans:
            lines.append(
                f"- **{cn}**: G-mean={np.mean(gmeans):.4f}±{np.std(gmeans):.4f}, "
                f"Time={np.mean(times):.1f}s"
            )

    # Passthrough verification
    lines.extend(["", "## Passthrough Verification", ""])
    fifo_gmeans = [r["gmean"] for r in results
                   if "FIFO" in r.get("config", "") and "error" not in r]
    conf_gmeans = [r["gmean"] for r in results
                   if "Confidence" in r.get("config", "") and "error" not in r]
    if fifo_gmeans and conf_gmeans:
        fifo_m = np.mean(fifo_gmeans)
        conf_m = np.mean(conf_gmeans)
        diff = abs(fifo_m - conf_m)
        if diff < 0.0001:
            lines.append(
                f"**WARNING**: FIFO ({fifo_m:.4f}) ≈ Confidence ({conf_m:.4f}) "
                f"— possible passthrough bug!"
            )
        else:
            lines.append(
                f"OK: FIFO ({fifo_m:.4f}) ≠ Confidence ({conf_m:.4f}), diff={diff:.4f}"
            )

    lines.append("")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written to {summary_path}")


def generate_phase3_summary(results: list[dict], output_dir: Path):
    """Generate Phase 3 diagnostics summary."""
    summary_path = output_dir / "r4_phase_3_summary.md"

    lines = [
        "# R4 Phase 3: Registry Diagnostics",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"## {r['dataset']}/{r['config']} — ERROR: {r['error']}")
            continue

        lines.extend([
            f"## {r['dataset']} / {r['scenario']} / {r['config']}",
            f"- G-mean: {r['gmean']:.4f}",
            f"- AR: {r['attack_recall']:.4f}, BR: {r['benign_recall']:.4f}",
            "",
            "### Diagnostics Timeline",
            "",
            "| Batch | Spearman(age,inf) | Coverage | Mean Age | Mean Inf | G-mean | AR | BR | n_benign | n_attack |",
            "|-------|-------------------|----------|----------|----------|--------|----|----|----------|----------|",
        ])

        for snap in r.get("diagnostics", []):
            sp = snap.get("age_influence_spearman", float("nan"))
            sp_str = f"{sp:.4f}" if not (sp != sp) else "N/A"  # NaN check
            lines.append(
                f"| {snap.get('batch', '?')}/{snap.get('total_batches', '?')} "
                f"| {sp_str} "
                f"| {snap.get('influence_coverage', 0):.2%} "
                f"| {snap.get('mean_age', 0):.0f} "
                f"| {snap.get('mean_influence', 0):.4f} "
                f"| {snap.get('running_gmean', 0):.4f} "
                f"| {snap.get('running_ar', 0):.4f} "
                f"| {snap.get('running_br', 0):.4f} "
                f"| {int(snap.get('n_benign', 0))} "
                f"| {int(snap.get('n_attack', 0))} |"
            )

        lines.extend(["", "### Key Findings", ""])

        # Analyze Spearman trend
        spearman_vals = [
            s.get("age_influence_spearman", float("nan"))
            for s in r.get("diagnostics", [])
        ]
        valid_sp = [v for v in spearman_vals if v == v]  # filter NaN
        if valid_sp:
            mean_sp = np.mean(valid_sp)
            lines.append(f"- Mean Spearman(age, influence): **{mean_sp:.4f}**")
            if abs(mean_sp) > 0.7:
                lines.append("  → **Strong correlation**: age ≈ influence → FIFO is a good proxy")
            elif abs(mean_sp) > 0.3:
                lines.append("  → **Moderate correlation**: some overlap between age and influence")
            else:
                lines.append("  → **Weak correlation**: age and influence are independent → Confidence should help")

        lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Phase 3 summary written to {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="R4: WHAT Matters Diagnostic")
    parser.add_argument("--phase", choices=["1", "2", "3"], default="1")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sanity", action="store_true", help="Quick sanity check")
    parser.add_argument("--no-anoshift", action="store_true", help="Skip ANoShift dataset")
    parser.add_argument("--configs", nargs="+", type=str, default=None,
                        help="Filter configs by name")
    parser.add_argument("--output-dir", type=str, default="results/paper_v2_rerun")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sanity:
        logger.info("=== SANITY CHECK MODE ===")
        run_specs = [("nslkdd", "moderate_sudden")]
        results = run_experiment_loop(
            CONFIGS_PHASE_1, run_specs, [42], output_dir, "r4_sanity", resume=False,
        )
        generate_summary(
            results, output_dir, "r4_sanity",
            [c.name for c in CONFIGS_PHASE_1],
        )
        return

    if args.phase == "1":
        configs = CONFIGS_PHASE_1
        if args.configs:
            configs = [c for c in CONFIGS_PHASE_1 if c.name in args.configs]
            if not configs:
                logger.error(f"No matching configs: {args.configs}")
                logger.info(f"Available: {[c.name for c in CONFIGS_PHASE_1]}")
                return

        # 3 NIDS × 4 scenarios + optional ANoShift
        run_specs = []
        for ds in NIDS_DATASETS:
            for sc in NIDS_SCENARIOS:
                run_specs.append((ds, sc))
        if not args.no_anoshift:
            run_specs.append(("anoshift", "temporal_10year"))

        logger.info(
            f"Phase 1: {len(configs)} configs × {len(run_specs)} combos × "
            f"{len(args.seeds)} seeds = {len(configs) * len(run_specs) * len(args.seeds)} runs"
        )
        results = run_experiment_loop(
            configs, run_specs, args.seeds, output_dir,
            "r4_phase_1", resume=args.resume,
        )
        generate_summary(
            results, output_dir, "r4_phase_1",
            [c.name for c in CONFIGS_PHASE_1],
        )

    elif args.phase == "2":
        configs = CONFIGS_PHASE_2
        if args.configs:
            configs = [c for c in CONFIGS_PHASE_2 if c.name in args.configs]
            if not configs:
                logger.error(f"No matching configs: {args.configs}")
                logger.info(f"Available: {[c.name for c in CONFIGS_PHASE_2]}")
                return

        # moderate_sudden only
        run_specs = [(ds, "moderate_sudden") for ds in NIDS_DATASETS]

        logger.info(
            f"Phase 2: {len(configs)} configs × {len(run_specs)} combos × "
            f"{len(args.seeds)} seeds = {len(configs) * len(run_specs) * len(args.seeds)} runs"
        )
        results = run_experiment_loop(
            configs, run_specs, args.seeds, output_dir,
            "r4_phase_2", resume=args.resume,
        )
        generate_summary(
            results, output_dir, "r4_phase_2",
            [c.name for c in CONFIGS_PHASE_2],
        )

    elif args.phase == "3":
        seed = args.seeds[0] if args.seeds else 42
        logger.info(f"Phase 3: Instrumented diagnostics (seed={seed})")
        results = run_phase_3(output_dir, seed=seed)
        generate_phase3_summary(results, output_dir)


if __name__ == "__main__":
    main()
