"""
R1-Grid: Influence Weight × Minority Protection Grid Search

Reuses paper_v2_rerun infrastructure (SCENARIOS, create_model, run_single, etc.)

Grid:
  influence_weight: [0.0, 0.3, 0.5, 0.7]
  minority_protection: [0.0, 0.05, 0.1, 0.2, 0.3]
  class_weight: always 0.0
  age_weight: 1.0 - influence_weight

20 configs × 4 scenarios × 3 datasets × 5 seeds = 1200 runs
~8s per SUDA run → ~2.7 hours total

Usage:
    uv run python -m src.experiments.r1_grid_search [--resume]
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.experiments.paper_v2_rerun import (
    ModelConfig,
    DATASETS,
    SCENARIOS,
    SEEDS_5,
    run_single,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/paper_v2_rerun/r1_grid_log.txt"),
    ],
)
logger = logging.getLogger(__name__)


# --- Grid Definition ---
INFLUENCE_WEIGHTS = [0.0, 0.3, 0.5, 0.7]
MINORITY_PROTECTIONS = [0.0, 0.05, 0.1, 0.2, 0.3]


def make_config_name(iw: float, mp: float) -> str:
    """e.g. iw00_mp000, iw03_mp010, iw07_mp030"""
    iw_str = f"iw{int(iw*10):02d}"
    mp_str = f"mp{int(mp*100):03d}"
    return f"{iw_str}_{mp_str}"


def build_configs() -> list[ModelConfig]:
    """Build all 20 grid configs (SUDA only, no ARF)."""
    configs = []
    for iw in INFLUENCE_WEIGHTS:
        for mp in MINORITY_PROTECTIONS:
            aw = round(1.0 - iw, 2)
            name = make_config_name(iw, mp)
            params = {
                "budget_enabled": True,
                "budget_max_samples": 3000,
                "budget_eviction_batch": 100,
                "budget_age_weight": aw,
                "budget_influence_weight": iw,
                "budget_class_weight": 0.0,
                "budget_minority_protection": mp,
                "budget_skip_forest_forget": False,
            }
            if iw > 0:
                params["budget_use_feature_distance"] = True
            configs.append(ModelConfig(name, "suda", params))
    return configs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path("results/paper_v2_rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "r1_grid_checkpoint.json"

    # Load checkpoint (key-value format for easy lookup)
    checkpoint = {}
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        logger.info(f"Resuming from checkpoint: {len(checkpoint)} completed runs")

    configs = build_configs()
    scenarios = list(SCENARIOS.keys())
    seeds = SEEDS_5
    total_runs = len(configs) * len(DATASETS) * len(scenarios) * len(seeds)

    logger.info("=" * 80)
    logger.info("R1-Grid: Influence Weight × Minority Protection Grid Search")
    logger.info(f"  {len(configs)} configs × {len(scenarios)} scenarios × {len(DATASETS)} datasets × {len(seeds)} seeds = {total_runs} runs")
    logger.info(f"  influence_weight: {INFLUENCE_WEIGHTS}")
    logger.info(f"  minority_protection: {MINORITY_PROTECTIONS}")
    logger.info("=" * 80)

    run_idx = 0
    for dataset in DATASETS:
        for scenario_name in scenarios:
            for config in configs:
                for seed in seeds:
                    run_idx += 1
                    key = f"{dataset}/{scenario_name}/{config.name}/seed={seed}"

                    if key in checkpoint:
                        logger.info(f"[{run_idx}/{total_runs}] SKIP (cached) {key}")
                        continue

                    logger.info(f"[{run_idx}/{total_runs}] {key}")
                    t0 = time.time()

                    try:
                        result = run_single(dataset, scenario_name, config, seed)
                        elapsed = round(time.time() - t0, 1)

                        checkpoint[key] = {
                            "gmean": result["gmean"],
                            "f1": result["f1"],
                            "attack_recall": result["attack_recall"],
                            "benign_recall": result["benign_recall"],
                            "time": elapsed,
                            "budget_evicted": result.get("budget_evicted", 0),
                        }

                        logger.info(
                            f"  -> G-mean={result['gmean']:.4f}, "
                            f"F1={result['f1']:.4f}, "
                            f"AtkR={result['attack_recall']:.4f}, "
                            f"BenR={result['benign_recall']:.4f}, "
                            f"Time={elapsed}s"
                        )

                        # Save checkpoint every 10 runs
                        if run_idx % 10 == 0:
                            checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
                    except Exception as e:
                        logger.error(f"  -> ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    # Final save
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    # Generate summary
    generate_summary(checkpoint, output_dir)
    logger.info("Done!")


def generate_summary(checkpoint: dict, output_dir: Path):
    """Generate grid search summary with heatmap tables."""
    from collections import defaultdict

    scenarios = list(SCENARIOS.keys())

    lines = ["# R1-Grid: Influence Weight × Minority Protection Results\n"]
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Total runs: {len(checkpoint)}\n")

    # --- Per-dataset heatmap (averaged over all scenarios) ---
    lines.append("## Overall G-mean (averaged over 4 scenarios × 5 seeds)\n")

    for ds in DATASETS:
        lines.append(f"### {ds}\n")
        lines.append("| iw \\ mp | " + " | ".join(f"{mp}" for mp in MINORITY_PROTECTIONS) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(MINORITY_PROTECTIONS)) + "|")

        for iw in INFLUENCE_WEIGHTS:
            row = [f"**{iw}**"]
            for mp in MINORITY_PROTECTIONS:
                config_name = make_config_name(iw, mp)
                gmeans = []
                for sc in scenarios:
                    for seed in SEEDS_5:
                        key = f"{ds}/{sc}/{config_name}/seed={seed}"
                        if key in checkpoint:
                            gmeans.append(checkpoint[key]["gmean"])
                if gmeans:
                    avg = np.mean(gmeans)
                    row.append(f"{avg:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- Per-dataset × scenario detail ---
    lines.append("\n## Per-Scenario Breakdown\n")

    for ds in DATASETS:
        for sc in scenarios:
            lines.append(f"### {ds}/{sc}\n")
            lines.append("| iw \\ mp | " + " | ".join(f"{mp}" for mp in MINORITY_PROTECTIONS) + " |")
            lines.append("|--------|" + "|".join(["--------"] * len(MINORITY_PROTECTIONS)) + "|")

            best_gmean, best_name = 0, ""
            for iw in INFLUENCE_WEIGHTS:
                row = [f"**{iw}**"]
                for mp in MINORITY_PROTECTIONS:
                    config_name = make_config_name(iw, mp)
                    gmeans = []
                    for seed in SEEDS_5:
                        key = f"{ds}/{sc}/{config_name}/seed={seed}"
                        if key in checkpoint:
                            gmeans.append(checkpoint[key]["gmean"])
                    if gmeans:
                        avg = np.mean(gmeans)
                        if avg > best_gmean:
                            best_gmean = avg
                            best_name = config_name
                        row.append(f"{avg:.4f}")
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")
            lines.append(f"\nBest: **{best_name}** (G-mean={best_gmean:.4f})\n")

    # --- Best config per dataset-scenario ---
    lines.append("\n## Best Config Summary\n")
    lines.append("| Dataset | Scenario | Best Config | G-mean | iw | mp |")
    lines.append("|---------|----------|-------------|--------|----|----|")

    for ds in DATASETS:
        for sc in scenarios:
            best_gmean, best_iw, best_mp = 0, 0, 0
            for iw in INFLUENCE_WEIGHTS:
                for mp in MINORITY_PROTECTIONS:
                    config_name = make_config_name(iw, mp)
                    gmeans = []
                    for seed in SEEDS_5:
                        key = f"{ds}/{sc}/{config_name}/seed={seed}"
                        if key in checkpoint:
                            gmeans.append(checkpoint[key]["gmean"])
                    if gmeans and np.mean(gmeans) > best_gmean:
                        best_gmean = np.mean(gmeans)
                        best_iw = iw
                        best_mp = mp
            lines.append(f"| {ds} | {sc} | {make_config_name(best_iw, best_mp)} | {best_gmean:.4f} | {best_iw} | {best_mp} |")

    summary_path = output_dir / "r1_grid_summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
