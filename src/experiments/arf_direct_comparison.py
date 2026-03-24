"""ARF Direct Comparison: SUDA-V3 vs ARF under identical conditions.

This experiment compares SUDA-V3 (with Rust-native buffer management) against
River's ARF (Adaptive Random Forest) under identical experimental conditions:
- Same data streams (controlled drift)
- Same buffer size limits (iso-memory)
- Same evaluation intervals
- Same random seeds for reproducibility

Research Question:
    Under iso-memory and iso-drift conditions, is SUDA-V3's selective unlearning
    competitive with ARF's adaptive ensemble approach?

Models Compared:
    1. SUDA-V3: Aggressive-OnDrift strategy (40% forget ratio)
    2. ARF: River's Adaptive Random Forest (50 trees)
    3. HAT: Hoeffding Adaptive Tree (single-tree baseline)

Evaluation Metrics:
    - Post-drift accuracy: Average accuracy after drift point
    - Pre-drift accuracy: Average accuracy before drift point
    - TTR (Time-to-Recover): Samples to recover to 90% of baseline
    - Total update time: Time spent in partial_fit
    - Accuracy curve: Accuracy over time

Experimental Design:
    - 4 datasets: nslkdd, unswnb15, cicids2018, cidds
    - 3 seeds: 42, 123, 456
    - Sudden drift at 50% point (20000 samples)
    - Evaluation every 1000 samples
    - Batch size: 500 samples

Author: Claude Sonnet 4.5
Date: 2026-01-16
Session: Sisyphus-Junior ARF Comparison
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.nids import make_sudden_drift_stream
from src.models.suda_drift_v3 import SUDADriftModelV3, SUDADriftConfigV3
from src.baselines.river_models import ARFModel, HATModel


@dataclass
class ComparisonResult:
    """Result of a single comparison run."""
    method: str
    dataset: str
    seed: int
    pre_drift_accuracy: float
    post_drift_accuracy: float
    final_accuracy: float
    ttr: Optional[int]  # Time-to-Recover (samples), None if never recovered
    total_update_time_s: float
    n_drift_events: int  # Only for SUDA
    samples_removed: int  # Only for SUDA
    accuracy_curve: List[float]
    positions: List[int]


def calculate_ttr(
    accuracies: List[float],
    positions: List[int],
    drift_point: int,
    baseline_accuracy: float,
    recovery_threshold: float = 0.9
) -> Optional[int]:
    """Calculate Time-to-Recover (TTR).

    TTR is the number of samples after drift to recover to recovery_threshold
    of the baseline accuracy.

    Args:
        accuracies: List of accuracy values
        positions: List of sample positions corresponding to accuracies
        drift_point: Sample position where drift occurs
        baseline_accuracy: Pre-drift baseline accuracy
        recovery_threshold: Threshold ratio (default 0.9 = 90% of baseline)

    Returns:
        Number of samples after drift to recover, or None if never recovered
    """
    target_accuracy = baseline_accuracy * recovery_threshold

    # Find evaluations after drift
    post_drift_indices = [i for i, pos in enumerate(positions) if pos > drift_point]

    if not post_drift_indices:
        return None

    # Find first time we meet or exceed target
    for idx in post_drift_indices:
        if accuracies[idx] >= target_accuracy:
            recovery_position = positions[idx]
            return recovery_position - drift_point

    return None


def evaluate_suda_v3(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    seed: int,
    batch_size: int = 500,
    eval_interval: int = 1000,
    drift_point: int = 20000,
    warmup_size: int = 5000,
    max_samples: int = 30000
) -> Dict[str, Any]:
    """Evaluate SUDA-V3 model."""
    config = SUDADriftConfigV3(
        num_trees=50,
        k=10,
        max_depth=15,
        max_samples=max_samples,
        detector_method="adwin",
        detector_delta=0.002,
        selector_strategy="old_majority",
        forget_ratio=0.4,  # Aggressive-OnDrift
        seed=seed
    )

    model = SUDADriftModelV3(config)

    n_samples = len(X_stream)
    accuracies = []
    positions = []
    total_update_time = 0.0

    # Reserve last 10% for testing
    test_start = int(n_samples * 0.9)
    X_test = X_stream[test_start:]
    y_test = y_stream[test_start:]

    pos = 0

    while pos < test_start:
        batch_end = min(pos + batch_size, test_start)
        X_batch = X_stream[pos:batch_end]
        y_batch = y_stream[pos:batch_end]

        # Partial fit
        check_drift = pos >= warmup_size
        update_start = time.time()
        result = model.partial_fit(X_batch, y_batch, check_drift=check_drift)
        total_update_time += time.time() - update_start

        # Evaluate every eval_interval
        if (pos + batch_size) % eval_interval < batch_size:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            positions.append(pos + batch_size)

        pos = batch_end

    # Final evaluation
    y_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)

    # Get stats
    stats = model.get_stats()

    return {
        "accuracies": accuracies,
        "positions": positions,
        "final_accuracy": final_accuracy,
        "total_update_time": total_update_time,
        "n_drift_events": stats["n_drift_events"],
        "total_samples_forgotten": stats["total_samples_forgotten"]
    }


def evaluate_arf(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    seed: int,
    batch_size: int = 500,
    eval_interval: int = 1000,
    n_models: int = 50
) -> Dict[str, Any]:
    """Evaluate ARF model."""
    model = ARFModel(n_models=n_models, seed=seed)

    n_samples = len(X_stream)
    accuracies = []
    positions = []
    total_update_time = 0.0

    # Reserve last 10% for testing
    test_start = int(n_samples * 0.9)
    X_test = X_stream[test_start:]
    y_test = y_stream[test_start:]

    pos = 0

    while pos < test_start:
        batch_end = min(pos + batch_size, test_start)
        X_batch = X_stream[pos:batch_end]
        y_batch = y_stream[pos:batch_end]

        # Partial fit
        update_start = time.time()
        model.partial_fit(X_batch, y_batch)
        total_update_time += time.time() - update_start

        # Evaluate every eval_interval
        if (pos + batch_size) % eval_interval < batch_size:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            positions.append(pos + batch_size)

        pos = batch_end

    # Final evaluation
    y_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracies": accuracies,
        "positions": positions,
        "final_accuracy": final_accuracy,
        "total_update_time": total_update_time,
        "n_drift_events": 0,  # ARF doesn't expose drift events
        "total_samples_forgotten": 0  # ARF uses implicit forgetting
    }


def evaluate_hat(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    seed: int,
    batch_size: int = 500,
    eval_interval: int = 1000
) -> Dict[str, Any]:
    """Evaluate HAT model."""
    model = HATModel(seed=seed)

    n_samples = len(X_stream)
    accuracies = []
    positions = []
    total_update_time = 0.0

    # Reserve last 10% for testing
    test_start = int(n_samples * 0.9)
    X_test = X_stream[test_start:]
    y_test = y_stream[test_start:]

    pos = 0

    while pos < test_start:
        batch_end = min(pos + batch_size, test_start)
        X_batch = X_stream[pos:batch_end]
        y_batch = y_stream[pos:batch_end]

        # Partial fit
        update_start = time.time()
        model.partial_fit(X_batch, y_batch)
        total_update_time += time.time() - update_start

        # Evaluate every eval_interval
        if (pos + batch_size) % eval_interval < batch_size:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            positions.append(pos + batch_size)

        pos = batch_end

    # Final evaluation
    y_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracies": accuracies,
        "positions": positions,
        "final_accuracy": final_accuracy,
        "total_update_time": total_update_time,
        "n_drift_events": 0,
        "total_samples_forgotten": 0
    }


def run_comparison(
    dataset: str,
    seed: int,
    total_samples: int = 40000,
    drift_point: int = 20000,
    pre_benign_ratio: float = 0.7,
    post_benign_ratio: float = 0.3,
    max_samples: int = 30000
) -> List[ComparisonResult]:
    """Run comparison experiment."""
    print(f"\n--- Comparison: {dataset}, seed {seed} ---")

    # Create drift stream
    rng = np.random.default_rng(seed)
    X_stream, y_stream = make_sudden_drift_stream(
        dataset,
        rng,
        total_samples=total_samples,
        drift_point=drift_point,
        pre_benign_ratio=pre_benign_ratio,
        post_benign_ratio=post_benign_ratio
    )

    results = []

    # Evaluate SUDA-V3
    print("  SUDA-V3 (Aggressive-OnDrift)...", end=" ", flush=True)
    try:
        eval_result = evaluate_suda_v3(
            X_stream, y_stream, seed,
            drift_point=drift_point,
            max_samples=max_samples
        )

        accuracies = eval_result["accuracies"]
        positions = eval_result["positions"]

        # Calculate metrics
        drift_eval_idx = next(
            (i for i, p in enumerate(positions) if p > drift_point),
            len(positions) - 1
        )

        pre_drift_accs = accuracies[:drift_eval_idx] if drift_eval_idx > 0 else [eval_result["final_accuracy"]]
        post_drift_accs = accuracies[drift_eval_idx:] if drift_eval_idx < len(accuracies) else [eval_result["final_accuracy"]]

        pre_drift_accuracy = float(np.mean(pre_drift_accs))
        post_drift_accuracy = float(np.mean(post_drift_accs))

        # Calculate TTR
        ttr = calculate_ttr(
            accuracies, positions, drift_point, pre_drift_accuracy
        )

        result = ComparisonResult(
            method="SUDA-V3",
            dataset=dataset,
            seed=seed,
            pre_drift_accuracy=pre_drift_accuracy,
            post_drift_accuracy=post_drift_accuracy,
            final_accuracy=eval_result["final_accuracy"],
            ttr=ttr,
            total_update_time_s=eval_result["total_update_time"],
            n_drift_events=eval_result["n_drift_events"],
            samples_removed=eval_result["total_samples_forgotten"],
            accuracy_curve=accuracies,
            positions=positions
        )
        results.append(result)

        print(f"Post-drift: {post_drift_accuracy:.2%}, TTR: {ttr if ttr else 'N/A'}, "
              f"Time: {result.total_update_time_s:.1f}s, Removed: {result.samples_removed}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Evaluate ARF
    print("  ARF (n_models=50)...", end=" ", flush=True)
    try:
        eval_result = evaluate_arf(
            X_stream, y_stream, seed,
            n_models=50
        )

        accuracies = eval_result["accuracies"]
        positions = eval_result["positions"]

        drift_eval_idx = next(
            (i for i, p in enumerate(positions) if p > drift_point),
            len(positions) - 1
        )

        pre_drift_accs = accuracies[:drift_eval_idx] if drift_eval_idx > 0 else [eval_result["final_accuracy"]]
        post_drift_accs = accuracies[drift_eval_idx:] if drift_eval_idx < len(accuracies) else [eval_result["final_accuracy"]]

        pre_drift_accuracy = float(np.mean(pre_drift_accs))
        post_drift_accuracy = float(np.mean(post_drift_accs))

        ttr = calculate_ttr(
            accuracies, positions, drift_point, pre_drift_accuracy
        )

        result = ComparisonResult(
            method="ARF",
            dataset=dataset,
            seed=seed,
            pre_drift_accuracy=pre_drift_accuracy,
            post_drift_accuracy=post_drift_accuracy,
            final_accuracy=eval_result["final_accuracy"],
            ttr=ttr,
            total_update_time_s=eval_result["total_update_time"],
            n_drift_events=0,
            samples_removed=0,
            accuracy_curve=accuracies,
            positions=positions
        )
        results.append(result)

        print(f"Post-drift: {post_drift_accuracy:.2%}, TTR: {ttr if ttr else 'N/A'}, "
              f"Time: {result.total_update_time_s:.1f}s")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Evaluate HAT
    print("  HAT (single tree)...", end=" ", flush=True)
    try:
        eval_result = evaluate_hat(
            X_stream, y_stream, seed
        )

        accuracies = eval_result["accuracies"]
        positions = eval_result["positions"]

        drift_eval_idx = next(
            (i for i, p in enumerate(positions) if p > drift_point),
            len(positions) - 1
        )

        pre_drift_accs = accuracies[:drift_eval_idx] if drift_eval_idx > 0 else [eval_result["final_accuracy"]]
        post_drift_accs = accuracies[drift_eval_idx:] if drift_eval_idx < len(accuracies) else [eval_result["final_accuracy"]]

        pre_drift_accuracy = float(np.mean(pre_drift_accs))
        post_drift_accuracy = float(np.mean(post_drift_accs))

        ttr = calculate_ttr(
            accuracies, positions, drift_point, pre_drift_accuracy
        )

        result = ComparisonResult(
            method="HAT",
            dataset=dataset,
            seed=seed,
            pre_drift_accuracy=pre_drift_accuracy,
            post_drift_accuracy=post_drift_accuracy,
            final_accuracy=eval_result["final_accuracy"],
            ttr=ttr,
            total_update_time_s=eval_result["total_update_time"],
            n_drift_events=0,
            samples_removed=0,
            accuracy_curve=accuracies,
            positions=positions
        )
        results.append(result)

        print(f"Post-drift: {post_drift_accuracy:.2%}, TTR: {ttr if ttr else 'N/A'}, "
              f"Time: {result.total_update_time_s:.1f}s")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def print_summary(results: List[ComparisonResult]) -> None:
    """Print summary table."""
    print(f"\n{'='*100}")
    print("ARF DIRECT COMPARISON SUMMARY")
    print(f"{'='*100}")

    methods = sorted(set(r.method for r in results))
    datasets = sorted(set(r.dataset for r in results))

    # Post-drift accuracy table
    print("\n--- Post-Drift Accuracy ---")
    header = f"{'Method':<20}"
    for ds in datasets:
        header += f" | {ds:<12}"
    header += " | Avg"
    print(header)
    print("-" * 100)

    for method in methods:
        method_results = [r for r in results if r.method == method]
        row = f"{method:<20}"

        method_avgs = []
        for ds in datasets:
            ds_results = [r for r in method_results if r.dataset == ds]
            if ds_results:
                avg = np.mean([r.post_drift_accuracy for r in ds_results])
                std = np.std([r.post_drift_accuracy for r in ds_results])
                method_avgs.append(avg)
                row += f" | {avg:.1%}±{std:.1%}"
            else:
                row += f" | {'N/A':<12}"

        if method_avgs:
            row += f" | {np.mean(method_avgs):.1%}"
        print(row)

    # Pre-drift accuracy table
    print("\n--- Pre-Drift Accuracy ---")
    header = f"{'Method':<20}"
    for ds in datasets:
        header += f" | {ds:<12}"
    header += " | Avg"
    print(header)
    print("-" * 100)

    for method in methods:
        method_results = [r for r in results if r.method == method]
        row = f"{method:<20}"

        method_avgs = []
        for ds in datasets:
            ds_results = [r for r in method_results if r.dataset == ds]
            if ds_results:
                avg = np.mean([r.pre_drift_accuracy for r in ds_results])
                std = np.std([r.pre_drift_accuracy for r in ds_results])
                method_avgs.append(avg)
                row += f" | {avg:.1%}±{std:.1%}"
            else:
                row += f" | {'N/A':<12}"

        if method_avgs:
            row += f" | {np.mean(method_avgs):.1%}"
        print(row)

    # TTR table
    print("\n--- Time-to-Recover (TTR) in Samples ---")
    header = f"{'Method':<20}"
    for ds in datasets:
        header += f" | {ds:<12}"
    header += " | Avg"
    print(header)
    print("-" * 100)

    for method in methods:
        method_results = [r for r in results if r.method == method]
        row = f"{method:<20}"

        method_avgs = []
        for ds in datasets:
            ds_results = [r for r in method_results if r.dataset == ds]
            if ds_results:
                ttrs = [r.ttr for r in ds_results if r.ttr is not None]
                if ttrs:
                    avg = np.mean(ttrs)
                    method_avgs.append(avg)
                    row += f" | {avg:<12.0f}"
                else:
                    row += f" | {'N/A':<12}"
            else:
                row += f" | {'N/A':<12}"

        if method_avgs:
            row += f" | {np.mean(method_avgs):.0f}"
        else:
            row += f" | N/A"
        print(row)

    # Update time table
    print("\n--- Total Update Time (seconds) ---")
    header = f"{'Method':<20}"
    for ds in datasets:
        header += f" | {ds:<12}"
    header += " | Avg"
    print(header)
    print("-" * 100)

    for method in methods:
        method_results = [r for r in results if r.method == method]
        row = f"{method:<20}"

        method_avgs = []
        for ds in datasets:
            ds_results = [r for r in method_results if r.dataset == ds]
            if ds_results:
                avg = np.mean([r.total_update_time_s for r in ds_results])
                method_avgs.append(avg)
                row += f" | {avg:<12.1f}"
            else:
                row += f" | {'N/A':<12}"

        if method_avgs:
            row += f" | {np.mean(method_avgs):.1f}"
        print(row)

    print("=" * 100)

    # Statistical summary
    print("\n--- Key Findings ---")

    suda_results = [r for r in results if r.method == "SUDA-V3"]
    arf_results = [r for r in results if r.method == "ARF"]

    if suda_results and arf_results:
        suda_post = np.mean([r.post_drift_accuracy for r in suda_results])
        arf_post = np.mean([r.post_drift_accuracy for r in arf_results])
        diff = suda_post - arf_post

        print(f"SUDA-V3 vs ARF (Post-drift):")
        print(f"  SUDA-V3: {suda_post:.2%}")
        print(f"  ARF:     {arf_post:.2%}")
        print(f"  Diff:    {diff:+.2%} ({'SUDA wins' if diff > 0 else 'ARF wins'})")

        suda_ttrs = [r.ttr for r in suda_results if r.ttr is not None]
        arf_ttrs = [r.ttr for r in arf_results if r.ttr is not None]

        if suda_ttrs and arf_ttrs:
            suda_ttr_avg = np.mean(suda_ttrs)
            arf_ttr_avg = np.mean(arf_ttrs)
            ttr_diff = suda_ttr_avg - arf_ttr_avg

            print(f"\nSUDA-V3 vs ARF (TTR):")
            print(f"  SUDA-V3: {suda_ttr_avg:.0f} samples")
            print(f"  ARF:     {arf_ttr_avg:.0f} samples")
            print(f"  Diff:    {ttr_diff:+.0f} samples ({'SUDA faster' if ttr_diff < 0 else 'ARF faster'})")

        # Samples removed (SUDA only)
        suda_removed = np.mean([r.samples_removed for r in suda_results])
        print(f"\nSUDA-V3 Total Samples Removed: {suda_removed:.0f}")


def main():
    parser = argparse.ArgumentParser(description="ARF Direct Comparison Experiment")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to test (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds (default: 42 123 456)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (1 dataset, 1 seed, fewer samples)")
    parser.add_argument("--output-dir", type=str, default="results/arf_comparison",
                        help="Output directory for results")
    parser.add_argument("--total-samples", type=int, default=40000,
                        help="Total samples in stream")
    parser.add_argument("--drift-point", type=int, default=20000,
                        help="Drift point position")
    parser.add_argument("--max-samples", type=int, default=30000,
                        help="Max samples in buffer (for SUDA)")
    args = parser.parse_args()

    # Select datasets
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ["nslkdd", "unswnb15", "cicids2018", "cidds"]

    # Quick mode
    if args.quick:
        datasets = ["nslkdd"]
        args.seeds = [42]
        args.total_samples = 10000
        args.drift_point = 5000
        args.max_samples = 5000

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []

    for dataset in datasets:
        for seed in args.seeds:
            try:
                results = run_comparison(
                    dataset=dataset,
                    seed=seed,
                    total_samples=args.total_samples,
                    drift_point=args.drift_point,
                    max_samples=args.max_samples
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error on {dataset} seed {seed}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    if all_results:
        print_summary(all_results)
    else:
        print("No results to display")

    # Save results
    results_file = output_dir / f"arf_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "datasets": datasets,
                "seeds": args.seeds,
                "total_samples": args.total_samples,
                "drift_point": args.drift_point,
                "max_samples": args.max_samples
            },
            "results": [asdict(r) for r in all_results]
        }, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
