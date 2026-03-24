"""Buffer Size Sensitivity Analysis for SUDA-V3.

Research Questions:
1. Does larger buffer always help post-drift accuracy?
2. Is there a sweet spot for buffer size?
3. How does SUDA scale compared to ARF with more memory?
4. What's the accuracy-per-KB efficiency trade-off?

Methodology:
- Test buffer sizes: 10K, 20K, 30K, 50K
- Use best strategy: Aggressive-OnDrift (old_majority, 40% forget_ratio)
- Run on all 4 datasets (nslkdd, unswnb15, cicids2018, cidds)
- Use 3 seeds (42, 123, 456)
- Include ARF baseline for comparison
- Measure: Post-drift acc, memory efficiency, training time

Author: Claude Opus 4.5
Date: 2026-01-16
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.nids import make_sudden_drift_stream
from src.models.suda_drift_v3 import SUDADriftModelV3, SUDADriftConfigV3
from src.baselines.river_models import ARFModel


@dataclass
class BufferSizeResult:
    """Result of buffer size experiment."""
    method: str
    buffer_size: int
    dataset: str
    seed: int

    # Accuracy metrics
    post_drift_accuracy: float
    pre_drift_accuracy: float
    final_accuracy: float

    # Efficiency metrics
    memory_kb: float
    accuracy_per_kb: float  # Post-drift accuracy / memory KB

    # Performance metrics
    total_time_s: float
    samples_per_second: float

    # Drift metrics
    n_drift_events: int
    samples_removed: int

    # Detailed accuracy curve
    accuracy_curve: List[float]
    positions: List[int]


class ARFWrapper:
    """Wrapper for ARF to match SUDA interface."""

    def __init__(self, max_samples: int, seed: int = 42):
        self.model = ARFModel(n_models=50, seed=seed)
        self.max_samples = max_samples
        self.is_fitted = False
        self._n_drift_events = 0

    def partial_fit(self, X: np.ndarray, y: np.ndarray, check_drift: bool = True) -> Dict[str, Any]:
        """Train on batch."""
        result = {
            "drift_detected": False,
            "samples_removed": 0,
            "n_samples": 0
        }

        self.model.partial_fit(X, y)
        self.is_fitted = True

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            return np.zeros(len(X), dtype=np.int64)

        return self.model.predict(X)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "n_drift_events": self._n_drift_events,
            "total_removed": 0,
            "buffer_size": self.max_samples
        }


def estimate_memory_kb(method: str, buffer_size: int, n_features: int) -> float:
    """Estimate memory usage in KB.

    SUDA:
    - Rust buffer: 24 bytes/sample (id, position, label)
    - Python features: n_features * 4 bytes/sample
    - Forest: 50 trees with ~buffer_size/10 nodes each, ~100 bytes/node

    ARF:
    - River trees: ~50 trees, similar node structure
    - Internal buffer: variable, assume similar to SUDA
    """
    if "ARF" in method:
        # ARF memory (conservative estimate)
        base_memory = 1024  # River overhead
        trees_memory = 50 * (buffer_size / 10) * 100  # 50 trees
        buffer_memory = buffer_size * (n_features * 4 + 24)
        return (base_memory + trees_memory + buffer_memory) / 1024
    else:
        # SUDA memory
        rust_buffer = buffer_size * 24
        python_features = buffer_size * n_features * 4
        forest = 50 * (buffer_size / 10) * 100
        return (rust_buffer + python_features + forest) / 1024


def evaluate_model_on_stream(
    model,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    batch_size: int = 500,
    eval_interval: int = 1000,
    drift_point: int = 20000,
    warmup_size: int = 5000
) -> Dict[str, Any]:
    """Evaluate model on stream with detailed metrics."""
    n_samples = len(X_stream)
    accuracies = []
    positions = []
    start_time = time.time()

    # Use last 10% as test set
    test_start = int(n_samples * 0.9)
    X_test = X_stream[test_start:]
    y_test = y_stream[test_start:]

    pos = 0
    total_removed = 0

    while pos < test_start:
        batch_end = min(pos + batch_size, test_start)
        X_batch = X_stream[pos:batch_end]
        y_batch = y_stream[pos:batch_end]

        check_drift = pos >= warmup_size
        result = model.partial_fit(X_batch, y_batch, check_drift=check_drift)
        total_removed += result.get("samples_removed", 0)

        # Periodic evaluation
        if (pos + batch_size) % eval_interval < batch_size:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            positions.append(pos + batch_size)

        pos = batch_end

    total_time = time.time() - start_time

    # Final evaluation
    y_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)

    # Calculate pre/post drift metrics
    drift_eval_idx = next((i for i, p in enumerate(positions) if p > drift_point), len(positions) - 1)

    pre_drift_accs = accuracies[:drift_eval_idx] if drift_eval_idx > 0 else [final_accuracy]
    post_drift_accs = accuracies[drift_eval_idx:] if drift_eval_idx < len(accuracies) else [final_accuracy]

    pre_drift_accuracy = float(np.mean(pre_drift_accs))
    post_drift_accuracy = float(np.mean(post_drift_accs))

    stats = model.get_stats()

    return {
        "final_accuracy": final_accuracy,
        "post_drift_accuracy": post_drift_accuracy,
        "pre_drift_accuracy": pre_drift_accuracy,
        "total_time_s": total_time,
        "samples_per_second": test_start / total_time,
        "n_drift_events": stats.get("n_drift_events", 0),
        "samples_removed": stats.get("total_removed", total_removed),
        "accuracy_curve": accuracies,
        "positions": positions
    }


def run_buffer_size_experiment(
    dataset: str,
    buffer_size: int,
    seed: int,
    total_samples: int = 40000,
    drift_point: int = 20000
) -> List[BufferSizeResult]:
    """Run experiment with specific buffer size."""
    print(f"\n--- Buffer Size {buffer_size}: {dataset}, seed {seed} ---")

    # Generate stream
    rng = np.random.default_rng(seed)
    X_stream, y_stream = make_sudden_drift_stream(
        dataset, rng, total_samples=total_samples, drift_point=drift_point
    )

    n_features = X_stream.shape[1]
    results = []

    # 1. SUDA-V3 with Aggressive-OnDrift strategy
    print(f"  SUDA-V3 (buffer={buffer_size})...", end=" ", flush=True)
    try:
        model = SUDADriftModelV3(SUDADriftConfigV3(
            num_trees=50,
            k=10,
            max_depth=15,
            max_samples=buffer_size,
            detector_method="adwin",
            detector_delta=0.002,
            selector_strategy="old_majority",
            forget_ratio=0.4,  # Aggressive
            seed=seed
        ))

        eval_result = evaluate_model_on_stream(
            model, X_stream, y_stream, drift_point=drift_point
        )

        memory_kb = estimate_memory_kb("SUDA-V3", buffer_size, n_features)
        accuracy_per_kb = eval_result["post_drift_accuracy"] / memory_kb

        result = BufferSizeResult(
            method="SUDA-V3-Aggressive",
            buffer_size=buffer_size,
            dataset=dataset,
            seed=seed,
            post_drift_accuracy=eval_result["post_drift_accuracy"],
            pre_drift_accuracy=eval_result["pre_drift_accuracy"],
            final_accuracy=eval_result["final_accuracy"],
            memory_kb=memory_kb,
            accuracy_per_kb=accuracy_per_kb,
            total_time_s=eval_result["total_time_s"],
            samples_per_second=eval_result["samples_per_second"],
            n_drift_events=eval_result["n_drift_events"],
            samples_removed=eval_result["samples_removed"],
            accuracy_curve=eval_result["accuracy_curve"],
            positions=eval_result["positions"]
        )
        results.append(result)

        print(f"Post-drift: {result.post_drift_accuracy:.2%}, "
              f"Mem: {memory_kb:.1f}KB, "
              f"Eff: {accuracy_per_kb*1000:.2f}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 2. ARF baseline (for comparison)
    print(f"  ARF (buffer={buffer_size})...", end=" ", flush=True)
    try:
        model = ARFWrapper(max_samples=buffer_size, seed=seed)

        eval_result = evaluate_model_on_stream(
            model, X_stream, y_stream, drift_point=drift_point
        )

        memory_kb = estimate_memory_kb("ARF", buffer_size, n_features)
        accuracy_per_kb = eval_result["post_drift_accuracy"] / memory_kb

        result = BufferSizeResult(
            method="ARF",
            buffer_size=buffer_size,
            dataset=dataset,
            seed=seed,
            post_drift_accuracy=eval_result["post_drift_accuracy"],
            pre_drift_accuracy=eval_result["pre_drift_accuracy"],
            final_accuracy=eval_result["final_accuracy"],
            memory_kb=memory_kb,
            accuracy_per_kb=accuracy_per_kb,
            total_time_s=eval_result["total_time_s"],
            samples_per_second=eval_result["samples_per_second"],
            n_drift_events=eval_result["n_drift_events"],
            samples_removed=eval_result["samples_removed"],
            accuracy_curve=eval_result["accuracy_curve"],
            positions=eval_result["positions"]
        )
        results.append(result)

        print(f"Post-drift: {result.post_drift_accuracy:.2%}, "
              f"Mem: {memory_kb:.1f}KB, "
              f"Eff: {accuracy_per_kb*1000:.2f}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def print_summary(results: List[BufferSizeResult]) -> None:
    """Print comprehensive summary."""
    print(f"\n{'='*100}")
    print("BUFFER SIZE SENSITIVITY ANALYSIS - SUMMARY")
    print(f"{'='*100}")

    methods = sorted(set(r.method for r in results))
    datasets = sorted(set(r.dataset for r in results))
    buffer_sizes = sorted(set(r.buffer_size for r in results))

    # 1. Post-drift accuracy by buffer size
    print("\n1. POST-DRIFT ACCURACY BY BUFFER SIZE")
    print(f"{'Method':<25} | ", end="")
    for bs in buffer_sizes:
        print(f"{bs:>8} | ", end="")
    print("Avg")
    print("-" * 100)

    for method in methods:
        for ds in datasets:
            row = f"{method[:20]:<20} ({ds[:4]})"
            ds_results = [r for r in results if r.method == method and r.dataset == ds]

            accs = []
            for bs in buffer_sizes:
                bs_results = [r for r in ds_results if r.buffer_size == bs]
                if bs_results:
                    avg = np.mean([r.post_drift_accuracy for r in bs_results])
                    accs.append(avg)
                    row += f" | {avg:>6.1%}"
                else:
                    row += f" | {'N/A':>6}"

            if accs:
                row += f" | {np.mean(accs):.1%}"
            print(row)

    # 2. Memory efficiency (accuracy per KB)
    print("\n2. MEMORY EFFICIENCY (Accuracy per KB × 1000)")
    print(f"{'Method':<25} | ", end="")
    for bs in buffer_sizes:
        print(f"{bs:>8} | ", end="")
    print("Avg")
    print("-" * 100)

    for method in methods:
        for ds in datasets:
            row = f"{method[:20]:<20} ({ds[:4]})"
            ds_results = [r for r in results if r.method == method and r.dataset == ds]

            effs = []
            for bs in buffer_sizes:
                bs_results = [r for r in ds_results if r.buffer_size == bs]
                if bs_results:
                    avg_eff = np.mean([r.accuracy_per_kb * 1000 for r in bs_results])
                    effs.append(avg_eff)
                    row += f" | {avg_eff:>6.2f}"
                else:
                    row += f" | {'N/A':>6}"

            if effs:
                row += f" | {np.mean(effs):.2f}"
            print(row)

    # 3. Training time analysis
    print("\n3. TRAINING TIME (seconds)")
    print(f"{'Method':<25} | ", end="")
    for bs in buffer_sizes:
        print(f"{bs:>8} | ", end="")
    print("Avg")
    print("-" * 100)

    for method in methods:
        row = f"{method:<25}"
        times = []

        for bs in buffer_sizes:
            bs_results = [r for r in results if r.method == method and r.buffer_size == bs]
            if bs_results:
                avg_time = np.mean([r.total_time_s for r in bs_results])
                times.append(avg_time)
                row += f" | {avg_time:>6.1f}s"
            else:
                row += f" | {'N/A':>7}"

        if times:
            row += f" | {np.mean(times):.1f}s"
        print(row)

    # 4. Key insights
    print("\n4. KEY INSIGHTS")
    print("-" * 100)

    # Find best buffer size for each method
    for method in methods:
        method_results = [r for r in results if r.method == method]

        if not method_results:
            continue

        # Group by buffer size
        bs_to_acc = {}
        for bs in buffer_sizes:
            bs_results = [r for r in method_results if r.buffer_size == bs]
            if bs_results:
                bs_to_acc[bs] = np.mean([r.post_drift_accuracy for r in bs_results])

        if bs_to_acc:
            best_bs = max(bs_to_acc, key=bs_to_acc.get)
            best_acc = bs_to_acc[best_bs]

            # Check if there's a plateau (within 0.5% of best)
            plateau_sizes = [bs for bs, acc in bs_to_acc.items() if acc >= best_acc - 0.005]

            print(f"\n{method}:")
            print(f"  Best buffer: {best_bs:,} samples ({best_acc:.2%} post-drift)")

            if len(plateau_sizes) > 1:
                smallest_good = min(plateau_sizes)
                print(f"  Sweet spot: {smallest_good:,} samples (within 0.5% of best, saves memory)")

            # Compare 10K vs 50K
            if 10000 in bs_to_acc and 50000 in bs_to_acc:
                gain = bs_to_acc[50000] - bs_to_acc[10000]
                print(f"  10K→50K gain: {gain:+.1%} accuracy (5× memory)")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Buffer Size Sensitivity Analysis")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset (or all if not specified)")
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[10000, 20000, 30000, 50000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--output-dir", type=str, default="results/buffer_sensitivity")
    args = parser.parse_args()

    # Configure datasets
    datasets = [args.dataset] if args.dataset else ["nslkdd", "unswnb15", "cicids2018", "cidds"]

    # Quick mode
    if args.quick:
        total_samples = 10000
        drift_point = 5000
        args.seeds = [42]
        datasets = ["nslkdd"]
        args.buffer_sizes = [5000, 10000]
        print("\n[Quick test mode: nslkdd, 10K samples, 2 buffer sizes, 1 seed]")
    else:
        total_samples = 40000
        drift_point = 20000

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    total_experiments = len(datasets) * len(args.buffer_sizes) * len(args.seeds)
    current = 0

    print(f"\nRunning {total_experiments} experiments...")
    print(f"Datasets: {datasets}")
    print(f"Buffer sizes: {args.buffer_sizes}")
    print(f"Seeds: {args.seeds}")

    for dataset in datasets:
        for buffer_size in args.buffer_sizes:
            for seed in args.seeds:
                current += 1
                print(f"\n[Experiment {current}/{total_experiments}]")

                try:
                    results = run_buffer_size_experiment(
                        dataset, buffer_size, seed, total_samples, drift_point
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"ERROR: {dataset} / {buffer_size} / {seed}: {e}")
                    import traceback
                    traceback.print_exc()

    # Print summary
    if all_results:
        print_summary(all_results)
    else:
        print("\nNo results collected!")
        return

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"buffer_sensitivity_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "datasets": datasets,
            "buffer_sizes": args.buffer_sizes,
            "seeds": args.seeds,
            "total_samples": total_samples,
            "drift_point": drift_point,
            "results": [asdict(r) for r in all_results]
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")
    print(f"✓ Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
