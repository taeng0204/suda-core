"""Quick Unlearning Contribution Test

Tests whether unlearning actually contributes to performance:
- V9 Normal: Unlearning enabled (threshold 0.02)
- V9 NoUnlearn: Unlearning disabled (threshold 1.0 = never triggers)

Usage:
    uv run python src/experiments/unlearning_contribution_test.py
"""

import numpy as np
from tqdm import tqdm
from src.data.nids import make_sudden_drift_stream
from src.models.suda_v9 import SUDAV9

def compute_gmean(y_true, y_pred):
    """Compute G-mean."""
    from sklearn.metrics import confusion_matrix
    if len(np.unique(y_true)) < 2:
        return 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2:
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(tpr * tnr)

def run_experiment(dataset_name: str, seed: int, enable_unlearning: bool):
    """Run single experiment."""
    rng = np.random.default_rng(seed)
    
    # Create drift stream: pre-drift has 99% benign (1% attack), post-drift has 90% benign (10% attack)
    X_stream, y_stream = make_sudden_drift_stream(
        dataset_name,
        rng,
        total_samples=20000,
        drift_point=10000,
        pre_benign_ratio=0.99,  # 1% attack before drift
        post_benign_ratio=0.90,  # 10% attack after drift
    )
    
    num_features = X_stream.shape[1]
    
    # V9 with/without unlearning
    # To disable unlearning: set very high thresholds
    model = SUDAV9(
        num_features=num_features,
        num_trees=50,
        max_depth=15,
        adaptive_k_enabled=True,
        k_min=3,
        k_max=50,
        gmean_drop_threshold=0.02 if enable_unlearning else 1.0,
        recall_drop_threshold=0.05 if enable_unlearning else 1.0,
        detector_delta=0.002 if enable_unlearning else 0.0001,  # Very sensitive = never triggers
        seed=seed,
    )
    
    # Process stream
    batch_size = 500
    n_samples = len(y_stream)
    
    all_y_true = []
    all_y_pred = []
    unlearn_count = 0
    unlearn_samples = 0
    
    # Initial fit with first 1000 samples (warmup)
    warmup_size = 1000
    model.fit(X_stream[:warmup_size], y_stream[:warmup_size])
    
    # Pre-drift metrics
    pre_drift_gmean = []
    # Post-drift metrics
    post_drift_gmean = []
    drift_point_batch = 10000 // batch_size
    
    for i in range(warmup_size, n_samples, batch_size):
        X_batch = X_stream[i:i+batch_size]
        y_batch = y_stream[i:i+batch_size]
        
        result = model.partial_fit(X_batch, y_batch)
        
        all_y_true.extend(y_batch)
        all_y_pred.extend(result.predictions)
        
        batch_idx = i // batch_size
        batch_gmean = compute_gmean(y_batch, result.predictions)
        
        if batch_idx < drift_point_batch:
            pre_drift_gmean.append(batch_gmean)
        else:
            post_drift_gmean.append(batch_gmean)
        
        if result.unlearning_triggered:
            unlearn_count += 1
            if result.unlearning_event:
                unlearn_samples += result.unlearning_event.get('num_forgotten', 0)
    
    # Compute final metrics
    final_gmean = compute_gmean(np.array(all_y_true), np.array(all_y_pred))
    avg_pre_drift = np.mean(pre_drift_gmean) if pre_drift_gmean else 0
    avg_post_drift = np.mean(post_drift_gmean) if post_drift_gmean else 0
    
    return {
        'final_gmean': final_gmean,
        'pre_drift_gmean': avg_pre_drift,
        'post_drift_gmean': avg_post_drift,
        'unlearn_count': unlearn_count,
        'unlearn_samples': unlearn_samples,
    }

def main():
    datasets = ['nslkdd', 'unswnb15']
    seeds = [42, 123, 456]
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        for seed in tqdm(seeds, desc=f"{dataset}"):
            # With unlearning
            res_unlearn = run_experiment(dataset, seed, enable_unlearning=True)
            res_unlearn['config'] = 'V9-Unlearn'
            res_unlearn['dataset'] = dataset
            res_unlearn['seed'] = seed
            
            # Without unlearning  
            res_no_unlearn = run_experiment(dataset, seed, enable_unlearning=False)
            res_no_unlearn['config'] = 'V9-NoUnlearn'
            res_no_unlearn['dataset'] = dataset
            res_no_unlearn['seed'] = seed
            
            results.append(res_unlearn)
            results.append(res_no_unlearn)
            
            print(f"\n  Seed {seed}:")
            print(f"    V9-Unlearn:   G-mean={res_unlearn['final_gmean']:.4f}, "
                  f"Post-drift={res_unlearn['post_drift_gmean']:.4f}, "
                  f"Triggers={res_unlearn['unlearn_count']}")
            print(f"    V9-NoUnlearn: G-mean={res_no_unlearn['final_gmean']:.4f}, "
                  f"Post-drift={res_no_unlearn['post_drift_gmean']:.4f}, "
                  f"Triggers={res_no_unlearn['unlearn_count']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Unlearning Contribution")
    print("="*60)
    
    for dataset in datasets:
        unlearn_results = [r for r in results if r['dataset'] == dataset and r['config'] == 'V9-Unlearn']
        no_unlearn_results = [r for r in results if r['dataset'] == dataset and r['config'] == 'V9-NoUnlearn']
        
        avg_unlearn = np.mean([r['final_gmean'] for r in unlearn_results])
        avg_no_unlearn = np.mean([r['final_gmean'] for r in no_unlearn_results])
        avg_post_unlearn = np.mean([r['post_drift_gmean'] for r in unlearn_results])
        avg_post_no_unlearn = np.mean([r['post_drift_gmean'] for r in no_unlearn_results])
        
        diff = avg_unlearn - avg_no_unlearn
        
        print(f"\n{dataset}:")
        print(f"  V9-Unlearn:   Final G-mean={avg_unlearn:.4f}, Post-drift={avg_post_unlearn:.4f}")
        print(f"  V9-NoUnlearn: Final G-mean={avg_no_unlearn:.4f}, Post-drift={avg_post_no_unlearn:.4f}")
        print(f"  Difference:   {diff:+.4f} ({'Unlearning helps!' if diff > 0.01 else 'No significant difference' if abs(diff) < 0.01 else 'Unlearning hurts?'})")
    
    # Save results
    import json
    from pathlib import Path
    from datetime import datetime
    
    output_dir = Path("note/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "260128_unlearning_contribution_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
