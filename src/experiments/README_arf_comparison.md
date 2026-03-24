# ARF Direct Comparison Experiment

## Overview

This experiment compares SUDA-V3 (Selective Unlearning for Drift Adaptation) against River's Adaptive Random Forest (ARF) under identical experimental conditions.

## Purpose

Answer the research question: **Under iso-memory and iso-drift conditions, is SUDA-V3's selective unlearning competitive with ARF's adaptive ensemble approach?**

## Models Compared

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **SUDA-V3** | Selective unlearning with Rust-native buffer | 50 trees, 40% forget ratio, Aggressive-OnDrift |
| **ARF** | Adaptive Random Forest (River) | 50 models |
| **HAT** | Hoeffding Adaptive Tree (baseline) | Single tree |

## Experimental Setup

### Identical Conditions
- **Same data streams**: Controlled drift using `make_sudden_drift_stream`
- **Same buffer limits**: SUDA-V3 max_samples enforces iso-memory constraint
- **Same evaluation intervals**: Every 1000 samples
- **Same random seeds**: 42, 123, 456 for reproducibility

### Default Parameters
```
total_samples = 40000
drift_point = 20000 (50% point)
pre_benign_ratio = 0.7
post_benign_ratio = 0.3
max_samples = 30000 (SUDA buffer size)
batch_size = 500
eval_interval = 1000
```

### Datasets
- nslkdd (balanced)
- unswnb15 (attack-heavy)
- cicids2018 (extreme attack bias)
- cidds (extreme benign bias)

## Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Post-drift Accuracy** | Average accuracy after drift point | Higher = better adaptation |
| **Pre-drift Accuracy** | Average accuracy before drift point | Baseline performance |
| **TTR (Time-to-Recover)** | Samples to recover to 90% of baseline | Lower = faster recovery |
| **Total Update Time** | Time spent in `partial_fit` | Lower = faster training |

## Usage

### Quick Test (1 dataset, 1 seed, 10K samples)
```bash
uv run python src/experiments/arf_direct_comparison.py --quick
```

### Full Experiment (4 datasets, 3 seeds, 40K samples)
```bash
uv run python src/experiments/arf_direct_comparison.py
```

### Single Dataset
```bash
uv run python src/experiments/arf_direct_comparison.py --dataset nslkdd
```

### Custom Configuration
```bash
uv run python src/experiments/arf_direct_comparison.py \
    --dataset unswnb15 \
    --seeds 42 123 456 789 \
    --total-samples 50000 \
    --drift-point 25000 \
    --max-samples 40000
```

## Output

### Console Output
The script prints:
1. Progress for each dataset/seed combination
2. Summary tables comparing all methods:
   - Post-drift accuracy (mean ± std)
   - Pre-drift accuracy (mean ± std)
   - TTR (Time-to-Recover)
   - Total update time
3. Key findings (SUDA vs ARF comparison)

### JSON Results
Results are saved to `results/arf_comparison/arf_comparison_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2026-01-16T08:30:00",
  "config": {
    "datasets": ["nslkdd", "unswnb15", "cicids2018", "cidds"],
    "seeds": [42, 123, 456],
    "total_samples": 40000,
    "drift_point": 20000,
    "max_samples": 30000
  },
  "results": [
    {
      "method": "SUDA-V3",
      "dataset": "nslkdd",
      "seed": 42,
      "pre_drift_accuracy": 0.95,
      "post_drift_accuracy": 0.89,
      "final_accuracy": 0.91,
      "ttr": 5000,
      "total_update_time_s": 12.3,
      "n_drift_events": 3,
      "samples_removed": 8400,
      "accuracy_curve": [...],
      "positions": [...]
    },
    ...
  ]
}
```

## Interpretation Guide

### Success Criteria
- **SUDA competitive**: Post-drift accuracy within 2% of ARF
- **Faster adaptation**: TTR lower than ARF
- **Transparency advantage**: SUDA reports drift events and samples removed

### Expected Outcomes
- **Balanced datasets** (nslkdd): Similar performance
- **Imbalanced datasets** (cicids2018, cidds): SUDA may excel due to targeted removal
- **Attack-heavy** (unswnb15): Tests resilience to minority class removal

## Research Methodology Compliance

This experiment follows SUDA project guidelines (CLAUDE.md Section 10):

- ✅ Multiple seeds (3 minimum)
- ✅ All 4 datasets tested
- ✅ Statistical reporting (mean ± std)
- ✅ No circular reasoning (real downstream metrics)
- ✅ Proper baselines (ARF, HAT)
- ✅ Transparent result logging

## Dependencies

Required:
- SUDA project environment (`uv sync`)
- suda-core built (`uv run maturin develop --release --manifest-path suda-core/Cargo.toml`)
- Datasets downloaded (`uv run python datasets/download.py --all`)

## Notes

### ARF Implicit Forgetting
ARF doesn't expose explicit unlearning mechanisms. It adapts through:
- Background model replacement (worst performers dropped)
- Implicit decay via ADWIN drift detection
- No controllable buffer size (grows unbounded)

### SUDA Explicit Forgetting
SUDA provides:
- Explicit drift event logging
- Sample-level removal tracking
- Controllable memory footprint
- Selective removal strategies

### TTR Calculation
TTR (Time-to-Recover) measures samples needed to reach 90% of pre-drift baseline accuracy. `None` indicates the model never recovered within the stream.

## Author
Claude Sonnet 4.5
Date: 2026-01-16
Session: Sisyphus-Junior ARF Comparison
