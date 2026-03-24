# SUDA: Selective Unlearning for Drift Adaptation

> **Sample-level exact forgetting** for concept drift adaptation in Network Intrusion Detection Systems (NIDS).

SUDA is a streaming random forest framework that adapts to concept drift by **precisely forgetting outdated samples** rather than replacing entire trees. Built with a high-performance Rust core (PyO3) and Python experiment interface.

## Key Idea

Existing adaptive random forests (ARF) handle concept drift by **replacing entire trees** when drift is detected — a coarse-grained approach that discards useful knowledge along with outdated patterns.

SUDA takes a different approach: **What if we selectively forget individual samples instead?**

- **Exact Forgetting**: Remove specific samples from the forest in O(k·depth), preserving the rest
- **Budget-based Eviction**: Maintain a fixed-size sample registry; evict oldest samples via FIFO
- **Adaptive-k Redundancy**: Protect minority classes (down to 1% attack ratio) with up to 70x redundancy
- **Age-based Subtree Refresh**: Periodically rebuild stale splits to address "Structural Debt"

## Key Findings (~3,400 experiments)

| Finding | Detail |
|---------|--------|
| **HOW > WHAT** | The forgetting *mechanism* matters more than the *selection strategy* — registry removal without `forest.forget()` has **zero effect** (36/36 scenarios) |
| **+8.6~31.4% G-mean** | Over baseline ARF on synthetic drift scenarios (p<0.001, Cohen's d=1.78) |
| **Structural Debt** | `forget()` alone cannot update split thresholds → performance ceiling under gradual drift. Solved by `split_max_age` (+5.2%p) |
| **~10x speedup** | Rust core vs. pure Python baseline |

## Architecture

```
Python API                          Rust Core (suda-core)
──────────                          ─────────────────────
SUDA.fit(X, y)          →          StreamingController::fit()
                                     ├─ DynFrsForest::fit_weighted()  (batch)
                                     ├─ enable_streaming()            (activate split updates)
                                     └─ Registry::register()          (track samples)

SUDA.partial_fit(X, y)  →          StreamingController::stream_batch()
                                     ├─ predict → update_metrics      (test-then-train)
                                     ├─ add_samples_streaming()       (insert + update splits)
                                     ├─ enforce_budget()              (evict if |R| > B)
                                     │   └─ forget_batch()            (exact unlearning)
                                     └─ develop_streaming()           (rebuild LazyTag nodes)
```

## Quick Start

```python
from src.models.suda import SUDA

model = SUDA(
    num_features=41,
    num_trees=50,
    k=10,
    max_depth=15,
    budget_enabled=True,
    budget_max_samples=3000,
    adaptive_k_enabled=True,
    k_min=1, k_max=70,
)

model.fit(X_warmup, y_warmup)

for X_batch, y_batch in data_stream:
    result = model.partial_fit(X_batch, y_batch)
    print(f"G-mean: {result.metrics['gmean']:.4f}")
```

## Build

```bash
# Rust core
cd suda-core && cargo test

# Python package (requires maturin)
uv run maturin develop --release -m suda-core/Cargo.toml

# Download datasets
uv run python datasets/download.py --all

# Run experiments
uv run python -m src.experiments.r1_grid_search --datasets nslkdd --seeds 42
```

## Datasets

All datasets are publicly available and downloaded via `datasets/download.py`.

| Dataset | Samples | Features | Attack % | Source |
|---------|---------|----------|----------|--------|
| NSL-KDD | 148,517 | 41 | 48.1% | [UNB](https://www.unb.ca/cic/datasets/nsl.html) |
| UNSW-NB15 | 257,673 | 42 | 63.9% | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| CIC-IDS2018 | 43,036 | 78 | 96.7% | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) |
| ANoShift | ~50,000 | 15 | 60.3% | [ANoShift](https://github.com/bit-ml/ANoShift) |

## Project Structure

```
suda-public/
├── suda-core/           # Rust core (PyO3 bindings)
│   ├── src/
│   │   ├── controller.rs    # StreamingController — main pipeline
│   │   ├── forest.rs        # DynFrsForest — OCC(k) + exact unlearning
│   │   ├── tree.rs          # DynFrsTree — batch + streaming learning
│   │   ├── registry.rs      # InfluenceRegistry — sample tracking + eviction
│   │   ├── node.rs          # Node — Internal/Leaf, LazyTag
│   │   ├── streaming.rs     # Streaming split updates
│   │   └── metrics/         # G-mean, accuracy tracking
│   ├── tests/
│   └── Cargo.toml
├── src/                 # Python interface
│   ├── models/suda.py       # SUDA Python wrapper
│   ├── data/nids.py         # Dataset loaders
│   ├── baselines/           # ARF, SRP baselines
│   └── experiments/         # Experiment scripts
├── tests/               # Python tests
├── datasets/            # Download scripts (data not included)
└── pyproject.toml
```

## Citation

Paper in preparation. If you use this code, please cite:

```bibtex
@misc{lim2026suda,
  title={SUDA: Selective Unlearning for Drift Adaptation in Network Intrusion Detection},
  author={Taein Lim},
  year={2026},
  note={In preparation}
}
```

## License

MIT
