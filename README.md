# suda-core

**A memory-safe Rust implementation of exact-unlearning Random Forests, with certified retrain-equivalence.**

`suda-core` is the core engine underneath the **SUDA** (Selective Unlearning for Drift Adaptation) research project. It provides the data structures and algorithms for *exact* machine unlearning in Random Forests — removing individual training samples so that the resulting model is **provably identical to one retrained from scratch without those samples**, in `O(k · depth)` instead of a full retrain.

This crate is the reusable foundation on top of which the SUDA experiments (streaming adaptation, influence-based eviction, budget management) are built.

---

## Attribution

This is an independent Rust reimplementation and extension of the **DynFrs** algorithm:

> Shurong Wang, Zhuoyang Shen, Xinbao Qiao, Tongning Zhang, Meng Zhang.
> **DynFrs: An Efficient Framework for Machine Unlearning in Random Forest.**
> *ICLR 2025.* <https://openreview.net/forum?id=nsCOeCLR8e>

```bibtex
@inproceedings{wang2025dynfrs,
  title     = {DynFrs: An Efficient Framework for Machine Unlearning in Random Forest},
  author    = {Shurong Wang and Zhuoyang Shen and Xinbao Qiao and Tongning Zhang and Meng Zhang},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url        = {https://openreview.net/forum?id=nsCOeCLR8e}
}
```

`suda-core` does **not** claim to improve on the DynFrs *method*. Its contribution is an **implementation**: memory safety, an extensive test suite that certifies the retrain-equivalence guarantee in code, and additional streaming / eviction features (below).

---

## What it provides

| Capability | Description |
|------------|-------------|
| **Exact unlearning** | `forget` / `forget_batch` remove samples with retrain-equivalent results (DynFrs guarantee), verified by tests — see [Correctness](#correctness). |
| **OCC(q) sampling** | Each training sample is assigned to at most `k` trees, bounding per-sample unlearning cost to `O(k · depth)`. |
| **LZY (lazy) tag rebuild** | Deletions mark subtrees `Dirty`/`Rebuild` and are reconciled in a bottom-up `develop()` pass, amortizing batch deletions. |
| **Streaming / incremental add** | Online sample insertion with Hoeffding-bound split decisions for concept-drift settings. |
| **Influence-based eviction** | An `InfluenceRegistry` tracks per-sample influence and supports budget-based continuous eviction to bound memory. |
| **Class-aware `k`** | Optional fixed two-value minority redundancy (`minority_k`, disabled by default). |
| **Python bindings** | Optional PyO3 module (`suda_core`) exposing the streaming controller to Python. |

## Correctness

The retrain-equivalence guarantee is checked in code, not just asserted: `tests/critical_correctness.rs` verifies that the model after `forget` is distribution-equivalent to one retrained from scratch on the remaining samples (DynFrs Theorem 1), alongside lazy-rebuild routing and streaming-state consistency. Run the suite with `cargo test`.

## Architecture

```
src/
├── forest.rs        DynFrsForest — OCC(q) ensemble, parallel fit (rayon), exact forget
├── tree.rs          DynFrsTree — split search, lazy-tag rebuild, streaming inserts
├── node.rs          Node (Internal/Leaf) + LazyTag state machine (Clean→Dirty→Rebuild)
├── registry.rs      InfluenceRegistry — influence tracking + budget eviction
├── controller.rs    StreamingController — end-to-end streaming loop + PyO3 surface
├── streaming.rs     Hoeffding-bound streaming split statistics
├── split*.rs        Split representation and candidate statistics
├── scan.rs          Cache-friendly / branchless partitioning
├── dataset.rs       Dataset abstraction, sample.rs — Sample traits
└── metrics/         Streaming confusion-matrix / metrics tracking
```

## Build & test

```bash
cargo build --release       # optimized (LTO) library
cargo test                  # 99 tests
cargo clippy --all-targets  # lint (0 warnings)
```

### Python bindings (optional)

Built with [maturin](https://github.com/PyO3/maturin):

```bash
maturin develop --release --features extension-module
```

```python
from suda_core import PyStreamingController
```

## Scope

This crate deliberately contains **only** the unlearning-tree engine and its correctness/performance concerns. Downstream benchmarking, datasets, and the SUDA research findings live in a separate project and are **not** part of this repository — so nothing here depends on any particular empirical result.

## License

MIT. See [LICENSE](LICENSE).
