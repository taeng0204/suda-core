//! SUDA Core: DynFrs-based Exact Unlearning Random Forest
//!
//! This library implements the DynFrs algorithm for efficient exact unlearning
//! in Random Forest models, optimized for Network Intrusion Detection Systems.
//!
//! # Key Features
//! - OCC(q) Sampling: Each sample appears in at most k trees
//! - LZY Tag: Lazy rebuild for batch deletions
//! - Exact Unlearning: Mathematically equivalent to retraining from scratch
//! - Budget-based Continuous Eviction: Maintains model freshness via exact forgetting
//! - Python Bindings: PyO3-based Python API

pub mod controller;
pub mod dataset;
pub mod forest;
pub mod metrics;
pub mod node;
pub mod registry;
pub mod sample;
pub mod scan;
pub mod split;
pub mod split_stats;
pub mod streaming;
pub mod tree;

// Re-exports
pub use controller::{
    FeatureProvider, SUDAConfig, SimpleFeatureStore, StreamingController, StreamingResult,
};
pub use dataset::{ArrayDataset, AttributeType, Dataset};
pub use forest::DynFrsForest;
pub use metrics::{ConfusionMatrix, MetricsTracker, StreamingMetrics};
pub use node::{LazyTag, Node};
pub use registry::{
    BudgetConfig, EvictionStats, InfluenceRegistry, SampleLifecycle, TrackedSample,
};
pub use sample::{ArraySample, Sample, VecSample};
pub use split::Split;
pub use split_stats::SplitStats;
pub use tree::DynFrsTree;

// Python module
use pyo3::prelude::*;

/// Python module for SUDA Core
#[pymodule]
fn suda_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<controller::PyStreamingController>()?;
    Ok(())
}
