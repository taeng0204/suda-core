//! Streaming Metrics Module
//!
//! Native Rust implementations of evaluation metrics for streaming classification.
//! Replaces Python/sklearn dependency with efficient Rust-native metrics.
//!
//! Key metrics for imbalanced NIDS data:
//! - G-mean: Geometric mean of TPR and TNR (most important for imbalanced data)
//! - Kappa: Cohen's Kappa (chance-corrected agreement)
//! - Balanced Accuracy: Average of TPR and TNR

pub mod streaming;

pub use streaming::{ConfusionMatrix, MetricsTracker, StreamingMetrics};
