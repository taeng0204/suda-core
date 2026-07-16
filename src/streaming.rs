//! Streaming learning support for DynFrs.
//!
//! This module implements the streaming/incremental learning capabilities
//! from the C++ DynFrs reference implementation. Key components:
//!
//! - `AttributeStats`: Tracks split statistics for incremental updates
//! - `SplitCandidate`: Represents a potential split with threshold counts
//! - Delayed rebuild mechanism for lazy tree restructuring
//!
//! Based on DynFrs.h lines 79-364 (attribute class) and 366-604 (node class).

use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::collections::HashMap;

/// Number of split candidates to try per attribute (p_count in C++).
pub const P_COUNT: usize = 10;

/// Number of threshold tries per candidate (p_tries in C++).
pub const P_TRIES: usize = 10;

/// Calculate Gini-based split score.
/// Formula: ((ls_s - ls_1) * ls_1 / ls_s + (rs_s - rs_1) * rs_1 / rs_s) / (ls_s + rs_s)
/// Lower score is better (more homogeneous split).
#[inline]
pub fn calc_score(ls_s: u32, ls_1: u32, rs_s: u32, rs_1: u32) -> f64 {
    if ls_s == 0 || rs_s == 0 {
        return f64::MAX;
    }

    let ls_s = ls_s as f64;
    let ls_1 = ls_1 as f64;
    let rs_s = rs_s as f64;
    let rs_1 = rs_1 as f64;

    let left_gini = (ls_s - ls_1) * ls_1 / ls_s;
    let right_gini = (rs_s - rs_1) * rs_1 / rs_s;

    (left_gini + right_gini) / (ls_s + rs_s)
}

/// A threshold candidate for a single attribute.
#[derive(Debug, Clone)]
pub struct ThresholdCandidate {
    /// The threshold value
    pub threshold: f32,
    /// Count of samples < threshold (left side, consistent with Split::goes_left)
    pub left_count: u32,
    /// Count of positive samples < threshold
    pub left_positive: u32,
}

impl ThresholdCandidate {
    pub fn new(threshold: f32, left_count: u32, left_positive: u32) -> Self {
        Self {
            threshold,
            left_count,
            left_positive,
        }
    }

    /// Calculate split score given total samples.
    pub fn score(&self, total_samples: u32, total_positive: u32) -> f64 {
        let rs_s = total_samples - self.left_count;
        let rs_1 = total_positive - self.left_positive;
        calc_score(self.left_count, self.left_positive, rs_s, rs_1)
    }
}

/// Statistics for a single attribute's split candidates.
#[derive(Debug, Clone)]
pub struct AttributeCandidate {
    /// Attribute index
    pub attr_idx: u8,
    /// Minimum value seen and its count
    pub min_val: f32,
    pub min_count: u32,
    /// Maximum value seen and its count
    pub max_val: f32,
    pub max_count: u32,
    /// Threshold candidates (sorted)
    pub thresholds: Vec<ThresholdCandidate>,
    /// Best split score and threshold
    pub best_score: f64,
    pub best_threshold: f32,
}

impl AttributeCandidate {
    pub fn new(attr_idx: u8) -> Self {
        Self {
            attr_idx,
            min_val: f32::MAX,
            min_count: 0,
            max_val: f32::MIN,
            max_count: 0,
            thresholds: Vec::new(),
            best_score: f64::MAX,
            best_threshold: 0.0,
        }
    }

    /// Initialize from a batch of samples.
    pub fn init_from_batch(
        attr_idx: u8,
        values: &[(f32, bool)], // (attribute_value, label)
        num_thresholds: usize,
        rng: &mut XorShiftRng,
    ) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        // Find min/max
        let mut min_val = values[0].0;
        let mut max_val = values[0].0;
        let mut min_count = 1u32;
        let mut max_count = 1u32;

        for &(v, _) in values.iter().skip(1) {
            if v < min_val {
                min_val = v;
                min_count = 1;
            } else if v > max_val {
                max_val = v;
                max_count = 1;
            } else if (v - min_val).abs() < f32::EPSILON {
                min_count += 1;
            } else if (v - max_val).abs() < f32::EPSILON {
                max_count += 1;
            }
        }

        // Check if constant (no valid split possible)
        if (max_val - min_val).abs() < f32::EPSILON {
            return None;
        }

        // Generate random thresholds
        let mut thresholds: Vec<f32> = (0..num_thresholds)
            .map(|_| rng.gen_range(min_val..max_val))
            .collect();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Count samples for each threshold
        let total_samples = values.len() as u32;
        let total_positive = values.iter().filter(|(_, l)| *l).count() as u32;

        let mut threshold_candidates = Vec::with_capacity(thresholds.len());
        let mut best_score = f64::MAX;
        let mut best_threshold = thresholds[0];

        for &thresh in &thresholds {
            let mut left_count = 0u32;
            let mut left_positive = 0u32;

            for &(v, label) in values {
                if v <= thresh {
                    left_count += 1;
                    if label {
                        left_positive += 1;
                    }
                }
            }

            let candidate = ThresholdCandidate::new(thresh, left_count, left_positive);
            let score = candidate.score(total_samples, total_positive);

            if score < best_score {
                best_score = score;
                best_threshold = thresh;
            }

            threshold_candidates.push(candidate);
        }

        Some(Self {
            attr_idx,
            min_val,
            min_count,
            max_val,
            max_count,
            thresholds: threshold_candidates,
            best_score,
            best_threshold,
        })
    }

    /// Update statistics when a sample is added.
    /// Returns true if the candidate was invalidated (range expanded).
    pub fn add_sample(
        &mut self,
        value: f32,
        label: bool,
        total_samples: u32,
        total_positive: u32,
    ) -> bool {
        // Check if range is expanded
        if value < self.min_val {
            self.min_val = value;
            self.min_count = 1;
            return true; // Candidate invalidated
        } else if value > self.max_val {
            self.max_val = value;
            self.max_count = 1;
            return true; // Candidate invalidated
        } else if (value - self.min_val).abs() < f32::EPSILON {
            self.min_count += 1;
        } else if (value - self.max_val).abs() < f32::EPSILON {
            self.max_count += 1;
        }

        // Update threshold counts
        // NOTE: Use strict `<` to match Split::goes_left() in split.rs
        self.best_score = f64::MAX;
        for candidate in &mut self.thresholds {
            if value <= candidate.threshold {
                candidate.left_count += 1;
                if label {
                    candidate.left_positive += 1;
                }
            }

            let score = candidate.score(total_samples, total_positive);
            if score < self.best_score {
                self.best_score = score;
                self.best_threshold = candidate.threshold;
            }
        }

        false // Candidate still valid
    }

    /// Update statistics when a sample is removed.
    /// Returns true if the candidate was invalidated (range shrunk to empty).
    pub fn remove_sample(
        &mut self,
        value: f32,
        label: bool,
        total_samples: u32,
        total_positive: u32,
    ) -> bool {
        // Check if removing min/max
        if (value - self.min_val).abs() < f32::EPSILON {
            self.min_count -= 1;
            if self.min_count == 0 {
                return true; // Candidate invalidated
            }
        } else if (value - self.max_val).abs() < f32::EPSILON {
            self.max_count -= 1;
            if self.max_count == 0 {
                return true; // Candidate invalidated
            }
        }

        // Update threshold counts
        // NOTE: Use strict `<` to match Split::goes_left() in split.rs
        self.best_score = f64::MAX;
        for candidate in &mut self.thresholds {
            if value <= candidate.threshold {
                candidate.left_count = candidate.left_count.saturating_sub(1);
                if label {
                    candidate.left_positive = candidate.left_positive.saturating_sub(1);
                }
            }

            let score = candidate.score(total_samples, total_positive);
            if score < self.best_score {
                self.best_score = score;
                self.best_threshold = candidate.threshold;
            }
        }

        false
    }
}

/// Streaming attribute statistics tracker for a node.
/// Equivalent to C++ `attribute` class in DynFrs.h:79-364.
#[derive(Debug, Clone)]
pub struct AttributeStats {
    /// Number of attributes
    num_attributes: u8,
    /// Total samples in this node
    pub total_samples: u32,
    /// Positive samples in this node
    pub total_positive: u32,
    /// Active candidates: attr_idx -> AttributeCandidate
    pub candidates: HashMap<u8, AttributeCandidate>,
    /// Attributes marked as constant (no valid split)
    constant_attrs: Vec<bool>,
    /// RNG for generating splits
    rng: XorShiftRng,
}

impl AttributeStats {
    /// Create new empty attribute stats.
    pub fn new(num_attributes: u8, seed: u64) -> Self {
        Self {
            num_attributes,
            total_samples: 0,
            total_positive: 0,
            candidates: HashMap::new(),
            constant_attrs: vec![false; num_attributes as usize],
            rng: XorShiftRng::seed_from_u64(seed),
        }
    }

    //   DynFrs random mode: tree-shared mt19937 (SUDA에서는 init 시점 seed로 충분).

    /// Initialize from a batch of samples.
    pub fn init_from_batch<F>(
        &mut self,
        samples: &[u64], // sample IDs
        get_features: F, // closure to get features: id -> &[f32]
        get_label: impl Fn(u64) -> bool,
        num_candidates: usize, // number of attribute candidates to try
    ) where
        F: Fn(u64) -> Vec<f32>,
    {
        self.total_samples = samples.len() as u32;
        self.total_positive = samples.iter().filter(|&&id| get_label(id)).count() as u32;
        self.candidates.clear();

        // Randomly select attributes to create candidates for
        let mut attr_indices: Vec<u8> = (0..self.num_attributes).collect();
        for i in 0..num_candidates.min(attr_indices.len()) {
            let j = self.rng.gen_range(i..attr_indices.len());
            attr_indices.swap(i, j);
        }

        for &attr_idx in attr_indices.iter().take(num_candidates) {
            // Collect values for this attribute
            let values: Vec<(f32, bool)> = samples
                .iter()
                .map(|&id| {
                    let features = get_features(id);
                    let value = features.get(attr_idx as usize).copied().unwrap_or(0.0);
                    (value, get_label(id))
                })
                .collect();

            // Try to create candidate
            if let Some(candidate) =
                AttributeCandidate::init_from_batch(attr_idx, &values, P_TRIES, &mut self.rng)
            {
                self.candidates.insert(attr_idx, candidate);
            } else {
                self.constant_attrs[attr_idx as usize] = true;
            }
        }
    }

    /// Add a sample and update all candidates.
    /// Returns the number of invalidated candidates.
    pub fn add_sample(&mut self, features: &[f32], label: bool) -> usize {
        self.total_samples += 1;
        if label {
            self.total_positive += 1;
        }

        let mut invalidated = Vec::new();

        for (&attr_idx, candidate) in &mut self.candidates {
            let value = features.get(attr_idx as usize).copied().unwrap_or(0.0);
            if candidate.add_sample(value, label, self.total_samples, self.total_positive) {
                invalidated.push(attr_idx);
            }
        }

        // Remove invalidated candidates
        for attr_idx in &invalidated {
            self.candidates.remove(attr_idx);
        }

        invalidated.len()
    }

    /// Remove a sample and update all candidates.
    /// Returns the number of invalidated candidates.
    pub fn remove_sample(&mut self, features: &[f32], label: bool) -> usize {
        self.total_samples = self.total_samples.saturating_sub(1);
        if label {
            self.total_positive = self.total_positive.saturating_sub(1);
        }

        let mut invalidated = Vec::new();

        for (&attr_idx, candidate) in &mut self.candidates {
            let value = features.get(attr_idx as usize).copied().unwrap_or(0.0);
            if candidate.remove_sample(value, label, self.total_samples, self.total_positive) {
                invalidated.push(attr_idx);
            }
        }

        for attr_idx in &invalidated {
            self.candidates.remove(attr_idx);
        }

        invalidated.len()
    }

    /// Find the best split among all candidates.
    /// Returns (attribute_index, threshold, score) or None if no valid split.
    pub fn find_best_split(&self) -> Option<(u8, f32, f64)> {
        let mut best: Option<(u8, f32, f64)> = None;

        for candidate in self.candidates.values() {
            if candidate.best_score < f64::MAX {
                match best {
                    None => {
                        best = Some((
                            candidate.attr_idx,
                            candidate.best_threshold,
                            candidate.best_score,
                        ));
                    }
                    Some((_, _, best_score)) if candidate.best_score < best_score => {
                        best = Some((
                            candidate.attr_idx,
                            candidate.best_threshold,
                            candidate.best_score,
                        ));
                    }
                    _ => {}
                }
            }
        }

        best
    }

    /// Check if this node should be a leaf.
    pub fn should_be_leaf(
        &self,
        min_samples_split: u32,
        max_depth: u32,
        current_depth: u32,
    ) -> bool {
        self.total_samples < min_samples_split
            || self.total_positive == 0
            || self.total_positive == self.total_samples
            || current_depth >= max_depth
    }

    /// Get the number of active candidates.
    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.total_samples = 0;
        self.total_positive = 0;
        self.candidates.clear();
        self.constant_attrs.fill(false);
    }
}

/// Delay tag for lazy rebuild (corresponds to C++ `delay` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelayTag {
    /// No pending operation (delay=0)
    #[default]
    None,
    /// Needs build (delay=1) - node has samples but no split computed
    NeedsBuild,
    /// Needs separate + build (delay=2) - split changed, need to re-partition
    NeedsSeparateAndBuild,
}

/// Streaming node state for incremental learning.
#[derive(Debug, Clone)]
pub struct StreamingNodeState {
    /// Attribute statistics tracker
    pub attr_stats: AttributeStats,
    /// Sample IDs in this node (for leaves or pending rebuild)
    pub sample_ids: Vec<u64>,
    /// Current split attribute (-1 if leaf)
    pub split_attr: Option<u8>,
    /// Current split threshold
    pub split_threshold: f32,
    /// Delay tag for lazy operations
    pub delay: DelayTag,
    /// Node depth
    pub depth: u32,
}

impl StreamingNodeState {
    pub fn new(num_attributes: u8, seed: u64, depth: u32) -> Self {
        Self {
            attr_stats: AttributeStats::new(num_attributes, seed),
            sample_ids: Vec::new(),
            split_attr: None,
            split_threshold: 0.0,
            delay: DelayTag::NeedsBuild,
            depth,
        }
    }

    /// Check if this is a leaf (no valid split).
    pub fn is_leaf(&self) -> bool {
        self.split_attr.is_none()
    }

    /// Check if the best split has changed from current.
    pub fn best_split_changed(&self) -> bool {
        if let Some(current_attr) = self.split_attr {
            if let Some((best_attr, best_thresh, _)) = self.attr_stats.find_best_split() {
                return best_attr != current_attr
                    || (best_thresh - self.split_threshold).abs() > f32::EPSILON;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_score() {
        // Perfect split: all left are positive, all right are negative
        let score = calc_score(50, 50, 50, 0);
        assert!(score < 0.01, "Perfect split should have near-zero score");

        // Random split: 50/50 in each side
        let score = calc_score(50, 25, 50, 25);
        assert!(score > 0.1, "Random split should have higher score");
    }

    #[test]
    fn test_threshold_candidate() {
        let candidate = ThresholdCandidate::new(5.0, 30, 20);
        let score = candidate.score(100, 50);
        assert!(score > 0.0, "Score should be positive");
    }

    #[test]
    fn test_attribute_candidate_init() {
        let values: Vec<(f32, bool)> = vec![
            (1.0, true),
            (2.0, true),
            (3.0, false),
            (8.0, false),
            (9.0, false),
        ];

        let mut rng = XorShiftRng::seed_from_u64(42);
        let candidate = AttributeCandidate::init_from_batch(0, &values, 5, &mut rng);

        assert!(candidate.is_some(), "Should create candidate");
        let c = candidate.unwrap();
        assert_eq!(c.attr_idx, 0);
        assert!((c.min_val - 1.0).abs() < f32::EPSILON);
        assert!((c.max_val - 9.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attribute_candidate_add_sample() {
        let values: Vec<(f32, bool)> = vec![(2.0, true), (4.0, true), (6.0, false), (8.0, false)];

        let mut rng = XorShiftRng::seed_from_u64(42);
        let mut candidate = AttributeCandidate::init_from_batch(0, &values, 3, &mut rng).unwrap();

        // Add sample within range
        let invalidated = candidate.add_sample(5.0, true, 5, 3);
        assert!(!invalidated, "Should not invalidate within range");

        // Add sample expanding range
        let invalidated = candidate.add_sample(10.0, false, 6, 3);
        assert!(invalidated, "Should invalidate when range expands");
    }

    #[test]
    fn test_attribute_stats() {
        let mut stats = AttributeStats::new(4, 42);
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.num_candidates(), 0);

        // Initialize with mock data
        let sample_ids: Vec<u64> = vec![0, 1, 2, 3, 4];
        let features: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0],
            vec![8.0, 9.0, 10.0, 11.0],
        ];
        let labels: Vec<bool> = vec![true, true, true, false, false];

        stats.init_from_batch(
            &sample_ids,
            |id| features[id as usize].clone(),
            |id| labels[id as usize],
            3,
        );

        assert_eq!(stats.total_samples, 5);
        assert_eq!(stats.total_positive, 3);
        assert!(stats.num_candidates() > 0, "Should have created candidates");
    }

    #[test]
    fn test_streaming_node_state() {
        let state = StreamingNodeState::new(4, 42, 0);
        assert!(state.is_leaf());
        assert_eq!(state.delay, DelayTag::NeedsBuild);
    }
}
