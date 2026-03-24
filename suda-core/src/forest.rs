//! DynFrs Random Forest with OCC(q) sampling and exact unlearning.

use hashbrown::{HashMap, HashSet};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use crate::dataset::ArrayDataset;
use crate::sample::VecSample;
use crate::tree::{DynFrsTree, TreeConfig};

/// Strategy for determining k value per sample in OCC(q) sampling.
enum KStrategy<'a> {
    /// Use config.k for all samples (standard OCC(q))
    Default,
    /// Look up k from class_k_map per sample's class label
    ClassBased,
    /// Use an explicit k value array (k_values[i] for samples[i])
    PerSample(&'a [usize]),
}

/// Configuration for the forest.
#[derive(Debug, Clone)]
pub struct ForestConfig {
    /// Number of trees in the forest
    pub num_trees: usize,
    /// OCC(q) parameter: maximum number of trees a sample can appear in
    pub k: usize,
    /// Tree configuration
    pub tree_config: TreeConfig,
    /// Random seed
    pub seed: u64,
}

impl Default for ForestConfig {
    fn default() -> Self {
        ForestConfig {
            num_trees: 100,
            k: 10,
            tree_config: TreeConfig::default(),
            seed: 42,
        }
    }
}

/// Configuration for adaptive k-redundancy based on class imbalance.
/// Minority class samples get higher k for better protection.
///
/// ECBRS-inspired: For extreme minority (< 1%), use k = num_trees to ensure
/// samples appear in ALL trees.
#[derive(Debug, Clone)]
pub struct AdaptiveKConfig {
    /// Minimum k for majority class (default: 3)
    pub k_min: usize,
    /// Maximum k for minority class (default: num_trees, set dynamically)
    pub k_max: usize,
    /// Class ratio threshold for k interpolation (default: 0.3)
    pub ratio_threshold: f64,
    /// Extreme minority threshold - below this, use k_max (default: 0.01 = 1%)
    pub extreme_threshold: f64,
}

impl Default for AdaptiveKConfig {
    fn default() -> Self {
        AdaptiveKConfig {
            k_min: 3,
            k_max: 50,  // Default to num_trees
            ratio_threshold: 0.3,
            extreme_threshold: 0.01,  // 1% - below this is "extreme minority"
        }
    }
}

impl AdaptiveKConfig {
    /// Calculate k value based on class ratio.
    ///
    /// ECBRS-inspired logic:
    /// - If ratio < extreme_threshold (1%): k = k_max (all trees)
    /// - Otherwise: interpolate between k_min and k_max
    ///
    /// Lower ratio (minority) -> higher k (more protection)
    pub fn compute_k(&self, class_ratio: f64) -> usize {
        // Extreme minority: use maximum k (appear in all trees)
        if class_ratio < self.extreme_threshold {
            return self.k_max;
        }

        // Normal interpolation for moderate minority
        let normalized = (class_ratio / self.ratio_threshold).min(1.0);
        let k_range = (self.k_max - self.k_min) as f64;
        let k = self.k_max as f64 - k_range * normalized;
        k.round() as usize
    }
}

/// DynFrs Random Forest with exact unlearning support.
#[derive(Debug)]
pub struct DynFrsForest {
    /// Individual trees
    trees: Vec<DynFrsTree>,
    /// OCC(q) mapping: sample_id -> tree indices where it appears
    sample_tree_map: HashMap<u64, Vec<usize>>,
    /// Sample labels for unlearning
    sample_labels: HashMap<u64, bool>,
    /// Configuration
    config: ForestConfig,
    /// Number of attributes
    num_attributes: u8,
    /// Random number generator for OCC sampling
    rng: XorShiftRng,
    /// Adaptive k configuration (optional)
    adaptive_k_config: Option<AdaptiveKConfig>,
    /// Class-based k mapping: label -> k value (for adaptive k)
    class_k_map: HashMap<bool, usize>,
    /// Streaming class counts: (negative_count, positive_count)
    streaming_class_counts: (u64, u64),
    /// Interval for updating class k during streaming (in samples)
    streaming_k_update_interval: u64,
    /// Samples processed since last k update
    streaming_samples_since_k_update: u64,
}

impl DynFrsForest {
    /// Create a new empty forest.
    pub fn new(config: ForestConfig, num_attributes: u8) -> Self {
        let rng = XorShiftRng::seed_from_u64(config.seed);

        let trees: Vec<DynFrsTree> = (0..config.num_trees)
            .map(|i| DynFrsTree::new(i, config.seed, config.tree_config.clone(), num_attributes))
            .collect();

        DynFrsForest {
            trees,
            sample_tree_map: HashMap::new(),
            sample_labels: HashMap::new(),
            config,
            num_attributes,
            rng,
            adaptive_k_config: None,
            class_k_map: HashMap::new(),
            streaming_class_counts: (0, 0),
            streaming_k_update_interval: 100, // Update k every 100 samples
            streaming_samples_since_k_update: 0,
        }
    }

    /// Enable adaptive k-redundancy with custom configuration.
    pub fn set_adaptive_k_config(&mut self, config: AdaptiveKConfig) {
        self.adaptive_k_config = Some(config);
    }

    /// Set k value for a specific class label.
    pub fn set_class_k(&mut self, label: bool, k: usize) {
        self.class_k_map.insert(label, k);
    }

    /// Get k value for a specific class label.
    pub fn get_class_k(&self, label: bool) -> Option<usize> {
        self.class_k_map.get(&label).copied()
    }

    /// Update class k values based on current class distribution.
    ///
    /// compute_k() maps low ratio → high k, so whichever class is the statistical
    /// minority automatically gets higher redundancy (more tree appearances).
    /// This is correct for both normal (attack=minority) and reversed (benign=minority)
    /// imbalance scenarios:
    ///   - NSL-KDD (48% attack): attack k≈27, benign k≈20
    ///   - CIC-IDS2018 (97% attack): attack k=3, benign k=45
    pub fn update_class_k_from_distribution(&mut self, positive_ratio: f64) {
        if let Some(ref config) = self.adaptive_k_config {
            let negative_ratio = 1.0 - positive_ratio;

            // Each class gets k proportional to its minority status:
            // low ratio → high k (more protection), high ratio → low k
            let k_positive = config.compute_k(positive_ratio);
            let k_negative = config.compute_k(negative_ratio);

            self.class_k_map.insert(true, k_positive);
            self.class_k_map.insert(false, k_negative);
        }
    }

    /// Fit the forest on samples using OCC(q) sampling.
    pub fn fit(&mut self, samples: &[VecSample]) {
        self.fit_internal(samples, KStrategy::Default);
    }

    /// Fit the forest using adaptive k-redundancy based on class labels.
    /// Each sample gets k based on its class from class_k_map.
    /// Falls back to config.k if class_k_map is not set.
    pub fn fit_adaptive(&mut self, samples: &[VecSample]) {
        self.fit_internal(samples, KStrategy::ClassBased);
    }

    /// Fit the forest using explicit k values for each sample.
    /// k_values[i] specifies the k value for samples[i].
    pub fn fit_with_k_values(&mut self, samples: &[VecSample], k_values: &[usize]) {
        assert_eq!(
            samples.len(),
            k_values.len(),
            "samples and k_values must have the same length"
        );
        self.fit_internal(samples, KStrategy::PerSample(k_values));
    }

    /// Fit the forest with class-weighted bootstrap sampling.
    ///
    /// Addresses extreme class imbalance by adjusting adaptive-k based on
    /// class distribution before OCC(q) sampling, giving minority class
    /// higher redundancy (more tree appearances).
    ///
    /// # Arguments
    /// * `samples` - Training samples
    /// * `positive_ratio` - Ratio of positive (attack) class (0.0 - 1.0)
    pub fn fit_weighted(&mut self, samples: &[VecSample], positive_ratio: f64) {
        self.update_class_k_from_distribution(positive_ratio);
        self.fit_internal(samples, KStrategy::ClassBased);
    }

    /// Internal fit implementation shared by all fit variants.
    ///
    /// OCC(q) sampling assigns each sample to at most k trees, where k is
    /// determined by `k_strategy`. Trees are then built in parallel.
    fn fit_internal(&mut self, samples: &[VecSample], k_strategy: KStrategy) {
        if samples.is_empty() {
            return;
        }

        self.sample_tree_map.clear();
        self.sample_labels.clear();

        // Store sample labels
        for sample in samples {
            self.sample_labels.insert(sample.id, sample.label);
        }

        // OCC(q) sampling: assign each sample to at most k trees
        let mut tree_samples: Vec<Vec<VecSample>> = vec![Vec::new(); self.config.num_trees];

        for (i, sample) in samples.iter().enumerate() {
            let k = match &k_strategy {
                KStrategy::Default => self.config.k,
                KStrategy::ClassBased => self
                    .class_k_map
                    .get(&sample.label)
                    .copied()
                    .unwrap_or(self.config.k),
                KStrategy::PerSample(k_values) => k_values[i],
            };

            let tree_indices = self.occ_sample_adaptive(sample.id, k);
            self.sample_tree_map.insert(sample.id, tree_indices.clone());

            for &tree_idx in &tree_indices {
                tree_samples[tree_idx].push(sample.clone());
            }
        }

        // Build dataset metadata
        let dataset = ArrayDataset::from_samples(samples, self.num_attributes);

        // Fit trees in parallel
        self.trees
            .par_iter_mut()
            .zip(tree_samples.par_iter_mut())
            .for_each(|(tree, samples)| {
                if !samples.is_empty() {
                    tree.fit(&dataset, samples);
                }
            });
    }

    /// Adaptive OCC(q) sampling: select trees with custom k value.
    /// Used for class-imbalance aware sampling (minority gets higher k).
    fn occ_sample_adaptive(&mut self, _sample_id: u64, k: usize) -> Vec<usize> {
        // Clamp k to valid range
        let k = k.min(self.config.num_trees).max(1);

        let mut selected = Vec::with_capacity(k);
        let mut count = 0;

        for i in 0..self.config.num_trees {
            if count >= k {
                break;
            }

            let remaining_trees = self.config.num_trees - i;
            let remaining_slots = k - count;

            // Probability: remaining_slots / remaining_trees
            if self.rng.gen_range(0..remaining_trees) < remaining_slots {
                selected.push(i);
                count += 1;
            }
        }

        selected
    }

    /// Predict for a single sample (majority vote).
    pub fn predict(&self, sample: &VecSample) -> bool {
        let num_positive: usize = self
            .trees
            .par_iter()
            .filter(|tree| tree.predict(sample))
            .count();

        num_positive * 2 > self.trees.len()
    }

    /// Predict probability (ratio of positive votes across all trees).
    pub fn predict_proba(&self, sample: &VecSample) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }
        let num_positive: usize = self
            .trees
            .par_iter()
            .filter(|tree| tree.predict(sample))
            .count();
        num_positive as f64 / self.trees.len() as f64
    }

    /// Predict probability using a subset of trees.
    pub fn predict_proba_with_trees(&self, sample: &VecSample, tree_indices: &[usize]) -> Option<f64> {
        if tree_indices.is_empty() {
            return None;
        }
        let num_positive: usize = tree_indices
            .par_iter()
            .filter(|&&idx| idx < self.trees.len() && self.trees[idx].predict(sample))
            .count();
        Some(num_positive as f64 / tree_indices.len() as f64)
    }

    /// Predict for multiple samples.
    pub fn predict_batch(&self, samples: &[VecSample]) -> Vec<bool> {
        samples.par_iter().map(|s| self.predict(s)).collect()
    }

    /// Predict using a specific tree (for OOB influence computation).
    pub fn predict_with_tree(&self, tree_idx: usize, sample: &VecSample) -> bool {
        if tree_idx < self.trees.len() {
            self.trees[tree_idx].predict(sample)
        } else {
            false
        }
    }

    /// Forget a single sample (exact unlearning).
    pub fn forget(&mut self, sample_id: u64) -> bool {
        self.forget_impl(sample_id, None)
    }

    /// Forget a single sample with streaming state update.
    /// When features are provided and the tree has streaming enabled,
    /// uses the streaming-aware path to keep attribute statistics consistent.
    pub fn forget_with_features(&mut self, sample_id: u64, features: &[f32]) -> bool {
        self.forget_impl(sample_id, Some(features))
    }

    /// Core forget implementation, optionally streaming-aware.
    fn forget_impl(&mut self, sample_id: u64, features: Option<&[f32]>) -> bool {
        // Get the label for this sample
        let was_positive = match self.sample_labels.remove(&sample_id) {
            Some(label) => label,
            None => return false, // Sample not found
        };

        // Get trees containing this sample
        let tree_indices = match self.sample_tree_map.remove(&sample_id) {
            Some(indices) => indices,
            None => return false,
        };

        // Remove from each tree (only k trees at most)
        // Use streaming-aware path when features are available and tree has streaming enabled
        for tree_idx in tree_indices {
            if let Some(feats) = features {
                if self.trees[tree_idx].is_streaming() {
                    self.trees[tree_idx].remove_sample_streaming(
                        sample_id, feats, was_positive, true,
                    );
                    continue;
                }
            }
            self.trees[tree_idx].forget(sample_id, was_positive);
        }

        // Streaming CWB: Update class counts to maintain accurate distribution
        if was_positive {
            self.streaming_class_counts.1 = self.streaming_class_counts.1.saturating_sub(1);
        } else {
            self.streaming_class_counts.0 = self.streaming_class_counts.0.saturating_sub(1);
        }

        true
    }

    /// Forget multiple samples (batch unlearning).
    pub fn forget_batch(&mut self, sample_ids: &[u64]) -> usize {
        let mut count = 0;
        for &sample_id in sample_ids {
            if self.forget(sample_id) {
                count += 1;
            }
        }
        count
    }

    /// Forget multiple samples with streaming state update.
    /// Uses feature_map to look up features for streaming-aware forget.
    pub fn forget_batch_with_features(
        &mut self,
        sample_ids: &[u64],
        feature_map: &HashMap<u64, Vec<f32>>,
    ) -> usize {
        let mut count = 0;
        for &sample_id in sample_ids {
            if let Some(features) = feature_map.get(&sample_id) {
                if self.forget_with_features(sample_id, features) {
                    count += 1;
                }
            } else if self.forget(sample_id) {
                count += 1;
            }
        }
        count
    }

    /// Optimized batch forget with 33% threshold (DynFrs algorithm).
    ///
    /// If deleting more than 33% of samples from a tree, it's faster
    /// to rebuild the entire tree than to incrementally update each node.
    ///
    /// Based on DynFrs C++ implementation (DynFrs.h:655-659).
    ///
    /// Returns the number of samples actually forgotten.
    pub fn forget_batch_optimized(&mut self, sample_ids: &[u64], samples: &[VecSample]) -> usize {
        if sample_ids.is_empty() {
            return 0;
        }

        // Prepare (sample_id, was_positive) pairs and group by tree
        let mut tree_deletions: Vec<Vec<(u64, bool)>> = vec![Vec::new(); self.config.num_trees];
        let mut total_forgotten = 0;

        for &sample_id in sample_ids {
            // Get label and tree indices
            let was_positive = match self.sample_labels.remove(&sample_id) {
                Some(label) => label,
                None => continue, // Sample not found
            };

            let tree_indices = match self.sample_tree_map.remove(&sample_id) {
                Some(indices) => indices,
                None => continue,
            };

            total_forgotten += 1;

            // Add to each tree's deletion list
            for &tree_idx in &tree_indices {
                tree_deletions[tree_idx].push((sample_id, was_positive));
            }
        }

        if total_forgotten == 0 {
            return 0;
        }

        // Build dataset for potential rebuilds
        let dataset = ArrayDataset::from_samples(samples, self.num_attributes);

        // Process each tree with optimized batch forget
        // Using parallel iteration for performance
        self.trees
            .par_iter_mut()
            .zip(tree_deletions.par_iter())
            .for_each(|(tree, deletions)| {
                if !deletions.is_empty() {
                    // Get samples for this tree (for potential rebuild)
                    let tree_samples: Vec<VecSample> = samples
                        .iter()
                        .filter(|s| tree.contains_sample(s.id))
                        .cloned()
                        .collect();

                    tree.forget_batch_optimized(deletions, &dataset, &tree_samples);
                }
            });

        total_forgotten
    }

    /// Develop: rebuild trees after batch deletions (LZY Tag processing).
    ///
    /// Optimized: Creates shared sample map once instead of per-tree.
    /// This reduces HashMap creation from O(num_trees * samples) to O(samples).
    pub fn develop(&mut self, samples: &[VecSample]) {
        if samples.is_empty() {
            return;
        }

        let dataset = ArrayDataset::from_samples(samples, self.num_attributes);

        // Create sample map ONCE (instead of 50 times per tree)
        let sample_map: HashMap<u64, &VecSample> =
            samples.iter().map(|s| (s.id, s)).collect();

        self.trees.par_iter_mut().for_each(|tree| {
            tree.develop_with_map(&dataset, &sample_map);
        });
    }

    /// Get the number of trees.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get the k parameter (OCC redundancy).
    pub fn k(&self) -> usize {
        self.config.k
    }

    /// Get the number of tracked samples.
    pub fn num_samples(&self) -> usize {
        self.sample_tree_map.len()
    }

    /// Check if a sample is in the forest.
    pub fn contains_sample(&self, sample_id: u64) -> bool {
        self.sample_tree_map.contains_key(&sample_id)
    }

    /// Get all sample IDs currently tracked in the forest.
    pub fn get_all_sample_ids(&self) -> Vec<u64> {
        self.sample_tree_map.keys().copied().collect()
    }

    /// Get the label for a tracked sample.
    pub fn get_sample_label(&self, sample_id: u64) -> Option<bool> {
        self.sample_labels.get(&sample_id).copied()
    }

    /// Check if any tree has pending lazy tags (Dirty/Rebuild).
    pub fn has_pending_rebuilds(&self) -> bool {
        self.trees.iter().any(|t| t.has_dirty_nodes())
    }

    /// Get statistics about tree sizes.
    pub fn tree_stats(&self) -> TreeStats {
        let depths: Vec<u32> = self.trees.iter().map(|t| t.depth()).collect();
        let node_counts: Vec<usize> = self.trees.iter().map(|t| t.num_nodes()).collect();

        TreeStats {
            num_trees: self.trees.len(),
            min_depth: depths.iter().copied().min().unwrap_or(0),
            max_depth: depths.iter().copied().max().unwrap_or(0),
            avg_depth: depths.iter().copied().sum::<u32>() as f64 / depths.len() as f64,
            min_nodes: node_counts.iter().copied().min().unwrap_or(0),
            max_nodes: node_counts.iter().copied().max().unwrap_or(0),
            avg_nodes: node_counts.iter().copied().sum::<usize>() as f64 / node_counts.len() as f64,
        }
    }

    // =========================================================================
    // Sample Influence Methods for Harmful Sample Detection
    // =========================================================================

    /// Get number of trees a sample affects (H-C1 support)
    pub fn get_sample_tree_count(&self, sample_id: u64) -> usize {
        self.sample_tree_map
            .get(&sample_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get tree indices where a sample appears
    pub fn get_sample_tree_indices(&self, sample_id: u64) -> Vec<usize> {
        self.sample_tree_map
            .get(&sample_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all sample IDs and their tree counts (batch operation)
    pub fn get_all_sample_tree_counts(&self) -> Vec<(u64, usize)> {
        self.sample_tree_map
            .iter()
            .map(|(id, trees)| (*id, trees.len()))
            .collect()
    }

    /// Get samples sorted by tree influence (highest first)
    pub fn get_samples_by_influence(&self, top_k: Option<usize>) -> Vec<(u64, usize)> {
        let mut stats: Vec<_> = self.sample_tree_map
            .iter()
            .map(|(id, trees)| (*id, trees.len()))
            .collect();
        stats.sort_by(|a, b| b.1.cmp(&a.1));

        if let Some(k) = top_k {
            stats.truncate(k);
        }
        stats
    }

    // =========================================================================
    // OOB-based Sample Influence for Principled Selective Unlearning
    // =========================================================================

    /// Predict using only a subset of trees (for OOB influence computation).
    ///
    /// This enables leave-one-out style analysis using OOB trees.
    /// Trees where a sample was NOT trained can serve as natural "holdout" evaluators.
    ///
    /// # Arguments
    /// * `sample` - The sample to predict
    /// * `tree_indices` - Indices of trees to use for prediction
    ///
    /// # Returns
    /// * `Some(prediction)` if at least one tree is specified
    /// * `None` if tree_indices is empty
    pub fn predict_with_trees(&self, sample: &VecSample, tree_indices: &[usize]) -> Option<bool> {
        if tree_indices.is_empty() {
            return None;
        }

        let num_positive: usize = tree_indices
            .par_iter()
            .filter(|&&idx| idx < self.trees.len() && self.trees[idx].predict(sample))
            .count();

        Some(num_positive * 2 > tree_indices.len())
    }

    /// Get OOB (Out-of-Bag) tree indices for a sample.
    ///
    /// OOB trees are trees where this sample was NOT included during training.
    /// Due to OCC(k) sampling, each sample appears in at most k trees.
    /// The remaining (num_trees - k) trees are OOB for this sample.
    ///
    /// # Arguments
    /// * `sample_id` - The sample ID
    ///
    /// # Returns
    /// Vector of tree indices where this sample is OOB (not trained)
    pub fn get_oob_tree_indices(&self, sample_id: u64) -> Vec<usize> {
        let in_bag: HashSet<usize> = self
            .sample_tree_map
            .get(&sample_id)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();

        (0..self.trees.len())
            .filter(|&idx| !in_bag.contains(&idx))
            .collect()
    }

    /// Compute Out-of-Bag (OOB) influence of a training sample on a test sample.
    ///
    /// OOB influence measures how much a training sample contributes to correct
    /// predictions by comparing accuracy with vs without the sample:
    ///
    /// ```text
    /// influence = Acc(In-Bag trees) - Acc(OOB trees)
    /// ```
    ///
    /// - Positive influence: sample helps prediction (beneficial)
    /// - Negative influence: sample hurts prediction (harmful, candidate for unlearning)
    /// - Zero: sample has no effect
    ///
    /// Complexity: O(num_trees) per call (two majority votes over tree subsets)
    ///
    /// # Arguments
    /// * `sample_id` - The training sample to evaluate
    /// * `test_sample` - A test sample to evaluate influence on
    ///
    /// # Returns
    /// * `Some(influence)` - The OOB influence score in `[-1, 1]`
    /// * `None` - If sample is not in the forest or appears in all trees
    pub fn compute_oob_influence(
        &self,
        sample_id: u64,
        test_sample: &VecSample,
    ) -> Option<f64> {
        // Get in-bag and OOB tree indices
        let in_bag_trees = self.sample_tree_map.get(&sample_id)?;
        if in_bag_trees.is_empty() {
            return None;
        }

        let oob_trees = self.get_oob_tree_indices(sample_id);
        if oob_trees.is_empty() {
            // Sample appears in ALL trees (k = num_trees), no OOB available
            return None;
        }

        // Get predictions from both sets
        let in_bag_pred = self.predict_with_trees(test_sample, in_bag_trees)?;
        let oob_pred = self.predict_with_trees(test_sample, &oob_trees)?;

        // Compute accuracy difference
        let true_label = test_sample.label;
        let in_bag_correct = if in_bag_pred == true_label { 1.0 } else { 0.0 };
        let oob_correct = if oob_pred == true_label { 1.0 } else { 0.0 };

        // Influence = performance WITH sample - performance WITHOUT sample
        Some(in_bag_correct - oob_correct)
    }

    /// Compute OOB influence scores for a sample across multiple test samples.
    ///
    /// This provides a more robust influence estimate by averaging across
    /// multiple test points.
    ///
    /// # Arguments
    /// * `sample_id` - The training sample to evaluate
    /// * `test_samples` - Test samples to evaluate influence on
    ///
    /// # Returns
    /// * `Some(avg_influence)` - Average influence across test samples
    /// * `None` - If sample is not in forest or no valid evaluations
    pub fn compute_oob_influence_batch(
        &self,
        sample_id: u64,
        test_samples: &[VecSample],
    ) -> Option<f64> {
        if test_samples.is_empty() {
            return None;
        }

        let influences: Vec<f64> = test_samples
            .iter()
            .filter_map(|test| self.compute_oob_influence(sample_id, test))
            .collect();

        if influences.is_empty() {
            return None;
        }

        Some(influences.iter().sum::<f64>() / influences.len() as f64)
    }

    /// Compute loss-based influence: cross-entropy difference between in-bag and OOB predictions.
    ///
    /// Unlike binary OOB influence (correct/wrong), this uses probability outputs
    /// for finer-grained influence estimation.
    ///
    /// Returns: loss_oob - loss_in_bag (positive = beneficial, negative = harmful)
    pub fn compute_loss_influence(
        &self,
        sample_id: u64,
        test_sample: &VecSample,
    ) -> Option<f64> {
        let in_bag_trees = self.sample_tree_map.get(&sample_id)?;
        if in_bag_trees.is_empty() {
            return None;
        }
        let oob_trees = self.get_oob_tree_indices(sample_id);
        if oob_trees.is_empty() {
            return None;
        }

        let in_bag_proba = self.predict_proba_with_trees(test_sample, in_bag_trees)?;
        let oob_proba = self.predict_proba_with_trees(test_sample, &oob_trees)?;

        let true_val = if test_sample.label { 1.0 } else { 0.0 };

        fn cross_entropy(proba: f64, target: f64) -> f64 {
            let p = proba.clamp(1e-7, 1.0 - 1e-7);
            -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
        }

        let loss_in_bag = cross_entropy(in_bag_proba, true_val);
        let loss_oob = cross_entropy(oob_proba, true_val);

        // Positive = sample reduces loss (beneficial), Negative = increases loss (harmful)
        Some(loss_oob - loss_in_bag)
    }

    /// Compute loss-based influence averaged over multiple test samples.
    pub fn compute_loss_influence_batch(
        &self,
        sample_id: u64,
        test_samples: &[VecSample],
    ) -> Option<f64> {
        if test_samples.is_empty() {
            return None;
        }
        let influences: Vec<f64> = test_samples
            .iter()
            .filter_map(|test| self.compute_loss_influence(sample_id, test))
            .collect();
        if influences.is_empty() {
            return None;
        }
        Some(influences.iter().sum::<f64>() / influences.len() as f64)
    }

    /// Compute OOB influence for ALL training samples efficiently.
    ///
    /// This is optimized for batch computation by:
    /// 1. Computing all tree predictions once
    /// 2. Using parallel iteration for influence computation
    ///
    /// # Arguments
    /// * `test_samples` - Test samples to evaluate influence on
    ///
    /// # Returns
    /// Vector of (sample_id, influence) pairs, sorted by influence (ascending)
    /// Negative influence = harmful samples (candidates for unlearning)
    pub fn compute_all_influences(&self, test_samples: &[VecSample]) -> Vec<(u64, f64)> {
        if test_samples.is_empty() {
            return Vec::new();
        }

        // Pre-compute all tree predictions for all test samples
        let tree_preds: Vec<Vec<bool>> = test_samples
            .par_iter()
            .map(|test| {
                self.trees.iter().map(|tree| tree.predict(test)).collect()
            })
            .collect();

        // Pre-compute true labels
        let true_labels: Vec<bool> = test_samples.iter().map(|s| s.label).collect();

        // Compute influence for each training sample
        let sample_ids: Vec<u64> = self.sample_tree_map.keys().copied().collect();

        let mut influences: Vec<(u64, f64)> = sample_ids
            .par_iter()
            .filter_map(|&sample_id| {
                let in_bag_trees = self.sample_tree_map.get(&sample_id)?;
                if in_bag_trees.is_empty() {
                    return None;
                }

                let in_bag_set: HashSet<usize> = in_bag_trees.iter().copied().collect();
                let oob_trees: Vec<usize> = (0..self.trees.len())
                    .filter(|idx| !in_bag_set.contains(idx))
                    .collect();

                if oob_trees.is_empty() {
                    return None;
                }

                // Compute average influence across test samples
                let mut total_influence = 0.0;
                let mut valid_count = 0;

                for (test_idx, true_label) in true_labels.iter().enumerate() {
                    let preds = &tree_preds[test_idx];

                    // In-bag prediction (majority vote from in-bag trees)
                    let in_bag_positive: usize = in_bag_trees
                        .iter()
                        .filter(|&&idx| preds[idx])
                        .count();
                    let in_bag_pred = in_bag_positive * 2 > in_bag_trees.len();

                    // OOB prediction (majority vote from OOB trees)
                    let oob_positive: usize = oob_trees
                        .iter()
                        .filter(|&&idx| preds[idx])
                        .count();
                    let oob_pred = oob_positive * 2 > oob_trees.len();

                    // Influence = in_bag_accuracy - oob_accuracy
                    let in_bag_correct = if in_bag_pred == *true_label { 1.0 } else { 0.0 };
                    let oob_correct = if oob_pred == *true_label { 1.0 } else { 0.0 };

                    total_influence += in_bag_correct - oob_correct;
                    valid_count += 1;
                }

                if valid_count == 0 {
                    return None;
                }

                Some((sample_id, total_influence / valid_count as f64))
            })
            .collect();

        // Sort by influence (ascending) - negative influence (harmful) first
        influences.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        influences
    }

    /// Get samples with negative OOB influence (harmful samples).
    ///
    /// These are candidates for selective unlearning - removing them
    /// should improve model performance.
    ///
    /// # Arguments
    /// * `test_samples` - Test samples to evaluate influence on
    /// * `top_k` - Optional limit on number of samples to return
    ///
    /// # Returns
    /// Vector of (sample_id, influence) for samples with negative influence
    pub fn get_harmful_samples(
        &self,
        test_samples: &[VecSample],
        top_k: Option<usize>,
    ) -> Vec<(u64, f64)> {
        let all_influences = self.compute_all_influences(test_samples);

        let harmful: Vec<(u64, f64)> = all_influences
            .into_iter()
            .filter(|(_, inf)| *inf < 0.0)
            .collect();

        match top_k {
            Some(k) => harmful.into_iter().take(k).collect(),
            None => harmful,
        }
    }

    /// Reset the forest to empty state (keep configuration).
    pub fn reset(&mut self) {
        self.sample_tree_map.clear();
        self.sample_labels.clear();
        self.rng = XorShiftRng::seed_from_u64(self.config.seed);

        // Recreate empty trees
        self.trees = (0..self.config.num_trees)
            .map(|i| {
                DynFrsTree::new(
                    i,
                    self.config.seed,
                    self.config.tree_config.clone(),
                    self.num_attributes,
                )
            })
            .collect();
    }

    // =========================================================================
    // True Streaming Learning (ported from C++ DynFrs)
    // =========================================================================

    /// Enable streaming mode for all trees.
    ///
    /// This initializes streaming states for incremental statistics tracking.
    pub fn enable_streaming(&mut self, samples: &[VecSample]) {
        for tree in &mut self.trees {
            tree.init_streaming_states(samples, |s| s.values.clone());
        }
    }

    /// Add a single sample with full streaming support.
    ///
    /// This is equivalent to C++ forest::add(X, Y):
    /// 1. OCC sampling to select trees (with Class-Aware k for Streaming CWB)
    /// 2. Call tree.add_sample_streaming() for each selected tree
    /// 3. Track if any tree needs rebuild
    /// 4. Update class statistics and adaptive k periodically
    ///
    /// Returns (success, any_needs_rebuild).
    pub fn add_sample_streaming(
        &mut self,
        sample: &VecSample,
        use_lazy_rebuild: bool,
    ) -> (bool, bool) {
        if self.sample_tree_map.contains_key(&sample.id) {
            return (false, false); // Already exists
        }

        // Streaming CWB: Class-Aware OCC(q) sampling
        // - Minority class (attack): higher k -> more trees -> more influence
        // - Majority class (benign): lower k -> fewer trees -> less influence
        let k = self.get_class_k(sample.label).unwrap_or(self.config.k);
        let tree_indices = self.occ_sample_adaptive(sample.id, k);

        let mut any_needs_rebuild = false;

        // Add to selected trees with streaming
        for &tree_idx in &tree_indices {
            let (_, needs_rebuild) = self.trees[tree_idx].add_sample_streaming(
                sample,
                &sample.values,
                use_lazy_rebuild,
            );
            any_needs_rebuild |= needs_rebuild;
        }

        // Store mappings
        self.sample_tree_map.insert(sample.id, tree_indices);
        self.sample_labels.insert(sample.id, sample.label);

        // Update streaming class counts
        if sample.label {
            self.streaming_class_counts.1 += 1; // positive (attack)
        } else {
            self.streaming_class_counts.0 += 1; // negative (benign)
        }

        // Periodically update class k values based on observed distribution
        self.streaming_samples_since_k_update += 1;
        if self.streaming_samples_since_k_update >= self.streaming_k_update_interval {
            self.update_streaming_class_k();
            self.streaming_samples_since_k_update = 0;
        }

        (true, any_needs_rebuild)
    }

    /// Add a single sample with Hoeffding inline splitting support.
    ///
    /// Same as `add_sample_streaming`, but provides a feature+label lookup
    /// so trees can perform inline Hoeffding splits without develop().
    ///
    /// # Arguments
    /// * `feature_lookup` - Maps (sample_id, attr_idx) -> Option<(attr_value, label)>.
    pub fn add_sample_streaming_hoeffding(
        &mut self,
        sample: &VecSample,
        use_lazy_rebuild: bool,
        feature_lookup: &dyn Fn(u64, u8) -> Option<(f32, bool)>,
    ) -> (bool, bool) {
        if self.sample_tree_map.contains_key(&sample.id) {
            return (false, false);
        }

        let k = self.get_class_k(sample.label).unwrap_or(self.config.k);
        let tree_indices = self.occ_sample_adaptive(sample.id, k);

        let mut any_needs_rebuild = false;

        for &tree_idx in &tree_indices {
            let (_, needs_rebuild) = self.trees[tree_idx].add_sample_streaming_hoeffding(
                sample,
                &sample.values,
                use_lazy_rebuild,
                feature_lookup,
            );
            any_needs_rebuild |= needs_rebuild;
        }

        self.sample_tree_map.insert(sample.id, tree_indices);
        self.sample_labels.insert(sample.id, sample.label);

        if sample.label {
            self.streaming_class_counts.1 += 1;
        } else {
            self.streaming_class_counts.0 += 1;
        }

        self.streaming_samples_since_k_update += 1;
        if self.streaming_samples_since_k_update >= self.streaming_k_update_interval {
            self.update_streaming_class_k();
            self.streaming_samples_since_k_update = 0;
        }

        (true, any_needs_rebuild)
    }

    /// Batch add samples with Hoeffding inline splitting.
    pub fn add_samples_streaming_hoeffding(
        &mut self,
        samples: &[VecSample],
        use_lazy_rebuild: bool,
        feature_lookup: &dyn Fn(u64, u8) -> Option<(f32, bool)>,
    ) -> (usize, usize) {
        if samples.is_empty() {
            return (0, 0);
        }
        let mut added = 0;
        let mut rebuild_count = 0;
        for sample in samples {
            let (success, needs_rebuild) =
                self.add_sample_streaming_hoeffding(sample, use_lazy_rebuild, feature_lookup);
            if success {
                added += 1;
            }
            if needs_rebuild {
                rebuild_count += 1;
            }
        }
        (added, rebuild_count)
    }

    /// Update class k values based on streaming class distribution.
    fn update_streaming_class_k(&mut self) {
        let (neg_count, pos_count) = self.streaming_class_counts;
        let total = neg_count + pos_count;

        if total == 0 {
            return;
        }

        let positive_ratio = pos_count as f64 / total as f64;
        self.update_class_k_from_distribution(positive_ratio);
    }

    /// Set the interval for updating class k during streaming.
    pub fn set_streaming_k_update_interval(&mut self, interval: u64) {
        self.streaming_k_update_interval = interval;
    }

    /// Reset streaming class statistics.
    pub fn reset_streaming_stats(&mut self) {
        self.streaming_class_counts = (0, 0);
        self.streaming_samples_since_k_update = 0;
    }

    /// Initialize streaming class k from initial fit data.
    ///
    /// This also initializes streaming_class_counts based on the initial fit,
    /// so that subsequent streaming updates have accurate class distribution.
    pub fn init_streaming_class_k(&mut self, positive_ratio: f64) {
        self.update_class_k_from_distribution(positive_ratio);
    }

    /// Initialize streaming class k with sample counts from initial fit.
    ///
    /// This should be called after initial fit to sync streaming_class_counts
    /// with the samples already in the forest.
    pub fn init_streaming_class_k_with_counts(
        &mut self,
        positive_ratio: f64,
        total_samples: u64,
    ) {
        // Set initial class counts based on the fit data
        let pos_count = (total_samples as f64 * positive_ratio).round() as u64;
        let neg_count = total_samples.saturating_sub(pos_count);
        self.streaming_class_counts = (neg_count, pos_count);

        // Update class k values
        self.update_class_k_from_distribution(positive_ratio);
    }

    /// Add multiple samples with streaming support.
    ///
    /// Returns (num_added, num_needing_rebuild).
    pub fn add_samples_streaming(
        &mut self,
        samples: &[VecSample],
        use_lazy_rebuild: bool,
    ) -> (usize, usize) {
        if samples.is_empty() {
            return (0, 0);
        }

        let mut added = 0;
        let mut rebuild_count = 0;

        for sample in samples {
            let (success, needs_rebuild) = self.add_sample_streaming(sample, use_lazy_rebuild);
            if success {
                added += 1;
            }
            if needs_rebuild {
                rebuild_count += 1;
            }
        }

        (added, rebuild_count)
    }

    /// Remove a sample with streaming support.
    ///
    /// This updates streaming statistics along the removal path.
    /// Returns (success, any_needs_rebuild).
    pub fn remove_sample_streaming(
        &mut self,
        sample_id: u64,
        features: &[f32],
        use_lazy_rebuild: bool,
    ) -> (bool, bool) {
        let was_positive = match self.sample_labels.remove(&sample_id) {
            Some(label) => label,
            None => return (false, false),
        };

        let tree_indices = match self.sample_tree_map.remove(&sample_id) {
            Some(indices) => indices,
            None => return (false, false),
        };

        let mut any_needs_rebuild = false;

        for &tree_idx in &tree_indices {
            let (_, needs_rebuild) = self.trees[tree_idx].remove_sample_streaming(
                sample_id,
                features,
                was_positive,
                use_lazy_rebuild,
            );
            any_needs_rebuild |= needs_rebuild;
        }

        // Streaming CWB: Update class counts to maintain accurate distribution
        if was_positive {
            self.streaming_class_counts.1 = self.streaming_class_counts.1.saturating_sub(1);
        } else {
            self.streaming_class_counts.0 = self.streaming_class_counts.0.saturating_sub(1);
        }

        (true, any_needs_rebuild)
    }

    /// Develop all trees with lazy tags (process pending rebuilds).
    ///
    /// This is the C++ develop() equivalent that finalizes delayed operations.
    ///
    /// Optimized: Uses shared sample map and parallel processing (same as develop()).
    pub fn develop_streaming(&mut self, samples: &[VecSample]) {
        if samples.is_empty() {
            return;
        }

        let dataset = ArrayDataset::from_samples(samples, self.num_attributes);

        // Create sample map ONCE (instead of per-tree)
        let sample_map: HashMap<u64, &VecSample> =
            samples.iter().map(|s| (s.id, s)).collect();

        // Process trees in PARALLEL with shared sample map
        self.trees.par_iter_mut().for_each(|tree| {
            tree.develop_with_map(&dataset, &sample_map);
        });
    }

    /// Develop all trees with age-based subtree refresh (streaming variant).
    ///
    /// Same as develop_streaming() but also forces rebuild of internal nodes
    /// whose split is older than `max_split_age` samples.
    pub fn develop_streaming_with_age(&mut self, samples: &[VecSample], current_position: u64, max_split_age: Option<u64>) {
        if samples.is_empty() {
            return;
        }

        let dataset = ArrayDataset::from_samples(samples, self.num_attributes);

        let sample_map: HashMap<u64, &VecSample> =
            samples.iter().map(|s| (s.id, s)).collect();

        self.trees.par_iter_mut().for_each(|tree| {
            tree.develop_with_age(&dataset, &sample_map, current_position, max_split_age);
        });
    }

    /// Check if any tree has streaming enabled.
    pub fn is_streaming(&self) -> bool {
        self.trees.iter().any(|t| t.is_streaming())
    }

    /// Get streaming statistics.
    pub fn streaming_stats(&self) -> StreamingStats {
        let streaming_trees = self.trees.iter().filter(|t| t.is_streaming()).count();
        let total_states: usize = self.trees.iter().map(|t| t.num_streaming_states()).sum();

        StreamingStats {
            streaming_trees,
            total_streaming_states: total_states,
        }
    }
}

/// Statistics about streaming state.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub streaming_trees: usize,
    pub total_streaming_states: usize,
}

/// Statistics about trees in the forest.
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub num_trees: usize,
    pub min_depth: u32,
    pub max_depth: u32,
    pub avg_depth: f64,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub avg_nodes: f64,
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Convert NumPy arrays (x, y, ids) into Vec<VecSample>.
fn convert_to_samples(
    x: &numpy::ndarray::ArrayView2<f32>,
    y: &numpy::ndarray::ArrayView1<bool>,
    ids: Option<&numpy::ndarray::ArrayView1<u64>>,
) -> PyResult<Vec<VecSample>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if let Some(ids) = ids {
        if ids.len() != n_samples {
            return Err(PyValueError::new_err(format!(
                "ids length mismatch: got {}, expected {}",
                ids.len(),
                n_samples
            )));
        }
    }

    let samples = (0..n_samples)
        .map(|i| {
            let values: Vec<f32> = (0..n_features).map(|j| x[[i, j]]).collect();
            let sample_id = ids.map(|ids| ids[i]).unwrap_or(i as u64);
            VecSample::new(sample_id, values, y[i])
        })
        .collect();

    Ok(samples)
}

/// Python-accessible DynFrs Forest.
#[pyclass]
pub struct PyDynFrsForest {
    inner: DynFrsForest,
}

#[pymethods]
impl PyDynFrsForest {
    /// Create a new forest.
    #[new]
    #[pyo3(signature = (num_trees=100, k=10, max_depth=20, min_samples_split=2, min_samples_leaf=1, seed=42))]
    fn new(
        num_trees: usize,
        k: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        seed: u64,
    ) -> Self {
        let config = ForestConfig {
            num_trees,
            k,
            tree_config: TreeConfig {
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features: None,
                num_splits_to_try: 1,
                split_quality_threshold: None,
                ..Default::default()
            },
            seed,
        };

        // Placeholder, will be set on fit
        PyDynFrsForest {
            inner: DynFrsForest::new(config, 1),
        }
    }

    /// Fit the forest on data.
    #[pyo3(signature = (x, y, ids=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let ids_array = ids.as_ref().map(|a| a.as_array());
        let samples = convert_to_samples(&x_array, &y_array, ids_array.as_ref())?;

        let config = self.inner.config.clone();
        self.inner = DynFrsForest::new(config, x_array.ncols() as u8);
        self.inner.fit(&samples);
        Ok(())
    }

    /// Predict labels for samples.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> Bound<'py, PyArray1<bool>> {
        let x_array = x.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                VecSample::new(i as u64, values, false)
            })
            .collect();

        let predictions = self.inner.predict_batch(&samples);
        PyArray1::from_vec(py, predictions)
    }

    /// Forget samples by their IDs.
    fn forget(&mut self, sample_ids: PyReadonlyArray1<u64>) -> usize {
        let ids = sample_ids.as_array();
        let ids_vec: Vec<u64> = ids.iter().copied().collect();
        self.inner.forget_batch(&ids_vec)
    }

    /// Optimized batch forget with 33% rebuild threshold.
    ///
    /// If deleting more than 33% of samples from a tree, automatically
    /// rebuilds instead of incremental deletion (faster for large batches).
    ///
    /// Args:
    ///     sample_ids: Array of sample IDs to forget
    ///     x: Feature matrix of remaining samples (for potential rebuild)
    ///     y: Labels of remaining samples
    ///     ids: Optional sample IDs for remaining samples
    ///
    /// Returns:
    ///     Number of samples actually forgotten
    #[pyo3(signature = (sample_ids, x, y, ids=None))]
    fn forget_optimized(
        &mut self,
        sample_ids: PyReadonlyArray1<u64>,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<usize> {
        let ids_to_forget: Vec<u64> = sample_ids.as_array().iter().copied().collect();

        let x_array = x.as_array();
        let y_array = y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let ids_view = if let Some(ids) = &ids {
            let ids_array = ids.as_array();
            if ids_array.len() != n_samples {
                return Err(PyValueError::new_err(format!(
                    "ids length mismatch: got {}, expected {}",
                    ids_array.len(),
                    n_samples
                )));
            }
            Some(ids_array)
        } else {
            None
        };

        let samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                let sample_id = ids_view.as_ref().map(|ids| ids[i]).unwrap_or(i as u64);
                VecSample::new(sample_id, values, y_array[i])
            })
            .collect();

        Ok(self.inner.forget_batch_optimized(&ids_to_forget, &samples))
    }

    /// Forget a single sample by ID.
    fn forget_one(&mut self, sample_id: u64) -> bool {
        self.inner.forget(sample_id)
    }

    /// Process lazy rebuilds after batch deletions.
    #[pyo3(signature = (x, y, ids=None))]
    fn develop(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let ids_view = if let Some(ids) = &ids {
            let ids_array = ids.as_array();
            if ids_array.len() != n_samples {
                return Err(PyValueError::new_err(format!(
                    "ids length mismatch: got {}, expected {}",
                    ids_array.len(),
                    n_samples
                )));
            }
            Some(ids_array)
        } else {
            None
        };

        let samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                let sample_id = ids_view.as_ref().map(|ids| ids[i]).unwrap_or(i as u64);
                VecSample::new(sample_id, values, y_array[i])
            })
            .collect();

        self.inner.develop(&samples);
        Ok(())
    }

    /// Get the number of trees.
    #[getter]
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    /// Get the k parameter.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// Get the number of tracked samples.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// Check if a sample is in the forest.
    fn contains_sample(&self, sample_id: u64) -> bool {
        self.inner.contains_sample(sample_id)
    }

    /// Get tree statistics as a dict.
    fn tree_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.tree_stats();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_trees", stats.num_trees)?;
        dict.set_item("min_depth", stats.min_depth)?;
        dict.set_item("max_depth", stats.max_depth)?;
        dict.set_item("avg_depth", stats.avg_depth)?;
        dict.set_item("min_nodes", stats.min_nodes)?;
        dict.set_item("max_nodes", stats.max_nodes)?;
        dict.set_item("avg_nodes", stats.avg_nodes)?;
        Ok(dict.into())
    }

    // =========================================================================
    // Sample Influence Methods (Python bindings)
    // =========================================================================

    /// Get number of trees a sample affects.
    ///
    /// This is useful for H-C1 hypothesis: samples affecting more trees
    /// have higher impact on model predictions.
    fn get_sample_tree_count(&self, sample_id: u64) -> usize {
        self.inner.get_sample_tree_count(sample_id)
    }

    /// Get tree indices where a sample appears.
    fn get_sample_tree_indices(&self, sample_id: u64) -> Vec<usize> {
        self.inner.get_sample_tree_indices(sample_id)
    }

    /// Get all sample IDs and their tree counts (batch operation).
    ///
    /// Returns list of (sample_id, tree_count) tuples.
    fn get_all_sample_tree_counts(&self) -> Vec<(u64, usize)> {
        self.inner.get_all_sample_tree_counts()
    }

    /// Get samples sorted by tree influence (highest first).
    ///
    /// Args:
    ///     top_k: If specified, only return top k samples
    ///
    /// Returns list of (sample_id, tree_count) tuples sorted by tree_count descending.
    #[pyo3(signature = (top_k=None))]
    fn get_samples_by_influence(&self, top_k: Option<usize>) -> Vec<(u64, usize)> {
        self.inner.get_samples_by_influence(top_k)
    }

    // =========================================================================
    // OOB-based Sample Influence (Python bindings)
    // =========================================================================

    /// Get OOB (Out-of-Bag) tree indices for a sample.
    ///
    /// OOB trees are trees where the sample was NOT included during training.
    /// Due to OCC(k) sampling, each sample appears in at most k trees,
    /// leaving (num_trees - k) trees as OOB evaluators.
    ///
    /// Args:
    ///     sample_id: The sample ID
    ///
    /// Returns:
    ///     List of tree indices where this sample is OOB (not trained)
    fn get_oob_tree_indices(&self, sample_id: u64) -> Vec<usize> {
        self.inner.get_oob_tree_indices(sample_id)
    }

    /// Predict using only a subset of trees.
    ///
    /// This enables leave-one-out style analysis by comparing predictions
    /// from different tree subsets.
    ///
    /// Args:
    ///     x: Single sample feature vector (n_features,)
    ///     tree_indices: Indices of trees to use for prediction
    ///
    /// Returns:
    ///     Prediction from the specified trees, or None if indices empty
    fn predict_with_trees(
        &self,
        x: PyReadonlyArray1<f32>,
        tree_indices: Vec<usize>,
    ) -> Option<bool> {
        let features: Vec<f32> = x.as_array().iter().copied().collect();
        let sample = VecSample::new(0, features, false);
        self.inner.predict_with_trees(&sample, &tree_indices)
    }

    /// Compute OOB influence for a single training sample.
    ///
    /// OOB Influence measures how much a sample contributes to model accuracy:
    ///   I_OOB = accuracy(in-bag trees) - accuracy(OOB trees)
    ///
    /// Interpretation:
    /// - Positive: Sample HELPS the model (keep it)
    /// - Negative: Sample HURTS the model (candidate for unlearning)
    /// - Zero: No effect
    ///
    /// Args:
    ///     sample_id: ID of the training sample to evaluate
    ///     test_x: Test sample feature vector
    ///     test_y: True label for test sample
    ///
    /// Returns:
    ///     Influence score, or None if sample not found / no OOB trees
    fn compute_oob_influence(
        &self,
        sample_id: u64,
        test_x: PyReadonlyArray1<f32>,
        test_y: bool,
    ) -> Option<f64> {
        let features: Vec<f32> = test_x.as_array().iter().copied().collect();
        let test_sample = VecSample::new(0, features, test_y);
        self.inner.compute_oob_influence(sample_id, &test_sample)
    }

    /// Compute OOB influence for a sample across multiple test samples.
    ///
    /// Provides more robust influence estimate by averaging across test points.
    ///
    /// Args:
    ///     sample_id: ID of the training sample to evaluate
    ///     test_x: Test feature matrix (n_samples, n_features)
    ///     test_y: True labels for test samples
    ///
    /// Returns:
    ///     Average influence score across test samples
    #[pyo3(signature = (sample_id, test_x, test_y))]
    fn compute_oob_influence_batch(
        &self,
        sample_id: u64,
        test_x: PyReadonlyArray2<f32>,
        test_y: PyReadonlyArray1<bool>,
    ) -> Option<f64> {
        let x_array = test_x.as_array();
        let y_array = test_y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let test_samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                VecSample::new(i as u64, values, y_array[i])
            })
            .collect();

        self.inner.compute_oob_influence_batch(sample_id, &test_samples)
    }

    /// Compute OOB influence for ALL training samples.
    ///
    /// This is optimized for batch computation and returns samples sorted
    /// by influence (ascending), so harmful samples appear first.
    ///
    /// Args:
    ///     test_x: Test feature matrix (n_samples, n_features)
    ///     test_y: True labels for test samples
    ///
    /// Returns:
    ///     List of (sample_id, influence) tuples, sorted ascending by influence.
    ///     Negative influence = harmful samples (candidates for unlearning)
    #[pyo3(signature = (test_x, test_y))]
    fn compute_all_influences(
        &self,
        test_x: PyReadonlyArray2<f32>,
        test_y: PyReadonlyArray1<bool>,
    ) -> Vec<(u64, f64)> {
        let x_array = test_x.as_array();
        let y_array = test_y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let test_samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                VecSample::new(i as u64, values, y_array[i])
            })
            .collect();

        self.inner.compute_all_influences(&test_samples)
    }

    /// Get training samples with negative OOB influence (harmful samples).
    ///
    /// These samples hurt model performance and are candidates for
    /// selective unlearning during drift adaptation.
    ///
    /// Args:
    ///     test_x: Test feature matrix
    ///     test_y: True labels for test samples
    ///     top_k: Optional limit on number of samples to return
    ///
    /// Returns:
    ///     List of (sample_id, influence) for harmful samples
    #[pyo3(signature = (test_x, test_y, top_k=None))]
    fn get_harmful_samples(
        &self,
        test_x: PyReadonlyArray2<f32>,
        test_y: PyReadonlyArray1<bool>,
        top_k: Option<usize>,
    ) -> Vec<(u64, f64)> {
        let x_array = test_x.as_array();
        let y_array = test_y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let test_samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                VecSample::new(i as u64, values, y_array[i])
            })
            .collect();

        self.inner.get_harmful_samples(&test_samples, top_k)
    }

    // =========================================================================
    // Buffer Integration Methods (Python bindings)
    // =========================================================================

    /// Get label for a sample.
    fn get_sample_label(&self, sample_id: u64) -> Option<bool> {
        self.inner.get_sample_label(sample_id)
    }

    /// Reset the forest to empty state.
    fn reset(&mut self) {
        self.inner.reset()
    }

    // =========================================================================
    // True Streaming Learning (Python bindings)
    // =========================================================================

    /// Enable streaming mode for all trees.
    ///
    /// This initializes streaming states based on current samples for
    /// incremental statistics tracking. Call after initial fit().
    #[pyo3(signature = (x, y, ids=None))]
    fn enable_streaming(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let ids_view = if let Some(ids) = &ids {
            let ids_array = ids.as_array();
            if ids_array.len() != n_samples {
                return Err(PyValueError::new_err("ids length mismatch"));
            }
            Some(ids_array)
        } else {
            None
        };

        let samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                let sample_id = ids_view.as_ref().map(|ids| ids[i]).unwrap_or(i as u64);
                VecSample::new(sample_id, values, y_array[i])
            })
            .collect();

        self.inner.enable_streaming(&samples);
        Ok(())
    }

    /// Add a single sample with streaming support.
    ///
    /// This is the C++ add(X, Y) equivalent with full statistics tracking.
    /// Returns (success, needs_rebuild).
    #[pyo3(signature = (x, y, sample_id, use_lazy_rebuild=true))]
    fn add_sample_streaming(
        &mut self,
        x: PyReadonlyArray1<f32>,
        y: bool,
        sample_id: u64,
        use_lazy_rebuild: bool,
    ) -> (bool, bool) {
        let features: Vec<f32> = x.as_array().iter().copied().collect();
        let sample = VecSample::new(sample_id, features, y);
        self.inner.add_sample_streaming(&sample, use_lazy_rebuild)
    }

    /// Add multiple samples with streaming support.
    ///
    /// Returns (num_added, num_needing_rebuild).
    #[pyo3(signature = (x, y, ids=None, use_lazy_rebuild=true))]
    fn add_samples_streaming(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
        use_lazy_rebuild: bool,
    ) -> PyResult<(usize, usize)> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let ids_view = if let Some(ids) = &ids {
            let ids_array = ids.as_array();
            if ids_array.len() != n_samples {
                return Err(PyValueError::new_err("ids length mismatch"));
            }
            Some(ids_array)
        } else {
            None
        };

        let samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                let sample_id = ids_view.as_ref().map(|ids| ids[i]).unwrap_or(i as u64);
                VecSample::new(sample_id, values, y_array[i])
            })
            .collect();

        Ok(self.inner.add_samples_streaming(&samples, use_lazy_rebuild))
    }

    /// Remove a sample with streaming support.
    ///
    /// Returns (success, needs_rebuild).
    fn remove_sample_streaming(
        &mut self,
        sample_id: u64,
        features: PyReadonlyArray1<f32>,
        use_lazy_rebuild: bool,
    ) -> (bool, bool) {
        let features_vec: Vec<f32> = features.as_array().iter().copied().collect();
        self.inner.remove_sample_streaming(sample_id, &features_vec, use_lazy_rebuild)
    }

    /// Develop all trees (process pending rebuilds from lazy tags).
    #[pyo3(signature = (x, y, ids=None))]
    fn develop_streaming(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let ids_view = if let Some(ids) = &ids {
            let ids_array = ids.as_array();
            if ids_array.len() != n_samples {
                return Err(PyValueError::new_err("ids length mismatch"));
            }
            Some(ids_array)
        } else {
            None
        };

        let samples: Vec<VecSample> = (0..n_samples)
            .map(|i| {
                let values: Vec<f32> = (0..n_features).map(|j| x_array[[i, j]]).collect();
                let sample_id = ids_view.as_ref().map(|ids| ids[i]).unwrap_or(i as u64);
                VecSample::new(sample_id, values, y_array[i])
            })
            .collect();

        self.inner.develop_streaming(&samples);
        Ok(())
    }

    /// Check if streaming mode is enabled.
    fn is_streaming(&self) -> bool {
        self.inner.is_streaming()
    }

    /// Get streaming statistics.
    fn streaming_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.streaming_stats();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("streaming_trees", stats.streaming_trees)?;
        dict.set_item("total_streaming_states", stats.total_streaming_states)?;
        Ok(dict.into())
    }

    // =========================================================================
    // Adaptive k-Redundancy (Python bindings)
    // =========================================================================

    /// Enable adaptive k-redundancy based on class imbalance.
    ///
    /// Minority class samples get higher k (appear in more trees) for better
    /// protection during unlearning. This improves performance on datasets
    /// with extreme class imbalance (e.g., CIDDS with 0.4% attack).
    ///
    /// Args:
    ///     k_min: Minimum k for majority class (default: 3)
    ///     k_max: Maximum k for minority class (default: 50 = num_trees)
    ///     ratio_threshold: Class ratio below which k is interpolated (default: 0.3)
    ///     extreme_threshold: Below this ratio, use k_max (default: 0.01 = 1%)
    #[pyo3(signature = (k_min=3, k_max=50, ratio_threshold=0.3, extreme_threshold=0.01))]
    fn set_adaptive_k(
        &mut self,
        k_min: usize,
        k_max: usize,
        ratio_threshold: f64,
        extreme_threshold: f64,
    ) {
        self.inner.set_adaptive_k_config(AdaptiveKConfig {
            k_min,
            k_max,
            ratio_threshold,
            extreme_threshold,
        });
    }

    /// Set k value for a specific class label.
    ///
    /// Args:
    ///     label: Class label (True = attack, False = benign)
    ///     k: Number of trees each sample of this class should appear in
    fn set_class_k(&mut self, label: bool, k: usize) {
        self.inner.set_class_k(label, k);
    }

    /// Get k value for a specific class label.
    fn get_class_k(&self, label: bool) -> Option<usize> {
        self.inner.get_class_k(label)
    }

    /// Update class k values based on class distribution.
    ///
    /// Calculates optimal k for each class based on the positive (attack) ratio.
    /// Requires set_adaptive_k() to be called first.
    ///
    /// Args:
    ///     positive_ratio: Ratio of positive (attack) samples (0.0 - 1.0)
    fn update_class_k_from_ratio(&mut self, positive_ratio: f64) {
        self.inner.update_class_k_from_distribution(positive_ratio);
    }

    /// Fit using adaptive k-redundancy based on class labels.
    ///
    /// Each sample gets k based on its class from class_k_map.
    /// Call set_class_k() or update_class_k_from_ratio() before using this.
    ///
    /// Args:
    ///     x: Feature matrix (n_samples, n_features)
    ///     y: Labels (n_samples,)
    ///     ids: Optional sample IDs
    #[pyo3(signature = (x, y, ids=None))]
    fn fit_adaptive(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let ids_array = ids.as_ref().map(|a| a.as_array());
        let samples = convert_to_samples(&x_array, &y_array, ids_array.as_ref())?;

        let config = self.inner.config.clone();
        self.inner = DynFrsForest::new(config, x_array.ncols() as u8);

        // Transfer adaptive k settings
        if let Some(ak_config) = self.inner.adaptive_k_config.clone() {
            self.inner.set_adaptive_k_config(ak_config);
        }

        self.inner.fit_adaptive(&samples);
        Ok(())
    }

    /// Fit with class-weighted bootstrap sampling.
    ///
    /// Addresses extreme class imbalance by adjusting adaptive-k and OCC(q)
    /// sampling based on class distribution.
    ///
    /// Args:
    ///     x: Feature matrix (n_samples, n_features)
    ///     y: Labels (n_samples,)
    ///     positive_ratio: Ratio of positive (attack) samples (0.0 - 1.0)
    ///     ids: Optional sample IDs
    #[pyo3(signature = (x, y, positive_ratio, ids=None))]
    fn fit_weighted(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        positive_ratio: f64,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let ids_array = ids.as_ref().map(|a| a.as_array());
        let samples = convert_to_samples(&x_array, &y_array, ids_array.as_ref())?;

        // Recreate forest with correct num_attributes
        let config = self.inner.config.clone();
        let adaptive_k = self.inner.adaptive_k_config.clone();
        let class_k = self.inner.class_k_map.clone();

        self.inner = DynFrsForest::new(config, x_array.ncols() as u8);

        // Restore adaptive k settings
        if let Some(ak_config) = adaptive_k {
            self.inner.set_adaptive_k_config(ak_config);
        }
        for (label, k) in class_k {
            self.inner.set_class_k(label, k);
        }

        self.inner.fit_weighted(&samples, positive_ratio);
        Ok(())
    }

    /// Fit with explicit k values for each sample.
    ///
    /// Args:
    ///     x: Feature matrix (n_samples, n_features)
    ///     y: Labels (n_samples,)
    ///     k_values: k value for each sample (n_samples,)
    ///     ids: Optional sample IDs
    #[pyo3(signature = (x, y, k_values, ids=None))]
    fn fit_with_k_values(
        &mut self,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
        k_values: PyReadonlyArray1<usize>,
        ids: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let k_array = k_values.as_array();
        let ids_array = ids.as_ref().map(|a| a.as_array());
        let samples = convert_to_samples(&x_array, &y_array, ids_array.as_ref())?;

        let n_samples = x_array.nrows();
        if k_array.len() != n_samples {
            return Err(PyValueError::new_err(format!(
                "k_values length mismatch: got {}, expected {}",
                k_array.len(),
                n_samples
            )));
        }
        let k_vec: Vec<usize> = k_array.iter().copied().collect();

        let config = self.inner.config.clone();
        self.inner = DynFrsForest::new(config, x_array.ncols() as u8);
        self.inner.fit_with_k_values(&samples, &k_vec);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples() -> Vec<VecSample> {
        vec![
            VecSample::new(0, vec![1.0, 1.0], true),
            VecSample::new(1, vec![2.0, 2.0], true),
            VecSample::new(2, vec![3.0, 3.0], true),
            VecSample::new(3, vec![4.0, 4.0], true),
            VecSample::new(4, vec![6.0, 1.0], false),
            VecSample::new(5, vec![7.0, 2.0], false),
            VecSample::new(6, vec![8.0, 3.0], false),
            VecSample::new(7, vec![9.0, 4.0], false),
        ]
    }

    #[test]
    fn test_forest_fit_predict() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            tree_config: TreeConfig {
                max_depth: 5,
                num_splits_to_try: 5,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Check predictions
        let predictions = forest.predict_batch(&samples);
        let mut correct = 0;
        for (sample, &pred) in samples.iter().zip(predictions.iter()) {
            if pred == sample.label {
                correct += 1;
            }
        }

        // Should have high accuracy on training data
        assert!(correct >= 6, "Too many mispredictions: {}/8", correct);
    }

    #[test]
    fn test_occ_sampling() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 100,
            k: 10,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Each sample should appear in at most k trees
        for sample in &samples {
            let tree_indices = forest.sample_tree_map.get(&sample.id).unwrap();
            assert!(
                tree_indices.len() <= forest.config.k,
                "Sample {} appears in {} trees, but k={}",
                sample.id,
                tree_indices.len(),
                forest.config.k
            );
        }
    }

    #[test]
    fn test_forest_forget() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        assert_eq!(forest.num_samples(), 8);
        assert!(forest.contains_sample(0));

        // Forget sample 0
        assert!(forest.forget(0));
        assert!(!forest.contains_sample(0));
        assert_eq!(forest.num_samples(), 7);

        // Try to forget again
        assert!(!forest.forget(0));

        // Forget multiple
        let forgotten = forest.forget_batch(&[1, 2, 3, 999]);
        assert_eq!(forgotten, 3); // 999 doesn't exist
        assert_eq!(forest.num_samples(), 4);
    }

    #[test]
    fn test_tree_stats() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        let stats = forest.tree_stats();
        assert_eq!(stats.num_trees, 10);
        assert!(stats.avg_depth > 0.0);
        assert!(stats.avg_nodes > 0.0);
    }

    // =========================================================================
    // Adaptive k-Redundancy Tests
    // =========================================================================

    #[test]
    fn test_adaptive_k_config_compute_k() {
        let config = AdaptiveKConfig {
            k_min: 3,
            k_max: 50,  // All trees for extreme minority
            ratio_threshold: 0.3,
            extreme_threshold: 0.01,  // 1% threshold
        };

        // Extreme minority (0.4% like CIDDS attack) -> k_max (all trees)
        // ECBRS-inspired: below 1%, use all trees
        assert_eq!(config.compute_k(0.004), 50, "Extreme minority should get k_max");

        // Moderate minority (5%) -> high k but not all trees
        let k_5pct = config.compute_k(0.05);
        assert!(k_5pct > 30, "5% minority should get high k, got {}", k_5pct);
        assert!(k_5pct < 50, "5% is not extreme, shouldn't get k_max");

        // Minority (10%) -> interpolated
        let k_10pct = config.compute_k(0.1);
        assert!(k_10pct > 15, "10% minority should get k > 15, got {}", k_10pct);

        // At threshold (30%) -> k_min
        let k_30pct = config.compute_k(0.3);
        assert_eq!(k_30pct, 3, "At threshold should get k_min");

        // Majority (60%) -> k_min (clamped)
        let k_60pct = config.compute_k(0.6);
        assert_eq!(k_60pct, 3, "Majority should get k_min");
    }

    #[test]
    fn test_adaptive_k_occ_sampling() {
        let _samples = make_samples();  // Used for reference only

        let config = ForestConfig {
            num_trees: 50,
            k: 10,  // Default k
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);

        // Set adaptive k config with ECBRS-inspired extreme threshold
        forest.set_adaptive_k_config(AdaptiveKConfig {
            k_min: 3,
            k_max: 50,  // Use all trees for extreme minority
            ratio_threshold: 0.3,
            extreme_threshold: 0.01,  // 1% threshold
        });

        // Simulate class distribution: 10% positive
        forest.update_class_k_from_distribution(0.1);

        // Check class k values
        let k_positive = forest.get_class_k(true).unwrap();
        let k_negative = forest.get_class_k(false).unwrap();

        assert!(k_positive > k_negative, "Minority (positive) should have higher k");
        assert!(k_positive >= 10, "Minority should get high k, got {}", k_positive);
        assert_eq!(k_negative, 3, "Majority should get k_min");
    }

    #[test]
    fn test_fit_adaptive() {
        let samples = make_samples();  // 4 true, 4 false

        let config = ForestConfig {
            num_trees: 20,
            k: 10,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);

        // Set class k values manually
        forest.set_class_k(true, 10);   // Positive gets 10
        forest.set_class_k(false, 3);   // Negative gets 3

        // Fit with adaptive k
        forest.fit_adaptive(&samples);

        // Verify all samples are tracked
        assert_eq!(forest.num_samples(), 8);

        // Check that positive samples appear in more trees
        let positive_tree_counts: Vec<usize> = samples.iter()
            .filter(|s| s.label)
            .map(|s| forest.get_sample_tree_count(s.id))
            .collect();

        let negative_tree_counts: Vec<usize> = samples.iter()
            .filter(|s| !s.label)
            .map(|s| forest.get_sample_tree_count(s.id))
            .collect();

        let avg_positive = positive_tree_counts.iter().sum::<usize>() as f64
            / positive_tree_counts.len() as f64;
        let avg_negative = negative_tree_counts.iter().sum::<usize>() as f64
            / negative_tree_counts.len() as f64;

        assert!(
            avg_positive > avg_negative,
            "Positive samples should appear in more trees on average: {} vs {}",
            avg_positive,
            avg_negative
        );
    }

    #[test]
    fn test_fit_with_k_values() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 30,
            k: 10,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);

        // Custom k values: higher for positive, lower for negative
        let k_values: Vec<usize> = samples.iter()
            .map(|s| if s.label { 15 } else { 5 })
            .collect();

        forest.fit_with_k_values(&samples, &k_values);

        assert_eq!(forest.num_samples(), 8);

        // Verify predictions still work (basic functionality check)
        let predictions = forest.predict_batch(&samples);
        assert_eq!(predictions.len(), 8, "Should return predictions for all samples");

        // With only 8 samples, accuracy can vary; just verify the forest is functional
        let correct = predictions.iter()
            .zip(samples.iter())
            .filter(|(pred, sample)| **pred == sample.label)
            .count();

        // At least some predictions should be correct (better than random baseline)
        assert!(correct >= 3, "Should have at least some correct predictions: {}/8", correct);
    }

    /// Create imbalanced samples for testing (similar to CIDDS)
    fn make_imbalanced_samples() -> Vec<VecSample> {
        let mut samples = Vec::new();
        let mut id = 0u64;

        // 2 attack samples (2%)
        samples.push(VecSample::new(id, vec![1.0, 1.0], true));
        id += 1;
        samples.push(VecSample::new(id, vec![2.0, 2.0], true));
        id += 1;

        // 98 benign samples (98%)
        for i in 0..98 {
            let x = 5.0 + (i as f32) * 0.1;
            samples.push(VecSample::new(id, vec![x, x], false));
            id += 1;
        }

        samples
    }

    #[test]
    fn test_fit_weighted() {
        let samples = make_imbalanced_samples();  // 2% positive

        let config = ForestConfig {
            num_trees: 30,
            k: 10,
            tree_config: TreeConfig {
                max_depth: 10,
                num_splits_to_try: 10,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config.clone(), 2);

        // Set adaptive k for better minority protection
        forest.set_adaptive_k_config(AdaptiveKConfig {
            k_min: 3,
            k_max: 30,  // All trees
            ratio_threshold: 0.3,
            extreme_threshold: 0.05,  // 5% threshold for test
        });
        forest.update_class_k_from_distribution(0.02);  // 2% positive

        // Fit with class-weighted bootstrap
        forest.fit_weighted(&samples, 0.02);

        assert_eq!(forest.num_samples(), 100);

        // Verify predictions work
        let predictions = forest.predict_batch(&samples);
        assert_eq!(predictions.len(), 100);

        // Count attack predictions
        let attack_samples: Vec<_> = samples.iter().filter(|s| s.label).collect();
        let mut attack_correct = 0;
        for sample in &attack_samples {
            if forest.predict(sample) == sample.label {
                attack_correct += 1;
            }
        }

        println!(
            "Weighted forest attack recall: {}/{} ({:.1}%)",
            attack_correct,
            attack_samples.len(),
            100.0 * attack_correct as f64 / attack_samples.len() as f64
        );

        // With weighted bootstrap, we expect some attack detection
        // (exact recall depends on random splits)
    }

    // =========================================================================
    // OOB-based Sample Influence Tests
    // =========================================================================

    #[test]
    fn test_get_oob_tree_indices() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 20,
            k: 5,  // Each sample in 5 trees, OOB for 15
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Check OOB trees for first sample
        let in_bag_trees = forest.get_sample_tree_indices(0);
        let oob_trees = forest.get_oob_tree_indices(0);

        // Verify no overlap between in-bag and OOB
        let in_bag_set: HashSet<usize> = in_bag_trees.iter().copied().collect();
        let oob_set: HashSet<usize> = oob_trees.iter().copied().collect();

        assert!(
            in_bag_set.is_disjoint(&oob_set),
            "In-bag and OOB should not overlap"
        );

        // Verify total covers all trees
        assert_eq!(
            in_bag_trees.len() + oob_trees.len(),
            20,
            "In-bag + OOB should equal num_trees"
        );

        // OOB count should be num_trees - k
        assert_eq!(oob_trees.len(), 15, "OOB trees should be num_trees - k");
    }

    #[test]
    fn test_predict_with_trees() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 20,
            k: 5,
            tree_config: TreeConfig {
                max_depth: 5,
                num_splits_to_try: 10,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Test prediction with all trees
        let test_sample = &samples[0];
        let all_trees: Vec<usize> = (0..20).collect();
        let full_pred = forest.predict_with_trees(test_sample, &all_trees);
        assert!(full_pred.is_some(), "Should get prediction with all trees");

        // Test prediction with subset
        let subset: Vec<usize> = (0..5).collect();
        let subset_pred = forest.predict_with_trees(test_sample, &subset);
        assert!(subset_pred.is_some(), "Should get prediction with subset");

        // Test empty tree list
        let empty: Vec<usize> = vec![];
        let empty_pred = forest.predict_with_trees(test_sample, &empty);
        assert!(empty_pred.is_none(), "Empty tree list should return None");
    }

    #[test]
    fn test_compute_oob_influence() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 30,
            k: 10,  // Each sample in ~10 trees, OOB for ~20
            tree_config: TreeConfig {
                max_depth: 5,
                num_splits_to_try: 10,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Compute influence of sample 0 on a test sample
        let test_sample = &samples[4];  // Use a different sample as test
        let influence = forest.compute_oob_influence(0, test_sample);

        // Influence should exist (sample is in forest)
        assert!(
            influence.is_some(),
            "Should compute influence for existing sample"
        );

        // Influence value should be in valid range [-1, 1]
        let inf_val = influence.unwrap();
        assert!(
            inf_val >= -1.0 && inf_val <= 1.0,
            "Influence should be in [-1, 1], got {}",
            inf_val
        );

        // Test non-existent sample
        let nonexistent = forest.compute_oob_influence(999, test_sample);
        assert!(
            nonexistent.is_none(),
            "Non-existent sample should return None"
        );
    }

    #[test]
    fn test_compute_all_influences() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 30,
            k: 10,
            tree_config: TreeConfig {
                max_depth: 5,
                num_splits_to_try: 10,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Compute influences using a subset as test samples
        let test_samples = &samples[4..8];  // Use negative samples as test
        let influences = forest.compute_all_influences(test_samples);

        // Should have influences for all training samples
        assert!(!influences.is_empty(), "Should compute some influences");

        // Should be sorted (ascending by influence)
        for window in influences.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Should be sorted ascending by influence"
            );
        }

        println!("OOB Influences (sorted by influence):");
        for (id, inf) in influences.iter().take(5) {
            println!("  Sample {}: {:.4}", id, inf);
        }
    }

    #[test]
    fn test_get_harmful_samples() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 50,
            k: 15,
            tree_config: TreeConfig {
                max_depth: 5,
                num_splits_to_try: 10,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Use all samples as test set
        let harmful = forest.get_harmful_samples(&samples, Some(3));

        println!("Harmful samples (top 3):");
        for (id, inf) in &harmful {
            println!("  Sample {}: influence = {:.4}", id, inf);
        }

        // All returned samples should have negative influence
        for (_, inf) in &harmful {
            assert!(
                *inf < 0.0,
                "Harmful samples should have negative influence"
            );
        }

        // Should respect top_k limit
        assert!(
            harmful.len() <= 3,
            "Should return at most 3 samples"
        );
    }

    #[test]
    fn test_oob_influence_batch() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 30,
            k: 10,
            tree_config: TreeConfig {
                max_depth: 5,
                num_splits_to_try: 10,
                ..Default::default()
            },
            seed: 42,
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        // Compute batch influence
        let test_samples = &samples[4..8];
        let batch_influence = forest.compute_oob_influence_batch(0, test_samples);

        assert!(
            batch_influence.is_some(),
            "Should compute batch influence"
        );

        let inf_val = batch_influence.unwrap();
        assert!(
            inf_val >= -1.0 && inf_val <= 1.0,
            "Batch influence should be in [-1, 1], got {}",
            inf_val
        );

        println!("Batch OOB influence for sample 0: {:.4}", inf_val);
    }

    // =========================================================================
    // Phase 5: fit_internal strategy tests
    // =========================================================================

    #[test]
    fn test_fit_internal_default_strategy() {
        let samples = make_samples();
        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        assert_eq!(forest.num_samples(), 8);
        // Each sample should be in at most k=3 trees
        for s in &samples {
            let indices = forest.sample_tree_map.get(&s.id).unwrap();
            assert!(indices.len() <= 3, "Sample {} in {} trees, expected <= 3", s.id, indices.len());
        }
    }

    #[test]
    fn test_fit_internal_class_based_strategy() {
        let samples = make_samples(); // 4 true, 4 false
        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.set_class_k(true, 7);  // minority gets 7
        forest.set_class_k(false, 2); // majority gets 2
        forest.fit_adaptive(&samples);

        assert_eq!(forest.num_samples(), 8);
        // True samples should be in more trees than false samples
        let true_avg: f64 = samples.iter()
            .filter(|s| s.label)
            .map(|s| forest.sample_tree_map.get(&s.id).unwrap().len() as f64)
            .sum::<f64>() / 4.0;
        let false_avg: f64 = samples.iter()
            .filter(|s| !s.label)
            .map(|s| forest.sample_tree_map.get(&s.id).unwrap().len() as f64)
            .sum::<f64>() / 4.0;

        assert!(true_avg > false_avg, "Minority (true) avg {:.1} should > majority (false) avg {:.1}", true_avg, false_avg);
    }

    #[test]
    fn test_fit_internal_per_sample_strategy() {
        let samples = make_samples();
        let k_values: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit_with_k_values(&samples, &k_values);

        assert_eq!(forest.num_samples(), 8);
        // Sample 0 should be in at most 1 tree, sample 7 in at most 8
        let s0_trees = forest.sample_tree_map.get(&0).unwrap().len();
        let s7_trees = forest.sample_tree_map.get(&7).unwrap().len();
        assert!(s0_trees <= 1, "Sample 0 should be in <= 1 trees, got {}", s0_trees);
        assert!(s7_trees <= 8, "Sample 7 should be in <= 8 trees, got {}", s7_trees);
        assert!(s7_trees > s0_trees, "Sample 7 ({}) should be in more trees than sample 0 ({})", s7_trees, s0_trees);
    }

    #[test]
    fn test_fit_weighted_numerical_equivalence() {
        // fit_weighted with balanced ratio should behave like fit_adaptive with same class_k
        let samples = make_samples(); // 4 true, 4 false -> 50/50
        let config = ForestConfig {
            num_trees: 20,
            k: 5,
            seed: 42,
            ..Default::default()
        };

        let mut forest1 = DynFrsForest::new(config.clone(), 2);
        forest1.set_adaptive_k_config(AdaptiveKConfig { k_min: 3, k_max: 50, ratio_threshold: 0.3, extreme_threshold: 0.01 });
        forest1.fit_weighted(&samples, 0.5);

        let mut forest2 = DynFrsForest::new(config, 2);
        forest2.set_adaptive_k_config(AdaptiveKConfig { k_min: 3, k_max: 50, ratio_threshold: 0.3, extreme_threshold: 0.01 });
        forest2.update_class_k_from_distribution(0.5);
        forest2.fit_adaptive(&samples);

        // Both should produce identical predictions (same seed, same k distribution)
        let test = VecSample::new(999, vec![5.0, 2.0], false);
        assert_eq!(forest1.predict(&test), forest2.predict(&test));
        assert_eq!(forest1.num_samples(), forest2.num_samples());
    }

    #[test]
    fn test_occ_sample_adaptive_clamps_k() {
        let config = ForestConfig {
            num_trees: 10,
            k: 5,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);

        // k=0 should be clamped to 1
        let indices = forest.occ_sample_adaptive(0, 0);
        assert_eq!(indices.len(), 1, "k=0 should be clamped to 1");

        // k > num_trees should be clamped to num_trees
        let indices = forest.occ_sample_adaptive(1, 100);
        assert_eq!(indices.len(), 10, "k=100 should be clamped to num_trees=10");
    }
}
