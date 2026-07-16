//! DynFrs Random Forest with OCC(q) sampling and exact unlearning.

use hashbrown::{HashMap, HashSet};
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
    /// Class-aware OCC(q) for the minority (positive/attack) class. 0 = disabled
    /// (use `k` for all classes). When > 0, positive-label samples are assigned to
    /// up to `minority_k` trees (typically larger than `k`) so the minority class is
    /// represented in enough trees to be learnable under extreme imbalance. This is a
    /// fixed two-value class-aware k — NOT the removed dynamic Adaptive-k.
    pub minority_k: usize,
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
            minority_k: 0,
            tree_config: TreeConfig::default(),
            seed: 42,
        }
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
    /// Diagnostic: cumulative count of add-path samples that triggered a
    /// `best_split_changed` rebuild mark (streaming subtree refresh on insert).
    add_rebuild_marks: u64,
    /// Diagnostic: cumulative count of forget-path samples that triggered a
    /// `best_split_changed` rebuild mark (streaming subtree refresh on removal).
    /// Q2-2 diagnostic: is exact forget's structure-refresh trigger actually firing?
    forget_rebuild_marks: u64,
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
            add_rebuild_marks: 0,
            forget_rebuild_marks: 0,
        }
    }

    /// Diagnostic accessor: cumulative (add-path, forget-path) rebuild marks.
    /// Used by Q2-2 diagnostic to confirm whether exact forget's `best_split_changed`
    /// subtree-refresh trigger fires under covariate drift (AnoShift). A near-zero
    /// forget count means the structure-refresh path is effectively dormant.
    pub fn rebuild_mark_counts(&self) -> (u64, u64) {
        (self.add_rebuild_marks, self.forget_rebuild_marks)
    }

    /// Fit the forest on samples using OCC(q) sampling.
    pub fn fit(&mut self, samples: &[VecSample]) {
        self.fit_internal(samples, KStrategy::Default);
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
                // Class-aware OCC: minority (positive) gets `minority_k` when enabled.
                KStrategy::Default => self.k_for_label(sample.label),
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

    /// Class-aware k: minority (positive/attack) label gets `minority_k` when enabled
    /// (> 0), otherwise the default `k`. Fixed two-value scheme (not dynamic Adaptive-k).
    #[inline]
    fn k_for_label(&self, label: bool) -> usize {
        if label && self.config.minority_k > 0 {
            self.config.minority_k
        } else {
            self.config.k
        }
    }

    /// OCC(q) sampling: select up to k trees for a sample.
    /// DynFrs 고정 k — 호출자가 k를 결정 (Default 또는 PerSample).
    fn occ_sample_adaptive(&mut self, sample_id: u64, k: usize) -> Vec<usize> {
        // Clamp k to valid range
        let k = k.min(self.config.num_trees).max(1);

        let mut selected = Vec::with_capacity(k);
        let mut count = 0;

        let _ = sample_id;
        for i in 0..self.config.num_trees {
            if count >= k {
                break;
            }

            let remaining_trees = self.config.num_trees - i;
            let remaining_slots = k - count;

            // Probability: remaining_slots / remaining_trees
            let draw = self.rng.gen_range(0..remaining_trees);
            if draw < remaining_slots {
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
    pub fn predict_proba_with_trees(
        &self,
        sample: &VecSample,
        tree_indices: &[usize],
    ) -> Option<f64> {
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

    /// Predict positive-vote ratio (probability) for multiple samples (immutable fast path).
    pub fn predict_proba_batch(&self, samples: &[VecSample]) -> Vec<f64> {
        samples.par_iter().map(|s| self.predict_proba(s)).collect()
    }

    /// Forget a single sample (DynFrs-compatible exact unlearning).
    ///
    /// Removes the sample from the leaf AND propagates the deletion along the
    /// path to root: internal node counts decrement, and (when streaming is
    /// enabled) AttributeStats reflect the removal so `best_split_changed`
    /// triggers lazy rebuild on the next query. Matches DynFrs
    /// `attribute::del` semantics (DynFrs.h:268-298).
    ///
    /// `features` is required so the streaming-aware path can locate the leaf
    /// by descent and update AttributeStats consistently.
    pub fn forget(&mut self, sample_id: u64, features: &[f32]) -> bool {
        let was_positive = match self.sample_labels.remove(&sample_id) {
            Some(label) => label,
            None => return false,
        };

        let tree_indices = match self.sample_tree_map.remove(&sample_id) {
            Some(indices) => indices,
            None => return false,
        };

        for tree_idx in tree_indices {
            let (_removed, needs_rebuild) = self.trees[tree_idx].remove_sample_streaming(
                sample_id,
                features,
                was_positive,
                true,
            );
            if needs_rebuild {
                self.forget_rebuild_marks += 1;
            }
        }
        true
    }

    /// Forget multiple samples (batch exact unlearning).
    ///
    /// `feature_map` must provide features for each sample id. Samples missing
    /// from `feature_map` are silently skipped — the caller owns feature
    /// retention via the sample registry.
    pub fn forget_batch(
        &mut self,
        sample_ids: &[u64],
        feature_map: &HashMap<u64, Vec<f32>>,
    ) -> usize {
        let mut count = 0;
        for &sample_id in sample_ids {
            if let Some(features) = feature_map.get(&sample_id) {
                if self.forget(sample_id, features) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get the number of trees.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Borrow a specific tree (test/audit/inspect use).
    /// A1: needed by lazy_resolve tests to inspect per-node LazyTag state.
    pub fn tree(&self, idx: usize) -> Option<&crate::tree::DynFrsTree> {
        self.trees.get(idx)
    }

    /// Mutable borrow of a specific tree (test/audit use).
    /// A1: needed to force LazyTag::Dirty for sequential-resolve scenarios.
    pub fn tree_mut(&mut self, idx: usize) -> Option<&mut crate::tree::DynFrsTree> {
        self.trees.get_mut(idx)
    }

    /// A1 GREEN: path-amortized lazy resolve predict (DynFrs qry() 정합).
    ///
    /// 각 query마다 트리 path 따라가며 만나는 노드의 LazyTag::Dirty/Rebuild를
    /// single-step으로 resolve (split + separate 한 단계). Off-path 노드는
    /// 그대로 Dirty 유지 → cost amortization.
    ///
    /// Fast path: 어떤 트리에도 pending이 없으면 기존 immutable predict_batch로
    /// 위임 (par_iter, mut overhead 0).
    pub fn predict_batch_with_lazy_resolve(
        &mut self,
        query_samples: &[VecSample],
        sample_map: &HashMap<u64, &VecSample>,
    ) -> Vec<bool> {
        if query_samples.is_empty() {
            return Vec::new();
        }

        // Fast path: pending 없으면 immutable par_iter predict (overhead 0)
        if !self.has_pending_rebuilds() {
            return self.predict_batch(query_samples);
        }

        // Slow path: tree 단위 mut iteration + per-query lazy resolve
        let active_samples: Vec<VecSample> = sample_map.values().map(|&s| s.clone()).collect();
        let dataset = ArrayDataset::from_samples(&active_samples, self.num_attributes);

        // 트리별 parallel: 각 트리에서 query별 lazy resolve
        let votes_per_tree: Vec<Vec<bool>> = self
            .trees
            .par_iter_mut()
            .map(|tree| {
                query_samples
                    .iter()
                    .map(|q| tree.predict_with_lazy_resolve(&dataset, sample_map, q))
                    .collect()
            })
            .collect();

        // Voting: 각 query마다 트리 결과 합산 (majority)
        let n_trees = self.trees.len();
        (0..query_samples.len())
            .map(|qi| {
                let positive_count = votes_per_tree.iter().filter(|votes| votes[qi]).count();
                positive_count * 2 > n_trees
            })
            .collect()
    }

    /// Probability version of `predict_batch_with_lazy_resolve` — identical lazy-resolve
    /// path (so forget's split rebuild IS reflected), but aggregates votes into a
    /// positive-vote *ratio* per query instead of a majority bool.
    ///
    /// gate(unlearning-as-attribution)용: forget 후 leaf-count 감소뿐 아니라
    /// split rebuild까지 반영한 정확한 확률을 얻어야 forget 효과를 과소평가하지 않음.
    pub fn predict_proba_batch_with_lazy_resolve(
        &mut self,
        query_samples: &[VecSample],
        sample_map: &HashMap<u64, &VecSample>,
    ) -> Vec<f64> {
        if query_samples.is_empty() {
            return Vec::new();
        }

        // Fast path: pending 없으면 immutable par_iter predict_proba (overhead 0)
        if !self.has_pending_rebuilds() {
            return query_samples
                .par_iter()
                .map(|s| self.predict_proba(s))
                .collect();
        }

        // Slow path: predict_batch_with_lazy_resolve와 동일한 tree-단위 lazy resolve
        let active_samples: Vec<VecSample> = sample_map.values().map(|&s| s.clone()).collect();
        let dataset = ArrayDataset::from_samples(&active_samples, self.num_attributes);

        let votes_per_tree: Vec<Vec<bool>> = self
            .trees
            .par_iter_mut()
            .map(|tree| {
                query_samples
                    .iter()
                    .map(|q| tree.predict_with_lazy_resolve(&dataset, sample_map, q))
                    .collect()
            })
            .collect();

        let n_trees = self.trees.len();
        (0..query_samples.len())
            .map(|qi| {
                let positive_count = votes_per_tree.iter().filter(|votes| votes[qi]).count();
                positive_count as f64 / n_trees as f64
            })
            .collect()
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
    // Sample Influence Methods (OOB only)
    // =========================================================================
    //
    //   get_samples_by_influence 제거 (호출 0건). get_sample_tree_indices만
    //   compute_oob_influence_batch에서 사용.

    /// Get tree indices where a sample appears (OOB influence computation 용).
    pub fn get_sample_tree_indices(&self, sample_id: u64) -> Vec<usize> {
        self.sample_tree_map
            .get(&sample_id)
            .cloned()
            .unwrap_or_default()
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
    pub fn compute_oob_influence(&self, sample_id: u64, test_sample: &VecSample) -> Option<f64> {
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
    pub fn compute_loss_influence(&self, sample_id: u64, test_sample: &VecSample) -> Option<f64> {
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
            .map(|test| self.trees.iter().map(|tree| tree.predict(test)).collect())
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
                    let in_bag_positive: usize =
                        in_bag_trees.iter().filter(|&&idx| preds[idx]).count();
                    let in_bag_pred = in_bag_positive * 2 > in_bag_trees.len();

                    // OOB prediction (majority vote from OOB trees)
                    let oob_positive: usize = oob_trees.iter().filter(|&&idx| preds[idx]).count();
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
    /// This is equivalent to DynFrs forest::add(X, Y):
    /// 1. OCC(q) sampling with fixed k=config.k to select trees
    /// 2. Call tree.add_sample_streaming() for each selected tree
    /// 3. Track if any tree needs rebuild
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

        // Class-aware OCC: streaming minority samples also get `minority_k` when enabled.
        let k = self.k_for_label(sample.label);
        let tree_indices = self.occ_sample_adaptive(sample.id, k);

        let mut any_needs_rebuild = false;

        // Add to selected trees with streaming
        for &tree_idx in &tree_indices {
            let (_, needs_rebuild) =
                self.trees[tree_idx].add_sample_streaming(sample, &sample.values, use_lazy_rebuild);
            any_needs_rebuild |= needs_rebuild;
        }

        // Store mappings
        self.sample_tree_map.insert(sample.id, tree_indices);
        self.sample_labels.insert(sample.id, sample.label);

        //   — Adaptive-k 본체 제거로 더 이상 사용 안 함.

        (true, any_needs_rebuild)
    }

    //   set_streaming_k_update_interval / reset_streaming_stats /
    //   init_streaming_class_k / init_streaming_class_k_with_counts 모두 제거.

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

        self.add_rebuild_marks += rebuild_count as u64;
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

        let _ = was_positive;

        (true, any_needs_rebuild)
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
//   Python interface는 controller::PyStreamingController(=SUDA)만 노출.
//   demo scripts (test_streaming.py, buffer_integration_example.py)는
//   scripts/legacy_pre_realignment/로 archive.
// =============================================================================

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
            minority_k: 0,
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
            minority_k: 0,
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
            minority_k: 0,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        assert_eq!(forest.num_samples(), 8);
        assert!(forest.contains_sample(0));

        let fmap: HashMap<u64, Vec<f32>> =
            samples.iter().map(|s| (s.id, s.values.clone())).collect();

        // Forget sample 0
        assert!(forest.forget(0, &fmap[&0]));
        assert!(!forest.contains_sample(0));
        assert_eq!(forest.num_samples(), 7);

        // Try to forget again
        assert!(!forest.forget(0, &fmap[&0]));

        // Forget multiple (id 999 is absent from both forest and fmap)
        let forgotten = forest.forget_batch(&[1, 2, 3, 999], &fmap);
        assert_eq!(forgotten, 3);
        assert_eq!(forest.num_samples(), 4);
    }

    #[test]
    fn test_tree_stats() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            minority_k: 0,
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
    //   test_adaptive_k_occ_sampling, test_fit_adaptive, test_fit_weighted).
    //   AdaptiveKConfig + 관련 API 본체 사라짐 → 시나리오 무효.
    // =========================================================================

    #[test]
    fn test_fit_with_k_values() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 30,
            k: 10,
            minority_k: 0,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);

        // Custom k values: higher for positive, lower for negative
        let k_values: Vec<usize> = samples
            .iter()
            .map(|s| if s.label { 15 } else { 5 })
            .collect();

        forest.fit_with_k_values(&samples, &k_values);

        assert_eq!(forest.num_samples(), 8);

        // Verify predictions still work (basic functionality check)
        let predictions = forest.predict_batch(&samples);
        assert_eq!(
            predictions.len(),
            8,
            "Should return predictions for all samples"
        );

        // With only 8 samples, accuracy can vary; just verify the forest is functional
        let correct = predictions
            .iter()
            .zip(samples.iter())
            .filter(|(pred, sample)| **pred == sample.label)
            .count();

        // At least some predictions should be correct (better than random baseline)
        assert!(
            correct >= 3,
            "Should have at least some correct predictions: {}/8",
            correct
        );
    }

    // =========================================================================
    // OOB-based Sample Influence Tests
    // =========================================================================

    #[test]
    fn test_get_oob_tree_indices() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 20,
            k: 5, // Each sample in 5 trees, OOB for 15
            minority_k: 0,
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
            minority_k: 0,
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
            k: 10, // Each sample in ~10 trees, OOB for ~20
            minority_k: 0,
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
        let test_sample = &samples[4]; // Use a different sample as test
        let influence = forest.compute_oob_influence(0, test_sample);

        // Influence should exist (sample is in forest)
        assert!(
            influence.is_some(),
            "Should compute influence for existing sample"
        );

        // Influence value should be in valid range [-1, 1]
        let inf_val = influence.unwrap();
        assert!(
            (-1.0..=1.0).contains(&inf_val),
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
            minority_k: 0,
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
        let test_samples = &samples[4..8]; // Use negative samples as test
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
            minority_k: 0,
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
            assert!(*inf < 0.0, "Harmful samples should have negative influence");
        }

        // Should respect top_k limit
        assert!(harmful.len() <= 3, "Should return at most 3 samples");
    }

    #[test]
    fn test_oob_influence_batch() {
        let samples = make_samples();

        let config = ForestConfig {
            num_trees: 30,
            k: 10,
            minority_k: 0,
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

        assert!(batch_influence.is_some(), "Should compute batch influence");

        let inf_val = batch_influence.unwrap();
        assert!(
            (-1.0..=1.0).contains(&inf_val),
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
            minority_k: 0,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit(&samples);

        assert_eq!(forest.num_samples(), 8);
        // Each sample should be in at most k=3 trees
        for s in &samples {
            let indices = forest.sample_tree_map.get(&s.id).unwrap();
            assert!(
                indices.len() <= 3,
                "Sample {} in {} trees, expected <= 3",
                s.id,
                indices.len()
            );
        }
    }

    #[test]
    fn test_fit_internal_per_sample_strategy() {
        let samples = make_samples();
        let k_values: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let config = ForestConfig {
            num_trees: 10,
            k: 3,
            minority_k: 0,
            seed: 42,
            ..Default::default()
        };

        let mut forest = DynFrsForest::new(config, 2);
        forest.fit_with_k_values(&samples, &k_values);

        assert_eq!(forest.num_samples(), 8);
        // Sample 0 should be in at most 1 tree, sample 7 in at most 8
        let s0_trees = forest.sample_tree_map.get(&0).unwrap().len();
        let s7_trees = forest.sample_tree_map.get(&7).unwrap().len();
        assert!(
            s0_trees <= 1,
            "Sample 0 should be in <= 1 trees, got {}",
            s0_trees
        );
        assert!(
            s7_trees <= 8,
            "Sample 7 should be in <= 8 trees, got {}",
            s7_trees
        );
        assert!(
            s7_trees > s0_trees,
            "Sample 7 ({}) should be in more trees than sample 0 ({})",
            s7_trees,
            s0_trees
        );
    }

    #[test]
    fn test_occ_sample_adaptive_clamps_k() {
        let config = ForestConfig {
            num_trees: 10,
            k: 5,
            minority_k: 0,
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
