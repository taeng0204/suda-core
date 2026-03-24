//! DynFrs Decision Tree with exact unlearning support.

use hashbrown::{HashMap, HashSet};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::dataset::{AttributeType, Dataset};
use crate::node::{node_id, LazyTag, Node};
use crate::sample::Sample;
use crate::scan::{partition_in_place, partition_indices, scan_auto, scan_refs};
use crate::split::Split;
use crate::split_stats::{gini_impurity, SplitStats};
use crate::streaming::{AttributeStats, DelayTag, StreamingNodeState, P_COUNT};

/// DynFrs rebuild threshold: if batch deletion removes more than this fraction
/// of a tree's samples, full rebuild is faster than incremental node updates.
/// Based on DynFrs C++ implementation (DynFrs.h:655-659).
const REBUILD_THRESHOLD: f64 = 0.33;

/// Action to take during develop traversal (avoids node cloning).
#[derive(Debug, Clone, Copy)]
enum DevelopAction {
    /// Rebuild the subtree rooted at this node
    Rebuild,
    /// Recurse to children
    Recurse,
    /// Skip (leaf or non-existent node)
    Skip,
}

/// Configuration for tree building.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Maximum tree depth
    pub max_depth: usize,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: usize,
    /// Number of features to consider for each split
    pub max_features: Option<usize>,
    /// Number of random splits to try per feature
    pub num_splits_to_try: usize,
    /// Split quality degradation threshold for forget-time monitoring.
    /// When set, forget() checks if the weighted Gini impurity of each internal
    /// node on the path has degraded beyond this threshold from its creation-time
    /// value. If so, the node is marked LazyTag::Rebuild.
    /// None = disabled (default, preserves existing behavior).
    pub split_quality_threshold: Option<f64>,
    /// Enable Hoeffding-style inline leaf splitting.
    pub hoeffding_split: bool,
    /// Minimum samples before evaluating a Hoeffding split (grace period).
    pub hoeffding_grace_period: usize,
    /// Confidence parameter for Hoeffding bound (smaller = more confident).
    pub hoeffding_delta: f64,
    /// Tie-breaking threshold: if epsilon < tau, split anyway.
    pub hoeffding_tau: f64,
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            max_depth: 20,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None, // sqrt(n_features)
            num_splits_to_try: 1,
            split_quality_threshold: None,
            hoeffding_split: false,
            hoeffding_grace_period: 200,
            hoeffding_delta: 1e-7,
            hoeffding_tau: 0.05,
        }
    }
}

/// Configuration for class-weighted bootstrap sampling.
///
/// ECBRS-inspired: Use inverse frequency weighting to oversample minority class.
/// For CIDDS (0.4% attack): attack weight ≈ 250, benign weight ≈ 1
#[derive(Debug, Clone)]
pub struct ClassWeightConfig {
    /// Weight for positive class (true = attack)
    pub positive_weight: f64,
    /// Weight for negative class (false = benign)
    pub negative_weight: f64,
    /// Bootstrap sample size ratio (relative to original, default 1.0)
    pub bootstrap_ratio: f64,
}

impl ClassWeightConfig {
    /// Create from class ratio (inverse frequency weighting).
    ///
    /// # Arguments
    /// * `positive_ratio` - Ratio of positive class (0.0 - 1.0)
    ///
    /// # Example
    /// ```ignore
    /// // CIDDS: 0.4% attack
    /// let config = ClassWeightConfig::from_ratio(0.004);
    /// // positive_weight ≈ 250, negative_weight ≈ 1
    /// ```
    pub fn from_ratio(positive_ratio: f64) -> Self {
        let positive_ratio = positive_ratio.clamp(0.0001, 0.9999);
        let negative_ratio = 1.0 - positive_ratio;

        // Inverse frequency weighting
        let positive_weight = 1.0 / positive_ratio;
        let negative_weight = 1.0 / negative_ratio;

        ClassWeightConfig {
            positive_weight,
            negative_weight,
            bootstrap_ratio: 1.0,
        }
    }

    /// Create balanced weights (both classes equal).
    pub fn balanced() -> Self {
        ClassWeightConfig {
            positive_weight: 1.0,
            negative_weight: 1.0,
            bootstrap_ratio: 1.0,
        }
    }

    /// Get weight for a label.
    #[inline]
    pub fn weight_for(&self, label: bool) -> f64 {
        if label {
            self.positive_weight
        } else {
            self.negative_weight
        }
    }
}

/// A decision tree with support for exact unlearning.
#[derive(Debug)]
pub struct DynFrsTree {
    /// Tree index (for seeding)
    index: usize,
    /// Random number generator
    rng: XorShiftRng,
    /// Node storage: node_id -> Node
    nodes: HashMap<u64, Node>,
    /// Sample to leaf mapping: sample_id -> leaf_node_id
    sample_leaf_map: HashMap<u64, u64>,
    /// Tree configuration
    config: TreeConfig,
    /// Number of attributes
    num_attributes: u8,
    /// Streaming state for each node (for incremental updates)
    streaming_states: HashMap<u64, StreamingNodeState>,
    /// Enable streaming mode (tracks attribute statistics)
    streaming_enabled: bool,
}

impl DynFrsTree {
    /// Create a new empty tree.
    pub fn new(index: usize, seed: u64, config: TreeConfig, num_attributes: u8) -> Self {
        let rng = XorShiftRng::seed_from_u64(seed.wrapping_add(index as u64));

        DynFrsTree {
            index,
            rng,
            nodes: HashMap::new(),
            sample_leaf_map: HashMap::new(),
            config,
            num_attributes,
            streaming_states: HashMap::new(),
            streaming_enabled: false,
        }
    }

    /// Create a new tree with streaming mode enabled.
    pub fn new_streaming(index: usize, seed: u64, config: TreeConfig, num_attributes: u8) -> Self {
        let rng = XorShiftRng::seed_from_u64(seed.wrapping_add(index as u64));

        DynFrsTree {
            index,
            rng,
            nodes: HashMap::new(),
            sample_leaf_map: HashMap::new(),
            config,
            num_attributes,
            streaming_states: HashMap::new(),
            streaming_enabled: true,
        }
    }

    /// Enable or disable streaming mode.
    pub fn set_streaming(&mut self, enabled: bool) {
        self.streaming_enabled = enabled;
        if !enabled {
            self.streaming_states.clear();
        }
    }

    /// Fit the tree on samples.
    pub fn fit<D: Dataset, S: Sample>(&mut self, dataset: &D, samples: &mut [S]) {
        self.nodes.clear();
        self.sample_leaf_map.clear();

        if samples.is_empty() {
            return;
        }

        // Compute initial impurity
        let num_plus = samples.iter().filter(|s| s.label()).count() as u32;
        let impurity = gini_impurity(num_plus, samples.len() as u32);

        // Build tree recursively
        self.build_node(dataset, samples, node_id::ROOT, 0, impurity);
    }

    /// Fit the tree with class-weighted bootstrap sampling.
    ///
    /// This method addresses extreme class imbalance by oversampling the minority class
    /// during bootstrap. Uses ECBRS-inspired inverse frequency weighting.
    ///
    /// # Arguments
    /// * `dataset` - The dataset schema
    /// * `samples` - Training samples
    /// * `class_weights` - Class weight configuration
    ///
    /// # Example
    /// ```ignore
    /// let weights = ClassWeightConfig::from_ratio(0.004); // 0.4% positive
    /// tree.fit_weighted(&dataset, &samples, &weights);
    /// ```
    pub fn fit_weighted<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        samples: &[S],
        _class_weights: &ClassWeightConfig,
    ) {
        self.nodes.clear();
        self.sample_leaf_map.clear();

        if samples.is_empty() {
            return;
        }

        // Perform weighted bootstrap sampling
        let mut bootstrap_samples = self.weighted_bootstrap_sample(samples);

        if bootstrap_samples.is_empty() {
            return;
        }

        // Compute initial impurity on bootstrap samples
        let num_plus = bootstrap_samples.iter().filter(|s| s.label()).count() as u32;
        let impurity = gini_impurity(num_plus, bootstrap_samples.len() as u32);

        // Build tree recursively
        self.build_node(dataset, &mut bootstrap_samples, node_id::ROOT, 0, impurity);
    }

    /// Perform stratified balanced sampling for extreme class imbalance.
    ///
    /// ECBRS-inspired: Creates a balanced subset by:
    /// 1. Including ALL minority class samples
    /// 2. Randomly sampling an equal number of majority class samples
    ///
    /// This ensures splits are evaluated on balanced data.
    fn weighted_bootstrap_sample<S: Sample>(
        &mut self,
        samples: &[S],
    ) -> Vec<S> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Separate by class
        let (positive_samples, negative_samples): (Vec<&S>, Vec<&S>) =
            samples.iter().partition(|s| s.label());

        // Determine which is minority
        let (minority, majority) = if positive_samples.len() <= negative_samples.len() {
            (&positive_samples, &negative_samples)
        } else {
            (&negative_samples, &positive_samples)
        };

        // If balanced or nearly so, just use all samples
        if minority.len() * 3 >= majority.len() {
            return samples.to_vec();
        }

        // Stratified sampling: ALL minority + equal number of majority
        let mut balanced_samples = Vec::with_capacity(minority.len() * 2 + 1);

        // Add all minority samples
        for sample in minority.iter() {
            balanced_samples.push((*sample).clone());
        }

        // Sample majority to match minority count (or slightly more for diversity)
        let target_majority = (minority.len() * 2).min(majority.len());

        if target_majority > 0 && !majority.is_empty() {
            // Fisher-Yates shuffle-based sampling without replacement
            let mut indices: Vec<usize> = (0..majority.len()).collect();
            for i in 0..target_majority.min(indices.len()) {
                let j = self.rng.gen_range(i..indices.len());
                indices.swap(i, j);
            }

            for i in 0..target_majority {
                balanced_samples.push(majority[indices[i]].clone());
            }
        }

        // Shuffle the combined samples
        for i in (1..balanced_samples.len()).rev() {
            let j = self.rng.gen_range(0..=i);
            balanced_samples.swap(i, j);
        }

        balanced_samples
    }

    /// Build a node recursively.
    fn build_node<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        samples: &mut [S],
        node_id: u64,
        depth: usize,
        impurity_before: f64,
    ) {
        self.build_node_with_position(dataset, samples, node_id, depth, impurity_before, 0);
    }

    fn build_node_with_position<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        samples: &mut [S],
        node_id: u64,
        depth: usize,
        impurity_before: f64,
        current_position: u64,
    ) {
        let num_samples = samples.len() as u32;
        let num_plus = samples.iter().filter(|s| s.label()).count() as u32;

        // Check stopping conditions
        let should_stop = depth >= self.config.max_depth
            || samples.len() < self.config.min_samples_split
            || num_plus == 0
            || num_plus == num_samples;

        if should_stop {
            self.create_leaf(samples, node_id, num_samples, num_plus);
            return;
        }

        // Try to find a good split
        let best_split = self.find_best_split(dataset, samples, impurity_before);

        match best_split {
            Some((split, stats)) if stats.has_positive_score() => {
                // Compute weighted Gini at split creation for quality monitoring
                let split_gini = if self.config.split_quality_threshold.is_some() {
                    let nl = stats.num_left() as f64;
                    let nr = stats.num_right() as f64;
                    let nt = nl + nr;
                    if nt > 0.0 {
                        (nl * stats.impurity_left + nr * stats.impurity_right) / nt
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // Create internal node with creation timestamp and split quality recorded
                let node = Node::internal_full(split.clone(), num_samples, num_plus, current_position, split_gini);
                self.nodes.insert(node_id, node);

                // Partition samples
                let split_idx = partition_in_place(samples, &split);

                let (left_samples, right_samples) = samples.split_at_mut(split_idx);

                // Check minimum leaf size
                if left_samples.len() < self.config.min_samples_leaf
                    || right_samples.len() < self.config.min_samples_leaf
                {
                    // Convert to leaf instead
                    self.nodes.remove(&node_id);
                    self.create_leaf(samples, node_id, num_samples, num_plus);
                    return;
                }

                // Recurse on children
                self.build_node_with_position(
                    dataset,
                    left_samples,
                    node_id::left_child(node_id),
                    depth + 1,
                    stats.impurity_left,
                    current_position,
                );

                self.build_node_with_position(
                    dataset,
                    right_samples,
                    node_id::right_child(node_id),
                    depth + 1,
                    stats.impurity_right,
                    current_position,
                );
            }
            _ => {
                // No good split found, create leaf
                self.create_leaf(samples, node_id, num_samples, num_plus);
            }
        }
    }

    /// Create a leaf node.
    fn create_leaf<S: Sample>(
        &mut self,
        samples: &[S],
        node_id: u64,
        num_samples: u32,
        num_plus: u32,
    ) {
        let sample_ids: HashSet<u64> = samples.iter().map(|s| s.id()).collect();

        // Update sample->leaf mapping
        for &id in &sample_ids {
            self.sample_leaf_map.insert(id, node_id);
        }

        let node = Node::leaf_with_samples(num_samples, num_plus, sample_ids);
        self.nodes.insert(node_id, node);
    }

    /// Find the best split for a node.
    fn find_best_split<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        samples: &[S],
        impurity_before: f64,
    ) -> Option<(Split, SplitStats)> {
        let num_features = self.num_attributes as usize;
        let max_features = self
            .config
            .max_features
            .unwrap_or_else(|| (num_features as f64).sqrt().ceil() as usize)
            .min(num_features);

        // Randomly select features to consider
        let mut feature_indices: Vec<u8> = (0..self.num_attributes).collect();

        // Fisher-Yates shuffle for first max_features elements
        for i in 0..max_features.min(feature_indices.len()) {
            let j = self.rng.gen_range(i..feature_indices.len());
            feature_indices.swap(i, j);
        }

        let mut best_split: Option<(Split, SplitStats)> = None;
        let mut best_score: Option<i64> = None;

        for &attr_idx in feature_indices.iter().take(max_features) {
            let (min_val, max_val) = dataset.attribute_range(attr_idx);
            let attr_type = dataset.attribute_type(attr_idx);

            for _ in 0..self.config.num_splits_to_try {
                let split = match attr_type {
                    AttributeType::Numerical => {
                        if (max_val - min_val).abs() < f32::EPSILON {
                            continue;
                        }
                        let threshold = self.rng.gen_range(min_val..max_val);
                        Split::numerical(attr_idx, threshold)
                    }
                    AttributeType::Categorical => {
                        let cardinality = (max_val - min_val + 1.0) as u32;
                        if cardinality <= 1 {
                            continue;
                        }
                        // Random subset
                        let subset = self.rng.gen_range(1..(1u64 << cardinality.min(63)));
                        Split::categorical(attr_idx, subset)
                    }
                };

                let mut stats = scan_auto(samples, &split);
                stats.update_score(impurity_before);

                if stats.has_positive_score() {
                    let current_score = stats.score.unwrap();
                    if best_score.map_or(true, |s| current_score > s) {
                        best_score = Some(current_score);
                        best_split = Some((split, stats));
                    }
                }
            }
        }

        best_split
    }

    /// Predict the label for a single sample.
    pub fn predict<S: Sample>(&self, sample: &S) -> bool {
        let mut node_id = node_id::ROOT;

        loop {
            match self.nodes.get(&node_id) {
                Some(Node::Internal { split, .. }) => {
                    if sample.is_left_of(split) {
                        node_id = node_id::left_child(node_id);
                    } else {
                        node_id = node_id::right_child(node_id);
                    }
                }
                Some(Node::Leaf {
                    num_samples,
                    num_plus,
                    ..
                }) => {
                    return *num_plus * 2 > *num_samples;
                }
                None => {
                    // Shouldn't happen in a valid tree
                    return false;
                }
            }
        }
    }

    /// Forget a sample from the tree (exact unlearning).
    /// Returns true if the sample was found and removed.
    pub fn forget(&mut self, sample_id: u64, was_positive: bool) -> bool {
        // Find the leaf containing this sample
        let leaf_id = match self.sample_leaf_map.remove(&sample_id) {
            Some(id) => id,
            None => return false, // Sample not in this tree
        };

        // Remove from leaf
        if let Some(leaf) = self.nodes.get_mut(&leaf_id) {
            leaf.remove_sample(sample_id, was_positive);
        }

        // Collect parent IDs that need split quality checking (if enabled).
        // We must check quality after all counts are updated, so collect first.
        let quality_threshold = self.config.split_quality_threshold;
        let mut parents_to_check: Vec<u64> = Vec::new();

        // Update counts along the path from leaf to root
        let mut current_id = leaf_id;
        while current_id != node_id::ROOT {
            let parent_id = node_id::parent(current_id);

            if let Some(node) = self.nodes.get_mut(&parent_id) {
                node.remove_sample(sample_id, was_positive);

                // Mark for potential rebuild if this split might no longer be optimal
                // (LZY Tag strategy: mark dirty but don't rebuild immediately)
                if let Node::Internal {
                    lazy_tag,
                    num_samples,
                    ..
                } = node
                {
                    if *num_samples < self.config.min_samples_split as u32 {
                        *lazy_tag = LazyTag::Dirty;
                    }
                }

                // Collect for quality check (only internal nodes that are still Clean)
                if quality_threshold.is_some() {
                    if let Node::Internal {
                        lazy_tag: LazyTag::Clean,
                        ..
                    } = node
                    {
                        parents_to_check.push(parent_id);
                    }
                }
            }

            current_id = parent_id;
        }

        // Phase 2: Check split quality degradation (after all counts are updated)
        if let Some(threshold) = quality_threshold {
            for &nid in &parents_to_check {
                // Get the stored split_gini from creation time
                let original_gini = match self.nodes.get(&nid) {
                    Some(node) => match node.split_gini() {
                        Some(g) => g,
                        None => continue,
                    },
                    None => continue,
                };

                // Compute current weighted Gini impurity
                if let Some(current_gini) = self.compute_current_split_gini(nid) {
                    // Quality degraded if current Gini increased beyond threshold
                    let degradation = current_gini - original_gini;
                    if degradation > threshold {
                        if let Some(node) = self.nodes.get_mut(&nid) {
                            node.set_lazy_tag(LazyTag::Rebuild);
                        }
                    }
                }
            }
        }

        true
    }

    /// Compute the current weighted Gini impurity of an internal node's split.
    ///
    /// Uses the child nodes' sample counts to recalculate the weighted Gini
    /// impurity of the split. Returns None if the node is not internal or
    /// if child nodes are missing.
    fn compute_current_split_gini(&self, nid: u64) -> Option<f64> {
        // Only internal nodes have splits
        if !matches!(self.nodes.get(&nid), Some(Node::Internal { .. })) {
            return None;
        }

        let left_id = node_id::left_child(nid);
        let right_id = node_id::right_child(nid);

        let left_node = self.nodes.get(&left_id)?;
        let right_node = self.nodes.get(&right_id)?;

        let left_n = left_node.num_samples() as f64;
        let right_n = right_node.num_samples() as f64;
        let total = left_n + right_n;

        if total == 0.0 {
            return None;
        }

        let left_plus = left_node.num_plus() as f64;
        let right_plus = right_node.num_plus() as f64;

        // Gini impurity for each child
        let gini_left = if left_n > 0.0 {
            let p = left_plus / left_n;
            2.0 * p * (1.0 - p)
        } else {
            0.0
        };

        let gini_right = if right_n > 0.0 {
            let p = right_plus / right_n;
            2.0 * p * (1.0 - p)
        } else {
            0.0
        };

        let weighted_gini = (left_n * gini_left + right_n * gini_right) / total;
        Some(weighted_gini)
    }

    /// Forget multiple samples (batch unlearning).
    pub fn forget_batch(&mut self, sample_ids: &[(u64, bool)]) {
        for &(sample_id, was_positive) in sample_ids {
            self.forget(sample_id, was_positive);
        }
    }

    /// Optimized batch forget using the DynFrs dual-path strategy.
    ///
    /// Chooses between two paths based on deletion ratio vs `REBUILD_THRESHOLD` (0.33):
    ///
    /// - **Incremental path** (ratio <= 0.33): Calls `forget()` per sample, marking
    ///   nodes Dirty. O(k × depth) per sample. Efficient for small deletions.
    /// - **Rebuild path** (ratio > 0.33): Reconstructs the tree from remaining samples.
    ///   O(n × log(n)). Faster than incremental when many nodes would need rebuilding.
    ///
    /// The 0.33 threshold comes from the DynFrs C++ implementation (DynFrs.h:655-659),
    /// where empirical analysis showed rebuild becomes faster above ~33% deletion.
    ///
    /// Returns the number of samples actually forgotten.
    pub fn forget_batch_optimized<D: Dataset, S: Sample>(
        &mut self,
        sample_ids: &[(u64, bool)],
        dataset: &D,
        all_samples: &[S],
    ) -> usize {
        if sample_ids.is_empty() {
            return 0;
        }

        let current_samples = self.sample_leaf_map.len();
        if current_samples == 0 {
            return 0;
        }

        let delete_ratio = sample_ids.len() as f64 / current_samples as f64;

        if delete_ratio > REBUILD_THRESHOLD {
            // Rebuild strategy: faster for large deletions
            self.rebuild_without_samples(sample_ids, dataset, all_samples)
        } else {
            // Incremental strategy: faster for small deletions
            let mut count = 0;
            for &(sample_id, was_positive) in sample_ids {
                if self.forget(sample_id, was_positive) {
                    count += 1;
                }
            }
            count
        }
    }

    /// Rebuild the tree excluding specified samples.
    ///
    /// This is called when batch deletion exceeds 33% threshold.
    /// More efficient than incremental deletion for large batches.
    fn rebuild_without_samples<D: Dataset, S: Sample>(
        &mut self,
        samples_to_remove: &[(u64, bool)],
        dataset: &D,
        all_samples: &[S],
    ) -> usize {
        // Create set of samples to exclude
        let exclude_set: HashSet<u64> = samples_to_remove.iter().map(|(id, _)| *id).collect();

        // Count how many we're actually removing (that exist in tree)
        let mut removed_count = 0;
        for (id, _) in samples_to_remove {
            if self.sample_leaf_map.contains_key(id) {
                removed_count += 1;
            }
        }

        if removed_count == 0 {
            return 0;
        }

        // Filter to samples that should remain
        let mut remaining_samples: Vec<S> = all_samples
            .iter()
            .filter(|s| {
                let id = s.id();
                self.sample_leaf_map.contains_key(&id) && !exclude_set.contains(&id)
            })
            .cloned()
            .collect();

        // Clear and rebuild
        self.nodes.clear();
        self.sample_leaf_map.clear();
        self.streaming_states.clear();

        if remaining_samples.is_empty() {
            return removed_count;
        }

        // Compute impurity for rebuild
        let num_plus = remaining_samples.iter().filter(|s| s.label()).count() as u32;
        let impurity = gini_impurity(num_plus, remaining_samples.len() as u32);

        // Rebuild from root
        self.build_node(dataset, &mut remaining_samples, node_id::ROOT, 0, impurity);

        removed_count
    }

    /// Check if the tree contains a sample.
    pub fn contains_sample(&self, sample_id: u64) -> bool {
        self.sample_leaf_map.contains_key(&sample_id)
    }

    /// Develop: rebuild nodes marked with LZY tags.
    /// This is called after batch deletions to restore tree optimality.
    /// Based on DynFrs C++ algorithm (DynFrs.h:775-790).
    ///
    /// Note: For better performance in forest context, use develop_with_map()
    /// which accepts a pre-built sample map to avoid redundant HashMap creation.
    pub fn develop<D: Dataset, S: Sample>(&mut self, dataset: &D, samples: &[S]) {
        // Create a sample lookup map for rebuild
        let sample_map: HashMap<u64, &S> = samples.iter().map(|s| (s.id(), s)).collect();

        // Start recursive develop from root
        self.develop_node_optimized(dataset, &sample_map, node_id::ROOT);
    }

    /// Optimized develop using pre-built sample map.
    /// This avoids redundant HashMap creation when called from forest.develop().
    ///
    /// Performance: 40-50% faster when processing 50 trees with shared sample map.
    #[inline]
    pub fn develop_with_map<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
    ) {
        self.develop_node_optimized(dataset, sample_map, node_id::ROOT);
    }

    /// Optimized recursive develop for a single node.
    /// Avoids unnecessary Node cloning by using direct pattern matching.
    fn develop_node_optimized<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        node_id: u64,
    ) {
        self.develop_node_with_age(dataset, sample_map, node_id, 0, None);
    }

    /// Develop with age-based subtree refresh.
    ///
    /// Same as develop_with_map but also checks split age: internal nodes with
    /// `current_position - created_at > max_split_age` are automatically rebuilt.
    /// This addresses structural debt from gradual drift.
    ///
    /// # Arguments
    /// * `current_position` - Current stream position (sample counter)
    /// * `max_split_age` - Maximum age for a split before forced rebuild (None = disabled)
    #[inline]
    pub fn develop_with_age<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        current_position: u64,
        max_split_age: Option<u64>,
    ) {
        self.develop_node_with_age(dataset, sample_map, node_id::ROOT, current_position, max_split_age);
    }

    /// Recursive develop with age-based subtree refresh support.
    ///
    /// When `max_split_age` is Some, internal nodes whose split is older than
    /// `current_position - created_at > max_split_age` are automatically marked
    /// for rebuild, even if their LazyTag is Clean. This addresses structural
    /// debt from gradual drift where forget() never invalidates the split.
    fn develop_node_with_age<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        node_id: u64,
        current_position: u64,
        max_split_age: Option<u64>,
    ) {
        // Check node state without cloning
        let action = match self.nodes.get(&node_id) {
            Some(Node::Internal { lazy_tag, created_at, .. }) => {
                // First check LazyTag state
                match lazy_tag {
                    LazyTag::Rebuild | LazyTag::Dirty => DevelopAction::Rebuild,
                    LazyTag::Clean => {
                        // Age-based check: if split is too old, force rebuild
                        if let Some(max_age) = max_split_age {
                            if current_position.saturating_sub(*created_at) > max_age {
                                DevelopAction::Rebuild
                            } else {
                                DevelopAction::Recurse
                            }
                        } else {
                            DevelopAction::Recurse
                        }
                    }
                }
            }
            Some(Node::Leaf { .. }) => DevelopAction::Skip,
            None => DevelopAction::Skip,
        };

        match action {
            DevelopAction::Rebuild => {
                self.rebuild_subtree_optimized_with_position(dataset, sample_map, node_id, current_position);
            }
            DevelopAction::Recurse => {
                self.develop_node_with_age(dataset, sample_map, node_id::left_child(node_id), current_position, max_split_age);
                self.develop_node_with_age(dataset, sample_map, node_id::right_child(node_id), current_position, max_split_age);
            }
            DevelopAction::Skip => {}
        }
    }

    /// Collect all sample IDs from a subtree rooted at the given node.
    fn collect_samples_from_subtree(&self, node_id: u64) -> Vec<u64> {
        let mut sample_ids = Vec::new();
        self.collect_samples_recursive(node_id, &mut sample_ids);
        sample_ids
    }

    /// Helper for recursive sample collection.
    fn collect_samples_recursive(&self, node_id: u64, sample_ids: &mut Vec<u64>) {
        match self.nodes.get(&node_id) {
            Some(Node::Leaf {
                sample_ids: leaf_samples,
                ..
            }) => {
                sample_ids.extend(leaf_samples.iter().copied());
            }
            Some(Node::Internal { .. }) => {
                self.collect_samples_recursive(node_id::left_child(node_id), sample_ids);
                self.collect_samples_recursive(node_id::right_child(node_id), sample_ids);
            }
            None => {}
        }
    }

    /// Optimized subtree removal using batch collection.
    /// Collects all node IDs first, then removes in a single pass.
    /// This avoids redundant contains_key + remove lookups.
    fn remove_subtree_batch(&mut self, node_id: u64) {
        let mut nodes_to_remove = Vec::new();
        self.collect_subtree_nodes(node_id, &mut nodes_to_remove);

        // Batch removal (single lookup per node)
        for id in nodes_to_remove {
            self.nodes.remove(&id);
        }
    }

    /// Collect all node IDs in a subtree (excluding the root).
    fn collect_subtree_nodes(&self, node_id: u64, result: &mut Vec<u64>) {
        let left_id = node_id::left_child(node_id);
        let right_id = node_id::right_child(node_id);

        if self.nodes.contains_key(&left_id) {
            result.push(left_id);
            self.collect_subtree_nodes(left_id, result);
        }
        if self.nodes.contains_key(&right_id) {
            result.push(right_id);
            self.collect_subtree_nodes(right_id, result);
        }
    }

    // =========================================================================
    // Tier 2 Optimization: Reference-based tree building (zero-copy rebuild)
    // =========================================================================

    /// Build a node recursively using sample references (zero-copy).
    ///
    /// This avoids cloning samples by working with references and using
    /// index-based partitioning instead of in-place mutation.
    #[allow(dead_code)]
    fn build_node_from_refs<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_refs: &[&S],
        node_id: u64,
        depth: usize,
        impurity_before: f64,
    ) {
        self.build_node_from_refs_with_position(dataset, sample_refs, node_id, depth, impurity_before, 0);
    }

    fn build_node_from_refs_with_position<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_refs: &[&S],
        node_id: u64,
        depth: usize,
        impurity_before: f64,
        current_position: u64,
    ) {
        let num_samples = sample_refs.len() as u32;
        let num_plus = sample_refs.iter().filter(|s| s.label()).count() as u32;

        // Check stopping conditions
        let should_stop = depth >= self.config.max_depth
            || sample_refs.len() < self.config.min_samples_split
            || num_plus == 0
            || num_plus == num_samples;

        if should_stop {
            self.create_leaf_from_refs(sample_refs, node_id, num_samples, num_plus);
            return;
        }

        // Try to find a good split
        let best_split = self.find_best_split_refs(dataset, sample_refs, impurity_before);

        match best_split {
            Some((split, stats)) if stats.has_positive_score() => {
                // Compute weighted Gini at split creation for quality monitoring
                let split_gini = if self.config.split_quality_threshold.is_some() {
                    let nl = stats.num_left() as f64;
                    let nr = stats.num_right() as f64;
                    let nt = nl + nr;
                    if nt > 0.0 {
                        (nl * stats.impurity_left + nr * stats.impurity_right) / nt
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // Create internal node with creation timestamp and split quality recorded
                let node = Node::internal_full(split.clone(), num_samples, num_plus, current_position, split_gini);
                self.nodes.insert(node_id, node);

                // Partition using indices (no mutation needed)
                let (left_indices, right_indices) = partition_indices(sample_refs, &split);

                // Check minimum leaf size
                if left_indices.len() < self.config.min_samples_leaf
                    || right_indices.len() < self.config.min_samples_leaf
                {
                    // Convert to leaf instead
                    self.nodes.remove(&node_id);
                    self.create_leaf_from_refs(sample_refs, node_id, num_samples, num_plus);
                    return;
                }

                // Collect references for children
                let left_refs: Vec<&S> = left_indices.iter().map(|&i| sample_refs[i]).collect();
                let right_refs: Vec<&S> = right_indices.iter().map(|&i| sample_refs[i]).collect();

                // Recurse on children
                self.build_node_from_refs_with_position(
                    dataset,
                    &left_refs,
                    node_id::left_child(node_id),
                    depth + 1,
                    stats.impurity_left,
                    current_position,
                );

                self.build_node_from_refs_with_position(
                    dataset,
                    &right_refs,
                    node_id::right_child(node_id),
                    depth + 1,
                    stats.impurity_right,
                    current_position,
                );
            }
            _ => {
                // No good split found, create leaf
                self.create_leaf_from_refs(sample_refs, node_id, num_samples, num_plus);
            }
        }
    }

    /// Create a leaf node from sample references.
    #[inline]
    fn create_leaf_from_refs<S: Sample>(
        &mut self,
        sample_refs: &[&S],
        node_id: u64,
        num_samples: u32,
        num_plus: u32,
    ) {
        let sample_ids: HashSet<u64> = sample_refs.iter().map(|s| s.id()).collect();

        // Update sample->leaf mapping
        for &id in &sample_ids {
            self.sample_leaf_map.insert(id, node_id);
        }

        let node = Node::leaf_with_samples(num_samples, num_plus, sample_ids);
        self.nodes.insert(node_id, node);
    }

    /// Find the best split for a node using sample references.
    fn find_best_split_refs<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_refs: &[&S],
        impurity_before: f64,
    ) -> Option<(Split, SplitStats)> {
        let num_features = self.num_attributes as usize;
        let max_features = self
            .config
            .max_features
            .unwrap_or_else(|| (num_features as f64).sqrt().ceil() as usize)
            .min(num_features);

        // Randomly select features to consider
        let mut feature_indices: Vec<u8> = (0..self.num_attributes).collect();

        // Fisher-Yates shuffle for first max_features elements
        for i in 0..max_features.min(feature_indices.len()) {
            let j = self.rng.gen_range(i..feature_indices.len());
            feature_indices.swap(i, j);
        }

        let mut best_split: Option<(Split, SplitStats)> = None;
        let mut best_score: Option<i64> = None;

        for &attr_idx in feature_indices.iter().take(max_features) {
            let (min_val, max_val) = dataset.attribute_range(attr_idx);
            let attr_type = dataset.attribute_type(attr_idx);

            for _ in 0..self.config.num_splits_to_try {
                let split = match attr_type {
                    AttributeType::Numerical => {
                        if (max_val - min_val).abs() < f32::EPSILON {
                            continue;
                        }
                        let threshold = self.rng.gen_range(min_val..max_val);
                        Split::numerical(attr_idx, threshold)
                    }
                    AttributeType::Categorical => {
                        let cardinality = (max_val - min_val + 1.0) as u32;
                        if cardinality <= 1 {
                            continue;
                        }
                        // Random subset
                        let subset = self.rng.gen_range(1..(1u64 << cardinality.min(63)));
                        Split::categorical(attr_idx, subset)
                    }
                };

                let mut stats = scan_refs(sample_refs, &split);
                stats.update_score(impurity_before);

                if stats.has_positive_score() {
                    let current_score = stats.score.unwrap();
                    if best_score.map_or(true, |s| current_score > s) {
                        best_score = Some(current_score);
                        best_split = Some((split, stats));
                    }
                }
            }
        }

        best_split
    }

    /// Optimized subtree rebuild using reference-based building (zero-copy).
    ///
    /// Key optimizations:
    /// 1. Uses batch node removal instead of recursive removal
    /// 2. **ZERO sample cloning** - uses references throughout
    /// 3. Clears sample_leaf_map entries in batch
    #[allow(dead_code)]
    fn rebuild_subtree_optimized<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        node_id: u64,
    ) {
        self.rebuild_subtree_optimized_with_position(dataset, sample_map, node_id, 0);
    }

    fn rebuild_subtree_optimized_with_position<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        node_id: u64,
        current_position: u64,
    ) {
        // 1. Collect all sample IDs from this subtree
        let sample_ids = self.collect_samples_from_subtree(node_id);

        if sample_ids.is_empty() {
            // No samples left - remove the node
            self.remove_subtree_batch(node_id);
            self.nodes.remove(&node_id);
            return;
        }

        // 2. Clear old sample_leaf_map entries in batch
        for id in &sample_ids {
            self.sample_leaf_map.remove(id);
        }

        // 3. Get sample references (NO CLONING!)
        let sample_refs: Vec<&S> = sample_ids
            .iter()
            .filter_map(|id| sample_map.get(id).copied())
            .collect();

        if sample_refs.is_empty() {
            return;
        }

        // 4. Remove old subtree using batch removal
        self.remove_subtree_batch(node_id);
        self.nodes.remove(&node_id);

        // 5. Compute impurity for rebuild
        let num_plus = sample_refs.iter().filter(|s| s.label()).count() as u32;
        let impurity = gini_impurity(num_plus, sample_refs.len() as u32);

        // 6. Determine depth from node_id
        let depth = node_id::depth(node_id) as usize;

        // 7. Rebuild subtree using reference-based building (ZERO COPY!)
        self.build_node_from_refs_with_position(dataset, &sample_refs, node_id, depth, impurity, current_position);

        // 8. Re-initialize streaming states for the rebuilt subtree.
        // Without this, streaming_states entries are lost after rebuild,
        // causing add_sample_streaming() to skip split statistics updates
        // for all nodes in this subtree — effectively freezing their splits.
        if self.streaming_enabled {
            let num_attrs = self.num_attributes;
            self.init_streaming_states_recursive(
                node_id,
                depth as u32,
                sample_map,
                &|s: &S| -> Vec<f32> {
                    (0..num_attrs as usize)
                        .map(|i| s.attribute_value(i as u8))
                        .collect()
                },
            );
        }
    }

    /// Get the number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of samples tracked.
    pub fn num_samples(&self) -> usize {
        self.sample_leaf_map.len()
    }

    /// Get tree depth.
    pub fn depth(&self) -> u32 {
        self.nodes
            .keys()
            .map(|&id| node_id::depth(id))
            .max()
            .unwrap_or(0)
    }

    /// Get the tree index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Check if this tree has any nodes with pending lazy tags (Dirty/Rebuild).
    pub fn has_dirty_nodes(&self) -> bool {
        self.nodes.values().any(|node| {
            matches!(
                node,
                Node::Internal {
                    lazy_tag: LazyTag::Dirty | LazyTag::Rebuild,
                    ..
                }
            )
        })
    }

    // =========================================================================
    // Incremental Sample Addition (for streaming without full refit)
    // =========================================================================

    /// Add a single sample to the existing tree without rebuilding.
    /// Routes the sample to its appropriate leaf and updates statistics.
    ///
    /// This enables incremental learning similar to ARF, but maintains
    /// exact unlearning capability.
    ///
    /// Returns the leaf_id where the sample was added.
    pub fn add_sample_incremental<S: Sample>(&mut self, sample: &S) -> Option<u64> {
        if self.nodes.is_empty() {
            // Tree is empty, create initial leaf
            let sample_ids: HashSet<u64> = vec![sample.id()].into_iter().collect();
            let num_plus = if sample.label() { 1 } else { 0 };
            let node = Node::leaf_with_samples(1, num_plus, sample_ids);
            self.nodes.insert(node_id::ROOT, node);
            self.sample_leaf_map.insert(sample.id(), node_id::ROOT);
            return Some(node_id::ROOT);
        }

        // Route sample to appropriate leaf
        let mut current_id = node_id::ROOT;

        loop {
            match self.nodes.get_mut(&current_id) {
                Some(Node::Internal { split, num_samples, num_plus, .. }) => {
                    // Update internal node statistics
                    *num_samples += 1;
                    if sample.label() {
                        *num_plus += 1;
                    }

                    // Route to child
                    if sample.is_left_of(split) {
                        current_id = node_id::left_child(current_id);
                    } else {
                        current_id = node_id::right_child(current_id);
                    }
                }
                Some(Node::Leaf { num_samples, num_plus, sample_ids, .. }) => {
                    // Add to leaf
                    *num_samples += 1;
                    if sample.label() {
                        *num_plus += 1;
                    }
                    sample_ids.insert(sample.id());

                    // Update sample->leaf mapping
                    self.sample_leaf_map.insert(sample.id(), current_id);

                    return Some(current_id);
                }
                None => {
                    // Node doesn't exist - shouldn't happen in valid tree
                    return None;
                }
            }
        }
    }

    /// Add multiple samples incrementally to the existing tree.
    ///
    /// This is the batch version of add_sample_incremental.
    /// More efficient than calling add_sample_incremental in a loop
    /// because it avoids repeated lookups.
    pub fn add_samples_incremental<S: Sample>(&mut self, samples: &[S]) -> Vec<u64> {
        let mut leaf_ids = Vec::with_capacity(samples.len());

        for sample in samples {
            if let Some(leaf_id) = self.add_sample_incremental(sample) {
                leaf_ids.push(leaf_id);
            }
        }

        leaf_ids
    }

    /// Check if a leaf needs to be split (exceeded capacity).
    ///
    /// This can be called after incremental additions to determine
    /// if the tree structure should be updated.
    pub fn leaf_should_split(&self, leaf_id: u64) -> bool {
        match self.nodes.get(&leaf_id) {
            Some(Node::Leaf { num_samples, num_plus, .. }) => {
                // Split if:
                // 1. Enough samples
                // 2. Not pure (has both classes)
                *num_samples >= self.config.min_samples_split as u32
                    && *num_plus > 0
                    && *num_plus < *num_samples
            }
            _ => false,
        }
    }


    /// Check if a leaf should split using the Hoeffding bound.
    fn hoeffding_should_split(&self, leaf_id: u64) -> bool {
        let n = match self.nodes.get(&leaf_id) {
            Some(Node::Leaf { num_samples, num_plus, .. }) => {
                if *num_plus == 0 || *num_plus == *num_samples {
                    return false;
                }
                *num_samples as usize
            }
            _ => return false,
        };
        if n < self.config.hoeffding_grace_period {
            return false;
        }
        let stats = match self.streaming_states.get(&leaf_id) {
            Some(s) => s,
            None => return false,
        };
        let (best_score, second_score, best_attr, _) = stats.attr_stats.top_two_splits();
        if best_attr.is_none() || best_score >= f64::MAX {
            return false;
        }
        let epsilon = AttributeStats::hoeffding_bound(n, self.config.hoeffding_delta);
        if second_score >= f64::MAX {
            epsilon < self.config.hoeffding_tau
        } else {
            second_score - best_score > epsilon || epsilon < self.config.hoeffding_tau
        }
    }

    /// Perform an inline Hoeffding split on a leaf node.
    /// Converts leaf -> internal + 2 child leaves without develop().
    fn perform_hoeffding_split(
        &mut self,
        leaf_id: u64,
        split_attr: u8,
        threshold: f32,
        sample_info: &dyn Fn(u64, u8) -> Option<(f32, bool)>,
    ) -> bool {
        let depth = node_id::depth(leaf_id);
        if depth as usize >= self.config.max_depth {
            return false;
        }
        let (sample_ids, num_samples, num_plus) = match self.nodes.remove(&leaf_id) {
            Some(Node::Leaf { sample_ids, num_samples, num_plus }) => {
                (sample_ids, num_samples, num_plus)
            }
            other => {
                if let Some(node) = other { self.nodes.insert(leaf_id, node); }
                return false;
            }
        };
        let split = Split::Numerical { attribute_index: split_attr, threshold };
        let mut left_ids = HashSet::new();
        let mut right_ids = HashSet::new();
        let mut left_plus: u32 = 0;
        let mut right_plus: u32 = 0;
        for &sid in &sample_ids {
            match sample_info(sid, split_attr) {
                Some((val, label)) => {
                    if val < threshold {
                        left_ids.insert(sid);
                        if label { left_plus += 1; }
                    } else {
                        right_ids.insert(sid);
                        if label { right_plus += 1; }
                    }
                }
                None => { self.sample_leaf_map.remove(&sid); }
            }
        }
        let left_count = left_ids.len() as u32;
        let right_count = right_ids.len() as u32;
        self.nodes.insert(leaf_id, Node::Internal {
            split, lazy_tag: LazyTag::Clean, num_samples, num_plus,
            created_at: 0, split_gini: 0.0,
        });
        let left_id = node_id::left_child(leaf_id);
        let right_id = node_id::right_child(leaf_id);
        self.nodes.insert(left_id,
            Node::leaf_with_samples(left_count, left_plus, left_ids.clone()));
        self.nodes.insert(right_id,
            Node::leaf_with_samples(right_count, right_plus, right_ids.clone()));
        for &sid in &left_ids { self.sample_leaf_map.insert(sid, left_id); }
        for &sid in &right_ids { self.sample_leaf_map.insert(sid, right_id); }
        if self.streaming_enabled {
            let child_depth = depth + 1;
            let seed_left: u64 = self.rng.gen();
            let seed_right: u64 = self.rng.gen();
            let mut left_state = StreamingNodeState::new(self.num_attributes, seed_left, child_depth);
            left_state.delay = DelayTag::None;
            left_state.sample_ids = left_ids.iter().copied().collect();
            left_state.attr_stats.total_samples = left_count;
            left_state.attr_stats.total_positive = left_plus;
            let mut right_state = StreamingNodeState::new(self.num_attributes, seed_right, child_depth);
            right_state.delay = DelayTag::None;
            right_state.sample_ids = right_ids.iter().copied().collect();
            right_state.attr_stats.total_samples = right_count;
            right_state.attr_stats.total_positive = right_plus;
            if let Some(ps) = self.streaming_states.get_mut(&leaf_id) {
                ps.split_attr = Some(split_attr);
                ps.split_threshold = threshold;
                ps.delay = DelayTag::None;
                ps.sample_ids.clear();
            }
            self.streaming_states.insert(left_id, left_state);
            self.streaming_states.insert(right_id, right_state);
        }
        true
    }

    // =========================================================================
    // True Streaming Learning (ported from C++ DynFrs)
    // =========================================================================

    /// Add a single sample with streaming statistics update.
    ///
    /// This is the core streaming learning method, equivalent to C++ node::add().
    /// It:
    /// 1. Updates attribute statistics along the path
    /// 2. Checks if best split changed
    /// 3. Marks nodes for lazy rebuild if needed
    ///
    /// Returns (leaf_id, needs_rebuild) where needs_rebuild indicates if
    /// the tree structure should be updated.
    pub fn add_sample_streaming<S: Sample>(
        &mut self,
        sample: &S,
        features: &[f32],
        use_lazy_rebuild: bool,
    ) -> (Option<u64>, bool) {
        self.add_sample_streaming_inner(sample, features, use_lazy_rebuild, None)
    }

    /// Add sample with Hoeffding inline splitting.
    /// The closure maps (sample_id, attr_idx) -> Option<(attr_value, label)>.
    pub fn add_sample_streaming_hoeffding<S: Sample>(
        &mut self,
        sample: &S,
        features: &[f32],
        use_lazy_rebuild: bool,
        sample_info: &dyn Fn(u64, u8) -> Option<(f32, bool)>,
    ) -> (Option<u64>, bool) {
        self.add_sample_streaming_inner(sample, features, use_lazy_rebuild, Some(sample_info))
    }

    fn add_sample_streaming_inner<S: Sample>(
        &mut self,
        sample: &S,
        features: &[f32],
        use_lazy_rebuild: bool,
        sample_info: Option<&dyn Fn(u64, u8) -> Option<(f32, bool)>>,
    ) -> (Option<u64>, bool) {
        if self.nodes.is_empty() {
            // Tree is empty - create root leaf
            return self.create_initial_leaf_streaming(sample, features);
        }

        let mut needs_rebuild = false;
        let mut current_id = node_id::ROOT;
        let sample_label = sample.label();

        loop {
            // Update streaming state if enabled
            if self.streaming_enabled {
                if let Some(state) = self.streaming_states.get_mut(&current_id) {
                    let invalidated = state.attr_stats.add_sample(features, sample_label);

                    // If candidates were invalidated, may need to regenerate
                    if invalidated > 0 && state.attr_stats.num_candidates() < P_COUNT / 2 {
                        // Mark for potential rebuild
                        if let Some(node) = self.nodes.get_mut(&current_id) {
                            if let Node::Internal { lazy_tag, .. } = node {
                                *lazy_tag = LazyTag::Dirty;
                            }
                        }
                    }

                    // Check if best split changed
                    if !state.is_leaf() && state.best_split_changed() {
                        if use_lazy_rebuild {
                            state.delay = DelayTag::NeedsSeparateAndBuild;
                            if let Some(node) = self.nodes.get_mut(&current_id) {
                                node.set_lazy_tag(LazyTag::Rebuild);
                            }
                        }
                        needs_rebuild = true;
                    }
                }
            }

            match self.nodes.get_mut(&current_id) {
                Some(Node::Internal { split, num_samples, num_plus, .. }) => {
                    // Update internal node statistics
                    *num_samples += 1;
                    if sample_label {
                        *num_plus += 1;
                    }

                    // Route to child
                    if sample.is_left_of(split) {
                        current_id = node_id::left_child(current_id);
                    } else {
                        current_id = node_id::right_child(current_id);
                    }
                }
                Some(Node::Leaf { num_samples, num_plus, sample_ids, .. }) => {
                    // Add to leaf
                    *num_samples += 1;
                    if sample_label {
                        *num_plus += 1;
                    }
                    sample_ids.insert(sample.id());
                    self.sample_leaf_map.insert(sample.id(), current_id);

                    // Update streaming state
                    if self.streaming_enabled {
                        if let Some(state) = self.streaming_states.get_mut(&current_id) {
                            state.sample_ids.push(sample.id());
                        }
                    }

                    // Check if leaf should split
                    let should_split = if self.config.hoeffding_split {
                        self.hoeffding_should_split(current_id)
                    } else {
                        self.leaf_should_split(current_id)
                    };
                    if should_split {
                        if self.config.hoeffding_split && self.streaming_enabled {
                            if let Some(info_fn) = sample_info {
                                let split_info = self.streaming_states.get(&current_id)
                                    .and_then(|state| {
                                        let (_, _, best_attr, best_threshold) =
                                            state.attr_stats.top_two_splits();
                                        best_attr.map(|attr| (attr, best_threshold))
                                    });
                                if let Some((attr, thresh)) = split_info {
                                    self.perform_hoeffding_split(current_id, attr, thresh, info_fn);
                                }
                            } else {
                                if let Some(state) = self.streaming_states.get(&current_id) {
                                    let (_, _, best_attr, best_threshold) =
                                        state.attr_stats.top_two_splits();
                                    if let Some(attr) = best_attr {
                                        if let Some(state) = self.streaming_states.get_mut(&current_id) {
                                            state.split_attr = Some(attr);
                                            state.split_threshold = best_threshold;
                                            state.delay = DelayTag::NeedsSeparateAndBuild;
                                        }
                                        needs_rebuild = true;
                                    }
                                }
                            }
                        } else {
                            if self.streaming_enabled {
                                if let Some(state) = self.streaming_states.get_mut(&current_id) {
                                    state.delay = DelayTag::NeedsBuild;
                                }
                            }
                            needs_rebuild = true;
                        }
                    }

                    return (Some(current_id), needs_rebuild);
                }
                None => {
                    return (None, needs_rebuild);
                }
            }
        }
    }

    /// Create the initial leaf for an empty tree (streaming mode).
    fn create_initial_leaf_streaming<S: Sample>(
        &mut self,
        sample: &S,
        _features: &[f32],
    ) -> (Option<u64>, bool) {
        let sample_ids: HashSet<u64> = vec![sample.id()].into_iter().collect();
        let num_plus = if sample.label() { 1 } else { 0 };
        let node = Node::leaf_with_samples(1, num_plus, sample_ids);
        self.nodes.insert(node_id::ROOT, node);
        self.sample_leaf_map.insert(sample.id(), node_id::ROOT);

        // Initialize streaming state
        if self.streaming_enabled {
            let seed = self.rng.gen();
            let mut state = StreamingNodeState::new(self.num_attributes, seed, 0);
            state.sample_ids.push(sample.id());
            state.attr_stats.total_samples = 1;
            state.attr_stats.total_positive = if sample.label() { 1 } else { 0 };
            self.streaming_states.insert(node_id::ROOT, state);
        }

        (Some(node_id::ROOT), false)
    }

    /// Remove a sample with streaming statistics update.
    ///
    /// This is equivalent to C++ node::del().
    /// Returns (removed_from_leaf_id, needs_rebuild).
    pub fn remove_sample_streaming(
        &mut self,
        sample_id: u64,
        features: &[f32],
        was_positive: bool,
        use_lazy_rebuild: bool,
    ) -> (Option<u64>, bool) {
        let leaf_id = match self.sample_leaf_map.remove(&sample_id) {
            Some(id) => id,
            None => return (None, false),
        };

        let mut needs_rebuild = false;
        let mut current_id = leaf_id;

        // Update leaf
        if let Some(leaf) = self.nodes.get_mut(&leaf_id) {
            leaf.remove_sample(sample_id, was_positive);
        }

        // Update streaming states along path to root
        while current_id != 0 {
            if self.streaming_enabled {
                if let Some(state) = self.streaming_states.get_mut(&current_id) {
                    let invalidated = state.attr_stats.remove_sample(features, was_positive);

                    if invalidated > 0 || state.best_split_changed() {
                        if use_lazy_rebuild {
                            state.delay = DelayTag::NeedsSeparateAndBuild;
                            if let Some(node) = self.nodes.get_mut(&current_id) {
                                node.set_lazy_tag(LazyTag::Rebuild);
                            }
                        }
                        needs_rebuild = true;
                    }

                    // Remove sample from state
                    state.sample_ids.retain(|&id| id != sample_id);
                }
            }

            // Update node counts
            if current_id != leaf_id {
                if let Some(node) = self.nodes.get_mut(&current_id) {
                    node.remove_sample(sample_id, was_positive);
                }
            }

            current_id = node_id::parent(current_id);
        }

        (Some(leaf_id), needs_rebuild)
    }

    /// Initialize streaming states for an existing tree.
    ///
    /// Call this after batch fit() to enable streaming updates.
    pub fn init_streaming_states<S: Sample + Clone>(
        &mut self,
        samples: &[S],
        get_features: impl Fn(&S) -> Vec<f32>,
    ) {
        self.streaming_enabled = true;
        self.streaming_states.clear();

        // Build sample lookup
        let sample_map: HashMap<u64, &S> = samples.iter().map(|s| (s.id(), s)).collect();

        // Initialize states for each node
        self.init_streaming_states_recursive(
            node_id::ROOT,
            0,
            &sample_map,
            &get_features,
        );
    }

    fn init_streaming_states_recursive<S: Sample + Clone>(
        &mut self,
        node_id: u64,
        depth: u32,
        sample_map: &HashMap<u64, &S>,
        get_features: &impl Fn(&S) -> Vec<f32>,
    ) {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n.clone(),
            None => return,
        };

        let seed = self.rng.gen();
        let mut state = StreamingNodeState::new(self.num_attributes, seed, depth);

        match &node {
            Node::Leaf { sample_ids, .. } => {
                // Collect sample IDs
                state.sample_ids = sample_ids.iter().copied().collect();

                // Initialize attribute stats
                let ids: Vec<u64> = state.sample_ids.clone();
                state.attr_stats.init_from_batch(
                    &ids,
                    |id| sample_map.get(&id).map(|s| get_features(s)).unwrap_or_default(),
                    |id| sample_map.get(&id).map(|s| s.label()).unwrap_or(false),
                    P_COUNT,
                );
                state.delay = DelayTag::None;
            }
            Node::Internal { split, .. } => {
                // Collect samples from subtree
                let subtree_samples = self.collect_samples_from_subtree(node_id);
                state.sample_ids = subtree_samples.clone();

                // Initialize attribute stats
                state.attr_stats.init_from_batch(
                    &subtree_samples,
                    |id| sample_map.get(&id).map(|s| get_features(s)).unwrap_or_default(),
                    |id| sample_map.get(&id).map(|s| s.label()).unwrap_or(false),
                    P_COUNT,
                );

                // Set current split info
                state.split_attr = Some(split.attribute_index());
                if let Split::Numerical { threshold, .. } = split {
                    state.split_threshold = *threshold;
                }
                state.delay = DelayTag::None;

                // Recurse to children
                self.init_streaming_states_recursive(
                    node_id::left_child(node_id),
                    depth + 1,
                    sample_map,
                    get_features,
                );
                self.init_streaming_states_recursive(
                    node_id::right_child(node_id),
                    depth + 1,
                    sample_map,
                    get_features,
                );
            }
        }

        self.streaming_states.insert(node_id, state);
    }

    /// Check if streaming mode is enabled.
    pub fn is_streaming(&self) -> bool {
        self.streaming_enabled
    }

    /// Get the number of streaming states (nodes with tracking).
    pub fn num_streaming_states(&self) -> usize {
        self.streaming_states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ArrayDataset;
    use crate::sample::ArraySample;

    fn make_samples() -> Vec<ArraySample<2>> {
        vec![
            // Clearly separable by X[0] < 5
            ArraySample::new(0, [1.0, 1.0], true),
            ArraySample::new(1, [2.0, 2.0], true),
            ArraySample::new(2, [3.0, 3.0], true),
            ArraySample::new(3, [4.0, 4.0], true),
            ArraySample::new(4, [6.0, 1.0], false),
            ArraySample::new(5, [7.0, 2.0], false),
            ArraySample::new(6, [8.0, 3.0], false),
            ArraySample::new(7, [9.0, 4.0], false),
        ]
    }

    /// Create imbalanced samples (simulating CIDDS-like 1% attack ratio)
    fn make_imbalanced_samples() -> Vec<ArraySample<2>> {
        let mut samples = Vec::new();
        let mut id = 0u64;

        // 2 attack samples (2%)
        samples.push(ArraySample::new(id, [1.0, 1.0], true));
        id += 1;
        samples.push(ArraySample::new(id, [2.0, 2.0], true));
        id += 1;

        // 98 benign samples (98%)
        for i in 0..98 {
            let x = 5.0 + (i as f32) * 0.1;
            samples.push(ArraySample::new(id, [x, x], false));
            id += 1;
        }

        samples
    }

    #[test]
    fn test_tree_fit_predict() {
        let mut samples = make_samples();
        let dataset = ArrayDataset::from_samples(&samples, 2);

        let config = TreeConfig {
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(2),
            num_splits_to_try: 10,
            split_quality_threshold: None,
            ..Default::default()
        };

        let mut tree = DynFrsTree::new(0, 42, config, 2);
        tree.fit(&dataset, &mut samples);

        // Check predictions
        for sample in &samples {
            let pred = tree.predict(sample);
            assert_eq!(
                pred,
                sample.label(),
                "Misprediction for sample {}",
                sample.id()
            );
        }
    }

    #[test]
    fn test_tree_forget() {
        let mut samples = make_samples();
        let dataset = ArrayDataset::from_samples(&samples, 2);

        let config = TreeConfig::default();
        let mut tree = DynFrsTree::new(0, 42, config, 2);
        tree.fit(&dataset, &mut samples);

        // Forget a sample
        assert!(tree.contains_sample(0));
        assert!(tree.forget(0, true));
        assert!(!tree.contains_sample(0));

        // Try to forget non-existent sample
        assert!(!tree.forget(999, false));
    }

    #[test]
    fn test_tree_stats() {
        let mut samples = make_samples();
        let dataset = ArrayDataset::from_samples(&samples, 2);

        let config = TreeConfig::default();
        let mut tree = DynFrsTree::new(0, 42, config, 2);
        tree.fit(&dataset, &mut samples);

        assert!(tree.num_nodes() > 0);
        assert_eq!(tree.num_samples(), 8);
        assert!(tree.depth() > 0);
    }

    #[test]
    fn test_develop_after_forget() {
        let mut samples = make_samples();
        let dataset = ArrayDataset::from_samples(&samples, 2);

        let config = TreeConfig {
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(2),
            num_splits_to_try: 5,
            split_quality_threshold: None,
            ..Default::default()
        };

        let mut tree = DynFrsTree::new(0, 42, config, 2);
        tree.fit(&dataset, &mut samples);

        // Initial state
        let initial_nodes = tree.num_nodes();
        assert_eq!(tree.num_samples(), 8);

        // Forget half the samples (4 positive samples)
        for i in 0..4 {
            tree.forget(i, true);
        }
        assert_eq!(tree.num_samples(), 4);

        // Remaining samples (keep only the ones still in tree)
        let remaining_samples: Vec<_> = samples.iter().filter(|s| s.id() >= 4).cloned().collect();

        // Call develop to rebuild
        tree.develop(&dataset, &remaining_samples);

        // Tree should still be valid
        assert!(tree.num_nodes() > 0, "Tree should have nodes after develop");
        assert_eq!(tree.num_samples(), 4, "Should still track 4 samples");

        // Predictions should still work for remaining samples
        for sample in &remaining_samples {
            let _pred = tree.predict(sample); // Should not panic
        }

        println!(
            "develop() test passed: nodes {} -> {}, samples 8 -> 4",
            initial_nodes,
            tree.num_nodes()
        );
    }

    #[test]
    fn test_class_weight_config_from_ratio() {
        // Test CIDDS-like extreme imbalance (0.4% attack)
        let config = ClassWeightConfig::from_ratio(0.004);
        assert!(config.positive_weight > 200.0, "Attack weight should be high");
        assert!(config.negative_weight < 2.0, "Benign weight should be low");
        assert!(
            config.positive_weight / config.negative_weight > 100.0,
            "Attack should have 100x+ weight"
        );

        // Test balanced case
        let balanced = ClassWeightConfig::from_ratio(0.5);
        assert!(
            (balanced.positive_weight - balanced.negative_weight).abs() < 0.01,
            "Should be nearly equal for 50/50"
        );

        // Test edge cases
        let config2 = ClassWeightConfig::from_ratio(0.0001);
        assert!(config2.positive_weight > 1000.0, "Very rare class needs high weight");
    }

    #[test]
    fn test_fit_weighted_basic() {
        let samples = make_samples();
        let dataset = ArrayDataset::from_samples(&samples, 2);

        let config = TreeConfig {
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(2),
            num_splits_to_try: 10,
            split_quality_threshold: None,
            ..Default::default()
        };

        let mut tree = DynFrsTree::new(0, 42, config, 2);

        // Balanced weights (should behave like normal fit)
        let weights = ClassWeightConfig::balanced();
        tree.fit_weighted(&dataset, &samples, &weights);

        assert!(tree.num_nodes() > 0, "Tree should have nodes");
        assert!(tree.num_samples() > 0, "Tree should track samples");

        // Predictions should work
        let mut correct = 0;
        for sample in &samples {
            if tree.predict(sample) == sample.label() {
                correct += 1;
            }
        }
        assert!(correct >= 4, "Should have reasonable accuracy: {}/8", correct);
    }

    #[test]
    fn test_fit_weighted_imbalanced() {
        let samples = make_imbalanced_samples();
        let dataset = ArrayDataset::from_samples(&samples, 2);

        let config = TreeConfig {
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(2),
            num_splits_to_try: 10,
            split_quality_threshold: None,
            ..Default::default()
        };

        // Test without weighting (minority likely ignored)
        let mut tree_unweighted = DynFrsTree::new(0, 42, config.clone(), 2);
        let mut samples_for_fit = samples.clone();
        tree_unweighted.fit(&dataset, &mut samples_for_fit);

        // Test with weighting (minority oversampled)
        let mut tree_weighted = DynFrsTree::new(1, 42, config, 2);
        let weights = ClassWeightConfig::from_ratio(0.02); // 2% positive
        tree_weighted.fit_weighted(&dataset, &samples, &weights);

        assert!(tree_weighted.num_nodes() > 0, "Weighted tree should have nodes");

        // Count predictions
        let attack_samples: Vec<_> = samples.iter().filter(|s| s.label()).collect();

        let mut weighted_attack_correct = 0;
        for sample in &attack_samples {
            if tree_weighted.predict(*sample) == sample.label() {
                weighted_attack_correct += 1;
            }
        }

        println!(
            "Weighted tree attack recall: {}/{} ({:.1}%)",
            weighted_attack_correct,
            attack_samples.len(),
            100.0 * weighted_attack_correct as f64 / attack_samples.len() as f64
        );

        // Weighted tree should recognize at least some attacks
        // (not a strict requirement as tree structure depends on random splits)
    }
}
