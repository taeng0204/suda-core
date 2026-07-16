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
use crate::streaming::{DelayTag, StreamingNodeState, P_COUNT};

/// Callback resolving a sample's (feature value, went-left) routing info during lazy rebuild.
type SampleInfoFn<'a> = &'a dyn Fn(u64, u8) -> Option<(f32, bool)>;

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
    //   DynFrs는 mt19937(rd()) random. SUDA도 random mode만 사용.
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            max_depth: 20,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None, // sqrt(n_features)
            num_splits_to_try: 1,
        }
    }
}

/// A decision tree with support for exact unlearning.
#[derive(Debug)]
pub struct DynFrsTree {
    /// Tree index (for seeding)
    index: usize,
    /// Random number generator (per-tree, seeded from seed + index)
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

    /// Build a node recursively.
    fn build_node<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        samples: &mut [S],
        node_id: u64,
        depth: usize,
        impurity_before: f64,
    ) {
        let num_samples = samples.len() as u32;
        let num_plus = samples.iter().filter(|s| s.label()).count() as u32;

        let should_stop = depth >= self.config.max_depth
            || samples.len() < self.config.min_samples_split
            || num_plus == 0
            || num_plus == num_samples;

        if should_stop {
            self.create_leaf(samples, node_id, num_samples, num_plus);
            return;
        }

        let best_split = self.find_best_split(dataset, samples, impurity_before, node_id);

        match best_split {
            Some((split, stats)) if stats.has_positive_score() => {
                let split_gini = 0.0;
                let node = Node::internal_full(split.clone(), num_samples, num_plus, split_gini);
                self.nodes.insert(node_id, node);

                let split_idx = partition_in_place(samples, &split);
                let (left_samples, right_samples) = samples.split_at_mut(split_idx);

                if left_samples.len() < self.config.min_samples_leaf
                    || right_samples.len() < self.config.min_samples_leaf
                {
                    self.nodes.remove(&node_id);
                    self.create_leaf(samples, node_id, num_samples, num_plus);
                    return;
                }

                self.build_node(
                    dataset,
                    left_samples,
                    node_id::left_child(node_id),
                    depth + 1,
                    stats.impurity_left,
                );

                self.build_node(
                    dataset,
                    right_samples,
                    node_id::right_child(node_id),
                    depth + 1,
                    stats.impurity_right,
                );
            }
            _ => {
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
    /// DynFrs random mode — tree-shared self.rng 사용 (mt19937 정합).
    fn find_best_split<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        samples: &[S],
        impurity_before: f64,
        _node_id: u64,
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

        let selected: Vec<u8> = feature_indices.iter().take(max_features).copied().collect();
        let mut node_ranges: Vec<(f32, f32)> = vec![(f32::MAX, f32::MIN); selected.len()];
        for s in samples {
            for (i, &attr_idx) in selected.iter().enumerate() {
                let v = s.attribute_value(attr_idx);
                if v < node_ranges[i].0 {
                    node_ranges[i].0 = v;
                }
                if v > node_ranges[i].1 {
                    node_ranges[i].1 = v;
                }
            }
        }

        for (i, &attr_idx) in selected.iter().enumerate() {
            let (min_val, max_val) = node_ranges[i];
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
                        let subset = self.rng.gen_range(1..(1u64 << cardinality.min(63)));
                        Split::categorical(attr_idx, subset)
                    }
                };

                let mut stats = scan_auto(samples, &split);
                stats.update_score(impurity_before);

                if stats.has_positive_score() {
                    let current_score = stats.score.unwrap();
                    if best_score.is_none_or(|s| current_score > s) {
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

    /// Check if the tree contains a sample.
    pub fn contains_sample(&self, sample_id: u64) -> bool {
        self.sample_leaf_map.contains_key(&sample_id)
    }

    //   develop_with_age / develop_node_with_age 모두 제거.
    //   A1 path-amortized lazy_resolve_this_node + predict_with_lazy_resolve로 대체.
    //   split_max_age path(_with_age 계열)는 PR-5 이미 controller 호출 끊김.
    //
    //   `rebuild_subtree_optimized_with_position`은 lazy_resolve_this_node가 호출 (live).

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
    fn build_node_from_refs<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_refs: &[&S],
        node_id: u64,
        depth: usize,
        impurity_before: f64,
    ) {
        let num_samples = sample_refs.len() as u32;
        let num_plus = sample_refs.iter().filter(|s| s.label()).count() as u32;

        let should_stop = depth >= self.config.max_depth
            || sample_refs.len() < self.config.min_samples_split
            || num_plus == 0
            || num_plus == num_samples;

        if should_stop {
            self.create_leaf_from_refs(sample_refs, node_id, num_samples, num_plus);
            return;
        }

        let best_split = self.find_best_split_refs(dataset, sample_refs, impurity_before, node_id);

        match best_split {
            Some((split, stats)) if stats.has_positive_score() => {
                let split_gini = 0.0;
                let node = Node::internal_full(split.clone(), num_samples, num_plus, split_gini);
                self.nodes.insert(node_id, node);

                let (left_indices, right_indices) = partition_indices(sample_refs, &split);

                if left_indices.len() < self.config.min_samples_leaf
                    || right_indices.len() < self.config.min_samples_leaf
                {
                    self.nodes.remove(&node_id);
                    self.create_leaf_from_refs(sample_refs, node_id, num_samples, num_plus);
                    return;
                }

                let left_refs: Vec<&S> = left_indices.iter().map(|&i| sample_refs[i]).collect();
                let right_refs: Vec<&S> = right_indices.iter().map(|&i| sample_refs[i]).collect();

                self.build_node_from_refs(
                    dataset,
                    &left_refs,
                    node_id::left_child(node_id),
                    depth + 1,
                    stats.impurity_left,
                );

                self.build_node_from_refs(
                    dataset,
                    &right_refs,
                    node_id::right_child(node_id),
                    depth + 1,
                    stats.impurity_right,
                );
            }
            _ => {
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
    /// Mirrors `find_best_split` (same RNG consumption order) so that, in
    /// deterministic mode, develop()-rebuild produces the same split as fit()
    /// for the same `node_id` and samples.
    fn find_best_split_refs<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_refs: &[&S],
        impurity_before: f64,
        _node_id: u64,
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

        let selected: Vec<u8> = feature_indices.iter().take(max_features).copied().collect();
        let mut node_ranges: Vec<(f32, f32)> = vec![(f32::MAX, f32::MIN); selected.len()];
        for s in sample_refs {
            for (i, &attr_idx) in selected.iter().enumerate() {
                let v = s.attribute_value(attr_idx);
                if v < node_ranges[i].0 {
                    node_ranges[i].0 = v;
                }
                if v > node_ranges[i].1 {
                    node_ranges[i].1 = v;
                }
            }
        }

        for (i, &attr_idx) in selected.iter().enumerate() {
            let (min_val, max_val) = node_ranges[i];
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
                        let subset = self.rng.gen_range(1..(1u64 << cardinality.min(63)));
                        Split::categorical(attr_idx, subset)
                    }
                };

                let mut stats = scan_refs(sample_refs, &split);
                stats.update_score(impurity_before);

                if stats.has_positive_score() {
                    let current_score = stats.score.unwrap();
                    if best_score.is_none_or(|s| current_score > s) {
                        best_score = Some(current_score);
                        best_split = Some((split, stats));
                    }
                }
            }
        }

        best_split
    }

    fn rebuild_subtree_optimized<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        node_id: u64,
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
        self.build_node_from_refs(dataset, &sample_refs, node_id, depth, impurity);

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

    // =========================================================================
    // A1: Path-amortized lazy resolve (DynFrs qry() 정합).
    //
    // Mapping:
    //   DynFrs split()        ↔ this 노드 best_split 결정 + child Internal 마킹
    //   DynFrs separate()     ↔ this 노드 sample을 새 threshold로 child Leaf에 분배
    //   DynFrs delay==1/2     ↔ LazyTag::Rebuild + DelayTag::NeedsBuild/NeedsSeparateAndBuild
    //   DynFrs qry() (line 558-564): if (delay) split()+separate(); 한 child만 재귀
    //
    // 핵심 차이 (rebuild_subtree_optimized_with_position과):
    //   subtree-rebuild: 자손 전부 재귀 build → max_depth까지 완성 subtree
    //   single-step    : 이 노드의 split만 결정, child는 fresh Leaf로 생성
    //                    → 자손의 resolve는 다음 query가 그 child로 갈 때
    // =========================================================================

    /// A1 lazy resolve primitive (Step 1 재설계, advisor 권고):
    /// **DynFrs split()+separate() 정확 정합 — 자식 subtree 보존**.
    ///
    /// 동작:
    /// 1. 이 노드의 streaming_states.sample_ids로 새 best_split 결정 (DynFrs split)
    /// 2. sample을 새 threshold로 left/right child에 분배 (DynFrs separate)
    /// 3. 자식 *subtree 구조는 그대로 유지* — child의 streaming_states.sample_ids만
    ///    overwrite + AttributeStats reset + LazyTag::Dirty 마킹
    /// 4. 자손은 lazy — 다음 query가 그 child로 routing되면 점진 resolve
    ///
    /// 이전 시도 (single-step + fresh Leaf)는 query depth가 얕아져 fail. 이번엔
    /// child *Internal subtree 보존*하므로 prediction depth 유지 + 점진 build.
    ///
    /// Fallback: streaming_states 없거나 sample 0이면 rebuild_subtree_optimized (안전).
    /// LazyTag::Clean 또는 Leaf 노드면 noop (idempotent).
    fn lazy_resolve_this_node<D: Dataset, S: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        node_id: u64,
    ) {
        let needs_resolve = matches!(
            self.nodes.get(&node_id),
            Some(Node::Internal { lazy_tag, .. })
                if matches!(lazy_tag, LazyTag::Dirty | LazyTag::Rebuild)
        );
        if !needs_resolve {
            return;
        }

        // streaming_states 없으면 (예: streaming_enabled=false) fallback to subtree rebuild
        let sample_ids: Vec<u64> = match self.streaming_states.get(&node_id) {
            Some(state) if !state.sample_ids.is_empty() => state.sample_ids.clone(),
            _ => {
                self.rebuild_subtree_optimized(dataset, sample_map, node_id);
                return;
            }
        };

        // sample refs 수집 (sample_map에 있는 것만)
        let sample_refs: Vec<&S> = sample_ids
            .iter()
            .filter_map(|id| sample_map.get(id).copied())
            .collect();
        if sample_refs.is_empty() {
            // sample_map에 features 없으면 fallback
            self.rebuild_subtree_optimized(dataset, sample_map, node_id);
            return;
        }

        let num_samples = sample_refs.len() as u32;
        let num_plus = sample_refs.iter().filter(|s| s.label()).count() as u32;

        // Fix 1 (DynFrs `leaf()` 정합): pure-class / sample 부족 / max_depth 도달 시
        //   *Leaf로 변환 + 자손 subtree 청소*. DynFrs `del(id)` line 595-596의
        //   `ls != nullptr && leaf() → concentrate` 정합 — Internal + 옛 split 그대로
        //   유지하는 SUDA-specific 의도적 divergence 제거 (회장님 명령 "최대한 정합").
        let depth = node_id::depth(node_id) as usize;
        let should_be_leaf = depth >= self.config.max_depth
            || (num_samples as usize) < self.config.min_samples_split
            || num_plus == 0
            || num_plus == num_samples;

        if should_be_leaf {
            // 1. 자손 subtree의 모든 sample_id 수집
            let descendant_samples = self.collect_samples_from_subtree(node_id);
            // 2. 자손의 sample_leaf_map 정리
            for old_id in &descendant_samples {
                self.sample_leaf_map.remove(old_id);
            }
            // 3. 자손 subtree 노드 + streaming_states 제거
            self.remove_subtree_batch(node_id);
            self.nodes.remove(&node_id);
            // streaming_states cleanup: 자손 노드들의 state 제거
            //   collect_samples_from_subtree는 sample_id만 수집. node_id 별도 수집.
            let mut to_remove = vec![node_id];
            let mut idx = 0;
            while idx < to_remove.len() {
                let nid = to_remove[idx];
                idx += 1;
                let left = node_id::left_child(nid);
                let right = node_id::right_child(nid);
                if self.streaming_states.contains_key(&left) {
                    to_remove.push(left);
                }
                if self.streaming_states.contains_key(&right) {
                    to_remove.push(right);
                }
            }
            for nid in &to_remove {
                self.streaming_states.remove(nid);
            }
            // 4. Leaf 생성 + sample_leaf_map 새로 등록
            let sample_ids_set: HashSet<u64> = sample_ids.iter().copied().collect();
            for &id in &sample_ids {
                self.sample_leaf_map.insert(id, node_id);
            }
            let leaf = Node::leaf_with_samples(num_samples, num_plus, sample_ids_set);
            self.nodes.insert(node_id, leaf);
            return;
        }

        let impurity = gini_impurity(num_plus, num_samples);

        // 1. 새 best_split 결정 (DynFrs split() 정합, node-seeded RNG)
        let best_split = self.find_best_split_refs(dataset, &sample_refs, impurity, node_id);
        let new_split = match best_split {
            Some((split, stats)) if stats.has_positive_score() => split,
            _ => {
                // 유효 split 없음 — 노드 그대로 두고 lazy_tag Clean
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.set_lazy_tag(LazyTag::Clean);
                }
                return;
            }
        };

        // 2. 새 threshold로 sample을 left/right child에 분배 (DynFrs separate() 정합)
        let (left_indices, right_indices) = partition_indices(&sample_refs, &new_split);
        let left_ids: Vec<u64> = left_indices.iter().map(|&i| sample_refs[i].id()).collect();
        let right_ids: Vec<u64> = right_indices.iter().map(|&i| sample_refs[i].id()).collect();
        let left_count = left_ids.len() as u32;
        let left_plus = left_indices
            .iter()
            .filter(|&&i| sample_refs[i].label())
            .count() as u32;
        let right_count = right_ids.len() as u32;
        let right_plus = right_indices
            .iter()
            .filter(|&&i| sample_refs[i].label())
            .count() as u32;

        // 3. 이 노드의 split + lazy_tag + counts 갱신 (자식 subtree는 그대로!)
        if let Some(Node::Internal {
            split,
            lazy_tag,
            num_samples: ns,
            num_plus: np,
            ..
        }) = self.nodes.get_mut(&node_id)
        {
            *split = new_split;
            *lazy_tag = LazyTag::Clean;
            *ns = num_samples;
            *np = num_plus;
        }

        // 4. 자식 노드의 sample_ids overwrite + AttributeStats 부분 regenerate (Step 2)
        //    + LazyTag::Dirty 마킹. 자식 subtree 구조는 그대로 — 다음 query 시 점진 resolve.
        //
        //    Step 2 (DynFrs gen_spl 정합):
        //    단순 reset 대신 init_from_batch로 *새 sample_ids로 candidates 다시 생성*.
        //    이전 destroy + 다음 add_sample 시점 채워지길 기다리는 게 아니라 *즉시 정확한 stats*.
        let left_id = node_id::left_child(node_id);
        let right_id = node_id::right_child(node_id);
        let num_attrs = self.num_attributes;

        // sample_map closure 빌드용 helper
        let build_features = |id: u64| -> Vec<f32> {
            sample_map
                .get(&id)
                .map(|s| {
                    (0..num_attrs as usize)
                        .map(|i| s.attribute_value(i as u8))
                        .collect()
                })
                .unwrap_or_else(|| vec![0.0; num_attrs as usize])
        };
        let get_label =
            |id: u64| -> bool { sample_map.get(&id).map(|s| s.label()).unwrap_or(false) };

        if let Some(left_state) = self.streaming_states.get_mut(&left_id) {
            left_state.sample_ids = left_ids.clone();
            // 부분 regenerate — sample_ids로 candidates 다시 생성 (DynFrs gen_spl 정합)
            left_state
                .attr_stats
                .init_from_batch(&left_ids, build_features, get_label, P_COUNT);
            left_state.delay = DelayTag::None;
        }
        if let Some(right_state) = self.streaming_states.get_mut(&right_id) {
            right_state.sample_ids = right_ids.clone();
            right_state
                .attr_stats
                .init_from_batch(&right_ids, build_features, get_label, P_COUNT);
            right_state.delay = DelayTag::None;
        }

        // 자식 Node 갱신 — Internal이면 counts + LazyTag::Dirty, Leaf면 sample_ids overwrite
        self.update_child_after_resplit(left_id, &left_ids, left_count, left_plus);
        self.update_child_after_resplit(right_id, &right_ids, right_count, right_plus);
    }

    /// Helper for lazy_resolve_this_node — child 노드 갱신 (Internal: Dirty / Leaf: overwrite)
    fn update_child_after_resplit(
        &mut self,
        child_id: u64,
        new_sample_ids: &[u64],
        num_samples: u32,
        num_plus: u32,
    ) {
        if let Some(node) = self.nodes.get_mut(&child_id) {
            match node {
                Node::Internal {
                    num_samples: ns,
                    num_plus: np,
                    lazy_tag,
                    ..
                } => {
                    *ns = num_samples;
                    *np = num_plus;
                    *lazy_tag = LazyTag::Dirty;
                }
                Node::Leaf {
                    num_samples: ns,
                    num_plus: np,
                    sample_ids,
                } => {
                    *ns = num_samples;
                    *np = num_plus;
                    // sample_leaf_map 갱신 — 기존 sample 제거 후 새로
                    let old_ids: Vec<u64> = sample_ids.iter().copied().collect();
                    for old_id in &old_ids {
                        self.sample_leaf_map.remove(old_id);
                    }
                    // Leaf의 sample_ids overwrite
                    if let Some(Node::Leaf { sample_ids, .. }) = self.nodes.get_mut(&child_id) {
                        sample_ids.clear();
                        for &id in new_sample_ids {
                            sample_ids.insert(id);
                        }
                    }
                    // 새 sample_leaf_map 등록
                    for &id in new_sample_ids {
                        self.sample_leaf_map.insert(id, child_id);
                    }
                }
            }
        }
    }

    /// A1 path-amortized predict: query path 따라가며 LazyTag 만나면 그 노드만 resolve.
    /// DynFrs.h:558-564 qry() 정합 (SUDA 자료구조 맞춰 subtree rebuild로 매핑).
    pub fn predict_with_lazy_resolve<D: Dataset, S: Sample, Q: Sample>(
        &mut self,
        dataset: &D,
        sample_map: &HashMap<u64, &S>,
        query: &Q,
    ) -> bool {
        let mut node_id = node_id::ROOT;
        loop {
            // 1. 이 노드 lazy resolve (DynFrs: if (delay) { split(); separate(); })
            self.lazy_resolve_this_node(dataset, sample_map, node_id);

            // 2. routing or leaf return (DynFrs: ls == nullptr ? return : recurse)
            match self.nodes.get(&node_id) {
                Some(Node::Internal { split, .. }) => {
                    if query.is_left_of(split) {
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
                None => return false,
            }
        }
    }

    /// Get the number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// A1 inspect: collect all internal node ids (test/audit).
    pub fn internal_node_ids(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter_map(|(&id, n)| {
                if matches!(n, Node::Internal { .. }) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// A1 inspect: get the LazyTag of a specific internal node.
    /// Returns None if node is Leaf or doesn't exist.
    pub fn lazy_tag_of(&self, node_id: u64) -> Option<LazyTag> {
        self.nodes.get(&node_id).and_then(|n| n.lazy_tag())
    }

    /// A1 inspect: count internal nodes by LazyTag state — (clean, dirty, rebuild).
    pub fn lazy_tag_counts(&self) -> (usize, usize, usize) {
        let mut clean = 0;
        let mut dirty = 0;
        let mut rebuild = 0;
        for n in self.nodes.values() {
            if let Some(tag) = n.lazy_tag() {
                match tag {
                    LazyTag::Clean => clean += 1,
                    LazyTag::Dirty => dirty += 1,
                    LazyTag::Rebuild => rebuild += 1,
                }
            }
        }
        (clean, dirty, rebuild)
    }

    /// A1 test helper: force a specific LazyTag on an internal node.
    /// Used by lazy_resolve tests to set up deterministic Dirty scenarios.
    pub fn force_lazy_tag_for_test(&mut self, node_id: u64, tag: LazyTag) -> bool {
        if let Some(n) = self.nodes.get_mut(&node_id) {
            n.set_lazy_tag(tag);
            true
        } else {
            false
        }
    }

    /// Get the number of samples tracked.
    pub fn num_samples(&self) -> usize {
        self.sample_leaf_map.len()
    }

    /// Get the sample count recorded at a specific node (Internal or Leaf).
    /// Returns None if the node does not exist. Used by tests to verify that
    /// DynFrs-compatible forget propagates count updates along the path.
    pub fn node_sample_count(&self, node_id: u64) -> Option<usize> {
        self.nodes.get(&node_id).map(|n| match n {
            Node::Internal { num_samples, .. } => *num_samples as usize,
            Node::Leaf { num_samples, .. } => *num_samples as usize,
        })
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
                Some(Node::Internal {
                    split,
                    num_samples,
                    num_plus,
                    ..
                }) => {
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
                Some(Node::Leaf {
                    num_samples,
                    num_plus,
                    sample_ids,
                    ..
                }) => {
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

    /// Check if a leaf needs to be split (exceeded capacity).
    ///
    /// This can be called after incremental additions to determine
    /// if the tree structure should be updated.
    pub fn leaf_should_split(&self, leaf_id: u64) -> bool {
        match self.nodes.get(&leaf_id) {
            Some(Node::Leaf {
                num_samples,
                num_plus,
                ..
            }) => {
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

    //   perform_hoeffding_split, add_sample_streaming_hoeffding).
    //   DynFrs는 ERT 알고리즘 사용 → Hoeffding 자체 불필요.

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

    fn add_sample_streaming_inner<S: Sample>(
        &mut self,
        sample: &S,
        features: &[f32],
        use_lazy_rebuild: bool,
        _sample_info: Option<SampleInfoFn>,
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
                        if let Some(Node::Internal { lazy_tag, .. }) =
                            self.nodes.get_mut(&current_id)
                        {
                            *lazy_tag = LazyTag::Dirty;
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
                Some(Node::Internal {
                    split,
                    num_samples,
                    num_plus,
                    ..
                }) => {
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
                Some(Node::Leaf {
                    num_samples,
                    num_plus,
                    sample_ids,
                    ..
                }) => {
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

                    if self.leaf_should_split(current_id) {
                        if self.streaming_enabled {
                            if let Some(state) = self.streaming_states.get_mut(&current_id) {
                                state.delay = DelayTag::NeedsBuild;
                            }
                        }
                        needs_rebuild = true;
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

        // Update streaming states + internal node counts along path to root.
        // C2 fix (DynFrs 정합, 회장님 결정 — 의도적 divergence 제거):
        // DynFrs attribute::del (DynFrs.h:268-298): n -= 1, n_1 -= Y for *all* path nodes.
        // 이전 SUDA는 leaf만 갱신, internal stale로 두어 forget 부정 효과 막으려 했음.
        // 그러나 그 디자인이 *realdrift effect 약화의 근원* — best_split_changed가
        // 정확한 internal count로 판정되어야 lazy resolve가 진짜 trigger.
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

                    state.sample_ids.retain(|&id| id != sample_id);
                }
            }

            // C2 fix: internal node count도 path 따라 갱신 (DynFrs.h:268-298 정합)
            if let Some(Node::Internal {
                num_samples,
                num_plus,
                ..
            }) = self.nodes.get_mut(&current_id)
            {
                *num_samples = num_samples.saturating_sub(1);
                if was_positive {
                    *num_plus = num_plus.saturating_sub(1);
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
        self.init_streaming_states_recursive(node_id::ROOT, 0, &sample_map, &get_features);
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
                    |id| {
                        sample_map
                            .get(&id)
                            .map(|s| get_features(s))
                            .unwrap_or_default()
                    },
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
                    |id| {
                        sample_map
                            .get(&id)
                            .map(|s| get_features(s))
                            .unwrap_or_default()
                    },
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

        // Forget a sample via the DynFrs-compatible streaming-aware path
        assert!(tree.contains_sample(0));
        let features_0: Vec<f32> = samples[0].values.to_vec();
        let (leaf, _) = tree.remove_sample_streaming(0, &features_0, true, true);
        assert!(leaf.is_some(), "Sample 0 should have been removed");
        assert!(!tree.contains_sample(0));

        // Try to forget non-existent sample
        let (leaf, _) = tree.remove_sample_streaming(999, &[0.0, 0.0], false, true);
        assert!(leaf.is_none(), "Non-existent sample should return None");
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
        };

        let mut tree = DynFrsTree::new(0, 42, config, 2);
        tree.fit(&dataset, &mut samples);

        // Initial state
        let initial_nodes = tree.num_nodes();
        assert_eq!(tree.num_samples(), 8);

        // Forget half the samples (4 positive samples) via streaming-aware path
        for i in 0..4 {
            let features: Vec<f32> = samples[i as usize].values.to_vec();
            tree.remove_sample_streaming(i, &features, true, true);
        }
        assert_eq!(tree.num_samples(), 4);

        // Remaining samples (keep only the ones still in tree)
        let remaining_samples: Vec<_> = samples.iter().filter(|s| s.id() >= 4).cloned().collect();

        let sample_map: HashMap<u64, &_> = remaining_samples.iter().map(|s| (s.id(), s)).collect();
        for sample in &remaining_samples {
            let _ = tree.predict_with_lazy_resolve(&dataset, &sample_map, sample);
        }

        // Tree should still be valid
        assert!(
            tree.num_nodes() > 0,
            "Tree should have nodes after lazy resolve"
        );
        assert_eq!(tree.num_samples(), 4, "Should still track 4 samples");

        // Predictions should still work for remaining samples
        for sample in &remaining_samples {
            let _pred = tree.predict(sample); // Should not panic
        }

        println!(
            "lazy_resolve test passed: nodes {} -> {}, samples 8 -> 4",
            initial_nodes,
            tree.num_nodes()
        );
    }

    //   test_fit_weighted_imbalanced 제거 (ClassWeightConfig + fit_weighted 사라짐).
}
