//! InfluenceRegistry - Full Sample Tracking for Selective Unlearning
//!
//! This module provides memory-efficient tracking of ALL samples for selective
//! unlearning. Every sample's influence across trees is tracked with budget-based
//! eviction to maintain model freshness.
//!
//! # Memory Management
//!
//! Fixed memory cap (default: 100MB) with oldest-first eviction when limit is reached.
//!
//! Full tracking: ~40 bytes/sample (id + trees + label + position + overhead)
//! At 100MB limit: ~250万 full samples

use hashbrown::HashMap;
use std::collections::BTreeMap;

/// Default memory limit: 100MB
const DEFAULT_MEMORY_LIMIT: usize = 100 * 1024 * 1024;

/// Configuration for sample budget management (continuous eviction).
///
/// When enabled, the registry maintains a maximum number of samples.
/// New samples that exceed the budget trigger eviction of lowest-quality
/// samples based on a composite score combining age, influence, and class.
///
/// This provides continuous value even in mild drift scenarios where
/// reactive triggers never fire.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Maximum number of samples in the registry
    pub max_samples: usize,
    /// Number of samples to evict when budget is exceeded
    pub eviction_batch_size: usize,
    /// Minority class protection: if a class has fewer than this ratio
    /// of total samples, its samples are protected from eviction
    pub minority_protection_ratio: f64,
    /// Weight for age component in eviction score (higher = prefer evicting old)
    pub age_weight: f64,
    /// Weight for influence component (higher = prefer evicting harmful)
    pub influence_weight: f64,
    /// Weight for class penalty (higher = more minority protection)
    pub class_weight: f64,
    /// Reservoir baseline: evict uniformly at random instead of by composite score.
    /// Ignores age/influence/class. Used as a CL baseline (random retention)
    /// to isolate the contribution of drift-aware selection. Default false.
    pub random_eviction: bool,
    /// Q2-1 ablation: keep class_penalty + protection_factor identical to the
    /// composite path, but replace the age signal (norm_age) with a uniform
    /// random value. Isolates "does age-based selection add anything beyond
    /// minority protection?". Default false.
    pub class_aware_random: bool,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            max_samples: 10000,
            eviction_batch_size: 100,
            minority_protection_ratio: 0.1,
            age_weight: 0.4,
            influence_weight: 0.4,
            class_weight: 0.2,
            random_eviction: false,
            class_aware_random: false,
        }
    }
}

/// Statistics from a budget eviction event.
#[derive(Debug, Clone, Default)]
pub struct EvictionStats {
    /// Number of samples evicted
    pub evicted_count: usize,
    /// Number of benign samples evicted
    pub evicted_benign: usize,
    /// Number of attack samples evicted
    pub evicted_attack: usize,
    /// Number of influence-degraded samples evicted
    pub evicted_degraded: usize,
    /// Number of evicted samples that had cached_influence != None
    pub evicted_with_influence: usize,
    /// Sum of influence scores of evicted samples (for avg calculation)
    pub evicted_influence_sum: f64,
}

/// Maximum number of history entries to retain per sample.
/// Keeps memory bounded for long-running streams.
const MAX_HISTORY_LEN: usize = 20;

/// Sample metadata stored in the registry.
#[derive(Debug, Clone)]
pub struct TrackedSample {
    /// Tree indices where this sample appears (OCC mapping)
    pub tree_indices: Vec<usize>,
    /// Class label (true = positive/attack, false = negative/benign)
    pub label: bool,
    /// Stream position when inserted
    pub position: u64,
    /// Cached influence score (None if not computed)
    pub cached_influence: Option<f64>,
    /// Previous influence score for tracking influence drift (positive→negative transitions)
    pub prev_influence: Option<f64>,

    // === NEW: History tracking for sample analysis ===
    /// Influence score history: (position, influence_score)
    /// Tracks how the sample's influence changes over time
    pub influence_history: Vec<(u64, f64)>,
    /// Removal rank history: (position, rank)
    /// Tracks how close the sample was to being removed
    pub removal_rank_history: Vec<(u64, usize)>,
}

impl TrackedSample {
    /// Create a new fully tracked sample.
    pub fn new(tree_indices: Vec<usize>, label: bool, position: u64) -> Self {
        Self {
            tree_indices,
            label,
            position,
            cached_influence: None,
            prev_influence: None,
            influence_history: Vec::new(),
            removal_rank_history: Vec::new(),
        }
    }

    /// Estimate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        // Base: 8 (tree_indices vec header) + 1 (label) + 9 (position Option) + 9 (cached_influence Option)
        // + 9 (prev_influence Option) + 24 (influence_history vec header) + 24 (removal_rank_history vec header)
        let base = 84;
        // Tree indices: 8 bytes each (usize)
        let tree_bytes = self.tree_indices.len() * std::mem::size_of::<usize>();
        // Influence history: 16 bytes each (u64 + f64)
        let influence_bytes = self.influence_history.len() * 16;
        // Removal rank history: 16 bytes each (u64 + usize)
        let rank_bytes = self.removal_rank_history.len() * 16;
        base + tree_bytes + influence_bytes + rank_bytes
    }

    /// Record influence score at current position.
    /// Keeps at most MAX_HISTORY_LEN entries (drops oldest on overflow).
    pub fn record_influence(&mut self, position: u64, influence: f64) {
        self.prev_influence = self.cached_influence;
        if self.influence_history.len() >= MAX_HISTORY_LEN {
            self.influence_history.drain(..1);
        }
        self.influence_history.push((position, influence));
        self.cached_influence = Some(influence);
    }

    /// Get the influence delta (current - previous). Negative = worsening.
    pub fn influence_delta(&self) -> Option<f64> {
        match (self.cached_influence, self.prev_influence) {
            (Some(current), Some(prev)) => Some(current - prev),
            _ => None,
        }
    }

    /// Check if influence has degraded from positive to negative.
    pub fn is_influence_degraded(&self) -> bool {
        match (self.cached_influence, self.prev_influence) {
            (Some(current), Some(prev)) => prev >= 0.0 && current < 0.0,
            _ => false,
        }
    }

    /// Record removal rank at current position.
    /// Keeps at most MAX_HISTORY_LEN entries (drops oldest on overflow).
    pub fn record_removal_rank(&mut self, position: u64, rank: usize) {
        if self.removal_rank_history.len() >= MAX_HISTORY_LEN {
            self.removal_rank_history.drain(..1);
        }
        self.removal_rank_history.push((position, rank));
    }

    /// Get the average influence over time.
    pub fn average_influence(&self) -> Option<f64> {
        if self.influence_history.is_empty() {
            return self.cached_influence;
        }
        let sum: f64 = self.influence_history.iter().map(|(_, inf)| inf).sum();
        Some(sum / self.influence_history.len() as f64)
    }

    /// Get the trend of influence (positive = improving, negative = worsening).
    pub fn influence_trend(&self) -> Option<f64> {
        if self.influence_history.len() < 2 {
            return None;
        }
        let n = self.influence_history.len();
        let first_half: f64 = self.influence_history[..n / 2]
            .iter()
            .map(|(_, inf)| inf)
            .sum();
        let second_half: f64 = self.influence_history[n / 2..]
            .iter()
            .map(|(_, inf)| inf)
            .sum();
        let first_avg = first_half / (n / 2) as f64;
        let second_avg = second_half / (n - n / 2) as f64;
        Some(second_avg - first_avg)
    }

    /// Get the number of times this sample was in top-N removal candidates.
    pub fn removal_risk_count(&self, top_n: usize) -> usize {
        self.removal_rank_history
            .iter()
            .filter(|(_, rank)| *rank < top_n)
            .count()
    }
}

/// Sample lifecycle data for analysis export.
#[derive(Debug, Clone)]
pub struct SampleLifecycle {
    /// Sample ID
    pub id: u64,
    /// Class label
    pub label: bool,
    /// Stream position when inserted
    pub insertion_position: u64,
    /// Influence score history: (position, score)
    pub influence_history: Vec<(u64, f64)>,
    /// Removal rank history: (position, rank)
    pub removal_rank_history: Vec<(u64, usize)>,
    /// Average influence score
    pub average_influence: Option<f64>,
    /// Influence trend (positive = improving)
    pub influence_trend: Option<f64>,
}

/// Full sample tracking registry for selective unlearning.
///
/// InfluenceRegistry tracks ALL samples with memory-efficient strategies
/// and budget-based eviction for continuous exact forgetting.
#[derive(Debug)]
pub struct InfluenceRegistry {
    /// Sample metadata: sample_id -> TrackedSample
    samples: HashMap<u64, TrackedSample>,
    /// Position index: position -> sample_id (for time-based operations)
    /// Uses BTreeMap for efficient range queries
    position_index: BTreeMap<u64, u64>,
    /// Maximum memory in bytes (default: 100MB). Oldest samples evicted when exceeded.
    max_bytes: usize,
    /// Current estimated memory usage in bytes
    current_bytes: usize,
    /// Total samples ever registered (monotonically increasing)
    total_registered: u64,
    /// Current stream position
    current_position: u64,
    /// Class counts [negative, positive]
    class_counts: [usize; 2],
    /// Budget configuration for continuous eviction (None = disabled)
    budget_config: Option<BudgetConfig>,
    /// Cumulative eviction statistics
    cumulative_eviction_stats: EvictionStats,
    /// Whether influence tracking is enabled (prev_influence updates)
    influence_tracking_enabled: bool,
}

impl InfluenceRegistry {
    /// Create a new registry with default memory limit (100MB).
    pub fn new() -> Self {
        Self::with_max_bytes(DEFAULT_MEMORY_LIMIT)
    }

    /// Create a new registry with the specified memory limit in bytes.
    /// Oldest samples are evicted when the limit is exceeded.
    pub fn with_fixed_limit(max_bytes: usize) -> Self {
        Self::with_max_bytes(max_bytes)
    }

    /// Create a new registry with the specified memory limit in bytes.
    pub fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            samples: HashMap::new(),
            position_index: BTreeMap::new(),
            max_bytes,
            current_bytes: 0,
            total_registered: 0,
            current_position: 0,
            class_counts: [0, 0],
            budget_config: None,
            cumulative_eviction_stats: EvictionStats::default(),
            influence_tracking_enabled: false,
        }
    }

    /// Enable sample budget management with the given configuration.
    pub fn set_budget_config(&mut self, config: BudgetConfig) {
        self.budget_config = Some(config);
    }

    /// Dynamically adjust the budget max_samples (for adaptive budget).
    pub fn set_budget_max_samples(&mut self, max_samples: usize) {
        if let Some(ref mut config) = self.budget_config {
            config.max_samples = max_samples;
        }
    }

    /// Enable influence tracking (prev_influence updates on set_influence).
    pub fn set_influence_tracking(&mut self, enabled: bool) {
        self.influence_tracking_enabled = enabled;
    }

    /// Register a new sample with its tree indices.
    ///
    /// Returns the evicted sample IDs if any (in fixed limit mode).
    pub fn register(&mut self, sample_id: u64, tree_indices: Vec<usize>, label: bool) -> Vec<u64> {
        self.register_internal(sample_id, tree_indices, label);
        self.enforce_limits()
    }

    /// Register a sample without triggering budget/memory enforcement.
    /// Use this in batch operations where enforcement should happen once at the end.
    fn register_internal(&mut self, sample_id: u64, tree_indices: Vec<usize>, label: bool) {
        let position = self.current_position;
        self.current_position += 1;
        self.total_registered += 1;

        let sample = TrackedSample::new(tree_indices, label, position);
        let sample_bytes = sample.memory_bytes();

        // Check if sample already exists (update case)
        if let Some(old_sample) = self.samples.get(&sample_id) {
            self.current_bytes -= old_sample.memory_bytes();
            self.class_counts[old_sample.label as usize] -= 1;
            self.position_index.remove(&old_sample.position);
        }

        // Insert new sample
        self.samples.insert(sample_id, sample);
        self.position_index.insert(position, sample_id);
        self.current_bytes += sample_bytes;
        self.class_counts[label as usize] += 1;
    }

    /// Enforce memory limits by evicting oldest samples.
    fn enforce_memory(&mut self) -> Vec<u64> {
        self.enforce_fixed_limit(self.max_bytes)
    }

    /// Register multiple samples (batch operation).
    /// Optimized: registers all samples first, then enforces budget once.
    pub fn register_batch(
        &mut self,
        sample_ids: &[u64],
        tree_indices_list: &[Vec<usize>],
        labels: &[bool],
    ) -> Vec<u64> {
        // Register all samples without per-sample budget enforcement
        for ((&id, indices), &label) in sample_ids
            .iter()
            .zip(tree_indices_list.iter())
            .zip(labels.iter())
        {
            self.register_internal(id, indices.clone(), label);
        }

        self.enforce_limits()
    }

    /// Register a batch using pure FIFO eviction (no composite scoring).
    ///
    /// Used during fit() warmup to ensure the registry's class distribution
    /// faithfully reflects the tail of the warmup data, without class-weighted
    /// scoring bias that can lock in the initial distribution.
    pub fn register_batch_fifo(
        &mut self,
        sample_ids: &[u64],
        tree_indices_list: &[Vec<usize>],
        labels: &[bool],
    ) -> Vec<u64> {
        for ((&id, indices), &label) in sample_ids
            .iter()
            .zip(tree_indices_list.iter())
            .zip(labels.iter())
        {
            self.register_internal(id, indices.clone(), label);
        }

        // Pure FIFO: evict oldest samples until within budget
        let max_samples = self
            .budget_config
            .as_ref()
            .map(|c| c.max_samples)
            .unwrap_or(usize::MAX);

        let mut evicted = Vec::new();
        while self.samples.len() > max_samples && !self.position_index.is_empty() {
            if let Some((&oldest_pos, &oldest_id)) = self.position_index.iter().next() {
                if let Some(sample) = self.samples.remove(&oldest_id) {
                    self.current_bytes -= sample.memory_bytes();
                    self.class_counts[sample.label as usize] -= 1;
                    self.position_index.remove(&oldest_pos);
                    evicted.push(oldest_id);
                }
            }
        }
        evicted
    }

    /// Enforce budget and memory limits, returning all evicted sample IDs.
    fn enforce_limits(&mut self) -> Vec<u64> {
        let budget_evicted = self.enforce_budget();
        let memory_evicted = self.enforce_memory();

        if budget_evicted.is_empty() {
            memory_evicted
        } else if memory_evicted.is_empty() {
            budget_evicted
        } else {
            let mut all = budget_evicted;
            all.extend(memory_evicted);
            all
        }
    }

    /// Remove a sample from the registry.
    pub fn remove(&mut self, sample_id: u64) -> bool {
        if let Some(sample) = self.samples.remove(&sample_id) {
            self.current_bytes -= sample.memory_bytes();
            self.class_counts[sample.label as usize] -= 1;
            self.position_index.remove(&sample.position);
            true
        } else {
            false
        }
    }

    /// Remove multiple samples (batch operation).
    pub fn remove_batch(&mut self, sample_ids: &[u64]) -> usize {
        let mut count = 0;
        for &id in sample_ids {
            if self.remove(id) {
                count += 1;
            }
        }
        count
    }

    /// Get sample by ID.
    pub fn get(&self, sample_id: u64) -> Option<&TrackedSample> {
        self.samples.get(&sample_id)
    }

    /// Get mutable sample by ID.
    pub fn get_mut(&mut self, sample_id: u64) -> Option<&mut TrackedSample> {
        self.samples.get_mut(&sample_id)
    }

    /// Check if a sample can be unlearned (has tree indices).
    pub fn can_unlearn(&self, sample_id: u64) -> bool {
        self.samples
            .get(&sample_id)
            .is_some_and(|s| !s.tree_indices.is_empty())
    }

    /// Get tree indices for a sample.
    pub fn get_tree_indices(&self, sample_id: u64) -> Option<&Vec<usize>> {
        self.samples.get(&sample_id).map(|s| &s.tree_indices)
    }

    /// All sample ids currently in the buffer (Q2-2 rebuild: snapshot of the window).
    pub fn current_ids(&self) -> Vec<u64> {
        self.samples.keys().copied().collect()
    }

    /// Overwrite tree indices for a sample (Q2-2 rebuild: reassign after forest rebuild).
    pub fn set_tree_indices(&mut self, sample_id: u64, tree_indices: Vec<usize>) {
        if let Some(sample) = self.samples.get_mut(&sample_id) {
            sample.tree_indices = tree_indices;
        }
    }

    /// Get label for a sample.
    pub fn get_label(&self, sample_id: u64) -> Option<bool> {
        self.samples.get(&sample_id).map(|s| s.label)
    }

    /// Update cached influence score for a sample.
    /// If influence tracking is enabled, also updates prev_influence.
    pub fn set_influence(&mut self, sample_id: u64, influence: f64) {
        if let Some(sample) = self.samples.get_mut(&sample_id) {
            if self.influence_tracking_enabled {
                sample.prev_influence = sample.cached_influence;
            }
            sample.cached_influence = Some(influence);
        }
    }

    /// Get cached influence score for a sample.
    pub fn get_influence(&self, sample_id: u64) -> Option<f64> {
        self.samples
            .get(&sample_id)
            .and_then(|s| s.cached_influence)
    }

    /// Snapshot of (sample_id, cached_influence) for all samples that have a
    /// cached influence score. Read-only; used by detection-recall analysis.
    pub fn cached_influences(&self) -> Vec<(u64, f64)> {
        self.samples
            .iter()
            .filter_map(|(&id, s)| s.cached_influence.map(|inf| (id, inf)))
            .collect()
    }

    /// Clear all cached influence scores.
    pub fn clear_influence_cache(&mut self) {
        for sample in self.samples.values_mut() {
            sample.cached_influence = None;
        }
    }

    // =========================================================================
    // History Tracking Methods (NEW for Sample Analysis)
    // =========================================================================

    /// Record influence score for a sample with history tracking.
    pub fn record_influence(&mut self, sample_id: u64, influence: f64) {
        if let Some(sample) = self.samples.get_mut(&sample_id) {
            sample.record_influence(self.current_position, influence);
        }
    }

    /// Record removal ranks for multiple samples.
    /// Called during selection to track how close samples are to being removed.
    pub fn record_removal_ranks(&mut self, ranked_sample_ids: &[u64]) {
        let position = self.current_position;
        for (rank, &sample_id) in ranked_sample_ids.iter().enumerate() {
            if let Some(sample) = self.samples.get_mut(&sample_id) {
                sample.record_removal_rank(position, rank);
            }
        }
    }

    /// Get samples with highest survival time (position - insertion position).
    pub fn get_longest_surviving_samples(&self, n: usize) -> Vec<(u64, u64)> {
        let mut samples: Vec<(u64, u64)> = self
            .samples
            .iter()
            .map(|(&id, s)| (id, self.current_position.saturating_sub(s.position)))
            .collect();

        samples.sort_by(|(_, a), (_, b)| b.cmp(a)); // Longest first
        samples.into_iter().take(n).collect()
    }

    /// Get samples with consistently positive influence (core concept samples).
    pub fn get_core_concept_samples(
        &self,
        min_observations: usize,
        min_avg_influence: f64,
    ) -> Vec<u64> {
        self.samples
            .iter()
            .filter(|(_, s)| {
                s.influence_history.len() >= min_observations
                    && s.average_influence()
                        .is_some_and(|avg| avg >= min_avg_influence)
            })
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get samples with declining influence (potential removal candidates).
    pub fn get_declining_influence_samples(&self, min_observations: usize) -> Vec<(u64, f64)> {
        let mut samples: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter(|(_, s)| s.influence_history.len() >= min_observations)
            .filter_map(|(&id, s)| s.influence_trend().map(|trend| (id, trend)))
            .filter(|(_, trend)| *trend < 0.0) // Negative trend = declining
            .collect();

        samples.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        samples
    }

    /// Get samples that are frequently at risk of removal.
    pub fn get_high_risk_samples(&self, top_n: usize, min_risk_count: usize) -> Vec<(u64, usize)> {
        let mut samples: Vec<(u64, usize)> = self
            .samples
            .iter()
            .map(|(&id, s)| (id, s.removal_risk_count(top_n)))
            .filter(|(_, count)| *count >= min_risk_count)
            .collect();

        samples.sort_by(|(_, a), (_, b)| b.cmp(a)); // Highest risk first
        samples
    }

    /// Get lifecycle data for all samples (for analysis export).
    pub fn get_lifecycle_data(&self) -> Vec<SampleLifecycle> {
        self.samples
            .iter()
            .map(|(&id, s)| SampleLifecycle {
                id,
                label: s.label,
                insertion_position: s.position,
                influence_history: s.influence_history.clone(),
                removal_rank_history: s.removal_rank_history.clone(),
                average_influence: s.average_influence(),
                influence_trend: s.influence_trend(),
            })
            .collect()
    }

    /// Get samples by influence stability (low variance = stable).
    pub fn get_stable_influence_samples(
        &self,
        max_variance: f64,
        min_observations: usize,
    ) -> Vec<u64> {
        self.samples
            .iter()
            .filter(|(_, s)| s.influence_history.len() >= min_observations)
            .filter(|(_, s)| {
                let avg = s.average_influence().unwrap_or(0.0);
                let variance: f64 = s
                    .influence_history
                    .iter()
                    .map(|(_, inf)| (inf - avg).powi(2))
                    .sum::<f64>()
                    / s.influence_history.len() as f64;
                variance <= max_variance
            })
            .map(|(&id, _)| id)
            .collect()
    }

    //   (호출 0건). feature-distance influence는 controller.update_feature_distance_scores에서
    //   별도 구현.

    // =========================================================================
    // Time Decay Selection Methods (Gradual Drift Adaptation)
    // =========================================================================

    /// Get time-weighted harmful samples for gradual drift adaptation.
    ///
    /// This method combines OOB influence with time penalty to prioritize removal of:
    /// 1. Old samples (naturally outdated for gradual drift)
    /// 2. Harmful samples (negative influence)
    ///
    /// Formula: weighted_score = influence - decay_rate * age
    /// - Negative influence + old age = lowest score = highest removal priority
    /// - Positive influence + recent = highest score = lowest removal priority
    ///
    /// # Arguments
    /// * `n` - Number of samples to select
    /// * `decay_rate` - Time decay rate (λ). Higher = faster decay.
    ///   - 0.0001: Very slow (100 age → -0.01 penalty)
    ///   - 0.001: Slow (100 age → -0.1 penalty)
    ///   - 0.01: Medium (100 age → -1.0 penalty)
    ///   - 0.1: Fast (100 age → -10.0 penalty)
    ///
    /// # Returns
    /// Vector of sample IDs sorted by weighted score (most removable first)
    pub fn get_time_weighted_harmful_samples(&self, n: usize, decay_rate: f64) -> Vec<u64> {
        let current_pos = self.current_position;

        let mut weighted: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter_map(|(&id, s)| {
                // Need cached influence
                let influence = s.cached_influence?;

                // Calculate age
                let age = (current_pos.saturating_sub(s.position)) as f64;

                // Weighted score: influence minus age penalty
                // - Harmful samples (negative influence) start with low score
                // - Old samples get additional penalty (more negative)
                // - Result: old harmful samples have lowest score → removed first
                //
                // Examples with decay_rate = 0.01:
                // - Old harmful (inf=-0.8, age=100): -0.8 - 1.0 = -1.8
                // - Recent harmful (inf=-0.5, age=10): -0.5 - 0.1 = -0.6
                // - Old helpful (inf=0.3, age=100): 0.3 - 1.0 = -0.7
                // - Recent helpful (inf=0.4, age=10): 0.4 - 0.1 = 0.3
                let weighted_score = influence - decay_rate * age;

                Some((id, weighted_score))
            })
            .collect();

        // Sort by weighted score ascending (most negative = highest removal priority)
        weighted.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        weighted.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get time-weighted harmful samples from a specific class.
    ///
    /// Same as `get_time_weighted_harmful_samples` but filtered by class label.
    pub fn get_time_weighted_harmful_samples_by_class(
        &self,
        label: bool,
        n: usize,
        decay_rate: f64,
    ) -> Vec<u64> {
        let current_pos = self.current_position;

        let mut weighted: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter(|(_, s)| s.label == label)
            .filter_map(|(&id, s)| {
                let influence = s.cached_influence?;

                let age = (current_pos.saturating_sub(s.position)) as f64;
                let weighted_score = influence - decay_rate * age;

                Some((id, weighted_score))
            })
            .collect();

        weighted.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        weighted.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get samples prioritizing old age with influence as tiebreaker.
    ///
    /// This is an alternative approach for gradual drift:
    /// Primary sort: age (oldest first)
    /// Secondary sort: influence (most harmful first for same age bracket)
    ///
    /// # Arguments
    /// * `n` - Number of samples to select
    /// * `age_bracket_size` - Samples within this many positions are considered same age
    pub fn get_age_prioritized_samples(&self, n: usize, age_bracket_size: u64) -> Vec<u64> {
        let current_pos = self.current_position;

        let mut samples: Vec<(u64, u64, f64)> = self
            .samples
            .iter()
            .map(|(&id, s)| {
                let age = current_pos.saturating_sub(s.position);
                let age_bracket = age / age_bracket_size.max(1);
                let influence = s.cached_influence.unwrap_or(0.0);

                (id, age_bracket, influence)
            })
            .collect();

        // Sort by age bracket (descending = oldest first), then by influence (ascending = most harmful)
        samples.sort_by(
            |(_, age_a, inf_a), (_, age_b, inf_b)| match age_b.cmp(age_a) {
                std::cmp::Ordering::Equal => inf_a
                    .partial_cmp(inf_b)
                    .unwrap_or(std::cmp::Ordering::Equal),
                other => other,
            },
        );

        samples.into_iter().take(n).map(|(id, _, _)| id).collect()
    }

    // =========================================================================
    // Selection Methods
    // =========================================================================

    /// Get oldest samples (by position).
    pub fn get_oldest_samples(&self, n: usize) -> Vec<u64> {
        self.position_index.values().take(n).copied().collect()
    }

    /// Get newest samples (by position).
    pub fn get_newest_samples(&self, n: usize) -> Vec<u64> {
        self.position_index
            .values()
            .rev()
            .take(n)
            .copied()
            .collect()
    }

    /// Get samples by class label.
    pub fn get_samples_by_class(&self, label: bool) -> Vec<u64> {
        self.samples
            .iter()
            .filter(|(_, s)| s.label == label)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get oldest samples of a specific class.
    pub fn get_oldest_samples_by_class(&self, label: bool, n: usize) -> Vec<u64> {
        let mut samples: Vec<(u64, u64)> = self
            .samples
            .iter()
            .filter(|(_, s)| s.label == label)
            .map(|(&id, s)| (id, s.position))
            .collect();

        samples.sort_by_key(|(_, pos)| *pos);
        samples.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get samples with lowest cached influence (most harmful).
    pub fn get_most_harmful_samples(&self, n: usize) -> Vec<u64> {
        let mut samples: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter_map(|(&id, s)| s.cached_influence.map(|inf| (id, inf)))
            .collect();

        // Sort by influence ascending (lowest/most negative first)
        samples.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        samples.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get samples with lowest cached influence from a specific class.
    pub fn get_most_harmful_samples_by_class(&self, label: bool, n: usize) -> Vec<u64> {
        let mut samples: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter(|(_, s)| s.label == label)
            .filter_map(|(&id, s)| s.cached_influence.map(|inf| (id, inf)))
            .collect();

        samples.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        samples.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get harmful samples from a specific class above a harm threshold.
    ///
    /// This method filters samples by their influence score, only returning samples
    /// with influence below the threshold (more negative = more harmful).
    ///
    /// # Arguments
    /// * `label` - The class label to filter by (true = attack, false = benign)
    /// * `n` - Maximum number of samples to return
    /// * `min_harm_threshold` - Only include samples with influence < this value (e.g., -0.1)
    ///
    /// # Returns
    /// A vector of sample IDs, sorted by influence (most harmful first)
    pub fn get_harmful_samples_above_threshold(
        &self,
        label: bool,
        n: usize,
        min_harm_threshold: f64,
    ) -> Vec<u64> {
        // Filter by class and influence threshold
        let mut harmful: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter(|(_, s)| s.label == label)
            .filter_map(|(&id, s)| {
                s.cached_influence.and_then(|influence| {
                    if influence < min_harm_threshold {
                        Some((id, influence))
                    } else {
                        None // Skip neutral/positive samples
                    }
                })
            })
            .collect();

        // Sort by influence ascending (most negative/harmful first)
        harmful.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        harmful.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get cached influence score for a sample (alias for get_influence).
    pub fn get_cached_influence(&self, sample_id: u64) -> Option<f64> {
        self.get_influence(sample_id)
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    // =========================================================================
    // Budget Management (Continuous Eviction)
    // =========================================================================

    /// Enforce budget constraint by evicting excess samples.
    ///
    /// When registry size exceeds `budget_config.max_samples`, computes composite
    /// eviction scores and removes the highest-scoring (least valuable) samples:
    ///
    ///   eviction_score = w_age × norm_age + w_influence × (-norm_influence) + w_class × class_penalty
    ///
    /// Minority class protection: samples from the minority class receive a penalty
    /// reduction when `minority_ratio < minority_protection_ratio`, preventing
    /// catastrophic loss of rare class examples.
    ///
    /// Returns IDs of evicted samples (for forest.forget_batch).
    fn enforce_budget(&mut self) -> Vec<u64> {
        let config = match &self.budget_config {
            Some(c) => c.clone(),
            None => return Vec::new(),
        };

        if self.samples.len() <= config.max_samples {
            return Vec::new();
        }

        let n_to_evict = (self.samples.len() - config.max_samples).max(config.eviction_batch_size);

        let evict_ids = self.compute_eviction_candidates(n_to_evict, &config);

        // Track stats
        let mut stats = EvictionStats {
            evicted_count: evict_ids.len(),
            ..Default::default()
        };

        for &id in &evict_ids {
            if let Some(sample) = self.samples.get(&id) {
                if sample.label {
                    stats.evicted_attack += 1;
                } else {
                    stats.evicted_benign += 1;
                }
                if sample.is_influence_degraded() {
                    stats.evicted_degraded += 1;
                }
                if let Some(inf) = sample.cached_influence {
                    stats.evicted_with_influence += 1;
                    stats.evicted_influence_sum += inf;
                }
            }
        }

        // Perform eviction
        for &id in &evict_ids {
            self.remove(id);
        }

        // Accumulate stats
        self.cumulative_eviction_stats.evicted_count += stats.evicted_count;
        self.cumulative_eviction_stats.evicted_benign += stats.evicted_benign;
        self.cumulative_eviction_stats.evicted_attack += stats.evicted_attack;
        self.cumulative_eviction_stats.evicted_degraded += stats.evicted_degraded;
        self.cumulative_eviction_stats.evicted_with_influence += stats.evicted_with_influence;
        self.cumulative_eviction_stats.evicted_influence_sum += stats.evicted_influence_sum;

        evict_ids
    }

    /// Compute composite eviction scores and return top-N candidates.
    ///
    /// Eviction score = w_age * norm_age + w_influence * (-norm_influence) + w_class * class_penalty
    ///   - High score = should be evicted
    ///   - Old samples → higher age score
    ///   - Harmful samples (negative influence) → higher influence score
    ///   - Minority class → lower score (protected)
    fn compute_eviction_candidates(&self, n: usize, config: &BudgetConfig) -> Vec<u64> {
        let current_pos = self.current_position;
        let total = self.samples.len();
        if total == 0 {
            return Vec::new();
        }

        // Reservoir baseline: evict uniformly at random (ignore age/influence/class).
        // Deterministic pseudo-random via hash(sample_id, current_pos) — reproducible
        // without RNG state. Isolates drift-aware selection's contribution vs random.
        if config.random_eviction {
            let mut hashed: Vec<(u64, u64)> = self
                .samples
                .keys()
                .map(|&id| {
                    // splitmix64-style mix of id and stream position for uniform spread
                    let mut z = id
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(current_pos.wrapping_mul(0xD1B54A32D192ED03));
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    (id, z ^ (z >> 31))
                })
                .collect();
            hashed.sort_by_key(|&(_, h)| h);
            return hashed.into_iter().take(n).map(|(id, _)| id).collect();
        }

        // Determine minority class for class-penalty scoring
        let minority_label = self.minority_class();

        // Compute max age for normalization
        let max_age = self
            .position_index
            .keys()
            .next()
            .map(|&oldest_pos| current_pos.saturating_sub(oldest_pos) as f64)
            .unwrap_or(1.0)
            .max(1.0);

        // Compute influence range for dynamic normalization (handles both OOB influence
        // in [-0.5, 0.5] and feature-distance in [-8.0, -0.1] without hard-coded clamp).
        //
        // Bug-1 (determinism fix): empty filter_map yields sentinel (f64::MAX, f64::MIN).
        // In that case (f64::MAX + f64::MIN) / 2.0 = 0.0, and (f64::MIN - 0.0) / 1e-9 = -inf,
        // making all sample scores -inf and eviction selection depend on HashMap iter order
        // (production non-determinism, hit when influence_update_interval > 0 in warmup
        // and especially when influence_strategy="none" — all cached_influence stay None).
        // Fix: detect sentinel and short-circuit influence component to 0.0 (neutral).
        let (inf_min, inf_max) = self
            .samples
            .values()
            .filter_map(|s| s.cached_influence)
            .fold((f64::MAX, f64::MIN), |(mn, mx), v| (mn.min(v), mx.max(v)));
        let has_any_influence = inf_min != f64::MAX;
        let inf_range = if has_any_influence {
            (inf_max - inf_min).max(1e-9)
        } else {
            1.0
        };

        let mut scored: Vec<(u64, f64)> = self
            .samples
            .iter()
            .map(|(&id, sample)| {
                let age = current_pos.saturating_sub(sample.position) as f64;

                let norm_age = age / max_age;

                // Q2-1 ablation: replace the age signal with a deterministic uniform
                // random value (splitmix64 of id + stream position), keeping class
                // penalty and protection_factor identical to the composite path.
                // Isolates whether age-based ordering adds anything beyond minority
                // protection. When disabled, age_signal == norm_age (no behavior change).
                let age_signal = if config.class_aware_random {
                    let mut z = id
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(current_pos.wrapping_mul(0xD1B54A32D192ED03));
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    let h = z ^ (z >> 31);
                    (h >> 11) as f64 / (1u64 << 53) as f64 // uniform [0,1)
                } else {
                    norm_age
                };

                // Influence component: normalize to [0, 1] based on actual data range.
                // Lower influence (more negative) → higher norm_influence → evict first.
                // Bug-1 fix: if no sample has cached_influence yet, neutralize this term
                // (norm_influence = 0.0) so age and class_penalty alone decide ordering.
                // Without this guard, all samples got norm_influence = -inf → all scores
                // collapsed to -inf → eviction picked arbitrarily via HashMap iter order.
                let norm_influence = if has_any_influence {
                    // Samples without cached value get midpoint (neutral within actual range).
                    let influence = sample.cached_influence.unwrap_or((inf_min + inf_max) / 2.0);
                    (inf_max - influence) / inf_range
                } else {
                    0.0
                };

                // Influence degradation bonus: samples that went from positive to negative
                let degradation_bonus = if sample.is_influence_degraded() {
                    0.3 // Extra priority for degraded samples
                } else if let Some(delta) = sample.influence_delta() {
                    if delta < -0.1 {
                        0.15
                    } else {
                        0.0
                    } // Bonus for rapidly worsening
                } else {
                    0.0
                };

                // Class penalty: majority class gets slight penalty (easier to evict)
                let class_penalty = if sample.label == minority_label {
                    -0.5
                } else {
                    0.0
                };

                let mut score = config.age_weight * age_signal
                    + config.influence_weight * (norm_influence + degradation_bonus)
                    + config.class_weight * class_penalty;

                // Adaptive minority protection: weaker protection that still allows
                // the registry distribution to evolve with the data stream.
                //
                // Problem with strong protection (score *= 0.1):
                //   Minority class becomes almost permanent in registry, preventing
                //   the class distribution from tracking the actual data stream.
                //   E.g., AnoShift registry stays at 88% attack forever even as
                //   the actual stream changes from 89% to 27% attack.
                //
                // Solution: scale protection proportionally to imbalance severity.
                //   - Extreme minority (<5%): strong protection (0.2x)
                //   - Moderate minority (5-20%): mild protection (0.5x)
                //   - Near-balanced (>20%): no special protection
                // 곱셈 스택(×0.01=100x 보호) 대신 *더 강한 보호 factor 한 번만 적용*.
                // 이전: extreme(×0.2) + PIHP(×0.05) = ×0.01 → minority core가 영구 잔존
                let sample_class_ratio =
                    self.class_counts[sample.label as usize] as f64 / total.max(1) as f64;

                let mut protection_factor: f64 = 1.0;
                if sample_class_ratio < 0.05 {
                    protection_factor = protection_factor.min(0.2); // extreme minority
                } else if sample_class_ratio < config.minority_protection_ratio {
                    protection_factor = protection_factor.min(0.5); // moderate minority
                }
                score *= protection_factor;

                (id, score)
            })
            .collect();

        // Sort by score descending (highest = evict first).
        // Bug-1 fix: tie-break by sample id ascending — guarantees deterministic
        // eviction order even when scores are equal (e.g., NaN/Inf, identical floats,
        // or all-None influence after the has_any_influence guard above).
        scored.sort_by(|(ia, a), (ib, b)| {
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ia.cmp(ib))
        });

        scored.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Get samples whose influence has degraded from positive to negative.
    /// These are samples that were once helpful but became harmful due to drift.
    pub fn get_influence_degraded_samples(&self) -> Vec<u64> {
        self.samples
            .iter()
            .filter(|(_, s)| s.is_influence_degraded())
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get samples with rapidly declining influence (large negative delta).
    pub fn get_rapidly_declining_samples(&self, min_delta: f64) -> Vec<(u64, f64)> {
        let mut result: Vec<(u64, f64)> = self
            .samples
            .iter()
            .filter_map(|(&id, s)| {
                s.influence_delta().and_then(|delta| {
                    if delta < min_delta {
                        Some((id, delta))
                    } else {
                        None
                    }
                })
            })
            .collect();

        result.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Get the cumulative eviction statistics.
    pub fn eviction_stats(&self) -> &EvictionStats {
        &self.cumulative_eviction_stats
    }

    /// Get the budget configuration.
    pub fn budget_config(&self) -> Option<&BudgetConfig> {
        self.budget_config.as_ref()
    }

    /// Check if budget mode is enabled.
    pub fn budget_enabled(&self) -> bool {
        self.budget_config.is_some()
    }

    /// Enforce fixed memory limit by evicting oldest samples.
    fn enforce_fixed_limit(&mut self, max_bytes: usize) -> Vec<u64> {
        let mut evicted = Vec::new();

        while self.current_bytes > max_bytes && !self.position_index.is_empty() {
            // Get oldest sample
            if let Some((&oldest_pos, &oldest_id)) = self.position_index.iter().next() {
                if let Some(sample) = self.samples.remove(&oldest_id) {
                    self.current_bytes -= sample.memory_bytes();
                    self.class_counts[sample.label as usize] -= 1;
                    self.position_index.remove(&oldest_pos);
                    evicted.push(oldest_id);
                }
            } else {
                break;
            }
        }

        evicted
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get current number of tracked samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get current memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Get memory usage in megabytes.
    pub fn memory_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get class counts [negative, positive].
    pub fn class_counts(&self) -> [usize; 2] {
        self.class_counts
    }

    /// Get minority class label.
    pub fn minority_class(&self) -> bool {
        self.class_counts[1] < self.class_counts[0]
    }

    /// Get majority class label.
    pub fn majority_class(&self) -> bool {
        self.class_counts[1] >= self.class_counts[0]
    }

    /// Get class imbalance ratio (majority / minority).
    pub fn imbalance_ratio(&self) -> f64 {
        let min = self.class_counts[0].min(self.class_counts[1]) as f64;
        let max = self.class_counts[0].max(self.class_counts[1]) as f64;
        if min > 0.0 {
            max / min
        } else {
            f64::INFINITY
        }
    }

    /// Get total samples ever registered.
    pub fn total_registered(&self) -> u64 {
        self.total_registered
    }

    /// Get current stream position.
    pub fn current_position(&self) -> u64 {
        self.current_position
    }

    /// Get influence coverage: (samples with cached_influence != None, total samples).
    pub fn influence_coverage(&self) -> (usize, usize) {
        let with_influence = self
            .samples
            .values()
            .filter(|s| s.cached_influence.is_some())
            .count();
        (with_influence, self.samples.len())
    }

    /// Get all sample IDs.
    pub fn sample_ids(&self) -> Vec<u64> {
        self.samples.keys().copied().collect()
    }

    /// Get up to N sample IDs using deterministic uniform sampling (no RNG needed).
    /// Uses step-based sampling across the position_index (BTreeMap) for
    /// deterministic ordering and even coverage.
    pub fn get_sample_ids_uniform(&self, n: usize) -> Vec<u64> {
        let total = self.position_index.len();
        if total == 0 {
            return Vec::new();
        }
        if n >= total {
            return self.position_index.values().copied().collect();
        }
        let step = total / n;
        self.position_index
            .values()
            .step_by(step.max(1))
            .take(n)
            .copied()
            .collect()
    }

    /// Iterate over all samples.
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &TrackedSample)> {
        self.samples.iter()
    }

    /// Get registry diagnostics for Phase 3 analysis.
    ///
    /// Returns a map with:
    /// - n_samples: total samples in registry
    /// - n_with_influence: samples with cached influence score
    /// - influence_coverage: fraction of samples with influence
    /// - mean_age: average age (current_position - insertion_position)
    /// - mean_influence: average cached influence
    /// - age_influence_spearman: Spearman rank correlation between age and influence
    /// - n_benign / n_attack: class distribution
    pub fn get_diagnostics(&self) -> std::collections::HashMap<String, f64> {
        let mut result = std::collections::HashMap::new();
        let current_pos = self.current_position;
        let total = self.samples.len();

        result.insert("n_samples".to_string(), total as f64);
        result.insert("n_benign".to_string(), self.class_counts[0] as f64);
        result.insert("n_attack".to_string(), self.class_counts[1] as f64);

        // Influence coverage
        let with_influence = self
            .samples
            .values()
            .filter(|s| s.cached_influence.is_some())
            .count();
        result.insert("n_with_influence".to_string(), with_influence as f64);
        result.insert(
            "influence_coverage".to_string(),
            if total > 0 {
                with_influence as f64 / total as f64
            } else {
                0.0
            },
        );

        // Collect (age, influence) pairs for samples with influence
        let pairs: Vec<(f64, f64)> = self
            .samples
            .values()
            .filter_map(|s| {
                let age = current_pos.saturating_sub(s.position) as f64;
                s.cached_influence.map(|inf| (age, inf))
            })
            .collect();

        let n = pairs.len();
        if n < 3 {
            result.insert("mean_age".to_string(), 0.0);
            result.insert("mean_influence".to_string(), 0.0);
            result.insert("age_influence_spearman".to_string(), f64::NAN);
            return result;
        }

        // Mean age and influence
        let mean_age: f64 = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
        let mean_inf: f64 = pairs.iter().map(|(_, i)| i).sum::<f64>() / n as f64;
        result.insert("mean_age".to_string(), mean_age);
        result.insert("mean_influence".to_string(), mean_inf);

        // Spearman rank correlation
        // 1. Rank ages
        let mut age_indexed: Vec<(usize, f64)> = pairs
            .iter()
            .enumerate()
            .map(|(i, (a, _))| (i, *a))
            .collect();
        age_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut age_ranks = vec![0.0f64; n];
        for (rank, &(idx, _)) in age_indexed.iter().enumerate() {
            age_ranks[idx] = rank as f64;
        }

        // 2. Rank influences
        let mut inf_indexed: Vec<(usize, f64)> = pairs
            .iter()
            .enumerate()
            .map(|(i, (_, inf))| (i, *inf))
            .collect();
        inf_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut inf_ranks = vec![0.0f64; n];
        for (rank, &(idx, _)) in inf_indexed.iter().enumerate() {
            inf_ranks[idx] = rank as f64;
        }

        // 3. Pearson correlation of ranks
        let mean_ar = age_ranks.iter().sum::<f64>() / n as f64;
        let mean_ir = inf_ranks.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_i = 0.0;
        for j in 0..n {
            let da = age_ranks[j] - mean_ar;
            let di = inf_ranks[j] - mean_ir;
            cov += da * di;
            var_a += da * da;
            var_i += di * di;
        }
        let spearman = if var_a > 0.0 && var_i > 0.0 {
            cov / (var_a.sqrt() * var_i.sqrt())
        } else {
            0.0
        };
        result.insert("age_influence_spearman".to_string(), spearman);

        result
    }

    /// Clear the registry.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.position_index.clear();
        self.current_bytes = 0;
        self.class_counts = [0, 0];
        self.cumulative_eviction_stats = EvictionStats::default();
        // Keep total_registered, current_position, budget_config, influence_tracking for continuity
    }
}

impl Default for InfluenceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_basic() {
        let mut registry = InfluenceRegistry::new();

        // Register samples
        registry.register(0, vec![0, 1, 2], true);
        registry.register(1, vec![3, 4], false);
        registry.register(2, vec![0, 2, 4], true);

        assert_eq!(registry.len(), 3);
        assert_eq!(registry.class_counts(), [1, 2]);

        // Check sample
        let sample = registry.get(0).unwrap();
        assert_eq!(sample.tree_indices, vec![0, 1, 2]);
        assert!(sample.label);
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = InfluenceRegistry::new();

        registry.register(0, vec![0, 1], true);
        registry.register(1, vec![2, 3], false);

        assert_eq!(registry.len(), 2);

        assert!(registry.remove(0));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.class_counts(), [1, 0]);

        assert!(!registry.remove(99)); // Non-existent
    }

    #[test]
    fn test_registry_fixed_limit() {
        // Very small limit to trigger eviction
        let mut registry = InfluenceRegistry::with_fixed_limit(200);

        // Each sample is ~40-50 bytes, so with 200 byte limit we can fit ~4-5
        for i in 0..10 {
            registry.register(i, vec![0, 1], i % 2 == 0);
        }

        // Should have evicted some samples
        assert!(registry.len() < 10);
        assert!(registry.memory_bytes() <= 250); // Allow some overhead
    }

    #[test]
    fn test_registry_oldest_samples() {
        let mut registry = InfluenceRegistry::new();

        for i in 0..5 {
            registry.register(i, vec![0], i % 2 == 0);
        }

        let oldest = registry.get_oldest_samples(3);
        assert_eq!(oldest, vec![0, 1, 2]);

        let newest = registry.get_newest_samples(2);
        assert_eq!(newest, vec![4, 3]);
    }

    #[test]
    fn test_registry_influence_selection() {
        let mut registry = InfluenceRegistry::new();

        for i in 0..5 {
            registry.register(i, vec![0], true);
        }

        // Set influence scores
        registry.set_influence(0, -0.5); // Harmful
        registry.set_influence(1, 0.2); // Helpful
        registry.set_influence(2, -0.8); // Most harmful
        registry.set_influence(3, 0.1); // Slightly helpful
        registry.set_influence(4, -0.3); // Somewhat harmful

        let harmful = registry.get_most_harmful_samples(3);
        assert_eq!(harmful[0], 2); // Most harmful first
        assert_eq!(harmful[1], 0);
        assert_eq!(harmful[2], 4);
    }

    #[test]
    fn test_registry_class_selection() {
        let mut registry = InfluenceRegistry::new();

        // Add samples: 0,2,4 = positive, 1,3 = negative
        for i in 0..5 {
            registry.register(i, vec![0], i % 2 == 0);
        }

        let positive = registry.get_samples_by_class(true);
        assert_eq!(positive.len(), 3);

        let negative = registry.get_samples_by_class(false);
        assert_eq!(negative.len(), 2);

        let oldest_positive = registry.get_oldest_samples_by_class(true, 2);
        assert_eq!(oldest_positive, vec![0, 2]);
    }

    #[test]
    fn test_registry_batch_operations() {
        let mut registry = InfluenceRegistry::new();

        let ids = vec![0, 1, 2];
        let tree_indices = vec![vec![0, 1], vec![2, 3], vec![0, 3]];
        let labels = vec![true, false, true];

        registry.register_batch(&ids, &tree_indices, &labels);

        assert_eq!(registry.len(), 3);
        assert_eq!(registry.class_counts(), [1, 2]);

        let removed = registry.remove_batch(&[0, 2]);
        assert_eq!(removed, 2);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_harmful_samples_above_threshold() {
        let mut registry = InfluenceRegistry::new();

        // Add 10 benign samples (false) with varying influence
        for i in 0..10 {
            registry.register(i, vec![0], false);
        }

        // Set influence scores
        registry.set_influence(0, -0.5); // Very harmful
        registry.set_influence(1, -0.3); // Harmful
        registry.set_influence(2, -0.15); // Harmful
        registry.set_influence(3, -0.08); // Slightly harmful (borderline)
        registry.set_influence(4, 0.0); // Neutral
        registry.set_influence(5, 0.1); // Helpful
        registry.set_influence(6, -0.12); // Harmful
        registry.set_influence(7, -0.05); // Slightly harmful (borderline)
        registry.set_influence(8, 0.2); // Helpful
        registry.set_influence(9, -0.25); // Harmful

        // Get harmful samples with threshold -0.1 (only include if influence < -0.1)
        let harmful = registry.get_harmful_samples_above_threshold(false, 5, -0.1);

        // Should get samples with influence < -0.1: 0 (-0.5), 9 (-0.25), 1 (-0.3), 2 (-0.15), 6 (-0.12)
        assert_eq!(harmful.len(), 5);

        // Verify they are sorted by influence (most harmful first)
        assert_eq!(harmful[0], 0); // -0.5 (most harmful)
        assert_eq!(harmful[1], 1); // -0.3
        assert_eq!(harmful[2], 9); // -0.25

        // Samples 3, 4, 5, 7, 8 should NOT be included (influence >= -0.1)
        assert!(!harmful.contains(&3));
        assert!(!harmful.contains(&4));
        assert!(!harmful.contains(&5));
        assert!(!harmful.contains(&7));
        assert!(!harmful.contains(&8));
    }

    #[test]
    fn test_registry_budget_eviction() {
        let mut registry = InfluenceRegistry::new();
        registry.set_budget_config(BudgetConfig {
            max_samples: 10,
            eviction_batch_size: 3,
            minority_protection_ratio: 0.1,
            age_weight: 0.4,
            influence_weight: 0.4,
            class_weight: 0.2,
            random_eviction: false,
            class_aware_random: false,
        });

        // Add 15 samples (budget is 10)
        for i in 0..15 {
            registry.register(i, vec![0, 1], i % 3 == 0); // 1/3 are attack
        }

        // Budget should have kicked in
        assert!(
            registry.len() <= 13,
            "Registry should not exceed budget + batch: got {}",
            registry.len()
        );
    }

    #[test]
    fn test_registry_budget_minority_protection() {
        let mut registry = InfluenceRegistry::new();
        registry.set_budget_config(BudgetConfig {
            max_samples: 8,
            eviction_batch_size: 2,
            minority_protection_ratio: 0.2, // Protect if < 20%
            age_weight: 0.5,
            influence_weight: 0.3,
            class_weight: 0.2,
            random_eviction: false,
            class_aware_random: false,
        });

        // Add 9 benign + 1 attack (attack is 10% = minority)
        for i in 0..9 {
            registry.register(i, vec![0], false);
        }
        registry.register(9, vec![0], true); // Attack (minority)

        // After eviction, attack sample should be protected
        assert!(
            registry.get(9).is_some(),
            "Minority attack sample should be protected"
        );
    }

    #[test]
    fn test_influence_degradation_tracking() {
        let mut registry = InfluenceRegistry::new();
        registry.set_influence_tracking(true);

        registry.register(0, vec![0, 1], true);

        // Initially positive
        registry.set_influence(0, 0.5);
        assert!(!registry.get(0).unwrap().is_influence_degraded());

        // Degrades to negative
        registry.set_influence(0, -0.3);
        let sample = registry.get(0).unwrap();
        assert!(
            sample.is_influence_degraded(),
            "Should detect positive→negative transition"
        );
        assert!((sample.influence_delta().unwrap() - (-0.8)).abs() < 1e-10);
    }

    #[test]
    fn test_get_influence_degraded_samples() {
        let mut registry = InfluenceRegistry::new();
        registry.set_influence_tracking(true);

        for i in 0..5 {
            registry.register(i, vec![0], true);
        }

        // Sample 0: positive → negative (degraded)
        registry.set_influence(0, 0.5);
        registry.set_influence(0, -0.3);

        // Sample 1: negative → negative (not degraded from positive)
        registry.set_influence(1, -0.2);
        registry.set_influence(1, -0.5);

        // Sample 2: positive → positive (not degraded)
        registry.set_influence(2, 0.3);
        registry.set_influence(2, 0.1);

        let degraded = registry.get_influence_degraded_samples();
        assert_eq!(degraded.len(), 1);
        assert!(degraded.contains(&0));
    }

    #[test]
    fn test_registry_time_weighted_selection() {
        let mut registry = InfluenceRegistry::new();

        // Add samples at different positions
        // Older samples have lower IDs (registered first)
        for i in 0..10 {
            registry.register(i, vec![0], i % 2 == 0);
        }

        // Set influence scores:
        // - Sample 0: old, very harmful (-0.8)
        // - Sample 1: old, helpful (+0.3)
        // - Sample 8: recent, harmful (-0.5)
        // - Sample 9: recent, helpful (+0.4)
        registry.set_influence(0, -0.8); // Old, very harmful
        registry.set_influence(1, 0.3); // Old, helpful
        registry.set_influence(2, -0.3); // Medium-old, harmful
        registry.set_influence(3, 0.1); // Medium-old, helpful
        registry.set_influence(4, -0.2); // Medium, harmful
        registry.set_influence(5, 0.05); // Medium, slightly helpful
        registry.set_influence(6, -0.1); // Medium-recent, slightly harmful
        registry.set_influence(7, 0.2); // Medium-recent, helpful
        registry.set_influence(8, -0.5); // Recent, harmful
        registry.set_influence(9, 0.4); // Recent, helpful

        // Test time-weighted selection with medium decay
        let decay_rate = 0.1; // Fast decay for testing
        let selected = registry.get_time_weighted_harmful_samples(3, decay_rate);

        // With time decay, old harmful samples should be prioritized
        // Sample 0 is oldest and most harmful → should be first
        assert_eq!(selected[0], 0, "Oldest harmful sample should be first");

        // All selected should have negative influence
        for &id in &selected {
            let inf = registry.get_influence(id).unwrap();
            assert!(
                inf < 0.0,
                "Selected sample {} should have negative influence",
                id
            );
        }
    }

    #[test]
    fn test_registry_time_weighted_by_class() {
        let mut registry = InfluenceRegistry::new();

        // Add samples: 0-4 = false (benign), 5-9 = true (attack)
        for i in 0..5 {
            registry.register(i, vec![0], false);
        }
        for i in 5..10 {
            registry.register(i, vec![0], true);
        }

        // Set influence: all negative for testing
        for i in 0..10 {
            registry.set_influence(i, -0.1 * (10 - i) as f64);
        }

        // Select only from benign class (false)
        let benign_selected = registry.get_time_weighted_harmful_samples_by_class(false, 2, 0.01);

        // Should only contain benign samples (IDs 0-4)
        for &id in &benign_selected {
            assert!(id < 5, "Should only select from benign class");
            let label = registry.get_label(id).unwrap();
            assert!(!label, "Label should be false (benign)");
        }

        // Select only from attack class (true)
        let attack_selected = registry.get_time_weighted_harmful_samples_by_class(true, 2, 0.01);

        // Should only contain attack samples (IDs 5-9)
        for &id in &attack_selected {
            assert!(id >= 5, "Should only select from attack class");
            let label = registry.get_label(id).unwrap();
            assert!(label, "Label should be true (attack)");
        }
    }

    #[test]
    fn test_registry_age_prioritized_selection() {
        let mut registry = InfluenceRegistry::new();

        // Add 20 samples
        for i in 0..20 {
            registry.register(i, vec![0], i % 2 == 0);
        }

        // Set influence: vary by sample
        for i in 0..20 {
            let influence = if i % 3 == 0 { -0.5 } else { 0.1 };
            registry.set_influence(i, influence);
        }

        // Select with age brackets of 5
        let selected = registry.get_age_prioritized_samples(5, 5);

        // Oldest samples (0-4) should be selected first
        // Within those, harmful ones (influence -0.5) should come first
        assert!(
            selected.iter().all(|&id| id < 10),
            "Should prioritize older samples"
        );
    }

    // =========================================================================
    // Phase 5: Budget eviction and batch registration tests
    // =========================================================================

    #[test]
    fn test_evict_by_budget_protects_minority() {
        let mut registry = InfluenceRegistry::new();

        // Add 8 benign (majority) and 4 attack (minority) WITHOUT budget
        for i in 0..8 {
            registry.register(i, vec![0], false);
        }
        for i in 8..12 {
            registry.register(i, vec![0], true);
        }
        assert_eq!(registry.len(), 12);

        // Now set budget to trigger eviction on next register
        registry.set_budget_config(BudgetConfig {
            max_samples: 10,
            eviction_batch_size: 5,
            minority_protection_ratio: 0.3,
            age_weight: 0.4,
            influence_weight: 0.4,
            class_weight: 0.2,
            random_eviction: false,
            class_aware_random: false,
        });

        // Trigger budget enforcement via register_batch
        let _evicted = registry.register_batch(&[100, 101], &[vec![0], vec![0]], &[false, false]);

        // After eviction, minority (attack) should still be present
        let minority_remaining = (8..12).filter(|&id| registry.get(id).is_some()).count();
        let majority_remaining = (0..8).filter(|&id| registry.get(id).is_some()).count();

        assert!(
            minority_remaining >= 3,
            "Minority should be protected: {} remaining out of 4",
            minority_remaining
        );
        assert!(
            majority_remaining < 8,
            "Majority should bear most eviction: {} remaining out of 8",
            majority_remaining
        );
    }

    #[test]
    fn test_register_batch_returns_evicted_ids() {
        let mut registry = InfluenceRegistry::new();
        registry.set_budget_config(BudgetConfig {
            max_samples: 5,
            eviction_batch_size: 3,
            minority_protection_ratio: 0.1,
            age_weight: 1.0,
            influence_weight: 0.0,
            class_weight: 0.0,
            random_eviction: false,
            class_aware_random: false,
        });

        // Fill to budget
        for i in 0..5 {
            registry.register(i, vec![0], false);
        }
        assert_eq!(registry.len(), 5);

        // Register batch that exceeds budget
        let evicted = registry.register_batch(
            &[10, 11, 12],
            &[vec![0], vec![0], vec![0]],
            &[false, false, false],
        );

        // Should have evicted some samples
        assert!(!evicted.is_empty(), "Should return evicted IDs");
        // Evicted IDs should no longer be in registry
        for &id in &evicted {
            assert!(
                registry.get(id).is_none(),
                "Evicted sample {} should not be in registry",
                id
            );
        }
        // Registry should be at or near budget
        assert!(
            registry.len() <= 8,
            "Registry {} should be near budget 5",
            registry.len()
        );
    }
}
