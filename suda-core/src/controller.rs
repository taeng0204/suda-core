//! SUDA Streaming Controller
//!
//! The streaming controller integrates components for budget-based continuous
//! exact forgetting in a DynFrs Random Forest:
//!
//! - InfluenceRegistry: Full sample tracking with budget eviction
//! - DynFrsForest: The model with query-time lazy rebuild (OCC(q) + LZY tag)
//! - MetricsTracker: Streaming performance metrics
//! - SimpleFeatureStore: Feature storage for streaming forget operations
//!
//! # Architecture
//!
//! ```text
//! stream_batch(features, labels)
//!   1. PREDICT (test-then-train)
//!   2. UPDATE METRICS
//!   3. ADD NEW SAMPLES (budget eviction happens inside register())
//!   4. PERIODIC MAINTENANCE (develop, influence recomputation)
//! ```
//!
//! # Key Design Decision: Budget Continuous > Trigger Reactive
//!
//! Experiments (519 runs) showed that continuous budget-based eviction with
//! exact forgetting (forest.forget_batch) is the only path that matters:
//! - Trigger-based reactive unlearning: 0 events in production
//! - Proactive drift-aware unlearning: always disabled
//! - Budget continuous eviction: 47,000+ samples evicted
//!
//! This controller retains only the budget path.

use std::time::Instant;

use hashbrown::HashMap;

use crate::forest::{AdaptiveKConfig, DynFrsForest, ForestConfig};
use crate::metrics::{MetricsTracker, StreamingMetrics};
use crate::registry::InfluenceRegistry;
use crate::sample::VecSample;
use crate::tree::TreeConfig;

// ─── Feature Store (moved from selector module) ─────────────────────────────

/// Trait for providing features for a sample.
pub trait FeatureProvider: Send + Sync {
    /// Get features for a sample ID.
    fn get_features(&self, sample_id: u64) -> Option<Vec<f32>>;
}

/// Simple feature store that wraps a HashMap.
pub struct SimpleFeatureStore {
    features: HashMap<u64, Vec<f32>>,
}

impl SimpleFeatureStore {
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u64, features: Vec<f32>) {
        self.features.insert(id, features);
    }

    pub fn remove(&mut self, id: u64) -> Option<Vec<f32>> {
        self.features.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn clear(&mut self) {
        self.features.clear();
    }

    pub fn keys(&self) -> impl Iterator<Item = &u64> {
        self.features.keys()
    }
}

impl Default for SimpleFeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureProvider for SimpleFeatureStore {
    fn get_features(&self, sample_id: u64) -> Option<Vec<f32>> {
        self.features.get(&sample_id).cloned()
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the streaming controller.
///
/// Fields are organized into tiers:
/// - **Core**: Forest structure, warmup, adaptive-k (always needed)
/// - **Budget**: Budget management parameters (key mechanism)
/// - **Experimental**: Influence tracking, feature distance, window retrain
#[derive(Debug, Clone)]
pub struct SUDAConfig {
    // ─── Core: Forest & Streaming ───────────────────────────────────────────
    pub num_trees: usize,
    pub k: usize,
    pub max_depth: u32,
    pub num_features: u8,
    pub seed: u64,
    pub warmup_samples: usize,
    pub metrics_window: usize,
    /// Memory limit in bytes (default: 100MB). Oldest samples evicted when exceeded.
    pub memory_max_bytes: usize,
    /// Adaptive-k: minority class gets higher OCC(q) redundancy
    pub adaptive_k_enabled: bool,
    pub k_min: usize,
    pub k_max: usize,

    // ─── Budget Management (key mechanism, see HOW > WHAT) ──────────────────
    /// Enable continuous budget-based eviction
    pub budget_enabled: bool,
    /// Maximum samples in registry (optimal: 3000)
    pub budget_max_samples: usize,
    /// Samples to evict per batch when budget exceeded (default: 100)
    pub budget_eviction_batch: usize,
    /// Minority protection threshold (default: 0.1)
    pub budget_minority_protection: f64,
    /// Eviction score weights: age, influence, class penalty
    pub budget_age_weight: f64,
    pub budget_influence_weight: f64,
    pub budget_class_weight: f64,
    /// Ablation flag: skip forest.forget_batch() during eviction
    pub budget_skip_forest_forget: bool,
    /// Use feature-distance based eviction scoring instead of age-based FIFO.
    pub budget_use_feature_distance: bool,
    /// Window retrain mode: instead of exact forget, periodically retrain from buffer
    pub window_retrain_mode: bool,
    /// Window retrain interval: retrain every N eviction batches (default: 1)
    pub window_retrain_interval: u64,
    /// Use incremental retrain (reset + add_samples_streaming) instead of batch fit_weighted
    pub window_retrain_incremental: bool,

    // ─── Experimental: Influence Tracking ───────────────────────────────────
    /// Enable influence drift tracking (prev_influence updates)
    pub influence_tracking_enabled: bool,
    /// How often to recompute influence scores (every N batches, 0 = disabled)
    pub influence_update_interval: usize,
    /// Number of samples to recompute per update
    pub influence_sample_count: usize,
    /// Influence computation strategy:
    /// "none" = no influence (pure FIFO), "oob" = OOB influence,
    /// "loss" = cross-entropy loss, "confidence" = redundancy-based,
    /// "cumulative_oob" = EMA of OOB, "feature_distance" = centroid distance
    pub influence_strategy: String,
    /// Feature distance update interval (samples, default 2000)
    pub feat_dist_update_interval: u64,

    // ─── Experimental: Tree Rebuild ─────────────────────────────────────────
    /// Periodic develop() interval (0 = disabled, N = every N batches)
    pub develop_interval: u64,

    // ─── Split Quality Monitoring ────────────────────────────────────────────
    /// Split quality degradation threshold for forget-time monitoring.
    /// When set, forget() checks if each internal node's weighted Gini has
    /// degraded beyond this threshold from its creation-time value.
    /// None = disabled (default, preserves existing behavior).
    pub split_quality_threshold: Option<f64>,

    /// Maximum age (in samples) for a split before forced rebuild during develop().
    /// None = disabled (default). When set, internal nodes with
    /// `total_samples - created_at > split_max_age` are automatically rebuilt,
    /// addressing structural debt from gradual drift.
    pub split_max_age: Option<u64>,
}

impl Default for SUDAConfig {
    fn default() -> Self {
        Self {
            num_trees: 50,
            k: 10,
            max_depth: 15,
            num_features: 41,
            seed: 42,
            memory_max_bytes: 100 * 1024 * 1024, // 100MB
            warmup_samples: 1000,
            metrics_window: 1000,
            adaptive_k_enabled: true,
            k_min: 3,
            k_max: 50,

            // Budget management (disabled by default)
            budget_enabled: false,
            budget_max_samples: 10000,
            budget_eviction_batch: 100,
            budget_minority_protection: 0.1,

            // Influence tracking (disabled by default)
            influence_tracking_enabled: false,

            // Influence recomputation
            influence_update_interval: 10,
            influence_sample_count: 200,
            influence_strategy: "none".to_string(),
            feat_dist_update_interval: 2000,

            // Budget eviction weights
            budget_age_weight: 0.4,
            budget_influence_weight: 0.4,
            budget_class_weight: 0.2,

            // Ablation: skip forest forget
            budget_skip_forest_forget: false,
            // Feature-distance eviction (disabled by default, uses FIFO)
            budget_use_feature_distance: false,
            // Window retrain mode (disabled by default)
            window_retrain_mode: false,
            window_retrain_interval: 1,
            window_retrain_incremental: false,

            // Tree rebuild (develop) - every 5 batches by default
            develop_interval: 5,

            // Split quality monitoring (disabled by default)
            split_quality_threshold: None,

            // Age-based subtree refresh (disabled by default)
            split_max_age: None,
        }
    }
}

// ─── Streaming Result ───────────────────────────────────────────────────────

/// Result of processing a batch through the streaming controller.
#[derive(Debug, Clone)]
pub struct StreamingResult {
    /// Predictions for the batch (before training)
    pub predictions: Vec<bool>,
    /// Current metrics after processing
    pub metrics: StreamingMetrics,
    /// Current registry size
    pub registry_size: usize,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// Total samples processed
    pub total_samples: u64,
    /// Processing time in microseconds
    pub process_time_us: u64,
    /// Number of samples evicted by budget management in this batch
    pub budget_evicted: usize,
}

// ─── Streaming Controller ───────────────────────────────────────────────────

/// The streaming controller.
pub struct StreamingController {
    /// The random forest model
    forest: DynFrsForest,
    /// Sample tracking registry (with budget eviction)
    registry: InfluenceRegistry,
    /// Metrics tracker
    metrics: MetricsTracker,
    /// Feature store (for streaming forget operations)
    feature_store: SimpleFeatureStore,
    /// Configuration
    config: SUDAConfig,
    /// Current stream position
    position: u64,
    /// Total samples processed
    total_samples: u64,
    /// Whether the model is warmed up
    is_warmed_up: bool,
    /// Next sample ID to assign
    next_sample_id: u64,
    /// Whether initial fit has been done (via fit() method)
    initial_fit_done: bool,

    /// Counter for influence recomputation interval
    influence_update_counter: u64,
    /// Last batch samples stored for influence recomputation (used as test samples)
    last_batch_samples: Vec<VecSample>,

    /// Counter for periodic develop() calls to process pending LazyTags
    batches_since_develop: u64,
    /// Counter for window retrain interval
    evictions_since_retrain: u64,
    /// Counter for feature-distance update (samples since last update)
    feature_distance_counter: u64,
}

impl StreamingController {
    /// Create a new streaming controller.
    pub fn new(config: SUDAConfig) -> Self {
        // Create forest config
        let tree_config = TreeConfig {
            max_depth: config.max_depth as usize,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 10,
            split_quality_threshold: config.split_quality_threshold,
            ..Default::default()
        };

        let forest_config = ForestConfig {
            num_trees: config.num_trees,
            k: config.k,
            tree_config,
            seed: config.seed,
        };

        // Create forest
        let mut forest = DynFrsForest::new(forest_config, config.num_features);

        // Configure adaptive k if enabled
        if config.adaptive_k_enabled {
            forest.set_adaptive_k_config(AdaptiveKConfig {
                k_min: config.k_min,
                k_max: config.k_max,
                ratio_threshold: 0.3,
                extreme_threshold: 0.01,
            });
        }

        // Create components
        let mut registry = InfluenceRegistry::with_max_bytes(config.memory_max_bytes);

        // Configure budget management if enabled
        if config.budget_enabled {
            registry.set_budget_config(crate::registry::BudgetConfig {
                max_samples: config.budget_max_samples,
                eviction_batch_size: config.budget_eviction_batch,
                minority_protection_ratio: config.budget_minority_protection,
                age_weight: config.budget_age_weight,
                influence_weight: config.budget_influence_weight,
                class_weight: config.budget_class_weight,
            });
        }

        // Configure influence tracking
        if config.influence_tracking_enabled {
            registry.set_influence_tracking(true);
        }

        let metrics = if config.metrics_window > 0 {
            MetricsTracker::with_window(config.metrics_window)
        } else {
            MetricsTracker::new()
        };

        Self {
            forest,
            registry,
            metrics,
            feature_store: SimpleFeatureStore::new(),
            config,
            position: 0,
            total_samples: 0,
            is_warmed_up: false,
            next_sample_id: 0,
            initial_fit_done: false,
            influence_update_counter: 0,
            last_batch_samples: Vec::new(),
            batches_since_develop: 0,
            evictions_since_retrain: 0,
            feature_distance_counter: 0,
        }
    }

    /// Process a single batch of samples (main streaming entry point).
    ///
    /// Pipeline: Predict -> Update Metrics -> Add Samples (with budget eviction) -> Maintain
    pub fn stream_batch(&mut self, features: &[Vec<f32>], labels: &[bool]) -> StreamingResult {
        let start = Instant::now();
        let batch_size = features.len();

        if batch_size == 0 || labels.len() != batch_size {
            return StreamingResult {
                predictions: Vec::new(),
                metrics: self.metrics.current_metrics(),
                registry_size: self.registry.len(),
                memory_mb: self.registry.memory_mb(),
                total_samples: self.total_samples,
                process_time_us: 0,
                budget_evicted: 0,
            };
        }

        // Store current batch for influence recomputation
        self.store_batch_for_influence(features, labels);

        // 1. PREDICT (Test-Then-Train)
        let predictions = self.predict_batch(features);

        // 2. UPDATE METRICS
        self.metrics.update_batch(labels, &predictions);

        // 3. ADD NEW SAMPLES (budget eviction happens inside register())
        let eviction_before = self.registry.eviction_stats().evicted_count;
        self.add_batch(features, labels);
        let budget_evicted = self.registry.eviction_stats().evicted_count - eviction_before;

        // 4. PERIODIC MAINTENANCE (develop, influence, warmup)
        self.periodic_maintenance(batch_size);

        let elapsed = start.elapsed();

        StreamingResult {
            predictions,
            metrics: self.metrics.current_metrics(),
            registry_size: self.registry.len(),
            memory_mb: self.registry.memory_mb(),
            total_samples: self.total_samples,
            process_time_us: elapsed.as_micros() as u64,
            budget_evicted,
        }
    }

    /// Store current batch as test samples for periodic influence recomputation.
    fn store_batch_for_influence(&mut self, features: &[Vec<f32>], labels: &[bool]) {
        if self.config.budget_enabled && self.config.influence_update_interval > 0 {
            self.last_batch_samples = features
                .iter()
                .zip(labels.iter())
                .enumerate()
                .map(|(i, (f, &l))| VecSample {
                    id: self.position + i as u64,
                    values: f.clone(),
                    label: l,
                })
                .collect();
        }
    }

    /// Periodic maintenance: develop trees, recompute influences, update warmup.
    fn periodic_maintenance(&mut self, batch_size: usize) {
        if self.is_warmed_up {
            self.maybe_develop(false);

            // Periodic influence recomputation for budget eviction quality
            if self.config.budget_enabled && self.config.influence_update_interval > 0 {
                self.influence_update_counter += 1;
                if self.influence_update_counter % self.config.influence_update_interval as u64 == 0 {
                    self.update_sample_influences();
                }
            }
        }

        // Update warmup status
        self.total_samples += batch_size as u64;
        if !self.is_warmed_up && self.total_samples >= self.config.warmup_samples as u64 {
            self.is_warmed_up = true;
        }
    }

    /// Predict labels for a batch.
    fn predict_batch(&self, features: &[Vec<f32>]) -> Vec<bool> {
        let samples: Vec<VecSample> = features
            .iter()
            .enumerate()
            .map(|(i, f)| VecSample {
                id: i as u64,
                values: f.clone(),
                label: false,
            })
            .collect();

        self.forest.predict_batch(&samples)
    }

    /// Predict labels for a batch without updating any controller state.
    pub fn predict_batch_only(&self, features: &[Vec<f32>]) -> Vec<bool> {
        if features.is_empty() {
            return Vec::new();
        }
        self.predict_batch(features)
    }

    /// Recompute influence scores for a subset of registry samples.
    /// Dispatches to the appropriate strategy based on config.influence_strategy.
    fn update_sample_influences(&mut self) {
        match self.config.influence_strategy.as_str() {
            "oob" => self.update_oob_influences(),
            "loss" => self.update_loss_based_influences(),
            "confidence" => self.update_confidence_influences(),
            "cumulative_oob" => self.update_cumulative_oob_influences(),
            "feature_distance" => {}, // handled in add_batch via update_feature_distance_scores
            _ => {},  // "none" = no influence update
        }
    }

    /// OOB influence: existing behavior — compare in-bag vs OOB accuracy.
    fn update_oob_influences(&mut self) {
        if self.last_batch_samples.is_empty() {
            return;
        }
        let sample_ids = self.registry.get_sample_ids_uniform(
            self.config.influence_sample_count,
        );
        if sample_ids.is_empty() {
            return;
        }
        for &sample_id in &sample_ids {
            if let Some(influence) = self.forest.compute_oob_influence_batch(
                sample_id,
                &self.last_batch_samples,
            ) {
                self.registry.set_influence(sample_id, influence);
            }
        }
    }

    /// Loss-based influence: cross-entropy loss difference between in-bag and OOB.
    fn update_loss_based_influences(&mut self) {
        if self.last_batch_samples.is_empty() {
            return;
        }
        let sample_ids = self.registry.get_sample_ids_uniform(
            self.config.influence_sample_count,
        );
        if sample_ids.is_empty() {
            return;
        }
        for &sample_id in &sample_ids {
            if let Some(influence) = self.forest.compute_loss_influence_batch(
                sample_id,
                &self.last_batch_samples,
            ) {
                self.registry.set_influence(sample_id, influence);
            }
        }
    }

    /// Confidence-based: redundant samples (high model confidence) get low influence → evict first.
    fn update_confidence_influences(&mut self) {
        let sample_ids = self.registry.get_sample_ids_uniform(
            self.config.influence_sample_count,
        );
        if sample_ids.is_empty() {
            return;
        }
        for &sample_id in &sample_ids {
            if let Some(features) = self.feature_store.get_features(sample_id) {
                let label = self.registry.get_label(sample_id);
                if let Some(label) = label {
                    let sample = VecSample {
                        id: sample_id,
                        values: features,
                        label,
                    };
                    let proba = self.forest.predict_proba(&sample);
                    // correct_proba: how confident the model is in the correct answer
                    let correct_proba = if label { proba } else { 1.0 - proba };
                    // High confidence → low influence → evict first (redundant)
                    let influence = -(correct_proba); // [-1, 0]
                    self.registry.set_influence(sample_id, influence);
                }
            }
        }
    }

    /// Cumulative OOB: EMA of OOB influence for stable estimation.
    fn update_cumulative_oob_influences(&mut self) {
        if self.last_batch_samples.is_empty() {
            return;
        }
        let alpha = 0.3; // new value weight
        let sample_ids = self.registry.get_sample_ids_uniform(
            self.config.influence_sample_count,
        );
        if sample_ids.is_empty() {
            return;
        }
        for &sample_id in &sample_ids {
            if let Some(new_inf) = self.forest.compute_oob_influence_batch(
                sample_id,
                &self.last_batch_samples,
            ) {
                let old_inf = self.registry.get_influence(sample_id).unwrap_or(0.0);
                let ema = alpha * new_inf + (1.0 - alpha) * old_inf;
                self.registry.set_influence(sample_id, ema);
            }
        }
    }

    /// Add a batch of samples with full tracking.
    fn add_batch(&mut self, features: &[Vec<f32>], labels: &[bool]) {
        let samples: Vec<VecSample> = features
            .iter()
            .zip(labels.iter())
            .map(|(f, &l)| {
                let id = self.next_sample_id;
                self.next_sample_id += 1;

                let values = f.clone(); // Single clone
                self.feature_store.insert(id, values.clone()); // Share via clone of owned

                VecSample {
                    id,
                    values,
                    label: l,
                }
            })
            .collect();

        // Streaming add - always use streaming mode
        let (_num_added, _needs_rebuild) = self.forest.add_samples_streaming(&samples, true);

        // Feature-distance eviction: compute centroid and set distances as influence scores
        // before registration so that eviction scoring uses distance instead of age.
        // Only update periodically (every 2000 samples) AND when near budget capacity (>80%)
        // to avoid O(n) distance computation on every batch.
        if self.config.budget_use_feature_distance && self.config.budget_enabled {
            self.feature_distance_counter += samples.len() as u64;
            let near_capacity = self.feature_store.len() > (self.config.budget_max_samples * 4 / 5);
            let interval_reached = self.feature_distance_counter >= self.config.feat_dist_update_interval;
            if near_capacity && interval_reached {
                self.update_feature_distance_scores();
                self.feature_distance_counter = 0;
            }
        }

        // Register samples with their tree indices (batch optimized)
        let sample_ids: Vec<u64> = samples.iter().map(|s| s.id).collect();
        let tree_indices_list: Vec<Vec<usize>> = samples
            .iter()
            .map(|s| self.forest.get_sample_tree_indices(s.id))
            .collect();
        let labels: Vec<bool> = samples.iter().map(|s| s.label).collect();
        let all_evicted = self.registry.register_batch(&sample_ids, &tree_indices_list, &labels);
        self.position += samples.len() as u64;

        // Forget budget-evicted samples from the forest (critical for actual model impact)
        if !all_evicted.is_empty() {
            if self.config.window_retrain_mode {
                // Window Retrain path: remove from feature_store, then periodically retrain
                for &id in &all_evicted {
                    self.feature_store.remove(id);
                }
                self.evictions_since_retrain += 1;
                if self.evictions_since_retrain >= self.config.window_retrain_interval {
                    self.retrain_from_window();
                    self.evictions_since_retrain = 0;
                }
            } else if !self.config.budget_skip_forest_forget {
                // Exact Forget path: streaming-aware forget updates attribute statistics
                let feature_map: HashMap<u64, Vec<f32>> = all_evicted.iter()
                    .filter_map(|&id| self.feature_store.get_features(id).map(|f| (id, f.to_vec())))
                    .collect();
                self.forest.forget_batch_with_features(&all_evicted, &feature_map);
                // Also remove from feature store
                for &id in &all_evicted {
                    self.feature_store.remove(id);
                }
                // Force develop after budget eviction (forget invalidates splits)
                self.run_develop();
            } else {
                // No-Forest-Forget ablation: registry-only eviction
                for &id in &all_evicted {
                    self.feature_store.remove(id);
                }
            }
        }
    }

    /// Collect current samples from feature_store + forest for develop().
    /// Only collects samples that are both in the forest and feature_store.
    fn collect_current_samples(&self) -> Vec<VecSample> {
        self.forest
            .get_all_sample_ids()
            .into_iter()
            .filter_map(|id| {
                let features = self.feature_store.get_features(id)?;
                let label = self.forest.get_sample_label(id)?;
                Some(VecSample {
                    id,
                    values: features,
                    label,
                })
            })
            .collect()
    }

    /// Compute feature-distance scores for all samples in the registry.
    /// Sets negative Euclidean distance from recent centroid as cached_influence,
    /// so the eviction scoring treats far-from-centroid samples as eviction candidates.
    fn update_feature_distance_scores(&mut self) {
        let n_features = self.config.num_features as usize;
        let all_ids: Vec<u64> = self.feature_store.keys().cloned().collect();
        if all_ids.is_empty() || n_features == 0 {
            return;
        }

        // Compute centroid of the most recent 1000 samples by position.
        // Use select_nth_unstable for O(n) partitioning instead of O(n log n) sort.
        let n_recent = 1000.min(all_ids.len());
        let mut recent_positions: Vec<(u64, u64)> = all_ids.iter()
            .filter_map(|&id| {
                self.registry.get(id)
                    .map(|t| (id, t.position))
            })
            .collect();

        if recent_positions.len() > n_recent {
            // Partition: top n_recent by position (descending = largest positions)
            recent_positions.select_nth_unstable_by_key(
                n_recent,
                |&(_, pos)| std::cmp::Reverse(pos),
            );
            recent_positions.truncate(n_recent);
        }

        let mut centroid = vec![0.0f64; n_features];
        let mut cnt = 0usize;
        for &(id, _) in &recent_positions {
            if let Some(f) = self.feature_store.get_features(id) {
                for (c, &v) in centroid.iter_mut().zip(f.iter()) {
                    *c += v as f64;
                }
                cnt += 1;
            }
        }
        if cnt == 0 {
            return;
        }
        for c in centroid.iter_mut() {
            *c /= cnt as f64;
        }

        // Set negative distance as influence for all samples
        for &id in &all_ids {
            if let Some(f) = self.feature_store.get_features(id) {
                let dist: f64 = f.iter().zip(centroid.iter())
                    .map(|(&v, &c)| (v as f64 - c).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if let Some(tracked) = self.registry.get_mut(id) {
                    tracked.cached_influence = Some(-dist);
                }
            }
        }
    }

    /// Retrain the forest from the current window (feature_store samples).
    /// Used in window_retrain_mode as an alternative to exact forget.
    fn retrain_from_window(&mut self) {
        // Collect samples from feature_store + registry (labels come from registry)
        let samples: Vec<VecSample> = self.feature_store
            .keys()
            .filter_map(|&id| {
                let features = self.feature_store.get_features(id)?;
                let tracked = self.registry.get(id)?;
                Some(VecSample {
                    id,
                    values: features,
                    label: tracked.label,
                })
            })
            .collect();

        if samples.is_empty() {
            return;
        }

        // Compute class ratio for Adaptive-k
        let positive_count = samples.iter().filter(|s| s.label).count();
        let positive_ratio = positive_count as f64 / samples.len() as f64;

        if self.config.window_retrain_incremental {
            // Incremental retrain: reset trees + re-add samples via streaming
            self.forest.reset();
            self.forest.init_streaming_class_k_with_counts(positive_ratio, samples.len() as u64);
            self.forest.add_samples_streaming(&samples, true);
            // develop to process any pending LazyTags from streaming adds
            self.forest.develop_streaming(&samples);
        } else {
            // Batch retrain: full fit_weighted (destroys + rebuilds all trees)
            self.forest.fit_weighted(&samples, positive_ratio);
            self.forest.init_streaming_class_k_with_counts(positive_ratio, samples.len() as u64);
        }

        // Sync registry tree_indices with new OCC mapping after retrain
        for sample in &samples {
            let new_tree_indices = self.forest.get_sample_tree_indices(sample.id);
            if let Some(tracked) = self.registry.get_mut(sample.id) {
                tracked.tree_indices = new_tree_indices;
            }
        }
    }

    /// Run develop() to process pending LazyTags (Dirty/Rebuild) in trees.
    /// This rebuilds stale splits after add/forget operations.
    /// When split_max_age is configured, also rebuilds splits older than the threshold.
    fn run_develop(&mut self) {
        let has_age_refresh = self.config.split_max_age.is_some();

        // If no age refresh, skip if nothing pending
        if !has_age_refresh && !self.forest.has_pending_rebuilds() {
            return;
        }

        let samples = self.collect_current_samples();
        if !samples.is_empty() {
            if has_age_refresh {
                self.forest.develop_streaming_with_age(
                    &samples,
                    self.total_samples,
                    self.config.split_max_age,
                );
            } else {
                self.forest.develop_streaming(&samples);
            }
        }
        self.batches_since_develop = 0;
    }

    /// Conditionally run develop based on interval or force flag.
    fn maybe_develop(&mut self, force: bool) {
        if force {
            self.run_develop();
            return;
        }
        if self.config.develop_interval > 0 {
            self.batches_since_develop += 1;
            if self.batches_since_develop >= self.config.develop_interval {
                self.run_develop();
            }
        }
    }

    /// Pre-train on historical data.
    pub fn fit(&mut self, features: &[Vec<f32>], labels: &[bool]) -> StreamingMetrics {
        if features.is_empty() || labels.len() != features.len() {
            return self.metrics.current_metrics();
        }

        let n_samples = features.len();

        let samples: Vec<VecSample> = features
            .iter()
            .zip(labels.iter())
            .map(|(f, &l)| {
                let id = self.next_sample_id;
                self.next_sample_id += 1;

                let values = f.clone();
                self.feature_store.insert(id, values.clone());

                VecSample {
                    id,
                    values,
                    label: l,
                }
            })
            .collect();

        let positive_count = labels.iter().filter(|&&l| l).count();
        let positive_ratio = positive_count as f64 / n_samples as f64;

        #[cfg(debug_assertions)]
        eprintln!(
            "[SUDA] Pre-training: {} samples, {} positive ({:.2}%)",
            n_samples,
            positive_count,
            positive_ratio * 100.0
        );

        // fit_weighted handles Adaptive-k internally
        self.forest.fit_weighted(&samples, positive_ratio);

        // Initialize streaming class k for Adaptive-k (for subsequent streaming adds)
        self.forest.init_streaming_class_k_with_counts(positive_ratio, n_samples as u64);

        // Register all samples with tree indices.
        // During fit() (batch warmup), we intentionally do NOT remove evicted
        // samples from the forest. The forest was just built with all warmup data
        // and should retain that knowledge. The registry tracks the budget window
        // for subsequent streaming-phase eviction. Forest-level forget only
        // applies during streaming (stream_batch / process_batch).
        for sample in &samples {
            let tree_indices = self.forest.get_sample_tree_indices(sample.id);
            let _evicted = self.registry.register(sample.id, tree_indices, sample.label);
            self.position += 1;
        }

        // CRITICAL: Enable streaming mode for incremental split updates.
        //
        // This initializes per-node StreamingNodeState with attribute statistics
        // so that subsequent add_sample_streaming() calls can:
        //   1. Update split statistics along the insertion path
        //   2. Detect when the best split has changed → mark LazyTag::Rebuild
        //   3. Check if leaves should split (growth)
        //
        // Without this call, streaming_enabled remains false and splits are
        // frozen after fit(), causing significant performance degradation
        // on long streams (e.g., ANoShift: 0.90 → 0.94 with this fix).
        self.forest.enable_streaming(&samples);

        self.is_warmed_up = true;
        self.initial_fit_done = true;
        self.total_samples = n_samples as u64;

        // Evaluate on training data
        let predictions = self.forest.predict_batch(&samples);
        self.metrics.update_batch(labels, &predictions);

        self.metrics.current_metrics()
    }

    // =========================================================================
    // Getters
    // =========================================================================

    pub fn current_metrics(&self) -> StreamingMetrics {
        self.metrics.current_metrics()
    }

    pub fn cumulative_metrics(&self) -> StreamingMetrics {
        self.metrics.cumulative_metrics()
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    pub fn registry_size(&self) -> usize {
        self.registry.len()
    }

    pub fn memory_mb(&self) -> f64 {
        self.registry.memory_mb()
    }

    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up
    }

    pub fn is_pretrained(&self) -> bool {
        self.initial_fit_done && self.is_warmed_up
    }

    pub fn config(&self) -> &SUDAConfig {
        &self.config
    }

    /// Get reference to the registry (for analysis).
    pub fn registry(&self) -> &InfluenceRegistry {
        &self.registry
    }

    /// Get mutable reference to the registry (for analysis).
    pub fn registry_mut(&mut self) -> &mut InfluenceRegistry {
        &mut self.registry
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.forest.reset();
        self.metrics.reset();
        self.registry.clear();
        self.feature_store.clear();
        self.position = 0;
        self.total_samples = 0;
        self.next_sample_id = 0;
        self.is_warmed_up = false;
        self.initial_fit_done = false;
        self.influence_update_counter = 0;
        self.last_batch_samples.clear();
        self.batches_since_develop = 0;
        self.evictions_since_retrain = 0;
        self.feature_distance_counter = 0;
    }

    /// Check if budget mode is enabled.
    pub fn budget_enabled(&self) -> bool {
        self.config.budget_enabled
    }

    /// Get the total number of samples evicted by budget management.
    pub fn total_budget_evicted(&self) -> usize {
        self.registry.eviction_stats().evicted_count
    }

    /// Get budget eviction stats.
    pub fn budget_eviction_stats(&self) -> (usize, usize, usize, usize) {
        let stats = self.registry.eviction_stats();
        (stats.evicted_count, stats.evicted_benign, stats.evicted_attack, stats.evicted_degraded)
    }

    /// Check if influence tracking is enabled.
    pub fn influence_tracking_enabled(&self) -> bool {
        self.config.influence_tracking_enabled
    }

    /// Get count of influence-degraded samples.
    pub fn influence_degraded_count(&self) -> usize {
        self.registry.get_influence_degraded_samples().len()
    }

    /// Get influence coverage: (samples with influence, total samples).
    pub fn influence_coverage(&self) -> (usize, usize) {
        self.registry.influence_coverage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SUDAConfig::default();
        assert_eq!(config.num_trees, 50);
        assert_eq!(config.k, 10);
        assert!(config.adaptive_k_enabled);
    }

    #[test]
    fn test_controller_creation() {
        let config = SUDAConfig {
            num_features: 10,
            warmup_samples: 100,
            ..Default::default()
        };

        let controller = StreamingController::new(config);
        assert!(!controller.is_warmed_up());
        assert_eq!(controller.total_samples(), 0);
        assert_eq!(controller.registry_size(), 0);
    }

    #[test]
    fn test_controller_warmup() {
        let config = SUDAConfig {
            num_features: 4,
            warmup_samples: 10,
            ..Default::default()
        };

        let mut controller = StreamingController::new(config);

        let features: Vec<Vec<f32>> = (0..15).map(|i| vec![i as f32; 4]).collect();
        let labels: Vec<bool> = (0..15).map(|i| i % 2 == 0).collect();

        let result = controller.stream_batch(&features, &labels);

        assert!(controller.is_warmed_up());
        assert_eq!(controller.total_samples(), 15);
        assert_eq!(result.predictions.len(), 15);
    }

    #[test]
    fn test_controller_fit() {
        let config = SUDAConfig {
            num_features: 4,
            warmup_samples: 100,
            ..Default::default()
        };

        let mut controller = StreamingController::new(config);

        let features: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32; 4]).collect();
        let labels: Vec<bool> = (0..50).map(|i| i % 2 == 0).collect();

        let metrics = controller.fit(&features, &labels);

        assert!(controller.is_pretrained());
        assert_eq!(controller.registry_size(), 50);
        assert!(metrics.total_samples > 0);
    }

    #[test]
    fn test_predict_batch_only_has_no_side_effects() {
        let config = SUDAConfig {
            num_features: 4,
            warmup_samples: 100,
            ..Default::default()
        };

        let mut controller = StreamingController::new(config);

        let features: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32; 4]).collect();
        let labels: Vec<bool> = (0..30).map(|i| i % 2 == 0).collect();
        controller.fit(&features, &labels);

        let metrics_before = controller.current_metrics();
        let total_samples_before = controller.total_samples();
        let registry_size_before = controller.registry_size();
        let position_before = controller.position();

        let preds = controller.predict_batch_only(&features[..10]);

        assert_eq!(preds.len(), 10);
        assert_eq!(controller.total_samples(), total_samples_before);
        assert_eq!(controller.registry_size(), registry_size_before);
        assert_eq!(controller.position(), position_before);

        let metrics_after = controller.current_metrics();
        assert_eq!(metrics_after.total_samples, metrics_before.total_samples);
        assert_eq!(metrics_after.accuracy, metrics_before.accuracy);
        assert_eq!(metrics_after.gmean, metrics_before.gmean);
    }

    #[test]
    fn test_stream_batch_empty_input() {
        let config = SUDAConfig {
            num_features: 4,
            warmup_samples: 100,
            ..Default::default()
        };
        let mut controller = StreamingController::new(config);

        // Empty features
        let result = controller.stream_batch(&[], &[]);
        assert!(result.predictions.is_empty());
        assert_eq!(result.budget_evicted, 0);

        // Mismatched lengths
        let features: Vec<Vec<f32>> = vec![vec![1.0; 4]];
        let result = controller.stream_batch(&features, &[]);
        assert!(result.predictions.is_empty());
    }

    #[test]
    fn test_budget_eviction_during_streaming() {
        let config = SUDAConfig {
            num_features: 4,
            warmup_samples: 10,
            budget_enabled: true,
            budget_max_samples: 50,
            budget_eviction_batch: 10,
            ..Default::default()
        };
        let mut controller = StreamingController::new(config);

        // Fit with initial data
        let features: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32; 4]).collect();
        let labels: Vec<bool> = (0..30).map(|i| i % 3 == 0).collect();
        controller.fit(&features, &labels);

        // Stream enough to exceed budget
        let mut total_evicted = 0u64;
        for batch in 0..10 {
            let batch_features: Vec<Vec<f32>> = (0..10).map(|i| vec![(batch * 10 + i) as f32; 4]).collect();
            let batch_labels: Vec<bool> = (0..10).map(|i| i % 3 == 0).collect();
            let result = controller.stream_batch(&batch_features, &batch_labels);
            total_evicted += result.budget_evicted as u64;
        }

        // Registry should be around budget limit
        assert!(controller.registry_size() <= 60, "Registry {} should be near budget 50", controller.registry_size());
        assert!(total_evicted > 0, "Should have evicted some samples");
    }
}

// =============================================================================
// Python Bindings (PyO3)
// =============================================================================

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python-accessible streaming controller.
#[pyclass]
pub struct PyStreamingController {
    inner: StreamingController,
}

#[pymethods]
impl PyStreamingController {
    /// Create a new streaming controller from a config dict.
    ///
    /// Accepts a Python dict with config keys. Missing keys use defaults from SUDAConfig::default().
    #[new]
    fn new(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        // Helper: extract value from dict or use default
        fn get<'py, T: pyo3::FromPyObject<'py>>(
            dict: &Bound<'py, PyDict>,
            key: &str,
            default: T,
        ) -> PyResult<T> {
            match dict.get_item(key)? {
                Some(val) => val.extract::<T>(),
                None => Ok(default),
            }
        }

        // Core
        let num_features: u8 = get(config, "num_features", 41)?;
        let num_trees: usize = get(config, "num_trees", 50)?;
        let k: usize = get(config, "k", 10)?;
        let max_depth: u32 = get(config, "max_depth", 15)?;
        let memory_limit_mb: usize = get(config, "memory_limit_mb", 100)?;
        let seed: u64 = get(config, "seed", 42)?;
        let warmup_samples: usize = get(config, "warmup_samples", 1000)?;
        let metrics_window: usize = get(config, "metrics_window", 1000)?;
        let adaptive_k_enabled: bool = get(config, "adaptive_k_enabled", true)?;
        let k_min: usize = get(config, "k_min", 3)?;
        let k_max: usize = get(config, "k_max", 50)?;

        // Budget
        let budget_enabled: bool = get(config, "budget_enabled", false)?;
        let budget_max_samples: usize = get(config, "budget_max_samples", 10000)?;
        let budget_eviction_batch: usize = get(config, "budget_eviction_batch", 100)?;
        let budget_minority_protection: f64 = get(config, "budget_minority_protection", 0.1)?;
        let budget_age_weight: f64 = get(config, "budget_age_weight", 0.4)?;
        let budget_influence_weight: f64 = get(config, "budget_influence_weight", 0.4)?;
        let budget_class_weight: f64 = get(config, "budget_class_weight", 0.2)?;
        let budget_skip_forest_forget: bool = get(config, "budget_skip_forest_forget", false)?;
        let budget_use_feature_distance: bool = get(config, "budget_use_feature_distance", false)?;

        // Influence tracking
        let influence_tracking: bool = get(config, "influence_tracking", false)?;
        let influence_update_interval: usize = get(config, "influence_update_interval", 10)?;
        let influence_sample_count: usize = get(config, "influence_sample_count", 200)?;
        let influence_strategy: String = get(config, "influence_strategy", "none".to_string())?;
        let feat_dist_update_interval: u64 = get(config, "feat_dist_update_interval", 2000)?;

        // Window retrain
        let window_retrain_mode: bool = get(config, "window_retrain_mode", false)?;
        let window_retrain_interval: u64 = get(config, "window_retrain_interval", 1)?;
        let window_retrain_incremental: bool = get(config, "window_retrain_incremental", false)?;

        // Develop
        let develop_interval: u64 = get(config, "develop_interval", 5)?;

        // Split quality monitoring
        let split_quality_threshold: Option<f64> = get(config, "split_quality_threshold", None)?;

        // Age-based subtree refresh: None (0) = disabled
        let split_max_age_raw: u64 = get(config, "split_max_age", 0)?;
        let split_max_age = if split_max_age_raw == 0 { None } else { Some(split_max_age_raw) };

        // Memory limit: 0 means effectively unlimited (usize::MAX)
        let memory_max_bytes = if memory_limit_mb == 0 {
            usize::MAX
        } else {
            memory_limit_mb * 1024 * 1024
        };

        let config = SUDAConfig {
            num_trees,
            k,
            max_depth,
            num_features,
            seed,
            memory_max_bytes,
            warmup_samples,
            metrics_window,
            adaptive_k_enabled,
            k_min,
            k_max,
            budget_enabled,
            budget_max_samples,
            budget_eviction_batch,
            budget_minority_protection,
            budget_age_weight,
            budget_influence_weight,
            budget_class_weight,
            influence_tracking_enabled: influence_tracking,
            influence_update_interval,
            influence_sample_count,
            influence_strategy,
            feat_dist_update_interval,
            budget_skip_forest_forget,
            budget_use_feature_distance,
            window_retrain_mode,
            window_retrain_interval,
            window_retrain_incremental,
            develop_interval,
            split_quality_threshold,
            split_max_age,
        };

        Ok(PyStreamingController {
            inner: StreamingController::new(config),
        })
    }

    /// Process a batch of samples (single FFI call).
    ///
    /// Args:
    ///     X: Feature array (n_samples, n_features)
    ///     y: Label array (n_samples,)
    ///
    /// Returns:
    ///     dict with keys: predictions, metrics, registry_size, memory_mb,
    ///                     total_samples, process_time_us, budget_evicted
    fn stream_batch<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let features: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| (0..n_features).map(|j| x_array[[i, j]]).collect())
            .collect();

        let labels: Vec<bool> = y_array.iter().copied().collect();

        let result = self.inner.stream_batch(&features, &labels);

        // Convert to Python dict
        let dict = PyDict::new(py);

        dict.set_item("predictions", result.predictions)?;

        // Metrics as nested dict
        let metrics_dict = PyDict::new(py);
        metrics_dict.set_item("accuracy", result.metrics.accuracy)?;
        metrics_dict.set_item("balanced_accuracy", result.metrics.balanced_accuracy)?;
        metrics_dict.set_item("gmean", result.metrics.gmean)?;
        metrics_dict.set_item("kappa", result.metrics.kappa)?;
        metrics_dict.set_item("attack_recall", result.metrics.attack_recall)?;
        metrics_dict.set_item("benign_recall", result.metrics.benign_recall)?;
        metrics_dict.set_item("precision", result.metrics.precision)?;
        metrics_dict.set_item("f1_score", result.metrics.f1_score)?;
        metrics_dict.set_item("total_samples", result.metrics.total_samples)?;
        dict.set_item("metrics", metrics_dict)?;

        dict.set_item("registry_size", result.registry_size)?;
        dict.set_item("memory_mb", result.memory_mb)?;
        dict.set_item("total_samples", result.total_samples)?;
        dict.set_item("process_time_us", result.process_time_us)?;
        dict.set_item("budget_evicted", result.budget_evicted)?;

        Ok(dict)
    }

    /// Get current metrics.
    fn current_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self.inner.current_metrics();
        let dict = PyDict::new(py);
        dict.set_item("accuracy", metrics.accuracy)?;
        dict.set_item("balanced_accuracy", metrics.balanced_accuracy)?;
        dict.set_item("gmean", metrics.gmean)?;
        dict.set_item("kappa", metrics.kappa)?;
        dict.set_item("attack_recall", metrics.attack_recall)?;
        dict.set_item("benign_recall", metrics.benign_recall)?;
        dict.set_item("precision", metrics.precision)?;
        dict.set_item("f1_score", metrics.f1_score)?;
        dict.set_item("total_samples", metrics.total_samples)?;
        Ok(dict)
    }

    /// Pre-train the model on historical data.
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let features: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| (0..n_features).map(|j| x_array[[i, j]]).collect())
            .collect();

        let labels: Vec<bool> = y_array.iter().copied().collect();

        let metrics = self.inner.fit(&features, &labels);

        let dict = PyDict::new(py);
        dict.set_item("accuracy", metrics.accuracy)?;
        dict.set_item("balanced_accuracy", metrics.balanced_accuracy)?;
        dict.set_item("gmean", metrics.gmean)?;
        dict.set_item("kappa", metrics.kappa)?;
        dict.set_item("attack_recall", metrics.attack_recall)?;
        dict.set_item("benign_recall", metrics.benign_recall)?;
        dict.set_item("precision", metrics.precision)?;
        dict.set_item("f1_score", metrics.f1_score)?;
        dict.set_item("total_samples", metrics.total_samples)?;

        Ok(dict)
    }

    /// Predict labels without updating training state or metrics.
    fn predict_batch(
        &self,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Vec<bool>> {
        let x_array = x.as_array();
        let n_samples = x_array.nrows();
        let n_features = x_array.ncols();

        let features: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| (0..n_features).map(|j| x_array[[i, j]]).collect())
            .collect();

        Ok(self.inner.predict_batch_only(&features))
    }

    /// Get total samples processed.
    #[getter]
    fn total_samples(&self) -> u64 {
        self.inner.total_samples()
    }

    /// Get current registry size.
    #[getter]
    fn registry_size(&self) -> usize {
        self.inner.registry_size()
    }

    /// Get memory usage in MB.
    #[getter]
    fn memory_mb(&self) -> f64 {
        self.inner.memory_mb()
    }

    /// Check if the model is warmed up.
    #[getter]
    fn is_warmed_up(&self) -> bool {
        self.inner.is_warmed_up()
    }

    /// Get current stream position.
    #[getter]
    fn position(&self) -> u64 {
        self.inner.position()
    }

    /// Check if the model has been pre-trained.
    #[getter]
    fn is_pretrained(&self) -> bool {
        self.inner.is_pretrained()
    }

    /// Check if budget mode is enabled.
    #[getter]
    fn budget_enabled(&self) -> bool {
        self.inner.budget_enabled()
    }

    /// Get total samples evicted by budget management.
    #[getter]
    fn total_budget_evicted(&self) -> usize {
        self.inner.total_budget_evicted()
    }

    /// Get budget eviction stats as (total, benign, attack, degraded).
    fn get_budget_eviction_stats(&self) -> (usize, usize, usize, usize) {
        self.inner.budget_eviction_stats()
    }

    /// Get influence coverage: (samples_with_influence, total_samples).
    fn get_influence_coverage(&self) -> (usize, usize) {
        self.inner.influence_coverage()
    }

    /// Get extended budget eviction stats including influence diagnostics.
    fn get_budget_eviction_stats_extended(&self) -> (usize, usize, usize, usize, usize, f64) {
        let stats = self.inner.registry().eviction_stats();
        (
            stats.evicted_count,
            stats.evicted_benign,
            stats.evicted_attack,
            stats.evicted_degraded,
            stats.evicted_with_influence,
            stats.evicted_influence_sum,
        )
    }

    /// Check if influence tracking is enabled.
    #[getter]
    fn influence_tracking_enabled(&self) -> bool {
        self.inner.influence_tracking_enabled()
    }

    /// Get count of influence-degraded samples.
    #[getter]
    fn influence_degraded_count(&self) -> usize {
        self.inner.influence_degraded_count()
    }

    /// Reset the controller state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    // =========================================================================
    // Analysis API
    // =========================================================================

    /// Get lifecycle data for all tracked samples.
    fn get_lifecycle_data<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let data = self.inner.registry.get_lifecycle_data();
        let result: Vec<Bound<'py, PyDict>> = data
            .into_iter()
            .map(|s| {
                let dict = PyDict::new(py);
                dict.set_item("id", s.id).unwrap();
                dict.set_item("label", s.label).unwrap();
                dict.set_item("insertion_position", s.insertion_position).unwrap();
                dict.set_item("influence_history", s.influence_history.clone()).unwrap();
                dict.set_item("removal_rank_history", s.removal_rank_history.clone()).unwrap();
                dict.set_item("average_influence", s.average_influence).unwrap();
                dict.set_item("influence_trend", s.influence_trend).unwrap();
                dict
            })
            .collect();
        Ok(result)
    }

    /// Get core concept samples (consistently positive influence).
    fn get_core_concept_samples(&self, min_observations: usize, min_avg_influence: f64) -> Vec<u64> {
        self.inner.registry.get_core_concept_samples(min_observations, min_avg_influence)
    }

    /// Get samples with declining influence over time.
    fn get_declining_samples(&self, min_observations: usize) -> Vec<(u64, f64)> {
        self.inner.registry.get_declining_influence_samples(min_observations)
    }

    /// Get longest surviving samples.
    fn get_longest_surviving(&self, n: usize) -> Vec<(u64, u64)> {
        self.inner.registry.get_longest_surviving_samples(n)
    }

    /// Get high-risk samples (frequently near removal).
    fn get_high_risk_samples(&self, top_n: usize, min_risk_count: usize) -> Vec<(u64, usize)> {
        self.inner.registry.get_high_risk_samples(top_n, min_risk_count)
    }

    /// Get samples with stable influence (low variance).
    fn get_stable_samples(&self, max_variance: f64, min_observations: usize) -> Vec<u64> {
        self.inner.registry.get_stable_influence_samples(max_variance, min_observations)
    }

    /// Record influence scores for analysis (call during selection).
    fn record_influence_batch(&mut self, sample_ids: Vec<u64>, influences: Vec<f64>) {
        for (&id, &inf) in sample_ids.iter().zip(influences.iter()) {
            self.inner.registry.record_influence(id, inf);
        }
    }

    /// Record removal ranks for analysis (call during selection).
    fn record_removal_ranks(&mut self, ranked_ids: Vec<u64>) {
        self.inner.registry.record_removal_ranks(&ranked_ids);
    }

    /// Get class distribution in current registry.
    fn get_class_distribution(&self) -> (usize, usize) {
        let counts = self.inner.registry.class_counts();
        (counts[0], counts[1])
    }

    /// Get imbalance ratio (majority/minority).
    fn get_imbalance_ratio(&self) -> f64 {
        self.inner.registry.imbalance_ratio()
    }

    /// Get registry diagnostics for Phase 3 analysis.
    ///
    /// Returns a dict with age-influence Spearman correlation, influence coverage,
    /// mean age/influence, and class distribution.
    fn get_registry_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let diag = self.inner.registry.get_diagnostics();
        let dict = PyDict::new(py);
        for (key, value) in &diag {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    }
}
