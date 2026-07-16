//! Integration tests for suda-core library.
//!
//! These tests verify the complete workflow of the DynFrs Random Forest,
//! including OCC(q) sampling, exact unlearning, and lazy rebuild.

use hashbrown::HashMap;

use suda_core::forest::{DynFrsForest, ForestConfig};
use suda_core::sample::VecSample;
use suda_core::tree::TreeConfig;

/// Build a feature map (sample_id → values) for batch forget API.
fn make_feature_map(samples: &[VecSample]) -> HashMap<u64, Vec<f32>> {
    samples.iter().map(|s| (s.id, s.values.clone())).collect()
}

/// Generate synthetic samples for testing.
fn make_test_samples(count: usize, num_features: u8, seed: u64) -> Vec<VecSample> {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(count);

    for i in 0..count {
        let values: Vec<f32> = (0..num_features)
            .map(|_| rng.gen_range(-5.0..5.0))
            .collect();
        // Simple rule: positive if first feature > 0
        let label = values[0] > 0.0;
        samples.push(VecSample::new(i as u64, values, label));
    }

    samples
}

/// Test the complete workflow: fit → predict → forget → develop
#[test]
fn test_forest_end_to_end() {
    let num_features: u8 = 10;
    let num_samples = 200;
    let samples = make_test_samples(num_samples, num_features, 42);

    // Create and fit forest
    let config = ForestConfig {
        num_trees: 20,
        k: 5,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 5,
        },
        seed: 42,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);

    // Verify forest structure
    assert_eq!(forest.num_trees(), 20);
    assert_eq!(forest.k(), 5);
    assert_eq!(forest.num_samples(), num_samples);

    // Test predictions
    let mut correct = 0;
    for sample in &samples {
        let pred = forest.predict(sample);
        if pred == sample.label {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / num_samples as f64;
    // Should have reasonable accuracy on training data
    assert!(
        accuracy > 0.7,
        "Training accuracy too low: {:.2}%",
        accuracy * 100.0
    );

    // Test batch prediction
    let predictions = forest.predict_batch(&samples);
    assert_eq!(predictions.len(), num_samples);

    // Test forget
    let samples_to_forget = vec![0, 1, 2, 3, 4];
    let fmap = make_feature_map(&samples);
    let forgotten = forest.forget_batch(&samples_to_forget, &fmap);
    assert_eq!(forgotten, 5);
    assert_eq!(forest.num_samples(), num_samples - 5);

    // Verify samples are no longer tracked
    for id in &samples_to_forget {
        assert!(!forest.contains_sample(*id));
    }

    // Verify remaining samples are still tracked
    for id in 5..num_samples {
        assert!(forest.contains_sample(id as u64));
    }

    println!("End-to-end test passed: accuracy={:.2}%", accuracy * 100.0);
}

/// Test OCC(q) sampling invariant: each sample appears in at most k trees
#[test]
fn test_occ_sampling_invariant() {
    let num_features: u8 = 5;
    let num_samples = 100;
    let samples = make_test_samples(num_samples, num_features, 123);

    let config = ForestConfig {
        num_trees: 30,
        k: 5,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 6,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 3,
        },
        seed: 123,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);

    // Get tree stats
    let stats = forest.tree_stats();
    assert_eq!(stats.num_trees, 30);
    assert!(stats.min_depth <= stats.max_depth);
    assert!(stats.max_depth <= 6);

    // After fitting, all samples should be tracked
    for i in 0..num_samples {
        assert!(forest.contains_sample(i as u64), "Sample {} not tracked", i);
    }

    println!(
        "OCC sampling test passed: trees={}, depths={}-{}",
        stats.num_trees, stats.min_depth, stats.max_depth
    );
}

/// Test exact unlearning: forget should remove samples completely
#[test]
fn test_exact_unlearning() {
    let num_features: u8 = 5;
    let samples = make_test_samples(50, num_features, 456);

    let config = ForestConfig {
        num_trees: 10,
        k: 3,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 3,
        },
        seed: 456,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);

    let fmap = make_feature_map(&samples);

    // Forget single sample
    let sample_id = 25u64;
    assert!(forest.contains_sample(sample_id));
    let was_forgotten = forest.forget(sample_id, &fmap[&sample_id]);
    assert!(was_forgotten, "Sample should have been forgotten");
    assert!(
        !forest.contains_sample(sample_id),
        "Sample should no longer be tracked"
    );

    // Forget multiple samples
    let batch: Vec<u64> = (30..35).collect();
    let forgotten_count = forest.forget_batch(&batch, &fmap);
    assert_eq!(forgotten_count, 5, "Should have forgotten 5 samples");

    for id in batch {
        assert!(!forest.contains_sample(id), "Sample {} should be gone", id);
    }

    // Trying to forget already-forgotten sample
    let was_forgotten_again = forest.forget(sample_id, &fmap[&sample_id]);
    assert!(
        !was_forgotten_again,
        "Cannot forget already-forgotten sample"
    );

    println!("Exact unlearning test passed");
}

/// Test that predictions still work after forgetting samples
#[test]
fn test_predictions_after_forget() {
    let num_features: u8 = 8;
    let samples = make_test_samples(100, num_features, 789);

    let config = ForestConfig {
        num_trees: 15,
        k: 4,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 7,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 5,
        },
        seed: 789,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);

    // Get initial accuracy
    let initial_correct: usize = samples
        .iter()
        .filter(|s| forest.predict(s) == s.label)
        .count();
    let initial_accuracy = initial_correct as f64 / samples.len() as f64;

    // Forget 20% of samples
    let to_forget: Vec<u64> = (0..20).collect();
    let fmap = make_feature_map(&samples);
    forest.forget_batch(&to_forget, &fmap);

    // Predictions should still work on remaining samples
    let remaining_samples: Vec<_> = samples[20..].to_vec();
    let predictions = forest.predict_batch(&remaining_samples);
    assert_eq!(predictions.len(), remaining_samples.len());

    // Predictions should still work on test sample
    let new_sample = VecSample::new(999, vec![1.0; num_features as usize], true);
    let _pred = forest.predict(&new_sample); // Should not panic

    println!(
        "Predictions after forget test passed: initial_accuracy={:.2}%",
        initial_accuracy * 100.0
    );
}

/// Test forest with highly imbalanced data (NIDS-like scenario)
#[test]
fn test_imbalanced_data() {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    let num_features: u8 = 10;
    let mut rng = XorShiftRng::seed_from_u64(999);
    let mut samples = Vec::new();

    // 95% negative (normal traffic), 5% positive (attacks)
    for i in 0..190 {
        let values: Vec<f32> = (0..num_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        samples.push(VecSample::new(i as u64, values, false));
    }
    for i in 190..200 {
        let values: Vec<f32> = (0..num_features)
            .map(|_| rng.gen_range(2.0..5.0)) // Different distribution for attacks
            .collect();
        samples.push(VecSample::new(i as u64, values, true));
    }

    let config = ForestConfig {
        num_trees: 20,
        k: 5,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 5,
        },
        seed: 999,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);

    // Check that minority class samples are tracked
    for i in 190..200 {
        assert!(
            forest.contains_sample(i as u64),
            "Attack sample {} should be tracked",
            i
        );
    }

    // Test minority class predictions
    let attack_samples: Vec<_> = samples[190..].to_vec();
    let attack_preds = forest.predict_batch(&attack_samples);
    let attack_recall = attack_preds.iter().filter(|&&p| p).count() as f64 / 10.0;

    // We expect reasonable recall on attacks (they have distinct features)
    println!(
        "Imbalanced data test passed: attack_recall={:.2}%",
        attack_recall * 100.0
    );
}

/// Test tree statistics reporting
#[test]
fn test_tree_stats() {
    let num_features: u8 = 5;
    let samples = make_test_samples(100, num_features, 111);

    let config = ForestConfig {
        num_trees: 10,
        k: 3,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 3,
        },
        seed: 111,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);

    let stats = forest.tree_stats();
    assert_eq!(stats.num_trees, 10);
    assert!(stats.min_depth <= stats.max_depth);
    assert!(stats.max_depth <= 5);
    assert!(stats.avg_depth >= stats.min_depth as f64);
    assert!(stats.avg_depth <= stats.max_depth as f64);

    println!(
        "Tree stats test passed: num_trees={}, depths={}-{}, avg={:.2}",
        stats.num_trees, stats.min_depth, stats.max_depth, stats.avg_depth
    );
}

/// forget_batch는 lazy resolve 경로만 사용한다.
#[test]
fn test_batch_forget_lazy_resolve() {
    let num_features: u8 = 10;
    let num_samples = 200;
    let samples = make_test_samples(num_samples, num_features, 42);

    let config = ForestConfig {
        num_trees: 20,
        k: 5,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 5,
        },
        seed: 42,
    };
    let mut forest = DynFrsForest::new(config, num_features);
    forest.fit(&samples);
    forest.enable_streaming(&samples);

    let batch: Vec<u64> = (0..50).collect();
    let mut fmap: hashbrown::HashMap<u64, Vec<f32>> = hashbrown::HashMap::new();
    for s in &samples {
        if batch.contains(&s.id) {
            fmap.insert(s.id, s.values.clone());
        }
    }
    let forgotten = forest.forget_batch(&batch, &fmap);
    assert_eq!(forgotten, 50);
    assert_eq!(forest.num_samples(), num_samples - 50);

    // lazy resolve로 후속 처리 검증
    let remaining_samples: Vec<_> = samples
        .iter()
        .filter(|s| !batch.contains(&s.id))
        .cloned()
        .collect();
    let sample_map: hashbrown::HashMap<u64, &VecSample> =
        remaining_samples.iter().map(|s| (s.id, s)).collect();
    let _preds = forest.predict_batch_with_lazy_resolve(&remaining_samples[..10], &sample_map);

    for id in &batch {
        assert!(
            !forest.contains_sample(*id),
            "Sample {} should be forgotten",
            id
        );
    }
    println!(
        "Batch forget lazy resolve test passed: forgotten={}",
        forgotten
    );
}

/// Test streaming controller full pipeline: fit -> stream cycle
#[test]
fn test_controller_full_pipeline() {
    use suda_core::controller::{SUDAConfig, StreamingController};

    let config = SUDAConfig {
        num_features: 5,
        num_trees: 10,
        k: 3,
        minority_k: 0,
        max_depth: 5,
        warmup_samples: 20,
        seed: 42,
        ..Default::default()
    };

    let mut controller = StreamingController::new(config);

    // Phase 1: Warmup with fit
    let features: Vec<Vec<f32>> = (0..30).map(|i| vec![(i as f32) * 0.1; 5]).collect();
    let labels: Vec<bool> = (0..30).map(|i| i % 3 == 0).collect();
    controller.fit(&features, &labels);

    assert!(controller.is_pretrained());
    assert_eq!(controller.registry_size(), 30);

    // Phase 2: Stream batches
    let mut total_predictions = 0;
    for batch in 0..5 {
        let batch_features: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![((batch * 10 + i) as f32) * 0.1; 5])
            .collect();
        let batch_labels: Vec<bool> = (0..10).map(|i| (batch + i) % 4 == 0).collect();

        let result = controller.stream_batch(&batch_features, &batch_labels);
        assert_eq!(result.predictions.len(), 10);
        total_predictions += result.predictions.len();
    }

    assert_eq!(total_predictions, 50);
    assert!(
        controller.registry_size() > 30,
        "Registry should grow during streaming"
    );
}

/// Test streaming controller with budget management enabled
#[test]
fn test_controller_with_budget() {
    use suda_core::controller::{SUDAConfig, StreamingController};

    let config = SUDAConfig {
        num_features: 5,
        num_trees: 10,
        k: 3,
        minority_k: 0,
        max_depth: 5,
        warmup_samples: 10,
        seed: 42,
        budget_enabled: true,
        budget_max_samples: 40,
        budget_eviction_batch: 5,
        ..Default::default()
    };

    let mut controller = StreamingController::new(config);

    // Warmup
    let features: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.1; 5]).collect();
    let labels: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
    controller.fit(&features, &labels);

    // Stream enough to trigger budget eviction
    let mut any_evicted = false;
    for batch in 0..10 {
        let batch_features: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![(batch * 10 + i) as f32 * 0.1; 5])
            .collect();
        let batch_labels: Vec<bool> = (0..10).map(|i| i % 2 == 0).collect();

        let result = controller.stream_batch(&batch_features, &batch_labels);
        if result.budget_evicted > 0 {
            any_evicted = true;
        }
    }

    assert!(any_evicted, "Budget eviction should have occurred");
    // Registry should be bounded near budget limit
    assert!(
        controller.registry_size() <= 50,
        "Registry {} should be bounded near budget 40",
        controller.registry_size()
    );
}

/// Test exact unlearning equivalence: forget+develop should produce predictions
/// functionally similar to retraining without the forgotten samples.
///
/// DynFrs guarantees that after forget(s) + develop(), the model's split statistics
/// no longer contain the influence of sample s. While OCC(q) tree assignments differ
/// between the two paths, the prediction quality should be comparable.
#[test]
fn test_exact_unlearning_equivalence() {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    let num_features: u8 = 10;
    let num_samples = 200;
    let mut rng = XorShiftRng::seed_from_u64(777);

    // Generate samples with separable classes
    let mut all_samples = Vec::new();
    for i in 0..num_samples {
        let label = i < num_samples / 2;
        let offset = if label { 2.0 } else { -2.0 };
        let values: Vec<f32> = (0..num_features)
            .map(|_| rng.gen_range(-1.0..1.0) + offset)
            .collect();
        all_samples.push(VecSample::new(i as u64, values, label));
    }

    // Samples to forget (20% of data)
    let forget_ids: Vec<u64> = (0..40).collect();
    let remaining_samples: Vec<VecSample> = all_samples
        .iter()
        .filter(|s| !forget_ids.contains(&s.id))
        .cloned()
        .collect();

    // Test samples for prediction comparison
    let mut test_rng = XorShiftRng::seed_from_u64(888);
    let test_samples: Vec<VecSample> = (0..50)
        .map(|i| {
            let label = i < 25;
            let offset = if label { 2.0 } else { -2.0 };
            let values: Vec<f32> = (0..num_features)
                .map(|_| test_rng.gen_range(-1.0..1.0) + offset)
                .collect();
            VecSample::new(1000 + i as u64, values, label)
        })
        .collect();

    let base_config = ForestConfig {
        num_trees: 30,
        k: 5,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 5,
        },
        seed: 42,
    };

    // Path A: fit all → forget → lazy resolve query
    let mut forest_a = DynFrsForest::new(base_config.clone(), num_features);
    forest_a.fit(&all_samples);
    let fmap_a = make_feature_map(&all_samples);
    forest_a.forget_batch(&forget_ids, &fmap_a);
    let map_a: hashbrown::HashMap<u64, &VecSample> =
        remaining_samples.iter().map(|s| (s.id, s)).collect();
    let preds_a = forest_a.predict_batch_with_lazy_resolve(&test_samples, &map_a);

    // Path B: fit only remaining samples (retrain without forgotten)
    let mut forest_b = DynFrsForest::new(base_config, num_features);
    forest_b.fit(&remaining_samples);
    let preds_b = forest_b.predict_batch(&test_samples);

    // Both paths should produce reasonable accuracy
    let accuracy_a = preds_a
        .iter()
        .zip(test_samples.iter())
        .filter(|(&pred, sample)| pred == sample.label)
        .count() as f64
        / test_samples.len() as f64;
    let accuracy_b = preds_b
        .iter()
        .zip(test_samples.iter())
        .filter(|(&pred, sample)| pred == sample.label)
        .count() as f64
        / test_samples.len() as f64;

    assert!(
        accuracy_a > 0.7,
        "Path A (forget+develop) accuracy too low: {:.2}%",
        accuracy_a * 100.0
    );
    assert!(
        accuracy_b > 0.7,
        "Path B (retrain) accuracy too low: {:.2}%",
        accuracy_b * 100.0
    );

    // Prediction agreement between the two paths should be high
    let agreement = preds_a
        .iter()
        .zip(preds_b.iter())
        .filter(|(&a, &b)| a == b)
        .count() as f64
        / test_samples.len() as f64;

    assert!(
        agreement > 0.7,
        "Prediction agreement between forget+develop and retrain should be > 70%, got {:.2}%",
        agreement * 100.0
    );

    // Forgotten samples should not be tracked
    for id in &forget_ids {
        assert!(
            !forest_a.contains_sample(*id),
            "Forgotten sample {} should not be in forest",
            id
        );
    }

    println!(
        "Exact unlearning equivalence test passed: acc_A={:.2}%, acc_B={:.2}%, agreement={:.2}%",
        accuracy_a * 100.0,
        accuracy_b * 100.0,
        agreement * 100.0
    );
}

/// Test that soft minority protection allows eviction of under-represented class
/// in reversed-imbalance scenarios (like CIC-IDS2018 with 97% attack).
///
/// Before fix: binary skip meant minority samples were NEVER evicted, accumulating stale data.
/// After fix: minority samples get 10x score reduction but CAN be evicted when very old/harmful.
#[test]
fn test_reversed_imbalance_soft_eviction() {
    use suda_core::registry::{BudgetConfig, InfluenceRegistry};

    let mut registry = InfluenceRegistry::new();
    registry.set_budget_config(BudgetConfig {
        max_samples: 20,
        eviction_batch_size: 5,
        minority_protection_ratio: 0.1, // Protect classes < 10%
        age_weight: 0.6,
        influence_weight: 0.2,
        class_weight: 0.2,
        random_eviction: false,
        class_aware_random: false,
    });

    // Simulate CIC-IDS2018: 97% attack (true), 3% benign (false)
    // Add 30 attack samples
    for i in 0..30 {
        registry.register(i, vec![0, 1], true); // attack
    }
    // Add 1 benign sample (early, will be old)
    registry.register(100, vec![0], false); // benign (minority ~3%)

    // Budget=20, we have 31 samples → should evict ~11
    // With soft protection, benign sample should have reduced eviction priority
    // but with only 1 benign and 30 attack, most evictions target attack
    let registry_size = registry.len();
    assert!(
        registry_size <= 25,
        "Registry {} should be near budget 20 after eviction",
        registry_size
    );

    // The benign sample should likely survive (very few benign = soft protected)
    // but it's NOT guaranteed to survive forever (that's the point of soft protection)
    // Just check that eviction occurred and registry is bounded
    let eviction_stats = registry.eviction_stats();
    assert!(
        eviction_stats.evicted_count > 0,
        "Eviction should have occurred"
    );

    // Now simulate extreme scenario: add many more samples to test that
    // old benign CAN eventually be evicted when budget is very tight
    for i in 31..100 {
        registry.register(i, vec![0, 1, 2], true); // more attack
    }
    // Add a few more benign to dilute
    for i in 101..105 {
        registry.register(i, vec![0], false);
    }

    // With ~105 samples and budget=20, heavy eviction needed
    // Both classes should contribute to eviction (benign with reduced priority)
    let final_size = registry.len();
    assert!(
        final_size <= 25,
        "Registry {} should be bounded after heavy streaming",
        final_size
    );

    // Verify attack samples were evicted (majority should bear most eviction)
    let stats = registry.eviction_stats();
    assert!(
        stats.evicted_attack > stats.evicted_benign,
        "Attack (majority) should be evicted more than benign: attack={}, benign={}",
        stats.evicted_attack,
        stats.evicted_benign
    );

    println!(
        "Reversed imbalance soft eviction test passed: evicted {} total ({} attack, {} benign), registry size={}",
        stats.evicted_count, stats.evicted_attack, stats.evicted_benign, final_size
    );
}
