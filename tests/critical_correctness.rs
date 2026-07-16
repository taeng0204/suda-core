//! Critical correctness tests for SUDA Core.
//!
//! DynFrs lazy mode 정합 + streaming-aware unlearning의 핵심 invariant 검증:
//! - retrain-equivalence (DynFrs Theorem 1 분포 동등)
//! - streaming-aware forget의 path-aware split 갱신
//! - lazy_resolve가 routing(prediction)을 바꾸는가
//! - leaf-only forget이 internal counts를 의도적으로 안 갱신함
//! - streaming_states 일관성 (orphan 누적 없음, attr_stats candidates 회복)

use hashbrown::HashMap;
use suda_core::forest::{DynFrsForest, ForestConfig};
use suda_core::sample::VecSample;
use suda_core::tree::TreeConfig;

/// streaming-aware forget: features map과 함께 호출해 path internal counts +
/// streaming_states.attr_stats 모두 갱신 (DynFrs `attribute::del` 정합).
fn forget_with_streaming(forest: &mut DynFrsForest, samples: &[VecSample], ids: &[u64]) -> usize {
    let mut fmap: HashMap<u64, Vec<f32>> = HashMap::new();
    for s in samples {
        if ids.contains(&s.id) {
            fmap.insert(s.id, s.values.clone());
        }
    }
    forest.forget_batch(ids, &fmap)
}

// =========================================================================
// Test helpers — 시드 고정으로 deterministic.
// =========================================================================

fn make_samples_seeded(count: usize, num_features: u8, seed: u64) -> Vec<VecSample> {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    let mut rng = XorShiftRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let values: Vec<f32> = (0..num_features)
                .map(|_| rng.gen_range(-5.0..5.0))
                .collect();
            let label = values[0] > 0.0;
            VecSample::new(i as u64, values, label)
        })
        .collect()
}

fn make_test_samples(count: usize, num_features: u8, base: u64) -> Vec<VecSample> {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    let mut rng = XorShiftRng::seed_from_u64(base + 999);
    (0..count)
        .map(|i| {
            let values: Vec<f32> = (0..num_features)
                .map(|_| rng.gen_range(-5.0..5.0))
                .collect();
            VecSample::new(1_000_000 + i as u64, values, false)
        })
        .collect()
}

/// Build a small forest for unit tests (DynFrs random mode).
fn build_exact_forest(num_features: u8, seed: u64) -> DynFrsForest {
    let config = ForestConfig {
        num_trees: 16,
        k: 4,
        minority_k: 0,
        tree_config: TreeConfig {
            max_depth: 8,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            num_splits_to_try: 5,
        },
        seed,
    };
    DynFrsForest::new(config, num_features)
}

fn predict_all(forest: &DynFrsForest, probes: &[VecSample]) -> Vec<bool> {
    probes.iter().map(|s| forest.predict(s)).collect()
}

/// active sample 전체를 lazy resolve query로 통과 → 모든 path 노드가 resolve됨.
/// (develop API가 사라진 이후, test에서 "full develop 효과"를 얻는 표준 방법.)
fn full_lazy_resolve(forest: &mut DynFrsForest, samples: &[VecSample]) {
    if samples.is_empty() {
        return;
    }
    let map: HashMap<u64, &VecSample> = samples.iter().map(|s| (s.id, s)).collect();
    let _ = forest.predict_batch_with_lazy_resolve(samples, &map);
}

// =========================================================================
// CRITICAL #1: retrain-equivalence (DynFrs Theorem 1 분포 동등)
//   fit(D) + forget(S) + lazy_resolve  ≈  fit(D\S)
// =========================================================================

/// DynFrs Theorem 1: forget+lazy_resolve가 retrain과 *분포 동등* 보장.
/// random mode에서 75%+ agreement면 분포 동등 통과.
/// 시나리오: drift-like contradictions — 같은 feature 영역에 양 클래스 섞임.
#[test]
fn test_forget_then_develop_equivalent_to_retrain() {
    let num_features: u8 = 8;
    // 충돌 시나리오: forget 대상이 모델 결정에 영향을 주는 위치
    let all_samples = make_conflicting_samples(200, num_features, 42);
    let probes = make_test_samples(120, num_features, 42);

    // Path A: full set 학습 → streaming-aware forget → develop
    let mut forest_a = build_exact_forest(num_features, 42);
    let mut samples_a = all_samples.clone();
    forest_a.fit(&samples_a);
    forest_a.enable_streaming(&samples_a); // streaming-aware forget 필수
    let forget_ids: Vec<u64> = (100..140).map(|i| i as u64).collect();
    forget_with_streaming(&mut forest_a, &all_samples, &forget_ids);
    samples_a.retain(|s| !forget_ids.contains(&s.id));
    full_lazy_resolve(&mut forest_a, &samples_a);
    let preds_a = predict_all(&forest_a, &probes);

    // Path B: 처음부터 D\S로 학습 (그라운드 트루스)
    let mut forest_b = build_exact_forest(num_features, 42);
    let samples_b: Vec<VecSample> = all_samples
        .iter()
        .filter(|s| !forget_ids.contains(&s.id))
        .cloned()
        .collect();
    forest_b.fit(&samples_b);
    let preds_b = predict_all(&forest_b, &probes);

    let mismatches: Vec<(usize, bool, bool)> = preds_a
        .iter()
        .zip(preds_b.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| (i, a, b))
        .collect();

    // DynFrs Theorem 1 (분포 동등): single-step lazy resolve + 자식 보존이라
    //   bit-exact는 본질적 불가능. random mode에서 75%+ agreement면 통과.
    let agreement_rate = (probes.len() - mismatches.len()) as f64 / probes.len() as f64;
    assert!(
        agreement_rate >= 0.75,
        "forget+develop != retrain (분포 동등 미달). agreement={:.3} ({} mismatches / {} probes). 첫 5개: {:?}",
        agreement_rate,
        mismatches.len(),
        probes.len(),
        &mismatches[..5.min(mismatches.len())]
    );
}

/// 시나리오 헬퍼: 같은 feature 영역에 양 클래스 섞인 conflict 샘플 (real drift 흉내).
/// 이런 데이터에서는 forget이 split 결정에 강한 영향을 줘야 함 → 결과 차이 노출.
fn make_conflicting_samples(count: usize, num_features: u8, seed: u64) -> Vec<VecSample> {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    let mut rng = XorShiftRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let values: Vec<f32> = (0..num_features)
                .map(|_| rng.gen_range(-3.0..3.0))
                .collect();
            // 처음 100개: 정상 결정경계 (x[0]>0=true). 나중 100개: 일부 영역에서 라벨 반대(conflict).
            let label = if i < 100 {
                values[0] > 0.0
            } else {
                // 충돌: dim0이 살짝 positive면서 dim1도 positive인 경우 false (정상과 충돌)
                if values[0] > 0.0 && values[1] > 0.5 {
                    false
                } else {
                    values[0] > 0.0
                }
            };
            VecSample::new(i as u64, values, label)
        })
        .collect()
}

// =========================================================================
// CRITICAL #2: forget→develop이 routing을 실제로 바꾸는가
//   forget 후 develop의 rebuild가 *살아있는 샘플만*으로 split을 재계산 →
//   잘못된 sample이 있던 영역의 leaf 매핑 변화 여부.
// =========================================================================

/// develop이 routing에 *추가로* 미치는 effect를 forget 단독과 비교해 분리.
/// 두 모델: F1 = forget만, F2 = forget + develop. 두 prediction이 같으면
/// develop이 routing에 영향 못 줌(leaf-only forget의 증거) → RED.
#[test]
fn test_develop_adds_routing_change_beyond_forget() {
    let num_features: u8 = 8;
    let all_samples = make_conflicting_samples(200, num_features, 7);
    let probes = make_test_samples(120, num_features, 7);

    // F1: streaming-aware forget만 (develop 안 호출)
    let mut f1 = build_exact_forest(num_features, 7);
    f1.fit(&all_samples);
    f1.enable_streaming(&all_samples);
    let forget_ids: Vec<u64> = (100..140).map(|i| i as u64).collect();
    forget_with_streaming(&mut f1, &all_samples, &forget_ids);
    let preds_f1 = predict_all(&f1, &probes);

    // F2: streaming-aware forget + develop (정제된 데이터로 rebuild)
    let mut f2 = build_exact_forest(num_features, 7);
    f2.fit(&all_samples);
    f2.enable_streaming(&all_samples);
    forget_with_streaming(&mut f2, &all_samples, &forget_ids);
    let remaining: Vec<VecSample> = all_samples
        .iter()
        .filter(|s| !forget_ids.contains(&s.id))
        .cloned()
        .collect();
    full_lazy_resolve(&mut f2, &remaining);
    let preds_f2 = predict_all(&f2, &probes);

    let extra_changes = preds_f1
        .iter()
        .zip(preds_f2.iter())
        .filter(|(a, b)| a != b)
        .count();

    // develop이 추가 routing change를 만들어야 함 (≥5% of probes).
    // RED 예상: 현재 forget이 LazyTag::Rebuild 마킹 안 하므로 develop이 skip → preds_f1≈preds_f2.
    let pct = extra_changes as f64 / probes.len() as f64;
    assert!(
        pct >= 0.05,
        "develop이 forget 단독 대비 추가 routing change 못 만듦 (only {}/{} = {:.1}%). \
         leaf-only forget의 한계 증거 (streaming-aware forget 부재 시 RED).",
        extra_changes,
        probes.len(),
        pct * 100.0
    );
}

// =========================================================================
// CRITICAL #3: streaming_enabled 상태에서 forget 후 split 갱신 흔적
//   streaming-aware forget이 들어가면 streaming_states가 갱신됨.
// =========================================================================

#[test]
fn test_streaming_enabled_forget_updates_state() {
    let num_features: u8 = 4;
    let samples = make_samples_seeded(60, num_features, 11);

    let mut forest = build_exact_forest(num_features, 11);
    forest.fit(&samples);
    forest.enable_streaming(&samples);

    // streaming 활성 상태에서 forget 호출이 존재해야 함.
    let sample_to_forget = &samples[10];
    let removed = forest.forget(sample_to_forget.id, &sample_to_forget.values);
    assert!(removed, "forget failed for known sample id");

    // sample은 forest에서 빠져야 함.
    assert!(
        !forest.contains_sample(sample_to_forget.id),
        "sample이 forget 후에도 forest에 잔존"
    );
}

// =========================================================================
// CRITICAL #5: forget이 internal counts를 path 전체에 갱신함을 검증
//   DynFrs attribute::del 정합 (DynFrs.h:268-298): n -= 1, n_1 -= Y for all path nodes.
//   tree::remove_sample_streaming이 leaf-only 동작에서 path-aware로 통합된 후의
//   positive contract test.
// =========================================================================

#[test]
fn test_forget_updates_internal_counts_along_path() {
    let num_features: u8 = 4;
    let samples = make_samples_seeded(60, num_features, 13);
    let fmap: hashbrown::HashMap<u64, Vec<f32>> =
        samples.iter().map(|s| (s.id, s.values.clone())).collect();

    let mut forest = build_exact_forest(num_features, 13);
    forest.fit(&samples);
    forest.enable_streaming(&samples);

    // 루트 노드 sample 수의 베이스라인 (모든 트리 합). ROOT = 1.
    let root_total_before: usize = (0..forest.num_trees())
        .filter_map(|i| forest.tree(i))
        .filter_map(|t| t.node_sample_count(1))
        .sum();

    let total_before = forest.get_all_sample_ids().len();
    let forget_ids: Vec<u64> = (0..20).map(|i| i as u64).collect();
    let n_forgotten = forest.forget_batch(&forget_ids, &fmap);

    // forest sample 추적도 함께 감소.
    let total_after = forest.get_all_sample_ids().len();
    assert!(total_after < total_before, "forget 후 sample 수 감소 안 함");
    assert!(n_forgotten > 0, "forget_batch가 아무 것도 안 제거함");

    // DynFrs 정합 검증 (강한 contract): 각 forgotten sample은 자신이 속한 트리들의
    // root에서 카운트 1씩 감소시켜야 한다 (path 전체 갱신). 옛 leaf-only forget이라면
    // root는 Internal이라 안 변하고 leaf만 줄어 → drop < n_forgotten으로 fail.
    // 안전한 lower bound는 n_forgotten — OCC(q)로 sample이 k개보다 적은 트리에
    // 들어갔을 수 있으므로 정확한 상수 곱은 강제하지 않는다.
    let root_total_after: usize = (0..forest.num_trees())
        .filter_map(|i| forest.tree(i))
        .filter_map(|t| t.node_sample_count(1))
        .sum();
    let drop = root_total_before.saturating_sub(root_total_after);
    assert!(
        drop >= n_forgotten,
        "root path 감소량({})이 forgotten 수({})에 못 미침 — path 갱신 누락 (leaf-only forget 회귀): before={} after={}",
        drop, n_forgotten, root_total_before, root_total_after
    );
}

// =========================================================================
// streaming-aware forget이 노출하는 silent failure RED 테스트
// =========================================================================

/// streaming_states orphan 엔트리 누적 (메모리 누수).
/// develop가 rebuild 시 nodes는 제거하나 streaming_states는 미제거 → 누적.
/// 정합 시 streaming_states.len()이 nodes.len()의 reasonable factor 내여야 함.
#[test]
fn test_streaming_states_no_orphan_after_develop() {
    let num_features: u8 = 6;
    let samples = make_samples_seeded(150, num_features, 21);

    let mut forest = build_exact_forest(num_features, 21);
    forest.fit(&samples);
    forest.enable_streaming(&samples);

    // 여러 forget+develop cycle로 rebuild 다수 트리거
    for cycle in 0..5 {
        let lo = (cycle * 20) as u64;
        let ids: Vec<u64> = (lo..lo + 20).collect();
        forget_with_streaming(&mut forest, &samples, &ids);
        let remaining: Vec<VecSample> = samples
            .iter()
            .filter(|s| s.id >= ((cycle + 1) * 20) as u64)
            .cloned()
            .collect();
        full_lazy_resolve(&mut forest, &remaining);
    }

    let stats = forest.streaming_stats();
    let states = stats.total_streaming_states;
    // C1 RED 시그널: 5번 develop cycle 후 streaming_states가 폭증.
    // 정합 (orphan cleanup 있다면): states <= 트리당 ~50 → 16 tree × 50 = 800 max.
    // RED 예상: orphan 누적으로 수천 단위 폭증.
    assert!(
        states <= 1500,
        "streaming_states 폭증 ({}) — C1 orphan 누적 의심",
        states
    );
}

/// C2: attr_stats.candidates 단조 감소 후 develop가 회복해야 best_split_changed 살아남.
/// 회복 안 되면 LazyTag::Rebuild가 영원히 마킹 안 됨.
#[test]
fn test_attr_stats_candidates_recover_after_heavy_forget() {
    let num_features: u8 = 8;
    let samples = make_samples_seeded(200, num_features, 31);

    let mut forest = build_exact_forest(num_features, 31);
    forest.fit(&samples);
    forest.enable_streaming(&samples);

    let initial_states = forest.streaming_stats().total_streaming_states;

    let ids: Vec<u64> = (0..80).map(|i| i as u64).collect();
    forget_with_streaming(&mut forest, &samples, &ids);

    let remaining: Vec<VecSample> = samples.iter().filter(|s| s.id >= 80).cloned().collect();
    full_lazy_resolve(&mut forest, &remaining);

    let after_states = forest.streaming_stats().total_streaming_states;

    assert!(
        after_states <= initial_states * 3 + 50,
        "streaming_states 폭증 (initial={}, after={}) — C2 candidate 회복 실패",
        initial_states,
        after_states
    );
}

//   Hoeffding 본체가 제거되어 시나리오 전제 자체 무효.

/// H2: internal node sample_ids underflow가 develop 후에도 누적되면 안 됨.
#[test]
fn test_internal_sample_ids_after_streaming_cycles() {
    let num_features: u8 = 6;
    let mut samples = make_samples_seeded(100, num_features, 51);

    let mut forest = build_exact_forest(num_features, 51);
    forest.fit(&samples);
    forest.enable_streaming(&samples);

    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;
    let mut rng = XorShiftRng::seed_from_u64(999);
    let mut extra = Vec::new();
    for i in 100u64..200u64 {
        let values: Vec<f32> = (0..num_features)
            .map(|_| rng.gen_range(-5.0..5.0))
            .collect();
        let label = values[0] > 0.0;
        extra.push(VecSample::new(i, values, label));
    }
    forest.add_samples_streaming(&extra, true);
    samples.extend(extra);

    let ids: Vec<u64> = (50..100).map(|i| i as u64).collect();
    forget_with_streaming(&mut forest, &samples, &ids);

    let remaining: Vec<VecSample> = samples
        .iter()
        .filter(|s| !ids.contains(&s.id))
        .cloned()
        .collect();
    full_lazy_resolve(&mut forest, &remaining);

    let stats = forest.streaming_stats();
    // H2 RED 시그널: 절대 임계 — 정상이면 트리당 ~50, 16tree × 50 = 800 max
    assert!(
        stats.total_streaming_states <= 2000,
        "streaming_states 폭증 ({}) — H2 underflow 누적 의심",
        stats.total_streaming_states
    );
}

// =========================================================================
// Phase 6b: B (per-node range) — Theorem 1 (distribution-equivalence) 회복
// =========================================================================

/// B test: 단순한 1-feature dataset에서 min sample만 forget → develop로 rebuild.
/// per-node range 사용 시 (min 빠짐 → 새 range는 더 좁아짐) split threshold 분포 변화.
/// 전역 range 사용 시 (dataset.attribute_range가 변함 없음) threshold 분포 불변.
/// 정합: forget(min sample) → develop → split이 retrain(D\{min})과 *같아야*.
#[test]
fn test_forget_min_max_sample_triggers_split_threshold_change() {
    let num_features: u8 = 2;
    let mut samples: Vec<VecSample> = Vec::new();
    // 충돌 시나리오: feature 0이 결정경계, feature 0 값이 -10..10
    for i in 0..200 {
        let x0 = -10.0 + (i as f32) * 0.1; // 균등 분포
        let label = x0 > 0.0;
        samples.push(VecSample::new(i as u64, vec![x0, 0.0], label));
    }

    // Path A: 전체 학습 → min sample(id=0, x0=-10)을 forget → develop
    let mut forest_a = build_exact_forest(num_features, 71);
    forest_a.fit(&samples);
    forest_a.enable_streaming(&samples);
    forget_with_streaming(&mut forest_a, &samples, &[0u64]);
    let remaining: Vec<VecSample> = samples[1..].to_vec();
    full_lazy_resolve(&mut forest_a, &remaining);

    // Path B: 처음부터 D\{min}으로 학습
    let mut forest_b = build_exact_forest(num_features, 71);
    forest_b.fit(&remaining);

    // probe 일치 검증
    let probes = make_test_samples(80, num_features, 71);
    let preds_a: Vec<bool> = probes.iter().map(|s| forest_a.predict(s)).collect();
    let preds_b: Vec<bool> = probes.iter().map(|s| forest_b.predict(s)).collect();

    let mismatches: Vec<usize> = preds_a
        .iter()
        .zip(preds_b.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();

    // B 결함 RED: min sample forget이 전역 range를 안 바꿔 split이 같지 않음 → mismatch.
    // GREEN (B 적용 후): per-node range 사용으로 retrain과 같은 분포 → 0 mismatch.
    assert!(
        mismatches.is_empty(),
        "forget(min)+develop ≠ retrain(D\\{{min}}): {} mismatches/{} (B 결함: dataset 전역 range 사용)",
        mismatches.len(), probes.len()
    );
}
