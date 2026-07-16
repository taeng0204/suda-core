//! Streaming Metrics for Binary Classification
//!
//! Provides efficient incremental metrics computation for streaming classification.
//! Designed for NIDS (Network Intrusion Detection) with class imbalance.
//!
//! Key Features:
//! - Incremental updates (no need to store all predictions)
//! - Sliding window support for recent metrics
//! - All essential metrics for imbalanced classification

use std::collections::VecDeque;

/// Binary confusion matrix with incremental updates.
///
/// Layout: [TN, FP, FN, TP]
/// - TN (True Negative): Actual=0, Predicted=0 (benign correctly classified)
/// - FP (False Positive): Actual=0, Predicted=1 (benign misclassified as attack)
/// - FN (False Negative): Actual=1, Predicted=0 (attack misclassified as benign)
/// - TP (True Positive): Actual=1, Predicted=1 (attack correctly classified)
#[derive(Debug, Clone, Copy, Default)]
pub struct ConfusionMatrix {
    /// True Negatives (benign correctly classified)
    pub tn: u64,
    /// False Positives (benign misclassified as attack)
    pub fp: u64,
    /// False Negatives (attack missed, classified as benign)
    pub fn_: u64,
    /// True Positives (attack correctly detected)
    pub tp: u64,
}

impl ConfusionMatrix {
    /// Create a new empty confusion matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with a single prediction.
    ///
    /// # Arguments
    /// * `actual` - True label (true = attack/positive, false = benign/negative)
    /// * `predicted` - Predicted label
    #[inline]
    pub fn update(&mut self, actual: bool, predicted: bool) {
        match (actual, predicted) {
            (false, false) => self.tn += 1, // True Negative
            (false, true) => self.fp += 1,  // False Positive
            (true, false) => self.fn_ += 1, // False Negative
            (true, true) => self.tp += 1,   // True Positive
        }
    }

    /// Remove a prediction (for sliding window).
    #[inline]
    pub fn remove(&mut self, actual: bool, predicted: bool) {
        match (actual, predicted) {
            (false, false) => self.tn = self.tn.saturating_sub(1),
            (false, true) => self.fp = self.fp.saturating_sub(1),
            (true, false) => self.fn_ = self.fn_.saturating_sub(1),
            (true, true) => self.tp = self.tp.saturating_sub(1),
        }
    }

    /// Total number of samples.
    #[inline]
    pub fn total(&self) -> u64 {
        self.tn + self.fp + self.fn_ + self.tp
    }

    /// Total positive samples (attacks).
    #[inline]
    pub fn actual_positive(&self) -> u64 {
        self.tp + self.fn_
    }

    /// Total negative samples (benign).
    #[inline]
    pub fn actual_negative(&self) -> u64 {
        self.tn + self.fp
    }

    /// True Positive Rate (TPR) = Recall = Sensitivity
    /// TP / (TP + FN)
    #[inline]
    pub fn tpr(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 {
            1.0 // No positive samples, perfect by default
        } else {
            self.tp as f64 / denom as f64
        }
    }

    /// True Negative Rate (TNR) = Specificity
    /// TN / (TN + FP)
    #[inline]
    pub fn tnr(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom == 0 {
            1.0 // No negative samples, perfect by default
        } else {
            self.tn as f64 / denom as f64
        }
    }

    /// False Positive Rate (FPR) = 1 - TNR
    /// FP / (FP + TN)
    #[inline]
    pub fn fpr(&self) -> f64 {
        1.0 - self.tnr()
    }

    /// False Negative Rate (FNR) = 1 - TPR
    /// FN / (FN + TP)
    #[inline]
    pub fn fnr(&self) -> f64 {
        1.0 - self.tpr()
    }

    /// Precision = TP / (TP + FP)
    #[inline]
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            1.0 // No positive predictions
        } else {
            self.tp as f64 / denom as f64
        }
    }

    /// Accuracy = (TP + TN) / Total
    #[inline]
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            (self.tp + self.tn) as f64 / total as f64
        }
    }

    /// Balanced Accuracy = (TPR + TNR) / 2
    #[inline]
    pub fn balanced_accuracy(&self) -> f64 {
        (self.tpr() + self.tnr()) / 2.0
    }

    /// G-mean = sqrt(TPR * TNR)
    ///
    /// The most important metric for imbalanced NIDS data.
    /// - 99% benign, 1% attack: Trivial classifier (always benign) gets 99% accuracy but 0% G-mean
    /// - G-mean penalizes ignoring minority class
    #[inline]
    pub fn gmean(&self) -> f64 {
        let tpr = self.tpr();
        let tnr = self.tnr();

        // Handle edge case where either rate is 0
        if tpr == 0.0 || tnr == 0.0 {
            0.0
        } else {
            (tpr * tnr).sqrt()
        }
    }

    /// F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    #[inline]
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.tpr();

        let denom = precision + recall;
        if denom == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / denom
        }
    }

    /// Cohen's Kappa = (p_o - p_e) / (1 - p_e)
    /// where p_o = observed agreement, p_e = expected agreement by chance
    ///
    /// Kappa interpretation:
    /// - < 0: Less than chance agreement
    /// - 0-0.20: Slight agreement
    /// - 0.21-0.40: Fair agreement
    /// - 0.41-0.60: Moderate agreement
    /// - 0.61-0.80: Substantial agreement
    /// - 0.81-1.00: Almost perfect agreement
    #[inline]
    pub fn kappa(&self) -> f64 {
        let total = self.total() as f64;
        if total == 0.0 {
            return 0.0;
        }

        // Observed agreement
        let p_o = self.accuracy();

        // Expected agreement by chance
        // P(both predict positive) + P(both predict negative)
        let actual_pos = self.actual_positive() as f64 / total;
        let actual_neg = self.actual_negative() as f64 / total;
        let pred_pos = (self.tp + self.fp) as f64 / total;
        let pred_neg = (self.tn + self.fn_) as f64 / total;

        let p_e = actual_pos * pred_pos + actual_neg * pred_neg;

        if (1.0 - p_e).abs() < 1e-10 {
            // Perfect expected agreement
            if (p_o - 1.0).abs() < 1e-10 {
                1.0 // Perfect agreement
            } else {
                0.0 // Can't compute
            }
        } else {
            (p_o - p_e) / (1.0 - p_e)
        }
    }

    /// Merge with another confusion matrix.
    pub fn merge(&mut self, other: &ConfusionMatrix) {
        self.tn += other.tn;
        self.fp += other.fp;
        self.fn_ += other.fn_;
        self.tp += other.tp;
    }

    /// Reset to empty state.
    pub fn reset(&mut self) {
        self.tn = 0;
        self.fp = 0;
        self.fn_ = 0;
        self.tp = 0;
    }
}

/// Computed streaming metrics snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamingMetrics {
    /// Standard accuracy (TP + TN) / Total
    pub accuracy: f64,
    /// Balanced accuracy (TPR + TNR) / 2
    pub balanced_accuracy: f64,
    /// Geometric mean sqrt(TPR * TNR) - Most important for imbalanced data
    pub gmean: f64,
    /// Cohen's Kappa (chance-corrected agreement)
    pub kappa: f64,
    /// True Positive Rate = Recall = Attack detection rate
    pub attack_recall: f64,
    /// True Negative Rate = Specificity = Benign correct rate
    pub benign_recall: f64,
    /// Precision = TP / (TP + FP)
    pub precision: f64,
    /// F1-Score
    pub f1_score: f64,
    /// Total samples
    pub total_samples: u64,
}

impl StreamingMetrics {
    /// Compute metrics from a confusion matrix.
    pub fn from_confusion(cm: &ConfusionMatrix) -> Self {
        Self {
            accuracy: cm.accuracy(),
            balanced_accuracy: cm.balanced_accuracy(),
            gmean: cm.gmean(),
            kappa: cm.kappa(),
            attack_recall: cm.tpr(),
            benign_recall: cm.tnr(),
            precision: cm.precision(),
            f1_score: cm.f1_score(),
            total_samples: cm.total(),
        }
    }

    /// Create from direct predictions.
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = ConfusionMatrix::new();
        for (a, p) in actual.iter().zip(predicted.iter()) {
            cm.update(*a, *p);
        }
        Self::from_confusion(&cm)
    }
}

/// Tracks streaming metrics with optional sliding window.
///
/// Supports two modes:
/// 1. Cumulative: Track all-time metrics
/// 2. Sliding Window: Track recent N samples only
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    /// Cumulative confusion matrix (all-time)
    cumulative: ConfusionMatrix,
    /// Window confusion matrix (recent window_size samples)
    windowed: ConfusionMatrix,
    /// History for sliding window: (actual, predicted)
    history: VecDeque<(bool, bool)>,
    /// Window size (0 = no windowing)
    window_size: usize,
}

impl MetricsTracker {
    /// Create a new tracker without sliding window (cumulative only).
    pub fn new() -> Self {
        Self {
            cumulative: ConfusionMatrix::new(),
            windowed: ConfusionMatrix::new(),
            history: VecDeque::new(),
            window_size: 0,
        }
    }

    /// Create a new tracker with sliding window.
    ///
    /// # Arguments
    /// * `window_size` - Number of recent samples to track
    pub fn with_window(window_size: usize) -> Self {
        Self {
            cumulative: ConfusionMatrix::new(),
            windowed: ConfusionMatrix::new(),
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Update with a single prediction.
    pub fn update(&mut self, actual: bool, predicted: bool) {
        // Always update cumulative
        self.cumulative.update(actual, predicted);

        // Update windowed if enabled
        if self.window_size > 0 {
            // Evict oldest if window is full
            if self.history.len() >= self.window_size {
                if let Some((old_actual, old_predicted)) = self.history.pop_front() {
                    self.windowed.remove(old_actual, old_predicted);
                }
            }

            self.history.push_back((actual, predicted));
            self.windowed.update(actual, predicted);
        }
    }

    /// Update with a batch of predictions.
    ///
    /// Returns metrics computed after processing the batch.
    pub fn update_batch(&mut self, actual: &[bool], predicted: &[bool]) -> StreamingMetrics {
        for (a, p) in actual.iter().zip(predicted.iter()) {
            self.update(*a, *p);
        }
        self.current_metrics()
    }

    /// Get current metrics (from window if enabled, otherwise cumulative).
    pub fn current_metrics(&self) -> StreamingMetrics {
        if self.window_size > 0 && !self.history.is_empty() {
            StreamingMetrics::from_confusion(&self.windowed)
        } else {
            StreamingMetrics::from_confusion(&self.cumulative)
        }
    }

    /// Get cumulative (all-time) metrics.
    pub fn cumulative_metrics(&self) -> StreamingMetrics {
        StreamingMetrics::from_confusion(&self.cumulative)
    }

    /// Get windowed (recent) metrics.
    pub fn windowed_metrics(&self) -> Option<StreamingMetrics> {
        if self.window_size > 0 && !self.history.is_empty() {
            Some(StreamingMetrics::from_confusion(&self.windowed))
        } else {
            None
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.cumulative.reset();
        self.windowed.reset();
        self.history.clear();
    }

    /// Get total samples processed.
    pub fn total_samples(&self) -> u64 {
        self.cumulative.total()
    }

    /// Get window samples count.
    pub fn window_samples(&self) -> usize {
        self.history.len()
    }

    /// Get the raw cumulative confusion matrix.
    pub fn confusion_matrix(&self) -> &ConfusionMatrix {
        &self.cumulative
    }

    /// Get the raw windowed confusion matrix.
    pub fn windowed_confusion_matrix(&self) -> Option<&ConfusionMatrix> {
        if self.window_size > 0 {
            Some(&self.windowed)
        } else {
            None
        }
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_basic() {
        let mut cm = ConfusionMatrix::new();
        assert_eq!(cm.total(), 0);

        // Add some predictions
        cm.update(false, false); // TN
        cm.update(false, true); // FP
        cm.update(true, false); // FN
        cm.update(true, true); // TP

        assert_eq!(cm.tn, 1);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.fn_, 1);
        assert_eq!(cm.tp, 1);
        assert_eq!(cm.total(), 4);
    }

    #[test]
    fn test_confusion_matrix_rates() {
        let mut cm = ConfusionMatrix::new();

        // 80 TN, 20 FP (100 benign)
        // 30 FN, 70 TP (100 attacks)
        cm.tn = 80;
        cm.fp = 20;
        cm.fn_ = 30;
        cm.tp = 70;

        assert!((cm.tpr() - 0.70).abs() < 1e-10); // 70/100
        assert!((cm.tnr() - 0.80).abs() < 1e-10); // 80/100
        assert!((cm.accuracy() - 0.75).abs() < 1e-10); // 150/200
        assert!((cm.balanced_accuracy() - 0.75).abs() < 1e-10); // (0.7 + 0.8) / 2
    }

    #[test]
    fn test_gmean() {
        let mut cm = ConfusionMatrix::new();

        // Perfect classifier
        cm.tn = 100;
        cm.tp = 100;
        cm.fp = 0;
        cm.fn_ = 0;
        assert!((cm.gmean() - 1.0).abs() < 1e-10);

        // Always predicts benign (ignores attacks)
        cm.tn = 100;
        cm.fp = 0;
        cm.fn_ = 100; // All attacks missed
        cm.tp = 0;
        assert_eq!(cm.gmean(), 0.0); // TPR = 0

        // Balanced classifier
        cm.tn = 80;
        cm.fp = 20;
        cm.fn_ = 20;
        cm.tp = 80;
        let expected = (0.8f64 * 0.8f64).sqrt(); // sqrt(0.8 * 0.8) = 0.8
        assert!((cm.gmean() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kappa() {
        let mut cm = ConfusionMatrix::new();

        // Perfect classifier
        cm.tn = 100;
        cm.tp = 100;
        cm.fp = 0;
        cm.fn_ = 0;
        assert!((cm.kappa() - 1.0).abs() < 1e-10);

        // Random classifier (50/50)
        cm.tn = 50;
        cm.fp = 50;
        cm.fn_ = 50;
        cm.tp = 50;
        assert!(cm.kappa().abs() < 0.1); // Near zero kappa

        // Good classifier
        cm.tn = 85;
        cm.fp = 15;
        cm.fn_ = 10;
        cm.tp = 90;
        assert!(cm.kappa() > 0.7); // Substantial agreement
    }

    #[test]
    fn test_imbalanced_scenario() {
        let mut cm = ConfusionMatrix::new();

        // 99% benign, 1% attack - trivial "always benign" classifier
        cm.tn = 9900; // All benign correctly classified
        cm.fp = 0;
        cm.fn_ = 100; // All attacks missed
        cm.tp = 0;

        assert!((cm.accuracy() - 0.99).abs() < 1e-10); // 99% accuracy!
        assert_eq!(cm.tpr(), 0.0); // 0% attack recall
        assert_eq!(cm.gmean(), 0.0); // 0% G-mean (correctly captures failure)

        // Good classifier on same data
        cm.tn = 9850; // 50 false positives
        cm.fp = 50;
        cm.fn_ = 20; // 20 missed attacks
        cm.tp = 80; // 80% attack recall

        assert!(cm.accuracy() > 0.99); // Still high accuracy
        assert!((cm.tpr() - 0.80).abs() < 1e-10); // 80% attack recall
        assert!(cm.gmean() > 0.88); // High G-mean
    }

    #[test]
    fn test_metrics_tracker_cumulative() {
        let mut tracker = MetricsTracker::new();

        // Add predictions
        for _ in 0..100 {
            tracker.update(false, false); // TN
        }
        for _ in 0..50 {
            tracker.update(true, true); // TP
        }

        let metrics = tracker.current_metrics();
        assert_eq!(metrics.total_samples, 150);
        assert!((metrics.accuracy - 1.0).abs() < 1e-10); // All correct
        assert!((metrics.gmean - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_tracker_windowed() {
        let mut tracker = MetricsTracker::with_window(100);

        // Fill with perfect predictions
        for _ in 0..100 {
            tracker.update(false, false); // TN
        }

        let metrics = tracker.current_metrics();
        assert_eq!(metrics.total_samples, 100);

        // Now add all wrong predictions
        for _ in 0..100 {
            tracker.update(false, true); // FP (wrong)
        }

        // Window should only contain the wrong predictions now
        let windowed = tracker.windowed_metrics().unwrap();
        assert_eq!(windowed.total_samples, 100);
        assert!((windowed.accuracy - 0.0).abs() < 1e-10); // All wrong

        // Cumulative still includes the good ones
        let cumulative = tracker.cumulative_metrics();
        assert_eq!(cumulative.total_samples, 200);
        assert!((cumulative.accuracy - 0.5).abs() < 1e-10); // 50% overall
    }

    #[test]
    fn test_streaming_metrics_from_predictions() {
        let actual = vec![false, false, true, true, false];
        let predicted = vec![false, true, true, false, false];

        let metrics = StreamingMetrics::from_predictions(&actual, &predicted);

        assert_eq!(metrics.total_samples, 5);
        // TN=2, FP=1, FN=1, TP=1
        assert!((metrics.accuracy - 0.6).abs() < 1e-10); // 3/5
    }

    #[test]
    fn test_confusion_matrix_remove() {
        let mut cm = ConfusionMatrix::new();

        cm.update(true, true);
        cm.update(false, false);
        assert_eq!(cm.total(), 2);

        cm.remove(true, true);
        assert_eq!(cm.tp, 0);
        assert_eq!(cm.total(), 1);
    }
}
