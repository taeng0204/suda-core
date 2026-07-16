//! Split statistics for evaluating split quality.

use std::cmp::Ordering;

/// Statistics for evaluating a split.
#[derive(Debug, Clone, Copy)]
pub struct SplitStats {
    /// Number of positive samples going left
    pub num_plus_left: u32,
    /// Number of negative samples going left
    pub num_minus_left: u32,
    /// Number of positive samples going right
    pub num_plus_right: u32,
    /// Number of negative samples going right
    pub num_minus_right: u32,
    /// Gini impurity of left child
    pub impurity_left: f64,
    /// Gini impurity of right child
    pub impurity_right: f64,
    /// Split score (Gini gain * 10^12 as integer for stable comparison)
    pub score: Option<i64>,
}

impl SplitStats {
    /// Create new split statistics.
    pub fn new(
        num_plus_left: u32,
        num_minus_left: u32,
        num_plus_right: u32,
        num_minus_right: u32,
    ) -> Self {
        SplitStats {
            num_plus_left,
            num_minus_left,
            num_plus_right,
            num_minus_right,
            impurity_left: 0.0,
            impurity_right: 0.0,
            score: None,
        }
    }

    /// Create empty stats.
    pub fn empty() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Total samples on the left.
    #[inline]
    pub fn num_left(&self) -> u32 {
        self.num_plus_left + self.num_minus_left
    }

    /// Total samples on the right.
    #[inline]
    pub fn num_right(&self) -> u32 {
        self.num_plus_right + self.num_minus_right
    }

    /// Total samples.
    #[inline]
    pub fn num_total(&self) -> u32 {
        self.num_left() + self.num_right()
    }

    /// Total positive samples.
    #[inline]
    pub fn num_plus(&self) -> u32 {
        self.num_plus_left + self.num_plus_right
    }

    /// Check if the split has a positive score.
    #[inline]
    pub fn has_positive_score(&self) -> bool {
        matches!(self.score, Some(s) if s > 0)
    }

    /// Check if the split is valid (both sides non-empty).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.num_left() > 0 && self.num_right() > 0
    }

    /// Update the score based on parent impurity.
    pub fn update_score(&mut self, impurity_before: f64) {
        let (score, impurity_left, impurity_right) = compute_gini_gain(
            impurity_before,
            self.num_plus_left,
            self.num_minus_left,
            self.num_plus_right,
            self.num_minus_right,
        );

        self.score = score;
        self.impurity_left = impurity_left;
        self.impurity_right = impurity_right;
    }

    /// Update score computing impurity_before from current stats.
    pub fn update_score_with_impurity(&mut self) {
        let num_plus = self.num_plus();
        let num_total = self.num_total();

        if num_total == 0 {
            self.score = None;
            return;
        }

        let impurity_before = gini_impurity(num_plus, num_total);
        self.update_score(impurity_before);
    }

    /// Format as string for debugging.
    pub fn fmt(&self) -> String {
        format!(
            "(+L:{}, -L:{}, +R:{}, -R:{})",
            self.num_plus_left, self.num_minus_left, self.num_plus_right, self.num_minus_right
        )
    }
}

impl PartialEq for SplitStats {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for SplitStats {}

impl PartialOrd for SplitStats {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SplitStats {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.score, other.score) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        }
    }
}

/// Compute Gini impurity for a node.
#[inline]
pub fn gini_impurity(num_plus: u32, num_total: u32) -> f64 {
    if num_total == 0 {
        return 0.0;
    }
    let p_plus = num_plus as f64 / num_total as f64;
    2.0 * p_plus * (1.0 - p_plus)
}

/// Compute Gini gain for a split.
fn compute_gini_gain(
    impurity_before: f64,
    num_plus_left: u32,
    num_minus_left: u32,
    num_plus_right: u32,
    num_minus_right: u32,
) -> (Option<i64>, f64, f64) {
    let num_left = num_plus_left + num_minus_left;
    let num_right = num_plus_right + num_minus_right;

    // Invalid split: one side is empty
    if num_left == 0 || num_right == 0 {
        return (None, 0.0, 0.0);
    }

    let num_total = num_left + num_right;

    let gini_left = gini_impurity(num_plus_left, num_left);
    let gini_right = gini_impurity(num_plus_right, num_right);

    let weighted_impurity = (num_left as f64 / num_total as f64) * gini_left
        + (num_right as f64 / num_total as f64) * gini_right;

    let gain = impurity_before - weighted_impurity;

    // Convert to integer for stable comparison (multiply by 10^12)
    let score = (gain * 1_000_000_000_000.0) as i64;

    (Some(score), gini_left, gini_right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gini_impurity() {
        // Pure node
        assert_eq!(gini_impurity(0, 100), 0.0);
        assert_eq!(gini_impurity(100, 100), 0.0);

        // Balanced node (maximum impurity)
        let impurity = gini_impurity(50, 100);
        assert!((impurity - 0.5).abs() < 1e-10);

        // 25% positive
        let impurity = gini_impurity(25, 100);
        assert!((impurity - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_split_stats() {
        let mut stats = SplitStats::new(10, 40, 30, 20);

        // impurity_before for 40 positive out of 100
        let impurity_before = gini_impurity(40, 100);
        stats.update_score(impurity_before);

        assert!(stats.has_positive_score());
        assert!(stats.is_valid());
        assert_eq!(stats.num_left(), 50);
        assert_eq!(stats.num_right(), 50);
    }

    #[test]
    fn test_invalid_split() {
        let mut stats = SplitStats::new(0, 0, 50, 50);
        stats.update_score_with_impurity();

        assert!(!stats.has_positive_score());
        assert!(!stats.is_valid());
    }
}
