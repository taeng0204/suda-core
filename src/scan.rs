//! Split statistics scanning with optional SIMD optimization.

use crate::sample::Sample;
use crate::split::Split;
use crate::split_stats::SplitStats;

/// Scan samples and compute split statistics.
/// This is the scalar (non-SIMD) implementation.
pub fn scan<S: Sample>(samples: &[S], split: &Split) -> SplitStats {
    let mut num_plus_left: u32 = 0;
    let mut num_minus_left: u32 = 0;
    let mut num_plus_right: u32 = 0;
    let mut num_minus_right: u32 = 0;

    for sample in samples {
        let is_left = sample.is_left_of(split);
        let is_plus = sample.label();

        if is_left {
            if is_plus {
                num_plus_left += 1;
            } else {
                num_minus_left += 1;
            }
        } else if is_plus {
            num_plus_right += 1;
        } else {
            num_minus_right += 1;
        }
    }

    SplitStats::new(
        num_plus_left,
        num_minus_left,
        num_plus_right,
        num_minus_right,
    )
}

/// Optimized scan using branchless operations.
pub fn scan_branchless<S: Sample>(samples: &[S], split: &Split) -> SplitStats {
    let mut num_left: u32 = 0;
    let mut num_plus_left: u32 = 0;
    let mut num_plus_right: u32 = 0;

    for sample in samples {
        let is_left = sample.is_left_of(split);
        let is_plus = sample.label();

        num_left += is_left as u32;
        num_plus_left += (is_left & is_plus) as u32;
        num_plus_right += (!is_left & is_plus) as u32;
    }

    let num_minus_left = num_left - num_plus_left;
    let num_right = samples.len() as u32 - num_left;
    let num_minus_right = num_right - num_plus_right;

    SplitStats::new(
        num_plus_left,
        num_minus_left,
        num_plus_right,
        num_minus_right,
    )
}

/// Choose the best scan implementation based on sample count.
#[inline]
pub fn scan_auto<S: Sample>(samples: &[S], split: &Split) -> SplitStats {
    // For small sample counts, simple scan is faster
    if samples.len() < 32 {
        scan(samples, split)
    } else {
        scan_branchless(samples, split)
    }
}

/// Partition samples in-place and return the split point.
/// Samples [0..split_idx) go left, [split_idx..len) go right.
pub fn partition_in_place<S: Sample>(samples: &mut [S], split: &Split) -> usize {
    let mut write_idx = 0;

    for read_idx in 0..samples.len() {
        if samples[read_idx].is_left_of(split) {
            samples.swap(write_idx, read_idx);
            write_idx += 1;
        }
    }

    write_idx
}

/// Partition sample indices without modifying original data.
/// Returns (left_indices, right_indices) where each contains indices into the original slice.
///
/// This is used for reference-based tree building to avoid cloning samples.
#[inline]
pub fn partition_indices<S: Sample>(samples: &[&S], split: &Split) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::with_capacity(samples.len() / 2);
    let mut right = Vec::with_capacity(samples.len() / 2);

    for (idx, sample) in samples.iter().enumerate() {
        if sample.is_left_of(split) {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    (left, right)
}

/// Scan sample references and compute split statistics.
#[inline]
pub fn scan_refs<S: Sample>(samples: &[&S], split: &Split) -> SplitStats {
    let mut num_plus_left: u32 = 0;
    let mut num_minus_left: u32 = 0;
    let mut num_plus_right: u32 = 0;
    let mut num_minus_right: u32 = 0;

    for sample in samples {
        let is_left = sample.is_left_of(split);
        let is_plus = sample.label();

        if is_left {
            if is_plus {
                num_plus_left += 1;
            } else {
                num_minus_left += 1;
            }
        } else if is_plus {
            num_plus_right += 1;
        } else {
            num_minus_right += 1;
        }
    }

    SplitStats::new(
        num_plus_left,
        num_minus_left,
        num_plus_right,
        num_minus_right,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::ArraySample;

    fn make_samples() -> Vec<ArraySample<2>> {
        vec![
            ArraySample::new(0, [1.0, 5.0], true),  // left, positive
            ArraySample::new(1, [2.0, 3.0], false), // left, negative
            ArraySample::new(2, [8.0, 7.0], true),  // right, positive
            ArraySample::new(3, [9.0, 2.0], false), // right, negative
            ArraySample::new(4, [3.0, 6.0], true),  // left, positive
            ArraySample::new(5, [7.0, 1.0], false), // right, negative
        ]
    }

    #[test]
    fn test_scan() {
        let samples = make_samples();
        let split = Split::numerical(0, 5.0);

        let stats = scan(&samples, &split);

        assert_eq!(stats.num_plus_left, 2); // samples 0, 4
        assert_eq!(stats.num_minus_left, 1); // sample 1
        assert_eq!(stats.num_plus_right, 1); // sample 2
        assert_eq!(stats.num_minus_right, 2); // samples 3, 5
    }

    #[test]
    fn test_scan_branchless() {
        let samples = make_samples();
        let split = Split::numerical(0, 5.0);

        let stats = scan_branchless(&samples, &split);

        assert_eq!(stats.num_plus_left, 2);
        assert_eq!(stats.num_minus_left, 1);
        assert_eq!(stats.num_plus_right, 1);
        assert_eq!(stats.num_minus_right, 2);
    }

    #[test]
    fn test_partition_in_place() {
        let mut samples = make_samples();
        let split = Split::numerical(0, 5.0);

        let split_idx = partition_in_place(&mut samples, &split);

        assert_eq!(split_idx, 3);

        for s in &samples[..split_idx] {
            assert!(s.is_left_of(&split));
        }
        for s in &samples[split_idx..] {
            assert!(!s.is_left_of(&split));
        }
    }
}
