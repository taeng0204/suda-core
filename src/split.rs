//! Split definitions for decision tree nodes.

use std::fmt;

/// Represents a split condition in a decision tree node.
#[derive(Clone, Debug, PartialEq)]
pub enum Split {
    /// Numerical split: attribute_value < threshold goes left
    Numerical { attribute_index: u8, threshold: f32 },
    /// Categorical split: attribute_value in subset goes left
    Categorical {
        attribute_index: u8,
        /// Bitmask representing the subset (up to 64 categories)
        subset: u64,
    },
}

impl Split {
    /// Create a new numerical split.
    #[inline]
    pub fn numerical(attribute_index: u8, threshold: f32) -> Self {
        Split::Numerical {
            attribute_index,
            threshold,
        }
    }

    /// Create a new categorical split.
    #[inline]
    pub fn categorical(attribute_index: u8, subset: u64) -> Self {
        Split::Categorical {
            attribute_index,
            subset,
        }
    }

    /// Get the attribute index for this split.
    #[inline]
    pub fn attribute_index(&self) -> u8 {
        match self {
            Split::Numerical {
                attribute_index, ..
            } => *attribute_index,
            Split::Categorical {
                attribute_index, ..
            } => *attribute_index,
        }
    }

    /// Check if a value goes left for this split.
    /// Bug-2 (DynFrs 정합): `<=` 사용. DynFrs.h:286, 494, 563 모두 `<=`로 일관.
    /// 이전 `<`는 streaming.rs:210,247의 `<=` 통계와 경계값 sample에서 미세 불일치 야기.
    #[inline]
    pub fn goes_left(&self, value: f32) -> bool {
        match self {
            Split::Numerical { threshold, .. } => value <= *threshold,
            Split::Categorical { subset, .. } => {
                let category = value as u64;
                (*subset & (1u64 << category)) != 0
            }
        }
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Split::Numerical {
                attribute_index,
                threshold,
            } => {
                write!(f, "X[{}] < {:.4}", attribute_index, threshold)
            }
            Split::Categorical {
                attribute_index,
                subset,
            } => {
                write!(f, "X[{}] in {:b}", attribute_index, subset)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_split() {
        // Bug-2 (DynFrs 정합): goes_left은 value <= threshold이면 true.
        let split = Split::numerical(0, 5.0);
        assert!(split.goes_left(4.0));
        assert!(split.goes_left(5.0)); // 경계값: <=이므로 left
        assert!(!split.goes_left(6.0));
    }

    /// Bug-2 명시 검증: 경계값 sample이 streaming 통계(`<=`)와 routing(`<=`)에서 일관.
    /// 이전(`<`)에는 경계값 sample이 통계는 left에 집계되지만 routing은 right로 가서
    /// best_split_changed 평가와 실제 partition이 불일치. 미세 silent 버그.
    /// 이 test는 *DynFrs 정합 후의 동작*을 명시적으로 박아 미래 회귀 방지.
    #[test]
    fn test_boundary_value_routes_left_matching_stats() {
        let split = Split::numerical(0, 5.0);
        // 정확히 threshold 값 → left로 routing (streaming.rs:210,247 통계와 일관)
        assert!(
            split.goes_left(5.0),
            "value == threshold은 left (DynFrs.h:286,494,563 `<=`와 정합)"
        );
        // threshold 미만 → left
        assert!(split.goes_left(4.99));
        // threshold 초과 → right
        assert!(!split.goes_left(5.01));
    }

    #[test]
    fn test_categorical_split() {
        // Subset contains categories 0, 2, 3 (binary: 1101 = 13)
        let split = Split::categorical(1, 0b1101);
        assert!(split.goes_left(0.0)); // category 0 in subset
        assert!(!split.goes_left(1.0)); // category 1 not in subset
        assert!(split.goes_left(2.0)); // category 2 in subset
        assert!(split.goes_left(3.0)); // category 3 in subset
        assert!(!split.goes_left(4.0)); // category 4 not in subset
    }
}
