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
    #[inline]
    pub fn goes_left(&self, value: f32) -> bool {
        match self {
            Split::Numerical { threshold, .. } => value < *threshold,
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
        let split = Split::numerical(0, 5.0);
        assert!(split.goes_left(4.0));
        assert!(!split.goes_left(5.0));
        assert!(!split.goes_left(6.0));
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
