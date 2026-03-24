//! Sample trait and implementations for training data.

use crate::split::Split;

/// Trait representing a single training sample.
pub trait Sample: Clone + Send + Sync {
    /// Unique identifier for this sample.
    fn id(&self) -> u64;

    /// Get the value of an attribute.
    fn attribute_value(&self, index: u8) -> f32;

    /// Get the true label (true = positive class).
    fn label(&self) -> bool;

    /// Check if this sample goes left for a given split.
    #[inline]
    fn is_left_of(&self, split: &Split) -> bool {
        let value = self.attribute_value(split.attribute_index());
        split.goes_left(value)
    }
}

/// A simple sample implementation backed by a fixed-size array.
#[derive(Clone, Debug)]
pub struct ArraySample<const N: usize> {
    /// Unique sample ID
    pub id: u64,
    /// Attribute values
    pub values: [f32; N],
    /// True label
    pub label: bool,
}

impl<const N: usize> ArraySample<N> {
    /// Create a new sample.
    pub fn new(id: u64, values: [f32; N], label: bool) -> Self {
        ArraySample { id, values, label }
    }
}

impl<const N: usize> Sample for ArraySample<N> {
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    fn attribute_value(&self, index: u8) -> f32 {
        self.values[index as usize]
    }

    #[inline]
    fn label(&self) -> bool {
        self.label
    }
}

/// A dynamic sample backed by a Vec.
#[derive(Clone, Debug)]
pub struct VecSample {
    /// Unique sample ID
    pub id: u64,
    /// Attribute values
    pub values: Vec<f32>,
    /// True label
    pub label: bool,
}

impl VecSample {
    /// Create a new sample.
    pub fn new(id: u64, values: Vec<f32>, label: bool) -> Self {
        VecSample { id, values, label }
    }
}

impl Sample for VecSample {
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    fn attribute_value(&self, index: u8) -> f32 {
        self.values[index as usize]
    }

    #[inline]
    fn label(&self) -> bool {
        self.label
    }
}

/// A sample that references external data (zero-copy from numpy).
#[derive(Clone, Debug)]
pub struct RefSample<'a> {
    /// Unique sample ID
    pub id: u64,
    /// Reference to attribute values
    pub values: &'a [f32],
    /// True label
    pub label: bool,
}

impl<'a> RefSample<'a> {
    /// Create a new reference sample.
    pub fn new(id: u64, values: &'a [f32], label: bool) -> Self {
        RefSample { id, values, label }
    }
}

impl<'a> Sample for RefSample<'a> {
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    fn attribute_value(&self, index: u8) -> f32 {
        self.values[index as usize]
    }

    #[inline]
    fn label(&self) -> bool {
        self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_sample() {
        let sample = ArraySample::new(42, [1.0, 2.0, 3.0], true);
        assert_eq!(sample.id(), 42);
        assert_eq!(sample.attribute_value(0), 1.0);
        assert_eq!(sample.attribute_value(1), 2.0);
        assert_eq!(sample.attribute_value(2), 3.0);
        assert!(sample.label());
    }

    #[test]
    fn test_vec_sample() {
        let sample = VecSample::new(1, vec![0.5, 1.5, 2.5], false);
        assert_eq!(sample.id(), 1);
        assert_eq!(sample.attribute_value(2), 2.5);
        assert!(!sample.label());
    }

    #[test]
    fn test_is_left_of() {
        let sample = ArraySample::new(1, [5.0, 10.0], true);

        let split = Split::numerical(0, 6.0);
        assert!(sample.is_left_of(&split)); // 5.0 < 6.0

        let split = Split::numerical(0, 4.0);
        assert!(!sample.is_left_of(&split)); // 5.0 >= 4.0
    }
}
