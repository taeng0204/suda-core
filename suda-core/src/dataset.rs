//! Dataset trait and implementations for training data metadata.

/// Attribute type for splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeType {
    /// Continuous numerical attribute
    Numerical,
    /// Discrete categorical attribute
    Categorical,
}

/// Trait representing dataset metadata.
pub trait Dataset: Sync {
    /// Total number of records in the dataset.
    fn num_records(&self) -> usize;

    /// Number of positive samples in the dataset.
    fn num_plus(&self) -> usize;

    /// Number of attributes (features).
    fn num_attributes(&self) -> u8;

    /// Get the range (min, max) for an attribute.
    fn attribute_range(&self, index: u8) -> (f32, f32);

    /// Get the type of an attribute.
    fn attribute_type(&self, index: u8) -> AttributeType;

    /// Number of negative samples.
    #[inline]
    fn num_minus(&self) -> usize {
        self.num_records() - self.num_plus()
    }
}

/// A simple dataset implementation backed by arrays.
#[derive(Debug, Clone)]
pub struct ArrayDataset {
    num_records: usize,
    num_plus: usize,
    /// (min, max) for each attribute
    ranges: Vec<(f32, f32)>,
    /// Type for each attribute
    types: Vec<AttributeType>,
}

impl ArrayDataset {
    /// Create a new dataset with the given metadata.
    pub fn new(
        num_records: usize,
        num_plus: usize,
        ranges: Vec<(f32, f32)>,
        types: Vec<AttributeType>,
    ) -> Self {
        assert_eq!(ranges.len(), types.len());
        ArrayDataset {
            num_records,
            num_plus,
            ranges,
            types,
        }
    }

    /// Create dataset metadata from samples.
    pub fn from_samples<S: crate::sample::Sample>(samples: &[S], num_attributes: u8) -> Self {
        if samples.is_empty() {
            return ArrayDataset {
                num_records: 0,
                num_plus: 0,
                ranges: vec![(0.0, 1.0); num_attributes as usize],
                types: vec![AttributeType::Numerical; num_attributes as usize],
            };
        }

        let num_records = samples.len();
        let num_plus = samples.iter().filter(|s| s.label()).count();

        // Compute ranges
        let mut ranges = Vec::with_capacity(num_attributes as usize);
        for attr_idx in 0..num_attributes {
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for sample in samples {
                let val = sample.attribute_value(attr_idx);
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            ranges.push((min_val, max_val));
        }

        // Default to numerical for all attributes
        let types = vec![AttributeType::Numerical; num_attributes as usize];

        ArrayDataset {
            num_records,
            num_plus,
            ranges,
            types,
        }
    }

    /// Set the type for an attribute.
    pub fn set_attribute_type(&mut self, index: u8, attr_type: AttributeType) {
        self.types[index as usize] = attr_type;
    }
}

impl Dataset for ArrayDataset {
    #[inline]
    fn num_records(&self) -> usize {
        self.num_records
    }

    #[inline]
    fn num_plus(&self) -> usize {
        self.num_plus
    }

    #[inline]
    fn num_attributes(&self) -> u8 {
        self.ranges.len() as u8
    }

    #[inline]
    fn attribute_range(&self, index: u8) -> (f32, f32) {
        self.ranges[index as usize]
    }

    #[inline]
    fn attribute_type(&self, index: u8) -> AttributeType {
        self.types[index as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::ArraySample;

    #[test]
    fn test_array_dataset() {
        let dataset = ArrayDataset::new(
            100,
            30,
            vec![(0.0, 1.0), (0.0, 10.0)],
            vec![AttributeType::Numerical, AttributeType::Categorical],
        );

        assert_eq!(dataset.num_records(), 100);
        assert_eq!(dataset.num_plus(), 30);
        assert_eq!(dataset.num_minus(), 70);
        assert_eq!(dataset.num_attributes(), 2);
        assert_eq!(dataset.attribute_range(0), (0.0, 1.0));
        assert_eq!(dataset.attribute_type(1), AttributeType::Categorical);
    }

    #[test]
    fn test_from_samples() {
        let samples = vec![
            ArraySample::new(0, [1.0, 5.0], true),
            ArraySample::new(1, [3.0, 2.0], false),
            ArraySample::new(2, [2.0, 8.0], true),
        ];

        let dataset = ArrayDataset::from_samples(&samples, 2);

        assert_eq!(dataset.num_records(), 3);
        assert_eq!(dataset.num_plus(), 2);
        assert_eq!(dataset.attribute_range(0), (1.0, 3.0));
        assert_eq!(dataset.attribute_range(1), (2.0, 8.0));
    }
}
