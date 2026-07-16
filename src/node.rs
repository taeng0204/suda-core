//! Node structures for decision trees with LZY Tag support.

use crate::split::Split;
use hashbrown::HashSet;

/// LZY Tag state for lazy rebuild (DynFrs exact unlearning mechanism).
///
/// Lifecycle: Clean → Dirty → Rebuild → Clean
///
/// - `Clean`: Node is up-to-date, no pending changes.
/// - `Dirty`: A sample was deleted from this subtree. The split may no longer be
///   optimal but the tree is still structurally valid. Accumulates until `develop()`.
/// - `Rebuild`: Too many deletions accumulated (or split became degenerate).
///   The entire subtree must be rebuilt from remaining samples during `develop()`.
///
/// The `develop()` traversal processes tags bottom-up: Dirty nodes check if their
/// split is still valid (otherwise promote to Rebuild), and Rebuild nodes are
/// reconstructed from scratch using only the remaining samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LazyTag {
    /// No rebuild needed — node is up-to-date
    #[default]
    Clean,
    /// Sample(s) deleted — split may be suboptimal, pending develop()
    Dirty,
    /// Subtree needs complete rebuild from remaining samples
    Rebuild,
}

/// A node in the decision tree.
#[derive(Debug, Clone)]
pub enum Node {
    /// Internal node with a split
    Internal {
        /// The split condition
        split: Split,
        /// LZY Tag for lazy rebuild
        lazy_tag: LazyTag,
        /// Number of samples passing through this node
        num_samples: u32,
        /// Number of positive samples passing through
        num_plus: u32,
        /// Weighted Gini impurity at split creation time (현재 미사용, dead but kept for layout).
        split_gini: f64,
    },
    /// Leaf node with prediction
    Leaf {
        /// Total number of samples in this leaf
        num_samples: u32,
        /// Number of positive samples in this leaf
        num_plus: u32,
        /// Sample IDs in this leaf (for unlearning)
        sample_ids: HashSet<u64>,
    },
}

impl Node {
    /// Create a new internal node.
    pub fn internal(split: Split, num_samples: u32, num_plus: u32) -> Self {
        Node::Internal {
            split,
            lazy_tag: LazyTag::Clean,
            num_samples,
            num_plus,
            split_gini: 0.0,
        }
    }

    /// Create a new internal node (split_gini는 현재 미사용이지만 호출처 호환성 위해 인자 유지).
    pub fn internal_full(split: Split, num_samples: u32, num_plus: u32, split_gini: f64) -> Self {
        Node::Internal {
            split,
            lazy_tag: LazyTag::Clean,
            num_samples,
            num_plus,
            split_gini,
        }
    }

    /// Create a new leaf node.
    pub fn leaf(num_samples: u32, num_plus: u32) -> Self {
        Node::Leaf {
            num_samples,
            num_plus,
            sample_ids: HashSet::new(),
        }
    }

    /// Create a new leaf node with sample IDs.
    pub fn leaf_with_samples(num_samples: u32, num_plus: u32, sample_ids: HashSet<u64>) -> Self {
        Node::Leaf {
            num_samples,
            num_plus,
            sample_ids,
        }
    }

    /// Check if this is an internal node.
    #[inline]
    pub fn is_internal(&self) -> bool {
        matches!(self, Node::Internal { .. })
    }

    /// Check if this is a leaf node.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }

    /// Get the number of samples in this node.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        match self {
            Node::Internal { num_samples, .. } => *num_samples,
            Node::Leaf { num_samples, .. } => *num_samples,
        }
    }

    /// Get the number of positive samples.
    #[inline]
    pub fn num_plus(&self) -> u32 {
        match self {
            Node::Internal { num_plus, .. } => *num_plus,
            Node::Leaf { num_plus, .. } => *num_plus,
        }
    }

    /// Get the prediction for this node (majority vote).
    #[inline]
    pub fn predict(&self) -> bool {
        self.num_plus() * 2 > self.num_samples()
    }

    /// Get the split if this is an internal node.
    pub fn split(&self) -> Option<&Split> {
        match self {
            Node::Internal { split, .. } => Some(split),
            Node::Leaf { .. } => None,
        }
    }

    /// Get the lazy tag if this is an internal node.
    pub fn lazy_tag(&self) -> Option<LazyTag> {
        match self {
            Node::Internal { lazy_tag, .. } => Some(*lazy_tag),
            Node::Leaf { .. } => None,
        }
    }

    /// Get the split_gini if this is an internal node.
    pub fn split_gini(&self) -> Option<f64> {
        match self {
            Node::Internal { split_gini, .. } => Some(*split_gini),
            Node::Leaf { .. } => None,
        }
    }

    /// Set the lazy tag for an internal node.
    pub fn set_lazy_tag(&mut self, tag: LazyTag) {
        if let Node::Internal { lazy_tag, .. } = self {
            *lazy_tag = tag;
        }
    }

    /// Check if this node needs rebuild.
    pub fn needs_rebuild(&self) -> bool {
        matches!(
            self,
            Node::Internal {
                lazy_tag: LazyTag::Dirty | LazyTag::Rebuild,
                ..
            }
        )
    }

    /// Update counts after sample removal (for leaf nodes).
    /// Returns true if the sample was found and removed.
    pub fn remove_sample(&mut self, sample_id: u64, was_positive: bool) -> bool {
        match self {
            Node::Leaf {
                num_samples,
                num_plus,
                sample_ids,
            } => {
                if sample_ids.remove(&sample_id) {
                    *num_samples = num_samples.saturating_sub(1);
                    if was_positive {
                        *num_plus = num_plus.saturating_sub(1);
                    }
                    true
                } else {
                    false
                }
            }
            Node::Internal {
                num_samples,
                num_plus,
                ..
            } => {
                // For internal nodes, just update counts
                *num_samples = num_samples.saturating_sub(1);
                if was_positive {
                    *num_plus = num_plus.saturating_sub(1);
                }
                true
            }
        }
    }

    /// Add a sample to a leaf node.
    pub fn add_sample(&mut self, sample_id: u64, is_positive: bool) {
        match self {
            Node::Leaf {
                num_samples,
                num_plus,
                sample_ids,
            } => {
                sample_ids.insert(sample_id);
                *num_samples += 1;
                if is_positive {
                    *num_plus += 1;
                }
            }
            Node::Internal {
                num_samples,
                num_plus,
                ..
            } => {
                *num_samples += 1;
                if is_positive {
                    *num_plus += 1;
                }
            }
        }
    }
}

/// Node ID utilities.
/// We use a binary heap-like indexing: root = 1, left child = 2*id, right child = 2*id + 1
pub mod node_id {
    /// Root node ID.
    pub const ROOT: u64 = 1;

    /// Get the left child ID.
    #[inline]
    pub fn left_child(id: u64) -> u64 {
        id * 2
    }

    /// Get the right child ID.
    #[inline]
    pub fn right_child(id: u64) -> u64 {
        id * 2 + 1
    }

    /// Get the parent ID.
    #[inline]
    pub fn parent(id: u64) -> u64 {
        id / 2
    }

    /// Check if this is the left child.
    #[inline]
    pub fn is_left_child(id: u64) -> bool {
        id.is_multiple_of(2)
    }

    /// Check if this is the right child.
    #[inline]
    pub fn is_right_child(id: u64) -> bool {
        id % 2 == 1 && id != ROOT
    }

    /// Get the depth of a node (root = 0).
    #[inline]
    pub fn depth(id: u64) -> u32 {
        63 - id.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_node() {
        let split = Split::numerical(0, 5.0);
        let node = Node::internal(split, 100, 40);

        assert!(node.is_internal());
        assert!(!node.is_leaf());
        assert_eq!(node.num_samples(), 100);
        assert_eq!(node.num_plus(), 40);
        assert!(!node.predict()); // 40 < 50
    }

    #[test]
    fn test_leaf_node() {
        let mut sample_ids = HashSet::new();
        sample_ids.insert(1);
        sample_ids.insert(2);
        sample_ids.insert(3);

        let node = Node::leaf_with_samples(3, 2, sample_ids);

        assert!(node.is_leaf());
        assert!(node.predict()); // 2 > 1.5
    }

    #[test]
    fn test_remove_sample() {
        let mut sample_ids = HashSet::new();
        sample_ids.insert(1);
        sample_ids.insert(2);

        let mut node = Node::leaf_with_samples(2, 1, sample_ids);

        // Remove positive sample
        assert!(node.remove_sample(1, true));
        assert_eq!(node.num_samples(), 1);
        assert_eq!(node.num_plus(), 0);

        // Try to remove non-existent sample
        assert!(!node.remove_sample(999, false));
    }

    #[test]
    fn test_node_id_utils() {
        use node_id::*;

        assert_eq!(left_child(1), 2);
        assert_eq!(right_child(1), 3);
        assert_eq!(left_child(2), 4);
        assert_eq!(right_child(2), 5);

        assert_eq!(parent(2), 1);
        assert_eq!(parent(3), 1);
        assert_eq!(parent(4), 2);

        assert!(is_left_child(2));
        assert!(is_right_child(3));

        assert_eq!(depth(1), 0);
        assert_eq!(depth(2), 1);
        assert_eq!(depth(3), 1);
        assert_eq!(depth(4), 2);
    }
}
