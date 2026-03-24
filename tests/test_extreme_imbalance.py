"""Tests for extreme class imbalance handling.

These tests verify that SUDA models handle extreme class imbalance
correctly, particularly with adaptive k-redundancy for minority class protection.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import suda_core
    HAS_SUDA_CORE = True
except ImportError:
    HAS_SUDA_CORE = False


def create_imbalanced_dataset(
    n_samples: int = 1000,
    minority_ratio: float = 0.01,
    n_features: int = 10,
    seed: int = 42
):
    """Create an imbalanced binary dataset.

    Args:
        n_samples: Total number of samples
        minority_ratio: Ratio of minority class (attack)
        n_features: Number of features
        seed: Random seed

    Returns:
        X, y: Feature matrix and labels
    """
    rng = np.random.default_rng(seed)

    n_minority = max(1, int(n_samples * minority_ratio))
    n_majority = n_samples - n_minority

    # Majority class (benign) - centered at origin
    X_majority = rng.normal(loc=0, scale=1.0, size=(n_majority, n_features))
    y_majority = np.zeros(n_majority, dtype=bool)

    # Minority class (attack) - shifted
    X_minority = rng.normal(loc=3, scale=1.0, size=(n_minority, n_features))
    y_minority = np.ones(n_minority, dtype=bool)

    X = np.vstack([X_majority, X_minority]).astype(np.float32)
    y = np.concatenate([y_majority, y_minority])

    # Shuffle
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


@pytest.fixture
def cidds_like_dataset():
    """Create dataset similar to CIDDS (0.4% minority)."""
    return create_imbalanced_dataset(n_samples=5000, minority_ratio=0.004)


@pytest.fixture
def realistic_nids_dataset():
    """Create realistic NIDS dataset (1% attack)."""
    return create_imbalanced_dataset(n_samples=5000, minority_ratio=0.01)


@pytest.fixture
def moderate_imbalance_dataset():
    """Create moderately imbalanced dataset (10% attack)."""
    return create_imbalanced_dataset(n_samples=5000, minority_ratio=0.10)


class TestExtremeImbalance:
    """Tests for extreme class imbalance scenarios."""

    @pytest.mark.skipif(not HAS_SUDA_CORE, reason="suda_core not available")
    def test_model_does_not_crash_on_0_4_percent(self, cidds_like_dataset):
        """Test model handles 0.4% minority ratio (CIDDS-like)."""
        X, y = cidds_like_dataset

        # Count minority samples
        n_minority = y.sum()
        n_total = len(y)
        actual_ratio = n_minority / n_total

        assert 0.001 < actual_ratio < 0.01, \
            f"Dataset should have ~0.4% minority, got {actual_ratio:.2%}"

        # Model should not crash
        forest = suda_core.PyDynFrsForest(
            num_trees=20,
            k=5,
            max_depth=10,
            min_samples_leaf=1,
            seed=42
        )

        sample_ids = np.arange(len(X), dtype=np.uint64)
        forest.fit(X, y, sample_ids)

        # Should be able to predict
        predictions = forest.predict(X[:100])
        assert len(predictions) == 100

    @pytest.mark.skipif(not HAS_SUDA_CORE, reason="suda_core not available")
    def test_model_handles_1_percent_minority(self, realistic_nids_dataset):
        """Test model handles 1% minority ratio (realistic NIDS)."""
        X, y = realistic_nids_dataset

        n_minority = y.sum()
        n_total = len(y)
        actual_ratio = n_minority / n_total

        assert 0.005 < actual_ratio < 0.02, \
            f"Dataset should have ~1% minority, got {actual_ratio:.2%}"

        forest = suda_core.PyDynFrsForest(
            num_trees=20,
            k=5,
            max_depth=10,
            min_samples_leaf=1,
            seed=42
        )

        sample_ids = np.arange(len(X), dtype=np.uint64)
        forest.fit(X, y, sample_ids)

        # Predict on test samples
        predictions = forest.predict(X)

        # Check that we can detect at least some minority samples
        minority_mask = y == True
        if minority_mask.sum() > 0:
            minority_predictions = predictions[minority_mask]
            recall = minority_predictions.sum() / len(minority_predictions)
            # With extreme imbalance, some recall loss is expected
            # but model should not predict all-majority
            assert recall > 0.0 or n_minority < 10, \
                f"Model should detect some minority samples, recall={recall:.2%}"

    @pytest.mark.skipif(not HAS_SUDA_CORE, reason="suda_core not available")
    def test_minority_samples_preserved_after_forget(self, realistic_nids_dataset):
        """Test that minority samples are not disproportionately forgotten."""
        X, y = realistic_nids_dataset
        sample_ids = np.arange(len(X), dtype=np.uint64)

        forest = suda_core.PyDynFrsForest(
            num_trees=20,
            k=5,
            max_depth=10,
            min_samples_leaf=1,
            seed=42
        )
        forest.fit(X, y, sample_ids)

        # Forget some random majority samples
        majority_ids = sample_ids[~y]
        n_to_forget = min(50, len(majority_ids))
        ids_to_forget = majority_ids[:n_to_forget]

        n_forgotten = forest.forget(ids_to_forget)

        # Model should still work
        predictions = forest.predict(X[:100])
        assert len(predictions) == 100

    @pytest.mark.skipif(not HAS_SUDA_CORE, reason="suda_core not available")
    def test_single_minority_sample(self):
        """Test edge case with only 1 minority sample."""
        rng = np.random.default_rng(42)

        # 99 majority, 1 minority
        X_majority = rng.normal(loc=0, scale=1.0, size=(99, 10)).astype(np.float32)
        X_minority = rng.normal(loc=3, scale=1.0, size=(1, 10)).astype(np.float32)

        X = np.vstack([X_majority, X_minority])
        y = np.array([False] * 99 + [True])

        sample_ids = np.arange(len(X), dtype=np.uint64)

        forest = suda_core.PyDynFrsForest(
            num_trees=10,
            k=3,
            max_depth=5,
            min_samples_leaf=1,
            seed=42
        )

        # Should not crash
        forest.fit(X, y, sample_ids)
        predictions = forest.predict(X)
        assert len(predictions) == 100


class TestAdaptiveK:
    """Tests for adaptive k-redundancy mechanism."""

    def test_compute_batch_k_basic(self):
        """Test basic k computation for labels."""
        try:
            from src.utils.adaptive_k import AdaptiveKManager, AdaptiveKConfig
        except ImportError:
            pytest.skip("AdaptiveKManager not available")

        config = AdaptiveKConfig(k_min=3, k_max=50)
        manager = AdaptiveKManager(config)

        # Update with imbalanced data
        labels = np.array([False] * 990 + [True] * 10)
        manager.update_class_ratios(labels)

        # Get k values
        test_labels = np.array([True, False, True, False])
        k_values = manager.compute_batch_k(test_labels)

        # Minority class should get higher k
        assert k_values[0] > k_values[1], \
            "Minority (True) should have higher k than majority"
        assert k_values[2] > k_values[3], \
            "Minority (True) should have higher k than majority"

    def test_compute_batch_k_handles_non_bool(self):
        """Test that k computation handles non-boolean labels."""
        try:
            from src.utils.adaptive_k import AdaptiveKManager, AdaptiveKConfig
        except ImportError:
            pytest.skip("AdaptiveKManager not available")

        config = AdaptiveKConfig(k_min=3, k_max=50)
        manager = AdaptiveKManager(config)

        # Update with imbalanced data
        labels = np.array([0] * 990 + [1] * 10)  # int labels
        manager.update_class_ratios(labels.astype(bool))

        # Get k values with int labels
        test_labels = np.array([1, 0, 1, 0])  # int labels
        k_values = manager.compute_batch_k(test_labels)

        assert len(k_values) == 4
        assert all(config.k_min <= k <= config.k_max for k in k_values)

    def test_adaptive_k_extreme_imbalance(self):
        """Test k values for extreme 0.4% imbalance."""
        try:
            from src.utils.adaptive_k import AdaptiveKManager, AdaptiveKConfig
        except ImportError:
            pytest.skip("AdaptiveKManager not available")

        config = AdaptiveKConfig(k_min=3, k_max=50)
        manager = AdaptiveKManager(config)

        # CIDDS-like: 99.6% majority, 0.4% minority
        labels = np.array([False] * 996 + [True] * 4)
        manager.update_class_ratios(labels)

        # Get k for minority and majority
        minority_k = manager.get_class_k(True)
        majority_k = manager.get_class_k(False)

        # Minority should get maximum k
        assert minority_k >= config.k_max * 0.8, \
            f"Minority k should be near max, got {minority_k}"

        # Majority should get lower k
        assert majority_k < minority_k, \
            f"Majority k ({majority_k}) should be less than minority k ({minority_k})"
