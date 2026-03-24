"""Regression tests for SUDA pure inference behavior."""

import numpy as np

from src.models.suda import SUDA


def _make_data(n_samples: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(bool)
    return x, y


def test_predict_has_no_state_side_effects():
    """predict() must not mutate counters, metrics, or history."""
    model = SUDA(
        num_features=4,
        num_trees=10,
        k=4,
        max_depth=8,
        warmup_samples=0,
        metrics_window=128,
        seed=7,
    )

    x_train, y_train = _make_data(120, 4, seed=1)
    model.fit(x_train, y_train)

    metrics_before = model.get_metrics()
    total_samples_before = model.total_samples
    registry_size_before = model.registry_size
    metrics_history_len_before = len(model.metrics_history)

    x_test, _ = _make_data(32, 4, seed=2)
    preds = model.predict(x_test)

    assert preds.shape == (32,)
    assert preds.dtype == np.bool_
    assert model.total_samples == total_samples_before
    assert model.registry_size == registry_size_before
    assert len(model.metrics_history) == metrics_history_len_before
    assert model.get_metrics() == metrics_before


def test_repeated_predict_is_idempotent():
    """Running predict multiple times must keep state unchanged."""
    model = SUDA(
        num_features=3,
        num_trees=8,
        k=3,
        max_depth=6,
        warmup_samples=0,
        metrics_window=64,
        seed=11,
    )

    x_train, y_train = _make_data(80, 3, seed=3)
    model.fit(x_train, y_train)

    x_test, _ = _make_data(20, 3, seed=4)
    state_before = (
        model.total_samples,
        model.registry_size,
        model.get_metrics(),
    )

    _ = model.predict(x_test)
    _ = model.predict(x_test)

    state_after = (
        model.total_samples,
        model.registry_size,
        model.get_metrics(),
    )
    assert state_after == state_before
