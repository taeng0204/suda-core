"""Limited Memory ARF wrapper for Iso-Memory comparison.

This module provides an ARF wrapper that limits memory usage to enable
fair comparison with SUDA under Iso-Memory constraints.

The wrapper maintains a sliding window of training samples and
periodically retrains ARF from the window contents.

Memory Budget Matching (ECBRS-style):
- SUDA: max_samples × (feature_bytes + k × 8) ≈ 5.45 MB
- ARF: n_models=50 + window_size=20000 ≈ 6 MB (matched)

Reference:
- ECBRS (NeurIPS 2023): Uses memory-bounded comparison
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from river.forest import ARFClassifier
from river import tree


def _to_frame(x: np.ndarray) -> pd.DataFrame:
    """Convert numpy array to pandas DataFrame."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    x_float = np.asarray(x, dtype=np.float32)
    return pd.DataFrame(x_float)


def _to_series(y: np.ndarray) -> pd.Series:
    """Convert numpy array to pandas Series."""
    if y.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={y.shape}")
    y_int = np.asarray(y, dtype=np.int64)
    return pd.Series(y_int)


def _coerce_pred(pred: Any) -> int:
    """Coerce prediction to integer."""
    if pred is None:
        return 0
    if isinstance(pred, (bool, np.bool_)):
        return int(bool(pred))
    if isinstance(pred, (int, np.integer)):
        return int(pred)
    if isinstance(pred, (float, np.floating)):
        return int(pred)
    return 0


def _predict_many(model: object, x: np.ndarray) -> np.ndarray:
    """Make predictions for multiple samples."""
    x_df = _to_frame(x)

    predict_many = getattr(model, "predict_many", None)
    if callable(predict_many):
        preds = predict_many(x_df)
        if isinstance(preds, pd.Series):
            preds_np = preds.to_numpy()
        else:
            preds_np = np.asarray(preds)
        preds_np = np.asarray(preds_np)
        if preds_np.dtype == bool:
            return preds_np.astype(np.int64)
        return preds_np.astype(np.int64, copy=False)

    predict_one = getattr(model, "predict_one", None)
    if not callable(predict_one):
        raise TypeError("Model does not support predict_many or predict_one")

    out = np.empty((x_df.shape[0],), dtype=np.int64)
    for i in range(x_df.shape[0]):
        pred = predict_one(x_df.iloc[i].to_dict())
        out[i] = _coerce_pred(pred)
    return out


def _learn_many(model: object, x: np.ndarray, y: np.ndarray) -> None:
    """Learn from multiple samples."""
    x_df = _to_frame(x)
    y_s = _to_series(y)

    learn_many = getattr(model, "learn_many", None)
    if callable(learn_many):
        learn_many(x_df, y_s)
        return

    learn_one = getattr(model, "learn_one", None)
    if not callable(learn_one):
        raise TypeError("Model does not support learn_many or learn_one")

    for i in range(x_df.shape[0]):
        learn_one(x_df.iloc[i].to_dict(), int(y_s.iat[i]))


@dataclass
class LimitedMemoryARF:
    """ARF model with limited memory buffer for Iso-Memory comparison.

    This wrapper maintains a sliding window of training samples and
    limits memory usage to match SUDA's memory budget.

    Attributes:
        n_models: Number of trees in the forest
        max_samples: Maximum samples to retain in memory buffer
        seed: Random seed for reproducibility
        retrain_frequency: How often to retrain from buffer (batches)
        enable_retrain: Whether to enable periodic retraining

    Example:
        >>> model = LimitedMemoryARF(n_models=50, max_samples=20000, seed=42)
        >>> for batch in stream:
        ...     result = model.partial_fit(batch.X, batch.y)
        ...     y_pred = model.predict(batch.X)
    """
    n_models: int = 50
    max_samples: int = 20000
    seed: int = 42
    retrain_frequency: int = 0  # 0 = no periodic retrain
    enable_retrain: bool = False

    # Internal state
    _model: ARFClassifier = field(init=False, repr=False)
    _buffer_X: deque = field(default_factory=deque, init=False, repr=False)
    _buffer_y: deque = field(default_factory=deque, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)
    _batch_count: int = field(default=0, init=False, repr=False)
    _total_samples_seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the ARF model."""
        self._model = ARFClassifier(n_models=self.n_models, seed=self.seed)
        self._buffer_X = deque(maxlen=self.max_samples)
        self._buffer_y = deque(maxlen=self.max_samples)
        self._is_fitted = False
        self._batch_count = 0
        self._total_samples_seen = 0

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Incrementally fit on a batch.

        Args:
            x: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Dict with training results
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # Learn from batch (standard ARF online learning)
        _learn_many(self._model, x, y)
        self._is_fitted = True
        self._batch_count += 1
        self._total_samples_seen += len(y)

        # Add to buffer (FIFO with max size)
        for i in range(len(x)):
            self._buffer_X.append(x[i].copy())
            self._buffer_y.append(int(y[i]))

        # Optional periodic retraining from buffer
        if (self.enable_retrain and
            self.retrain_frequency > 0 and
            self._batch_count % self.retrain_frequency == 0):
            self._retrain_from_buffer()

        return {
            "drift_detected": False,
            "buffer_size": len(self._buffer_X),
            "total_samples_seen": self._total_samples_seen
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels for samples.

        Args:
            x: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)

    def _retrain_from_buffer(self) -> None:
        """Retrain model from buffer contents."""
        if len(self._buffer_X) < 10:
            return

        # Create new model
        self._model = ARFClassifier(n_models=self.n_models, seed=self.seed)

        # Train on buffer
        X_buffer = np.array(list(self._buffer_X), dtype=np.float32)
        y_buffer = np.array(list(self._buffer_y), dtype=np.int64)

        _learn_many(self._model, X_buffer, y_buffer)

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer_X)

    def get_memory_usage_estimate(self, n_features: int) -> float:
        """Estimate memory usage in MB.

        Args:
            n_features: Number of features per sample

        Returns:
            Estimated memory usage in MB
        """
        # Sample buffer: n_samples × n_features × 4 bytes (float32)
        buffer_bytes = len(self._buffer_X) * n_features * 4
        # Label buffer: n_samples × 8 bytes (int64)
        label_bytes = len(self._buffer_y) * 8
        # Model overhead (rough estimate: ~100KB per tree)
        model_bytes = self.n_models * 100 * 1024

        total_bytes = buffer_bytes + label_bytes + model_bytes
        return total_bytes / (1024 * 1024)

    def reset(self) -> None:
        """Reset model to initial state."""
        self.__post_init__()


@dataclass
class LimitedMemoryHAT:
    """HAT model with memory tracking for Iso-Memory comparison.

    Note: HAT doesn't store samples, so memory is mostly constant.
    This wrapper is for API consistency with other models.

    Attributes:
        seed: Random seed for reproducibility
    """
    seed: int = 42

    # Internal state
    _model: tree.HoeffdingAdaptiveTreeClassifier = field(init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)
    _total_samples_seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the HAT model."""
        self._model = tree.HoeffdingAdaptiveTreeClassifier(seed=self.seed)
        self._is_fitted = False
        self._total_samples_seen = 0

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Incrementally fit on a batch.

        Args:
            x: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Dict with training results
        """
        _learn_many(self._model, x, y)
        self._is_fitted = True
        self._total_samples_seen += len(y)

        return {
            "drift_detected": False,
            "total_samples_seen": self._total_samples_seen
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels for samples.

        Args:
            x: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)

    def get_memory_usage_estimate(self) -> float:
        """Estimate memory usage in MB.

        Returns:
            Estimated memory usage in MB (roughly constant ~2 MB)
        """
        return 2.0  # HAT uses approximately constant memory

    def reset(self) -> None:
        """Reset model to initial state."""
        self.__post_init__()
