from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from river import tree
from river.ensemble import SRPClassifier, LeveragingBaggingClassifier
from river.forest.adaptive_random_forest import ARFClassifier



def _to_frame(x: np.ndarray) -> pd.DataFrame:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    x_float = np.asarray(x, dtype=np.float32)
    return pd.DataFrame(x_float)


def _to_series(y: np.ndarray) -> pd.Series:
    if y.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={y.shape}")
    y_int = np.asarray(y, dtype=np.int64)
    return pd.Series(y_int)


def _coerce_pred(pred: Any) -> int:
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
class ARFModel:
    n_models: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        self._model = ARFClassifier(n_models=self.n_models, seed=self.seed)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class ARFWithOversampling:
    """ARF with minority oversampling in each batch.

    Replicates minority samples to target_ratio before learning,
    providing a fair comparison against SUDA's Adaptive-k (k_max=70).
    """

    n_models: int = 50
    seed: int = 42
    target_ratio: float = 0.3  # minority target proportion in batch

    def __post_init__(self) -> None:
        self._model = ARFClassifier(n_models=self.n_models, seed=self.seed)
        self._is_fitted = False
        self._rng = np.random.default_rng(self.seed)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            minority_cls = classes[np.argmin(counts)]
            minority_mask = y == minority_cls
            majority_mask = ~minority_mask
            n_minority = minority_mask.sum()
            n_majority = majority_mask.sum()

            # Replicate minority to reach target_ratio
            target_n = int(n_majority * self.target_ratio / (1 - self.target_ratio))
            if target_n > n_minority and n_minority > 0:
                extra_needed = target_n - n_minority
                minority_indices = np.where(minority_mask)[0]
                extra_indices = self._rng.choice(minority_indices, size=extra_needed, replace=True)
                x = np.concatenate([x, x[extra_indices]], axis=0)
                y = np.concatenate([y, y[extra_indices]], axis=0)

        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class ARFWithUndersampling:
    """ARF with majority undersampling in each batch."""

    n_models: int = 50
    seed: int = 42
    target_ratio: float = 0.5  # majority:minority ratio target

    def __post_init__(self) -> None:
        self._model = ARFClassifier(n_models=self.n_models, seed=self.seed)
        self._is_fitted = False
        self._rng = np.random.default_rng(self.seed)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            minority_cls = classes[np.argmin(counts)]
            majority_cls = classes[np.argmax(counts)]
            minority_mask = y == minority_cls
            majority_mask = y == majority_cls
            n_minority = minority_mask.sum()
            n_majority = majority_mask.sum()

            # Undersample majority to target ratio
            target_majority = int(n_minority / self.target_ratio)
            if target_majority < n_majority and n_minority > 0:
                majority_indices = np.where(majority_mask)[0]
                keep_indices = self._rng.choice(majority_indices, size=target_majority, replace=False)
                minority_indices = np.where(minority_mask)[0]
                selected = np.concatenate([keep_indices, minority_indices])
                x = x[selected]
                y = y[selected]

        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class ARFWithClassWeight:
    """ARF with class-weighted learning (repeat minority samples).

    Repeats minority samples minority_weight times during learning,
    analogous to SUDA's Adaptive-k with k_max.
    """

    n_models: int = 50
    seed: int = 42
    minority_weight: int = 10  # repeat minority this many times

    def __post_init__(self) -> None:
        self._model = ARFClassifier(n_models=self.n_models, seed=self.seed)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            minority_cls = classes[np.argmin(counts)]
            minority_mask = y == minority_cls
            n_minority = minority_mask.sum()

            if n_minority > 0:
                minority_x = x[minority_mask]
                minority_y = y[minority_mask]
                # Repeat minority samples (minority_weight - 1) extra times
                extra_x = np.tile(minority_x, (self.minority_weight - 1, 1))
                extra_y = np.tile(minority_y, self.minority_weight - 1)
                x = np.concatenate([x, extra_x], axis=0)
                y = np.concatenate([y, extra_y], axis=0)

        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class HATModel:
    seed: int = 42

    def __post_init__(self) -> None:
        self._model = tree.HoeffdingAdaptiveTreeClassifier(seed=self.seed)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class SRPModel:
    """Streaming Random Patches (Gomes et al., ICDM 2019)."""

    n_models: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        self._model = SRPClassifier(n_models=self.n_models, seed=self.seed)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class LeveragingBaggingModel:
    """Leveraging Bagging (Bifet et al., ECML-PKDD 2010)."""

    n_models: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        self._model = LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=self.n_models,
            seed=self.seed,
        )
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class EFDTModel:
    """Extremely Fast Decision Tree (Manapragada et al., KDD 2018)."""

    seed: int = 42

    def __post_init__(self) -> None:
        self._model = tree.ExtremelyFastDecisionTreeClassifier()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)


@dataclass
class HoeffdingTreeModel:
    """Hoeffding Tree / VFDT (Domingos & Hulten, KDD 2000)."""

    seed: int = 42

    def __post_init__(self) -> None:
        self._model = tree.HoeffdingTreeClassifier()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        _learn_many(self._model, x, y)
        self._is_fitted = True
        return {"drift_detected": False}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(x), dtype=np.int64)
        return _predict_many(self._model, x)
