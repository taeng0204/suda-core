"""
SUDA: Budget-Based Continuous Exact Forgetting for Streaming Random Forest

This module provides the Python interface to the SUDA streaming architecture,
which implements:
- Full sample tracking via InfluenceRegistry (with budget eviction)
- Budget-based continuous exact forgetting (forest.forget_batch on eviction)
- Adaptive-k: minority class gets higher OCC(q) redundancy
- Native metrics computation (G-mean, Kappa)

Key Design Decision (519 experiments):
  Budget continuous eviction is the only active path.
  Trigger-based reactive and proactive drift-aware paths were never activated
  in production and have been removed.

Usage:
    model = SUDA(num_features=41)
    result = model.partial_fit(X_batch, y_batch)

    # Result contains:
    # - predictions: np.ndarray
    # - metrics: dict (gmean, kappa, etc.)
    # - registry_size: int (samples tracked)
    # - memory_mb: float
    # - budget_evicted: int
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

from suda_core import PyStreamingController


@dataclass
class SUDAConfig:
    """Configuration for SUDA model.

    Fields are organized into tiers:
    - Core: Forest structure, warmup, adaptive-k (always needed)
    - Budget: Budget management parameters (key mechanism)
    - Experimental: Influence tracking, feature distance, window retrain
    """

    # --- Core: Forest & Streaming ----------------------------------------
    num_trees: int = 50
    k: int = 10
    max_depth: int = 15
    num_features: int = 41
    seed: int = 42
    warmup_samples: int = 1000
    metrics_window: int = 1000
    memory_limit_mb: int = 100  # 0 = unlimited
    adaptive_k_enabled: bool = True
    k_min: int = 3
    k_max: int = 50

    # --- Budget Management (key mechanism, see HOW > WHAT) ----------------
    budget_enabled: bool = False
    budget_max_samples: int = 10000           # Optimal: 3000
    budget_eviction_batch: int = 100
    budget_minority_protection: float = 0.1
    budget_age_weight: float = 0.4
    budget_influence_weight: float = 0.4
    budget_class_weight: float = 0.2
    budget_skip_forest_forget: bool = False   # Ablation: registry-only eviction
    budget_use_feature_distance: bool = False # Feature-distance eviction
    window_retrain_mode: bool = False        # Window retrain instead of exact forget
    window_retrain_interval: int = 1         # Retrain every N eviction batches
    window_retrain_incremental: bool = False # Use incremental (streaming) retrain vs batch

    # --- Experimental: Influence Tracking ---------------------------------
    influence_tracking: bool = False
    influence_update_interval: int = 10       # Recompute every N batches
    influence_sample_count: int = 200
    influence_strategy: str = "none"          # "none"|"oob"|"loss"|"confidence"|"cumulative_oob"|"feature_distance"
    feat_dist_update_interval: int = 2000     # Feature distance recomputation interval (samples)

    # --- Experimental: Tree Rebuild ---------------------------------------
    develop_interval: int = 5                 # 0 = disabled

    # --- Split Quality Monitoring -----------------------------------------
    split_quality_threshold: float | None = None  # None = disabled

    # --- Age-based Subtree Refresh ----------------------------------------
    split_max_age: int = 0                    # 0 = disabled, N = rebuild splits older than N samples


@dataclass
class StreamingResult:
    """Result of processing a batch through SUDA."""

    predictions: np.ndarray
    metrics: dict
    registry_size: int
    memory_mb: float
    total_samples: int
    process_time_us: int
    budget_evicted: int = 0

    @property
    def gmean(self) -> float:
        """Get G-mean (most important metric for imbalanced data)."""
        return self.metrics.get("gmean", 0.0)

    @property
    def kappa(self) -> float:
        """Get Cohen's Kappa."""
        return self.metrics.get("kappa", 0.0)

    @property
    def balanced_accuracy(self) -> float:
        """Get balanced accuracy."""
        return self.metrics.get("balanced_accuracy", 0.0)

    @property
    def attack_recall(self) -> float:
        """Get attack recall (TPR)."""
        return self.metrics.get("attack_recall", 0.0)

    @property
    def benign_recall(self) -> float:
        """Get benign recall (TNR)."""
        return self.metrics.get("benign_recall", 0.0)


class SUDA:
    """
    SUDA: Selective Unlearning for Drift Adaptation.

    DynFrs Random Forest with sample-level exact unlearning for concept drift
    adaptation in class-imbalanced network intrusion detection.

    Pipeline:
        1. fit(X, y)          → batch tree construction + enable_streaming()
        2. partial_fit(X, y)  → test-then-train streaming with:
           - Incremental split updates (streaming mode)
           - Budget-based exact forgetting (forest.forget_batch)
           - Adaptive-k minority protection (OCC(k) redundancy)

    Key mechanisms:
        - Budget eviction: maintains fixed-size sample registry (B)
        - Exact forgetting: forest.forget_batch() removes sample influence from trees
          (without this, registry-only removal has zero effect)
        - Adaptive-k: minority class gets k_max (up to 70), majority gets k_min (1)
        - Streaming splits: enable_streaming() allows splits to update incrementally
          (without this, splits are frozen after fit → severe performance loss)

    Example:
        >>> model = SUDA(num_features=41, budget_enabled=True, budget_max_samples=3000)
        >>> model.fit(X_warmup, y_warmup)  # builds forest + enables streaming
        >>> for X_batch, y_batch in data_stream:
        ...     result = model.partial_fit(X_batch, y_batch)
        ...     print(f"G-mean: {result.metrics['gmean']:.4f}")
    """

    def __init__(self, config: SUDAConfig | None = None, **kwargs):
        """
        Initialize SUDA model.

        Usage:
            # Option 1: Config object
            model = SUDA(config=SUDAConfig(num_features=78, budget_enabled=True))

            # Option 2: Keyword arguments (backward compatible)
            model = SUDA(num_features=78, budget_enabled=True)

        See SUDAConfig for all available parameters and their descriptions.
        """
        if config is not None:
            self.config = config
        else:
            self.config = SUDAConfig(**kwargs)

        # Single dict pass to Rust
        import dataclasses
        self._controller = PyStreamingController(dataclasses.asdict(self.config))

        # History for analysis
        self.metrics_history: list[dict] = []

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        record_history: bool = True,
    ) -> StreamingResult:
        """
        Process a batch of samples.

        This is the main entry point. It performs (all in Rust):
        1. Predict (test-then-train)
        2. Update metrics
        3. Add new samples with full tracking (budget eviction on overflow)
        4. Periodic maintenance (develop, influence recomputation)

        Args:
            X: Feature array (n_samples, n_features), float32
            y: Label array (n_samples,), bool
            record_history: Whether to record metrics history

        Returns:
            StreamingResult with predictions, metrics, and budget info
        """
        # Ensure correct types
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=bool)

        # Single FFI call!
        result_dict = self._controller.stream_batch(X, y)

        # Wrap in StreamingResult
        result = StreamingResult(
            predictions=np.array(result_dict["predictions"], dtype=bool),
            metrics=dict(result_dict["metrics"]),
            registry_size=result_dict["registry_size"],
            memory_mb=result_dict["memory_mb"],
            total_samples=result_dict["total_samples"],
            process_time_us=result_dict["process_time_us"],
            budget_evicted=result_dict.get("budget_evicted", 0),
        )

        # Record history
        if record_history:
            self.metrics_history.append(result.metrics.copy())

        return result

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Pre-train the model on historical data (batch training).

        This method performs offline training before streaming begins:
        1. Builds forest via fit_weighted() with OCC(k) + Adaptive-k
        2. Calls enable_streaming() to activate incremental split updates
        3. Registers all warmup samples in the registry

        After calling this, the model is ready for streaming (is_warmed_up = True).
        IMPORTANT: enable_streaming() is called automatically. Without it,
        splits would remain frozen and streaming performance degrades severely.

        Args:
            X: Feature array (n_samples, n_features), float32
            y: Label array (n_samples,), bool

        Returns:
            dict with training metrics (gmean, accuracy, etc.)
        """
        # Ensure correct types
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=bool)

        # Call Rust pre-training
        metrics = self._controller.fit(X, y)

        # Record initial metrics
        self.metrics_history.append(dict(metrics))

        return dict(metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for samples without training.

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        X = np.asarray(X, dtype=np.float32)
        predictions = self._controller.predict_batch(X)
        return np.asarray(predictions, dtype=bool)

    def stream_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Process a batch of samples and return raw result dict.

        This is a convenience method that returns the raw result dictionary
        instead of the StreamingResult object. Useful for benchmarks and
        analysis scripts that expect a dict interface.

        Args:
            X: Feature array (n_samples, n_features), float32
            y: Label array (n_samples,), bool

        Returns:
            dict with keys: predictions, metrics, registry_size, etc.
        """
        # Ensure correct types
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=bool)

        # Single FFI call - returns raw dict
        result_dict = self._controller.stream_batch(X, y)
        return result_dict

    def get_metrics(self) -> dict:
        """Get current metrics from the controller."""
        return dict(self._controller.current_metrics())

    def reset(self) -> None:
        """Reset the controller state (clears registry and metrics)."""
        self._controller.reset()
        self.metrics_history.clear()

    @property
    def total_samples(self) -> int:
        """Total samples processed."""
        return self._controller.total_samples

    @property
    def registry_size(self) -> int:
        """Current registry size (samples being tracked)."""
        return self._controller.registry_size

    @property
    def memory_mb(self) -> float:
        """Current memory usage in MB."""
        return self._controller.memory_mb

    @property
    def is_warmed_up(self) -> bool:
        """Whether the model is warmed up."""
        return self._controller.is_warmed_up

    @property
    def is_pretrained(self) -> bool:
        """Whether the model has been pre-trained with fit()."""
        return self._controller.is_pretrained

    @property
    def position(self) -> int:
        """Current stream position."""
        return self._controller.position

    @property
    def controller(self):
        """Access to the internal Rust controller for advanced analysis."""
        return self._controller

    # Budget Management properties

    @property
    def budget_enabled(self) -> bool:
        """Whether budget management is enabled."""
        return self._controller.budget_enabled

    @property
    def total_budget_evicted(self) -> int:
        """Total samples evicted by budget management."""
        return self._controller.total_budget_evicted

    def get_budget_eviction_stats(self) -> dict:
        """Get budget eviction statistics."""
        total, benign, attack, degraded = self._controller.get_budget_eviction_stats()
        return {
            "total": total,
            "benign": benign,
            "attack": attack,
            "degraded": degraded,
        }

    @property
    def influence_tracking_enabled(self) -> bool:
        """Whether influence tracking is enabled."""
        return self._controller.influence_tracking_enabled

    def get_influence_coverage(self) -> tuple[int, int]:
        """
        Get influence coverage: (samples_with_influence, total_samples).
        Use this to verify that influence recomputation is working.
        """
        try:
            return self._controller.get_influence_coverage()
        except Exception as e:
            logger.warning("Could not get influence coverage: %s", e)
            return (0, 0)

    def get_budget_eviction_stats_extended(self) -> dict:
        """Get extended budget eviction stats including influence diagnostics."""
        try:
            total, benign, attack, degraded, with_inf, inf_sum = (
                self._controller.get_budget_eviction_stats_extended()
            )
            return {
                "total": total,
                "benign": benign,
                "attack": attack,
                "degraded": degraded,
                "with_influence": with_inf,
                "avg_influence": inf_sum / max(with_inf, 1),
            }
        except Exception as e:
            logger.warning("Could not get extended stats: %s", e)
            return {}

    def get_registry_diagnostics(self) -> dict:
        """Get registry diagnostics for Phase 3 analysis.

        Returns:
            dict with keys: n_samples, n_with_influence, influence_coverage,
            mean_age, mean_influence, age_influence_spearman, n_benign, n_attack
        """
        try:
            return dict(self._controller.get_registry_diagnostics())
        except Exception as e:
            logger.warning("Could not get registry diagnostics: %s", e)
            return {}

    @property
    def influence_degraded_count(self) -> int:
        """Number of samples with degraded influence (positive->negative)."""
        return self._controller.influence_degraded_count

    def get_metrics_summary(self) -> dict:
        """
        Get summary statistics of metrics history.

        Returns:
            dict with mean, std, min, max for each metric
        """
        if not self.metrics_history:
            return {}

        summary = {}
        keys = self.metrics_history[0].keys()

        for key in keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        return summary

    def __repr__(self) -> str:
        memory_mode = "Unlimited" if self.config.memory_limit_mb == 0 else f"{self.config.memory_limit_mb}MB"
        return (
            f"SUDA(num_features={self.config.num_features}, "
            f"num_trees={self.config.num_trees}, "
            f"memory={memory_mode}, "
            f"registry={self.registry_size}, "
            f"total_samples={self.total_samples})"
        )
