"""NIDS dataset loaders for drift experiments.

Supported datasets:
- nslkdd: NSL-KDD (148K samples, 41 features)
- unswnb15: UNSW-NB15 (258K samples, 42 features)
- cicids2018: CIC-IDS2018 (43K samples, 78 features)
- cidds: CIDDS (94K samples, 5 features)

All datasets are stored as single .npy files with shape (n_samples, n_features + 1)
where the last column is the binary label (0=benign, 1=attack).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _datasets_dir() -> Path:
    return _project_root() / "datasets"


def _anoshift_dir() -> Path:
    """Get the AnoShift dataset directory."""
    return _datasets_dir() / "anoshift"


def _anoshift_single_path() -> Path:
    """Canonical single-file AnoShift path."""
    return _anoshift_dir() / "anoshift.npy"


def _anoshift_year_npy_paths() -> list[Path]:
    """Get available yearly AnoShift .npy files in chronological order."""
    return sorted(_anoshift_dir().glob("year_*.npy"))


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset metadata."""
    name: str
    n_samples: int
    n_features: int
    benign_count: int
    attack_count: int
    benign_ratio: float


def list_datasets() -> tuple[str, ...]:
    """List available datasets."""
    return ("nslkdd", "unswnb15", "cicids2018", "cidds", "anoshift")


def get_dataset_path(name: str) -> Path:
    """Get the path to a dataset file."""
    if name not in list_datasets():
        raise ValueError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    if name == "anoshift":
        single_path = _anoshift_single_path()
        if single_path.exists():
            return single_path
        year_paths = _anoshift_year_npy_paths()
        if year_paths:
            return year_paths[0]
        return single_path
    return _datasets_dir() / f"{name}.npy"


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset.

    Args:
        name: Dataset name (nslkdd, unswnb15, cicids2018, cidds)

    Returns:
        Tuple of (X, y) where X is features and y is binary labels
    """
    if name == "anoshift":
        single_path = _anoshift_single_path()
        if single_path.exists():
            data = np.load(single_path)
        else:
            year_paths = _anoshift_year_npy_paths()
            if not year_paths:
                raise FileNotFoundError(
                    "AnoShift dataset not found.\n"
                    f"Expected one of:\n"
                    f"  - {single_path}\n"
                    f"  - {_anoshift_dir() / 'year_2006.npy'} ... {_anoshift_dir() / 'year_2015.npy'}\n"
                    "Run: uv run python datasets/download.py --dataset anoshift"
                )
            data = np.vstack([np.load(path) for path in year_paths])
    else:
        path = get_dataset_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Run: uv run python datasets/download.py --dataset {name}"
            )
        data = np.load(path)

    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)

    return X, y


def get_dataset_info(name: str) -> DatasetInfo:
    """Get dataset statistics."""
    X, y = load_dataset(name)
    benign_count = int((y == 0).sum())
    attack_count = int((y == 1).sum())

    return DatasetInfo(
        name=name,
        n_samples=len(y),
        n_features=X.shape[1],
        benign_count=benign_count,
        attack_count=attack_count,
        benign_ratio=benign_count / len(y) if len(y) > 0 else 0.0,
    )


def make_stream(
    name: str,
    rng: np.random.Generator,
    *,
    n_samples: int | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a data stream from a dataset.

    Args:
        name: Dataset name
        rng: Random number generator
        n_samples: Number of samples to return (None = all)
        shuffle: Whether to shuffle the data

    Returns:
        Tuple of (X, y)
    """
    X, y = load_dataset(name)

    if shuffle:
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]

    if n_samples is not None and n_samples < len(y):
        X, y = X[:n_samples], y[:n_samples]

    return X, y


def create_imbalanced_dataset(
    name: str,
    attack_ratio: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a dataset with realistic class imbalance.

    Subsamples the attack class to achieve the target attack ratio while
    keeping all benign samples. This simulates real-world NIDS scenarios
    where attacks are rare (typically < 1%).

    Args:
        name: Dataset name (nslkdd, unswnb15, cicids2018)
        attack_ratio: Target attack ratio (default: 0.01 = 1%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) with the specified imbalance

    Raises:
        ValueError: If dataset has insufficient attack samples

    Example:
        >>> X, y = create_imbalanced_dataset("nslkdd", attack_ratio=0.01, seed=42)
        >>> print(f"Attack ratio: {y.mean():.2%}")  # ~1%
    """
    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    n_benign = len(X_benign)
    n_attack_original = len(X_attack)

    # attack_ratio = n_attack / (n_benign + n_attack)
    # => n_attack = attack_ratio * n_benign / (1 - attack_ratio)
    n_attack_target = int(attack_ratio * n_benign / (1 - attack_ratio))

    if n_attack_target < 10:
        raise ValueError(
            f"Target attack count too low ({n_attack_target}). "
            f"Consider using a higher attack_ratio or a different dataset."
        )

    if n_attack_target > n_attack_original:
        raise ValueError(
            f"Insufficient attack samples. Dataset has {n_attack_original:,} attacks "
            f"but target ratio requires {n_attack_target:,}. "
            f"Use a lower attack_ratio."
        )

    attack_idx = rng.choice(n_attack_original, size=n_attack_target, replace=False)
    X_attack_sampled = X_attack[attack_idx]
    y_attack_sampled = y_attack[attack_idx]

    X_out = np.vstack([X_benign, X_attack_sampled])
    y_out = np.hstack([y_benign, y_attack_sampled])

    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# =============================================================================
# Module-level sampling helper (replaces repeated inline closures)
# =============================================================================

def _sample_with_ratio(
    X_benign: np.ndarray,
    y_benign: np.ndarray,
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    n: int,
    attack_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n instances with given attack ratio, shuffled."""
    n_attack = int(n * attack_ratio)
    n_benign = n - n_attack
    b_idx = rng.choice(len(X_benign), size=n_benign, replace=len(X_benign) < n_benign)
    a_idx = rng.choice(len(X_attack), size=n_attack, replace=len(X_attack) < n_attack)
    X_part = np.vstack([X_benign[b_idx], X_attack[a_idx]])
    y_part = np.hstack([y_benign[b_idx], y_attack[a_idx]])
    perm = rng.permutation(len(y_part))
    return X_part[perm], y_part[perm]


def _load_and_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset and split into benign/attack arrays."""
    X, y = load_dataset(name)
    mask = y == 1
    return X[~mask], y[~mask], X[mask], y[mask]


# =============================================================================
# Drift Stream Factories
# =============================================================================

def make_realistic_drift_stream(
    name: str,
    *,
    pre_attack_ratio: float = 0.01,
    post_attack_ratio: float = 0.05,
    total_samples: int = 50000,
    drift_point: int = 25000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a drift stream with realistic class imbalance.

    Simulates a label shift scenario where attack ratio increases after drift.
    This is common in real NIDS when a new attack campaign begins.

    Args:
        name: Dataset name
        pre_attack_ratio: Attack ratio before drift (default: 1%)
        post_attack_ratio: Attack ratio after drift (default: 5%)
        total_samples: Total samples in stream
        drift_point: Position where drift occurs
        seed: Random seed

    Returns:
        Tuple of (X, y) stream

    Example:
        >>> X, y = make_realistic_drift_stream("nslkdd", pre_attack_ratio=0.01)
        >>> pre_ratio = y[:25000].mean()  # ~1%
        >>> post_ratio = y[25000:].mean()  # ~5%
    """
    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    X_pre, y_pre = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack, drift_point, pre_attack_ratio, rng
    )
    X_post, y_post = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack,
        total_samples - drift_point, post_attack_ratio, rng,
    )

    return np.vstack([X_pre, X_post]), np.hstack([y_pre, y_post])


def make_moderate_sudden_drift_stream(
    name: str,
    *,
    phase1_samples: int = 25000,
    phase2_samples: int = 25000,
    phase1_ratio: float = 0.01,
    phase2_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a moderate sudden drift stream (1% -> 20%).

    Not mild enough for Adaptive-k to fully handle, not extreme enough
    for reactive triggers to fire immediately. Tests whether continuous
    budget eviction provides value in this middle ground.

    Args:
        name: Dataset name
        phase1_samples: Samples in normal phase
        phase2_samples: Samples in drifted phase
        phase1_ratio: Attack ratio before drift
        phase2_ratio: Attack ratio after drift
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    X1, y1 = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack, phase1_samples, phase1_ratio, rng
    )
    X2, y2 = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack, phase2_samples, phase2_ratio, rng
    )

    metadata = {
        "scenario": "moderate_sudden",
        "phases": [phase1_ratio, phase2_ratio],
        "phase_boundaries": [phase1_samples],
        "total": phase1_samples + phase2_samples,
    }

    return np.vstack([X1, X2]), np.hstack([y1, y2]), metadata


def make_stepwise_drift_stream(
    name: str,
    *,
    phase_samples: int = 10000,
    phases: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a step-wise drift stream with accumulating stale samples.

    Default: 1% -> 10% -> 20% -> 10% -> 5%
    Each step creates stale samples from the previous concept.
    Tests whether budget eviction removes accumulated stale samples
    better than no eviction.

    Args:
        name: Dataset name
        phase_samples: Samples per phase
        phases: Attack ratios per phase
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    if phases is None:
        phases = [0.01, 0.10, 0.20, 0.10, 0.05]

    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    parts = []
    phase_boundaries = []
    current_pos = 0

    for attack_ratio in phases:
        X_p, y_p = _sample_with_ratio(
            X_benign, y_benign, X_attack, y_attack, phase_samples, attack_ratio, rng
        )
        parts.append((X_p, y_p))
        current_pos += phase_samples
        phase_boundaries.append(current_pos)

    X_stream = np.vstack([p[0] for p in parts])
    y_stream = np.hstack([p[1] for p in parts])

    metadata = {
        "scenario": "stepwise",
        "phases": phases,
        "phase_boundaries": phase_boundaries[:-1],
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


def make_gradual_ramp_stream(
    name: str,
    *,
    phase_samples: int = 10000,
    phases: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a gradual ramp drift stream.

    Default: 1% -> 5% -> 10% -> 15% -> 20%
    Each step is small (~5%) but cumulative effect is large (1%->20%).
    No individual step triggers reactive unlearning, but accumulated
    stale samples degrade performance. Budget eviction should help.

    Args:
        name: Dataset name
        phase_samples: Samples per phase
        phases: Attack ratios per phase
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    if phases is None:
        phases = [0.01, 0.05, 0.10, 0.15, 0.20]

    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    parts = []
    phase_boundaries = []
    current_pos = 0

    for attack_ratio in phases:
        X_p, y_p = _sample_with_ratio(
            X_benign, y_benign, X_attack, y_attack, phase_samples, attack_ratio, rng
        )
        parts.append((X_p, y_p))
        current_pos += phase_samples
        phase_boundaries.append(current_pos)

    X_stream = np.vstack([p[0] for p in parts])
    y_stream = np.hstack([p[1] for p in parts])

    metadata = {
        "scenario": "gradual_ramp",
        "phases": phases,
        "phase_boundaries": phase_boundaries[:-1],
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


def make_asymmetric_recovery_stream(
    name: str,
    *,
    phase1_samples: int = 20000,
    phase2_samples: int = 20000,
    phase3_samples: int = 20000,
    phase1_ratio: float = 0.01,
    phase2_ratio: float = 0.20,
    phase3_ratio: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create an asymmetric recovery stream.

    Default: 1% -> 20% -> 3%
    Recovery doesn't return to exact original (3% vs 1%).
    Tests whether budget eviction helps when the "new normal"
    is different from the original. Stale samples from Phase 1
    (1% world) should be cleaned out for Phase 3 (3% world).

    Args:
        name: Dataset name
        phase1_samples: Normal phase
        phase2_samples: Drift phase
        phase3_samples: Partial recovery phase
        phase1_ratio: Attack ratio phase 1
        phase2_ratio: Attack ratio phase 2
        phase3_ratio: Attack ratio phase 3 (doesn't fully recover)
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    X1, y1 = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack, phase1_samples, phase1_ratio, rng
    )
    X2, y2 = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack, phase2_samples, phase2_ratio, rng
    )
    X3, y3 = _sample_with_ratio(
        X_benign, y_benign, X_attack, y_attack, phase3_samples, phase3_ratio, rng
    )

    X_stream = np.vstack([X1, X2, X3])
    y_stream = np.hstack([y1, y2, y3])

    metadata = {
        "scenario": "asymmetric_recovery",
        "phases": [phase1_ratio, phase2_ratio, phase3_ratio],
        "phase_boundaries": [phase1_samples, phase1_samples + phase2_samples],
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


def make_attack_decrease_stream(
    name: str,
    *,
    phase_samples: int = 20000,
    phases: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create an attack decrease stream: 20% -> 5% -> 1%.

    This is the KEY scenario where Selective eviction should beat FIFO.
    FIFO loses useful attack samples as they age out, leaving only ~30 attack
    samples in a 3K budget at 1% ratio. Selective preserves high-influence
    attack samples regardless of age.

    Args:
        name: Dataset name
        phase_samples: Samples per phase
        phases: Attack ratios per phase (default: [0.20, 0.05, 0.01])
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    if phases is None:
        phases = [0.20, 0.05, 0.01]

    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    X_parts, y_parts = [], []
    boundaries = []
    total = 0
    for ratio in phases:
        # Ensure at least 1 attack sample at low ratios
        n = phase_samples
        n_attack = max(1, int(n * ratio))
        n_benign = n - n_attack
        b_idx = rng.choice(len(X_benign), size=n_benign, replace=len(X_benign) < n_benign)
        a_idx = rng.choice(len(X_attack), size=n_attack, replace=len(X_attack) < n_attack)
        X_part = np.vstack([X_benign[b_idx], X_attack[a_idx]])
        y_part = np.hstack([y_benign[b_idx], y_attack[a_idx]])
        perm = rng.permutation(len(y_part))
        X_parts.append(X_part[perm])
        y_parts.append(y_part[perm])
        total += len(y_part)
        boundaries.append(total)

    X_stream = np.vstack(X_parts)
    y_stream = np.hstack(y_parts)

    metadata = {
        "scenario": "attack_decrease",
        "phases": phases,
        "phase_boundaries": boundaries[:-1],
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


def make_noisy_burst_stream(
    name: str,
    *,
    clean1_samples: int = 20000,
    noise_samples: int = 5000,
    clean2_samples: int = 25000,
    attack_ratio: float = 0.01,
    noise_flip_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a stream with a noisy burst in the middle.

    Pattern: 1% clean (20K) -> 1% + 10% label noise (5K) -> 1% clean (25K)

    FIFO removes noise samples by age order (slow cleanup).
    Selective identifies negative-influence noise samples and evicts them first.

    Args:
        name: Dataset name
        clean1_samples: Samples before noise burst
        noise_samples: Samples in noise burst
        clean2_samples: Samples after noise burst
        attack_ratio: Baseline attack ratio
        noise_flip_ratio: Fraction of labels flipped during noise burst
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    def _sample(n: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        n_attack = max(1, int(n * ratio))
        n_benign = n - n_attack
        b_idx = rng.choice(len(X_benign), size=n_benign, replace=len(X_benign) < n_benign)
        a_idx = rng.choice(len(X_attack), size=n_attack, replace=len(X_attack) < n_attack)
        X_part = np.vstack([X_benign[b_idx], X_attack[a_idx]])
        y_part = np.hstack([y_benign[b_idx], y_attack[a_idx]])
        perm = rng.permutation(len(y_part))
        return X_part[perm], y_part[perm]

    X1, y1 = _sample(clean1_samples, attack_ratio)

    X2, y2 = _sample(noise_samples, attack_ratio)
    n_flip = int(len(y2) * noise_flip_ratio)
    flip_idx = rng.choice(len(y2), size=n_flip, replace=False)
    y2[flip_idx] = 1 - y2[flip_idx]

    X3, y3 = _sample(clean2_samples, attack_ratio)

    X_stream = np.vstack([X1, X2, X3])
    y_stream = np.hstack([y1, y2, y3])

    metadata = {
        "scenario": "noisy_burst",
        "phases": [attack_ratio, f"{attack_ratio}+{noise_flip_ratio}noise", attack_ratio],
        "phase_boundaries": [clean1_samples, clean1_samples + noise_samples],
        "noise_flip_count": n_flip,
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


def make_class_oscillation_stream(
    name: str,
    *,
    phase_samples: int = 10000,
    low_ratio: float = 0.02,
    high_ratio: float = 0.20,
    n_cycles: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a class oscillation stream: 2% -> 20% -> 2% -> 20% -> 2% -> 20%.

    FIFO forgets the previous concept completely when oscillating.
    Selective preserves high-influence samples from previous concepts,
    enabling faster recovery when the concept recurs.

    Args:
        name: Dataset name
        phase_samples: Samples per phase
        low_ratio: Low attack ratio
        high_ratio: High attack ratio
        n_cycles: Number of low->high cycles
        seed: Random seed

    Returns:
        Tuple of (X, y, metadata)
    """
    rng = np.random.default_rng(seed)
    X_benign, y_benign, X_attack, y_attack = _load_and_split(name)

    X_parts, y_parts = [], []
    phases = []
    boundaries = []
    total = 0
    for _cycle in range(n_cycles):
        for ratio in [low_ratio, high_ratio]:
            n_attack = max(1, int(phase_samples * ratio))
            n_benign = phase_samples - n_attack
            b_idx = rng.choice(len(X_benign), size=n_benign, replace=len(X_benign) < n_benign)
            a_idx = rng.choice(len(X_attack), size=n_attack, replace=len(X_attack) < n_attack)
            X_part = np.vstack([X_benign[b_idx], X_attack[a_idx]])
            y_part = np.hstack([y_benign[b_idx], y_attack[a_idx]])
            perm = rng.permutation(len(y_part))
            X_parts.append(X_part[perm])
            y_parts.append(y_part[perm])
            phases.append(ratio)
            total += len(y_part)
            boundaries.append(total)

    X_stream = np.vstack(X_parts)
    y_stream = np.hstack(y_parts)

    metadata = {
        "scenario": "class_oscillation",
        "phases": phases,
        "phase_boundaries": boundaries[:-1],
        "n_cycles": n_cycles,
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


# =============================================================================
# AnoShift Dataset (10-year temporal drift)
# =============================================================================

ANOSHIFT_YEARS = list(range(2006, 2016))  # 2006-2015


def load_anoshift_year(year: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a single year of AnoShift data.

    Args:
        year: Year to load (2006-2015)

    Returns:
        Tuple of (X, y) where X is features and y is binary labels (0=normal, 1=anomaly)

    Note:
        Kyoto-2016 AnoShift columns:
        - 0-4: Categorical (source/dest IP encoded, service, etc.)
        - 5-12: Numeric (ratios and counts)
        - 13: Connection state (categorical)
        - 14: Datetime (skipped)
        - 15-17: IDS alerts (skipped)
        - 18: Label (1=normal, -1/-2=anomaly)
        - 19: Protocol (categorical)

        We use columns 0-13, 19 as features (encoding categoricals).
    """
    if year not in ANOSHIFT_YEARS:
        raise ValueError(f"Invalid year {year}. Valid years: {ANOSHIFT_YEARS}")

    # Try npy format first (preprocessed, faster)
    npy_path = _anoshift_dir() / f"year_{year}.npy"
    if npy_path.exists():
        data = np.load(npy_path)
        X = data[:, :-1].astype(np.float32)
        y = data[:, -1].astype(np.int64)
        return X, y

    # Try parquet (original format, needs preprocessing)
    parquet_path = _anoshift_dir() / "Kyoto-2016_AnoShift" / "subset" / f"{year}_subset.parquet"
    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            from sklearn.preprocessing import LabelEncoder
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            cat_cols = [0, 1, 2, 3, 4, 13, 19]
            num_cols = [5, 6, 7, 8, 9, 10, 11, 12]
            label_col = 18

            features = []

            for col_idx in cat_cols:
                col = df.iloc[:, col_idx].astype(str)
                le = LabelEncoder()
                encoded = le.fit_transform(col)
                features.append(encoded.reshape(-1, 1))

            for col_idx in num_cols:
                col = df.iloc[:, col_idx]
                numeric = np.array([float(x) if x not in ('', 'NA', 'NaN') else 0.0 for x in col])
                features.append(numeric.reshape(-1, 1))

            X = np.hstack(features).astype(np.float32)

            labels = df.iloc[:, label_col].astype(str)
            y = np.array([0 if l == '1' else 1 for l in labels], dtype=np.int64)

            # Cache as npy for faster future loads
            np.save(npy_path, np.hstack([X, y.reshape(-1, 1)]))

            return X, y
        except ImportError:
            raise ImportError("pyarrow required for AnoShift. Run: uv add pyarrow")

    raise FileNotFoundError(
        f"AnoShift year {year} not found. Run:\n"
        f"  uv run python datasets/download.py --dataset anoshift"
    )


def load_anoshift_tasks(
    years: list[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Load AnoShift as a list of yearly tasks.

    This creates a natural temporal drift scenario where each task
    represents one year of network traffic data.

    Args:
        years: List of years to load (default: all 2006-2015)

    Returns:
        List of (X, y, year) tuples for each year
    """
    if years is None:
        years = ANOSHIFT_YEARS

    tasks = []
    for year in years:
        X, y = load_anoshift_year(year)
        tasks.append((X, y, year))

    return tasks


def make_anoshift_temporal_stream(
    *,
    years: list[int] | None = None,
    samples_per_year: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a temporal drift stream from AnoShift.

    Each year is a natural drift point, creating a long-term
    temporal drift scenario spanning 10 years.

    Args:
        years: Years to include (default: all 2006-2015)
        samples_per_year: Max samples per year (None = all)
        seed: Random seed for sampling

    Returns:
        Tuple of (X, y, metadata) where metadata contains:
        - years: List of years included
        - year_boundaries: Sample positions where each year ends
        - total: Total samples
    """
    if years is None:
        years = ANOSHIFT_YEARS

    rng = np.random.default_rng(seed)

    parts = []
    year_boundaries = []  # List of (year, start, end) tuples
    current_pos = 0

    for year in years:
        X, y = load_anoshift_year(year)

        if samples_per_year is not None and len(y) > samples_per_year:
            idx = rng.choice(len(y), size=samples_per_year, replace=False)
            X, y = X[idx], y[idx]

        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        start_pos = current_pos
        current_pos += len(y)
        year_boundaries.append((year, start_pos, current_pos))
        parts.append((X, y))

    X_stream = np.vstack([p[0] for p in parts])
    y_stream = np.hstack([p[1] for p in parts])

    metadata = {
        "scenario": "anoshift_temporal",
        "years": years,
        "year_boundaries": year_boundaries,
        "samples_per_year": [len(p[1]) for p in parts],
        "total": len(y_stream),
    }

    return X_stream, y_stream, metadata


# =============================================================================
# Attack Subtype Loading & Feature Drift Scenarios (2026-02-10)
# =============================================================================

# NSL-KDD attack type -> category mapping
_NSLKDD_ATTACK_CATEGORIES = {
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe attacks
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

# CIC-IDS2018 label -> category mapping
_CICIDS2018_CATEGORIES = {
    "FTP-BruteForce": "BruteForce", "SSH-Bruteforce": "BruteForce",
    "Brute Force -Web": "BruteForce", "Brute Force -XSS": "BruteForce",
    "DoS attacks-Hulk": "DoS", "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS", "DoS attacks-GoldenEye": "DoS",
    "DDOS attack-HOIC": "DDoS", "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "Bot": "Bot", "Infilteration": "Infiltration",
    "SQL Injection": "WebAttack",
}

# UNSW-NB15 categories used for feature drift (high-count only)
_UNSWNB15_DRIFT_CATEGORIES = [
    "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
]


def load_dataset_with_attack_types(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Load a dataset with attack subtype information.

    Reads raw CSV files to extract attack type categories, then applies
    the same preprocessing as the standard load_dataset() function.

    Args:
        name: Dataset name (nslkdd, unswnb15, cicids2018)

    Returns:
        Tuple of (X, y_binary, y_category, category_map) where:
        - X: Feature matrix (n_samples, n_features) matching load_dataset()
        - y_binary: Binary labels (0=benign, 1=attack)
        - y_category: Integer category labels (0=benign, 1..N=attack subtypes)
        - category_map: {int: str} mapping category IDs to names

    Raises:
        ValueError: If dataset not supported for attack type loading
    """
    import pandas as pd  # noqa: F401 (used in sub-functions)

    raw_dir = _datasets_dir() / "raw"

    if name == "nslkdd":
        return _load_nslkdd_with_attack_types(raw_dir)
    elif name == "unswnb15":
        return _load_unswnb15_with_attack_types(raw_dir)
    elif name == "cicids2018":
        return _load_cicids2018_with_attack_types(raw_dir)
    else:
        raise ValueError(
            f"Attack type loading not supported for '{name}'. "
            f"Supported: nslkdd, unswnb15, cicids2018"
        )


def _load_nslkdd_with_attack_types(
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Load NSL-KDD with attack categories: DoS, Probe, R2L, U2R."""
    import pandas as pd

    from datasets.download import NSLKDD_COLUMNS

    train_file = raw_dir / "KDDTrain+.txt"
    test_file = raw_dir / "KDDTest+.txt"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"NSL-KDD raw files not found in {raw_dir}. "
            "Run: uv run python datasets/download.py --dataset nslkdd"
        )

    train_df = pd.read_csv(train_file, header=None, names=NSLKDD_COLUMNS)
    test_df = pd.read_csv(test_file, header=None, names=NSLKDD_COLUMNS)
    df = pd.concat([train_df, test_df], ignore_index=True)

    def categorize(name: str) -> str:
        if name == "normal":
            return "normal"
        return _NSLKDD_ATTACK_CATEGORIES.get(name, "Unknown")

    df["attack_category"] = df["class"].apply(categorize)

    category_names = ["normal", "DoS", "Probe", "R2L", "U2R"]
    cat_to_id = {c: i for i, c in enumerate(category_names)}
    category_map = {i: c for i, c in enumerate(category_names)}

    y_category = df["attack_category"].map(
        lambda c: cat_to_id.get(c, cat_to_id.get("normal", 0))
    ).values.astype(np.int64)

    df["label"] = (df["class"] != "normal").astype(int)

    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes.astype(float)

    feature_cols = [c for c in df.columns
                    if c not in ["class", "difficulty", "label", "attack_category"]]

    X = df[feature_cols].values.astype(np.float32)
    y_binary = df["label"].values.astype(np.int64)

    return X, y_binary, y_category, category_map


def _load_unswnb15_with_attack_types(
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Load UNSW-NB15 with attack categories from attack_cat column."""
    import pandas as pd

    train_file = raw_dir / "train.csv"
    test_file = raw_dir / "test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 raw files not found in {raw_dir}. "
            "Run: uv run python datasets/download.py --dataset unswnb15"
        )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    df = pd.concat([train_df, test_df], ignore_index=True)

    df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()

    category_names = ["Normal"] + sorted(
        df["attack_cat"][df["attack_cat"] != "Normal"].unique().tolist()
    )
    cat_to_id = {c: i for i, c in enumerate(category_names)}
    category_map = {i: c for i, c in enumerate(category_names)}

    y_category = df["attack_cat"].map(cat_to_id).values.astype(np.int64)

    label_col = "label"
    if df[label_col].dtype == object:
        df["binary_label"] = (
            ~df[label_col].str.lower().isin(["normal", "benign", "0"])
        ).astype(int)
    else:
        df["binary_label"] = (df[label_col] != 0).astype(int)

    drop_cols = [label_col, "binary_label", "id", "attack_cat", "Attack", "attack"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes.astype(float)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_binary = df["binary_label"].values.astype(np.int64)

    return X, y_binary, y_category, category_map


def _load_cicids2018_with_attack_types(
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Load CIC-IDS2018 with attack categories from Label column."""
    import pandas as pd

    filepath = raw_dir / "cicids2018.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"CIC-IDS2018 raw file not found: {filepath}. "
            "Run: uv run python datasets/download.py --dataset cicids2018"
        )

    df = pd.read_csv(filepath, low_memory=False)

    label_col = df.columns[-1]
    df[label_col] = df[label_col].str.strip()

    def categorize(label: str) -> str:
        if label.lower() in ("benign", "normal"):
            return "Benign"
        return _CICIDS2018_CATEGORIES.get(label, label)

    df["attack_category"] = df[label_col].apply(categorize)

    category_names = ["Benign"] + sorted(
        df["attack_category"][df["attack_category"] != "Benign"].unique().tolist()
    )
    cat_to_id = {c: i for i, c in enumerate(category_names)}
    category_map = {i: c for i, c in enumerate(category_names)}

    y_category = df["attack_category"].map(cat_to_id).values.astype(np.int64)

    df["binary_label"] = (
        ~df[label_col].str.strip().str.lower().isin(["benign", "normal", "0"])
    ).astype(int)

    drop_cols = [label_col, "binary_label", "attack_category",
                 "Timestamp", "timestamp", "Flow ID", "Src IP", "Dst IP"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes.astype(float)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_binary = df["binary_label"].values.astype(np.int64)

    return X, y_binary, y_category, category_map


def make_attack_type_shift_stream(
    name: str,
    *,
    total_samples: int = 50000,
    warmup_ratio: float = 0.3,
    attack_ratio: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a feature drift stream via attack type shift.

    Attack ratio stays FIXED across all phases. Only the attack subtype
    changes between phases, creating pure feature drift without label shift.

    NSL-KDD example:
    - Phase 1 (warmup + early): Benign 99% + DoS 1%
    - Phase 2: Benign 99% + Probe 1%  (feature distribution changes)
    - Phase 3: Benign 99% + R2L 1%    (another feature shift)

    Args:
        name: Dataset name (nslkdd, unswnb15, cicids2018)
        total_samples: Total samples in stream
        warmup_ratio: Fraction used for warmup (model.fit)
        attack_ratio: Fixed attack ratio across all phases (default: 1%)
        seed: Random seed

    Returns:
        Tuple of (X, y_binary, metadata) where metadata contains:
        - scenario: "attack_type_shift"
        - attack_types: list of attack type names per phase
        - phase_boundaries: sample indices where phases change
        - attack_ratio: the fixed attack ratio
    """
    rng = np.random.default_rng(seed)
    X_all, y_binary, y_category, category_map = load_dataset_with_attack_types(name)

    benign_mask = y_binary == 0
    X_benign = X_all[benign_mask]

    attack_groups: dict[str, np.ndarray] = {}
    for cat_id, cat_name in category_map.items():
        if cat_id == 0:
            continue
        cat_mask = y_category == cat_id
        if cat_mask.sum() >= 50:
            attack_groups[cat_name] = X_all[cat_mask]

    if name == "nslkdd":
        phase_types = ["DoS", "Probe", "R2L"]
    elif name == "unswnb15":
        phase_types = ["Generic", "Exploits", "Fuzzers"]
    elif name == "cicids2018":
        phase_types = ["DoS", "BruteForce", "DDoS"]
    else:
        sorted_groups = sorted(attack_groups.items(), key=lambda x: len(x[1]), reverse=True)
        phase_types = [g[0] for g in sorted_groups[:3]]

    phase_types = [t for t in phase_types if t in attack_groups]
    if len(phase_types) < 2:
        raise ValueError(
            f"Not enough attack types for feature drift in {name}. "
            f"Available: {list(attack_groups.keys())}"
        )

    n_phases = len(phase_types)
    phase_samples = total_samples // n_phases

    parts_X = []
    parts_y = []
    phase_boundaries = []
    current_pos = 0

    for i, attack_type in enumerate(phase_types):
        n_phase = phase_samples if i < n_phases - 1 else total_samples - current_pos
        n_attack = int(n_phase * attack_ratio)
        n_benign = n_phase - n_attack

        b_idx = rng.choice(len(X_benign), size=n_benign, replace=len(X_benign) < n_benign)
        X_b = X_benign[b_idx]
        y_b = np.zeros(n_benign, dtype=np.int64)

        X_type = attack_groups[attack_type]
        a_idx = rng.choice(len(X_type), size=n_attack, replace=len(X_type) < n_attack)
        X_a = X_type[a_idx]
        y_a = np.ones(n_attack, dtype=np.int64)

        X_phase = np.vstack([X_b, X_a])
        y_phase = np.hstack([y_b, y_a])
        perm = rng.permutation(len(y_phase))
        parts_X.append(X_phase[perm])
        parts_y.append(y_phase[perm])

        current_pos += n_phase
        if i < n_phases - 1:
            phase_boundaries.append(current_pos)

    X_stream = np.vstack(parts_X)
    y_stream = np.hstack(parts_y)

    metadata = {
        "scenario": "attack_type_shift",
        "attack_types": phase_types,
        "phase_boundaries": phase_boundaries,
        "attack_ratio": attack_ratio,
        "n_phases": n_phases,
        "total": len(y_stream),
        "warmup_samples": int(total_samples * warmup_ratio),
    }

    return X_stream, y_stream, metadata


if __name__ == "__main__":
    print("=== Available Datasets ===\n")

    for name in list_datasets():
        path = get_dataset_path(name)
        if path.exists():
            info = get_dataset_info(name)
            print(f"{name}:")
            print(f"  Samples: {info.n_samples:,}")
            print(f"  Features: {info.n_features}")
            print(f"  Benign: {info.benign_count:,} ({info.benign_ratio:.1%})")
            print(f"  Attack: {info.attack_count:,} ({1-info.benign_ratio:.1%})")
            print()
        else:
            print(f"{name}: NOT FOUND (run download.py)")
            print()
