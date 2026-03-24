#!/usr/bin/env python3
"""Download NIDS datasets from public sources.

Usage:
    uv run python datasets/download.py --all
    uv run python datasets/download.py --dataset nslkdd
    uv run python datasets/download.py --dataset unswnb15
    uv run python datasets/download.py --dataset cicids2018
    uv run python datasets/download.py --dataset cidds
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

# Base directory for datasets
DATASETS_DIR = Path(__file__).parent


# =============================================================================
# Dataset Sources
# =============================================================================

NSLKDD_TRAIN_SOURCES = [
    (
        "github_jmnwong_root",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt",
    ),
    (
        "github_jmnwong_dataset",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/dataset/KDDTrain+.txt",
    ),
    (
        "github_defcom17",
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    ),
]

NSLKDD_TEST_SOURCES = [
    (
        "github_jmnwong_root",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt",
    ),
    (
        "github_jmnwong_dataset",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/dataset/KDDTest+.txt",
    ),
    (
        "github_defcom17",
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
    ),
]

UNSWNB15_SOURCES = [
    (
        "hf_mireu_train",
        "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv",
    ),
    (
        "hf_mireu_test",
        "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv",
    ),
]

CICIDS2018_SOURCES = [
    (
        "hf_najet",
        "https://huggingface.co/datasets/Najet-hamdi/CIC-IDS2018/resolve/main/dataset.csv",
    ),
]

CIDDS_SOURCES = [
    (
        "hf_caffeinatedcherrychic",
        "https://huggingface.co/datasets/caffeinatedcherrychic/cidds-aggregated/resolve/main/internal-output.csv",
    ),
]


# =============================================================================
# NSL-KDD Column Names (41 features + class + difficulty)
# =============================================================================

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"
]


# =============================================================================
# Download Utilities
# =============================================================================

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file from URL with progress."""
    print(f"  Downloading from {url}")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; SUDA/1.0)"}

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=120)
        response.raise_for_status()

        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None

        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)", end="")
                    else:
                        print(f"\r  Progress: {downloaded:,} bytes", end="")

            print()  # newline

        return True

    except requests.RequestException as e:
        print(f"  Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def try_download_sources(sources: list[tuple[str, str]], dest: Path) -> bool:
    """Try multiple sources until one succeeds."""
    for name, url in sources:
        print(f"  Trying source: {name}")
        if download_file(url, dest):
            return True
    return False


# =============================================================================
# Dataset Processors
# =============================================================================

def process_nslkdd(output_dir: Path) -> bool:
    """Download and process NSL-KDD dataset."""
    print("\n=== NSL-KDD Dataset ===")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download train
    train_file = raw_dir / "KDDTrain+.txt"
    if not train_file.exists():
        print("Downloading training set...")
        if not try_download_sources(NSLKDD_TRAIN_SOURCES, train_file):
            print("ERROR: Failed to download NSL-KDD training set")
            return False
    else:
        print(f"Training set already exists: {train_file}")

    # Download test
    test_file = raw_dir / "KDDTest+.txt"
    if not test_file.exists():
        print("Downloading test set...")
        if not try_download_sources(NSLKDD_TEST_SOURCES, test_file):
            print("ERROR: Failed to download NSL-KDD test set")
            return False
    else:
        print(f"Test set already exists: {test_file}")

    # Process
    print("Processing NSL-KDD data...")

    # Load data
    train_df = pd.read_csv(train_file, header=None, names=NSLKDD_COLUMNS)
    test_df = pd.read_csv(test_file, header=None, names=NSLKDD_COLUMNS)

    # Combine for processing
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Create binary label: normal=0, attack=1
    df["label"] = (df["class"] != "normal").astype(int)

    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes.astype(float)

    # Select features (exclude class, difficulty, label)
    feature_cols = [c for c in df.columns if c not in ["class", "difficulty", "label"]]

    # Convert to numpy
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Save as single file with label in last column
    data = np.column_stack([X, y])

    # Save
    save_path = output_dir / "nslkdd.npy"
    np.save(save_path, data)
    print(f"Saved: {save_path} (shape: {data.shape})")
    print(f"  Benign: {(y == 0).sum():,}, Attack: {(y == 1).sum():,}")

    return True


def process_unswnb15(output_dir: Path) -> bool:
    """Download and process UNSW-NB15 dataset."""
    print("\n=== UNSW-NB15 Dataset ===")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for name, url in UNSWNB15_SOURCES:
        filename = "train.csv" if "train" in name else "test.csv"
        filepath = raw_dir / filename

        if not filepath.exists():
            print(f"Downloading {filename}...")
            if not download_file(url, filepath):
                print(f"ERROR: Failed to download {filename}")
                return False
        else:
            print(f"File already exists: {filepath}")

        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"  Loaded {filename}: {len(df):,} rows")

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(df):,} rows")

    # Process
    print("Processing UNSW-NB15 data...")

    # Find label column (could be 'label', 'Label', 'attack_cat', etc.)
    label_col = None
    for candidate in ["label", "Label", "attack_cat", "attack"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        # Check for binary label column
        for col in df.columns:
            if df[col].nunique() <= 2 and set(df[col].unique()).issubset({0, 1, "0", "1", "normal", "attack"}):
                label_col = col
                break

    if label_col is None:
        print(f"WARNING: Could not find label column. Columns: {list(df.columns)}")
        # Try to use the last column
        label_col = df.columns[-1]
        print(f"Using last column as label: {label_col}")

    # Create binary label
    if df[label_col].dtype == object:
        # String labels
        df["binary_label"] = (~df[label_col].str.lower().isin(["normal", "benign", "0"])).astype(int)
    else:
        # Numeric labels
        df["binary_label"] = (df[label_col] != 0).astype(int)

    # Drop non-feature columns
    drop_cols = [label_col, "binary_label", "id", "attack_cat", "Attack", "attack"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Encode categorical columns
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes.astype(float)

    # Convert to numpy
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Save
    data = np.column_stack([X, y])
    save_path = output_dir / "unswnb15.npy"
    np.save(save_path, data)
    print(f"Saved: {save_path} (shape: {data.shape})")
    print(f"  Benign: {(y == 0).sum():,}, Attack: {(y == 1).sum():,}")

    return True


def process_cicids2018(output_dir: Path) -> bool:
    """Download and process CIC-IDS2018 dataset."""
    print("\n=== CIC-IDS2018 Dataset ===")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    filepath = raw_dir / "cicids2018.csv"

    if not filepath.exists():
        print("Downloading CIC-IDS2018 (this may take a while)...")
        if not download_file(CICIDS2018_SOURCES[0][1], filepath):
            print("ERROR: Failed to download CIC-IDS2018")
            return False
    else:
        print(f"File already exists: {filepath}")

    # Process
    print("Processing CIC-IDS2018 data...")

    # Read CSV (may be large)
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Find label column
    label_col = None
    for candidate in ["Label", "label", "class", "attack"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        print(f"WARNING: Could not find label column. Columns: {list(df.columns)}")
        label_col = df.columns[-1]
        print(f"Using last column as label: {label_col}")

    # Create binary label
    print(f"Label column: {label_col}")
    print(f"Label values: {df[label_col].value_counts().head(10)}")

    if df[label_col].dtype == object:
        # Case-insensitive matching for benign labels
        df["binary_label"] = (~df[label_col].str.strip().str.lower().isin(["benign", "normal", "0"])).astype(int)
    else:
        df["binary_label"] = (df[label_col] != 0).astype(int)

    # Drop non-feature columns
    drop_cols = [label_col, "binary_label", "Timestamp", "timestamp", "Flow ID", "Src IP", "Dst IP"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Handle categorical columns
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes.astype(float)

    # Convert to numpy
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Save
    data = np.column_stack([X, y])
    save_path = output_dir / "cicids2018.npy"
    np.save(save_path, data)
    print(f"Saved: {save_path} (shape: {data.shape})")
    print(f"  Benign: {(y == 0).sum():,}, Attack: {(y == 1).sum():,}")

    return True


def process_cidds(output_dir: Path) -> bool:
    """Download and process CIDDS dataset."""
    print("\n=== CIDDS Dataset ===")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    filepath = raw_dir / "cidds.csv"

    if not filepath.exists():
        print("Downloading CIDDS...")
        if not download_file(CIDDS_SOURCES[0][1], filepath):
            print("ERROR: Failed to download CIDDS")
            return False
    else:
        print(f"File already exists: {filepath}")

    # Process
    print("Processing CIDDS data...")

    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Find label column
    label_col = None
    for candidate in ["label", "Label", "class", "Class", "attack_type", "attackType"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        print(f"Columns: {list(df.columns)}")
        label_col = df.columns[-1]
        print(f"Using last column as label: {label_col}")

    print(f"Label column: {label_col}")
    print(f"Label values: {df[label_col].value_counts().head(10)}")

    # Create binary label
    # CIDDS labels are like "normal_ 32", "victim_  0", "attacker_  0"
    # normal_ prefix = benign, victim_/attacker_ prefix = attack
    if df[label_col].dtype == object:
        df["binary_label"] = (~df[label_col].str.strip().str.startswith("normal")).astype(int)
    else:
        df["binary_label"] = (df[label_col] != 0).astype(int)

    # Drop non-feature columns (IP addresses, timestamps, label)
    drop_cols = [label_col, "binary_label", "start_frame", "end_frame", "src_ip", "dst_ip"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Feature columns: {feature_cols}")

    # Handle categorical columns
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes.astype(float)

    # Convert to numpy
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int64)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Save
    data = np.column_stack([X, y])
    save_path = output_dir / "cidds.npy"
    np.save(save_path, data)
    print(f"Saved: {save_path} (shape: {data.shape})")
    print(f"  Benign: {(y == 0).sum():,}, Attack: {(y == 1).sum():,}")

    return True


# =============================================================================
# Main
# =============================================================================

DATASET_PROCESSORS: dict[str, Callable[[Path], bool]] = {
    "nslkdd": process_nslkdd,
    "unswnb15": process_unswnb15,
    "cicids2018": process_cicids2018,
    "cidds": process_cidds,
}


def main():
    parser = argparse.ArgumentParser(description="Download NIDS datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_PROCESSORS.keys()),
        help="Specific dataset to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASETS_DIR),
        help="Output directory",
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        print("\nError: Specify --dataset or --all")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(DATASET_PROCESSORS.keys()) if args.all else [args.dataset]

    results = {}
    for ds in datasets:
        processor = DATASET_PROCESSORS[ds]
        success = processor(output_dir)
        results[ds] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {ds}: {status}")

    # Check saved files
    print("\nSaved files:")
    for f in sorted(output_dir.glob("*.npy")):
        data = np.load(f)
        print(f"  {f.name}: shape={data.shape}")

    if all(results.values()):
        print("\nAll datasets downloaded successfully!")
        return 0
    else:
        print("\nSome datasets failed to download.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
