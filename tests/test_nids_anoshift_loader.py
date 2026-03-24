"""Tests for AnoShift path resolution and loading fallbacks."""

from pathlib import Path

import numpy as np
import pytest

from src.data import nids


def _save_npy(path: Path, rows: int, cols: int, label: int) -> None:
    x = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    y = np.full((rows, 1), label, dtype=np.int64)
    data = np.hstack([x, y])
    np.save(path, data)


def test_get_dataset_path_anoshift_prefers_single_file(tmp_path, monkeypatch):
    single = tmp_path / "anoshift.npy"
    _save_npy(single, rows=3, cols=4, label=0)

    monkeypatch.setattr(nids, "_anoshift_single_path", lambda: single)
    monkeypatch.setattr(nids, "_anoshift_year_npy_paths", lambda: [])

    assert nids.get_dataset_path("anoshift") == single


def test_get_dataset_path_anoshift_falls_back_to_year_files(tmp_path, monkeypatch):
    year = tmp_path / "year_2006.npy"
    _save_npy(year, rows=2, cols=3, label=1)

    monkeypatch.setattr(nids, "_anoshift_single_path", lambda: tmp_path / "anoshift.npy")
    monkeypatch.setattr(nids, "_anoshift_year_npy_paths", lambda: [year])

    assert nids.get_dataset_path("anoshift") == year


def test_load_dataset_anoshift_single_file(tmp_path, monkeypatch):
    single = tmp_path / "anoshift.npy"
    _save_npy(single, rows=5, cols=3, label=1)

    monkeypatch.setattr(nids, "_anoshift_single_path", lambda: single)
    monkeypatch.setattr(nids, "_anoshift_year_npy_paths", lambda: [])

    x, y = nids.load_dataset("anoshift")
    assert x.shape == (5, 3)
    assert y.shape == (5,)
    assert np.all(y == 1)


def test_load_dataset_anoshift_yearly_fallback(tmp_path, monkeypatch):
    y2006 = tmp_path / "year_2006.npy"
    y2007 = tmp_path / "year_2007.npy"
    _save_npy(y2006, rows=2, cols=2, label=0)
    _save_npy(y2007, rows=3, cols=2, label=1)

    monkeypatch.setattr(nids, "_anoshift_single_path", lambda: tmp_path / "anoshift.npy")
    monkeypatch.setattr(nids, "_anoshift_year_npy_paths", lambda: [y2006, y2007])

    x, y = nids.load_dataset("anoshift")
    assert x.shape == (5, 2)
    assert y.shape == (5,)
    assert np.array_equal(y, np.array([0, 0, 1, 1, 1], dtype=np.int64))


def test_load_dataset_anoshift_missing_raises_helpful_error(tmp_path, monkeypatch):
    anoshift_dir = tmp_path / "anoshift"
    anoshift_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(nids, "_anoshift_dir", lambda: anoshift_dir)
    monkeypatch.setattr(nids, "_anoshift_single_path", lambda: anoshift_dir / "anoshift.npy")
    monkeypatch.setattr(nids, "_anoshift_year_npy_paths", lambda: [])

    with pytest.raises(FileNotFoundError) as exc:
        nids.load_dataset("anoshift")

    msg = str(exc.value)
    assert "AnoShift dataset not found" in msg
    assert "datasets/download.py --dataset anoshift" in msg
