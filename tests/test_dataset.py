"""Tests for GenomicDataset: parquet -> (one-hot tensor, score) with correct
shape and N handling. Uses a tiny temp parquet, no real data.
"""
import pandas as pd
import pytest

from model import GenomicDataset


@pytest.fixture
def tiny_parquet(tmp_path):
    path = tmp_path / "tiny.parquet"
    pd.DataFrame({
        "sequence": ["ACGT", "NNNN", "acgt"],   # mixed case + all-N row
        "score": [0.5, 0.2, 0.7],
    }).to_parquet(path)
    return str(path)


def test_length_and_targets(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    assert len(ds) == 3
    _, y0 = ds[0]
    assert float(y0) == pytest.approx(0.5)


def test_item_shape_is_channels_first(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[0]
    assert tuple(x.shape) == (4, 4)              # (channels=4, length=4)


def test_base_one_hot_columns(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[0]                                  # "ACGT"
    assert x[:, 0].tolist() == [1, 0, 0, 0]       # A
    assert x[:, 3].tolist() == [0, 0, 0, 1]       # T


def test_all_n_row_is_zero(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[1]                                  # "NNNN"
    assert x.sum().item() == 0.0


def test_lowercase_is_upper_cased(tiny_parquet):
    ds = GenomicDataset(tiny_parquet)
    x, _ = ds[2]                                  # "acgt" -> treated as ACGT
    assert x[:, 0].tolist() == [1, 0, 0, 0]
