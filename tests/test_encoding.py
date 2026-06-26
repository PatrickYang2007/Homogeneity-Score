"""Tests for the DNA encoders in prepare_data.

A bug here silently corrupts every model input, so the mapping, the unknown/N
handling, and the output shapes are all pinned down.
"""
import numpy as np

from prepare_data import one_hot_encode, encode_indices


def test_one_hot_shape_and_mapping():
    oh = one_hot_encode(["ACGT"])
    assert oh.shape == (1, 4, 4)
    assert oh[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]  # A
    assert oh[0, 1].tolist() == [0.0, 1.0, 0.0, 0.0]  # C
    assert oh[0, 2].tolist() == [0.0, 0.0, 1.0, 0.0]  # G
    assert oh[0, 3].tolist() == [0.0, 0.0, 0.0, 1.0]  # T


def test_one_hot_n_is_all_zero():
    oh = one_hot_encode(["AN"])
    assert oh[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]   # A
    assert oh[0, 1].tolist() == [0.0, 0.0, 0.0, 0.0]   # N -> no information
    # any non-ACGT character is treated as unknown (all zero)
    assert one_hot_encode(["X"])[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_one_hot_batches():
    oh = one_hot_encode(["ACGT", "TGCA"])
    assert oh.shape == (2, 4, 4)
    assert oh[1, 0].tolist() == [0.0, 0.0, 0.0, 1.0]   # T


def test_encode_indices_values():
    arr = encode_indices(["ACGTN"])
    assert arr.shape == (1, 5)
    assert arr[0].tolist() == [0, 1, 2, 3, 4]           # A,C,G,T,N(other)=4


def test_encode_indices_is_case_insensitive():
    # encode_indices maps lowercase too (GenomicDataset upper-cases anyway).
    assert encode_indices(["acgt"])[0].tolist() == [0, 1, 2, 3]


def test_encode_indices_batch_shape():
    arr = encode_indices(["ACGT", "TGCA"])
    assert arr.shape == (2, 4)
    assert arr.dtype == np.int8
