"""Tests for extract_window — the centered-window + padding logic.

This is the most correctness-critical transform: an off-by-one here silently
misaligns every sequence with its label. Covers the passthrough cases, exact
centering, both-end padding, and the invariant that output length == window.
"""
import pytest

from prepare_data import extract_window


def test_window_none_returns_raw_slice(toy_chrom):
    assert extract_window(toy_chrom, 40, 60, None) == toy_chrom[40:60]


def test_window_not_wider_than_region_returns_raw_slice(toy_chrom):
    # window <= region width -> original slice unchanged
    assert extract_window(toy_chrom, 40, 60, 10) == toy_chrom[40:60]
    assert extract_window(toy_chrom, 40, 60, 20) == toy_chrom[40:60]


def test_centered_window_no_padding(toy_chrom):
    # region [40,60) center=50; window 40 -> [30,70)
    out = extract_window(toy_chrom, 40, 60, 40)
    assert out == toy_chrom[30:70]
    assert len(out) == 40


def test_left_padding_at_chrom_start(toy_chrom):
    # region [0,16) center=8; window 64 -> new_start=-24 -> 24 'N' on the left.
    # This is the exact chromosome-start case discussed in the design.
    out = extract_window(toy_chrom, 0, 16, 64)
    assert len(out) == 64
    assert out[:24] == "N" * 24
    assert out[24:] == toy_chrom[0:40]


def test_right_padding_at_chrom_end(toy_chrom):
    # chrom length 200; region [190,198) center=194; window 40 -> end=214 -> 14 'N'
    out = extract_window(toy_chrom, 190, 198, 40)
    assert len(out) == 40
    assert out[-14:] == "N" * 14
    assert out[:26] == toy_chrom[174:200]


def test_padding_on_both_sides(toy_chrom):
    # window 256 around a region near the start: pad left and right of a 200bp chrom
    out = extract_window(toy_chrom, 90, 106, 256)
    assert len(out) == 256
    assert out.count("N") == 256 - 200  # all real bases retained, rest padded


@pytest.mark.parametrize("window", [32, 64, 100, 256, 512])
def test_output_length_always_equals_window(toy_chrom, window):
    out = extract_window(toy_chrom, 90, 106, window)
    assert len(out) == window
