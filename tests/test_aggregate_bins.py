"""Tests for the summed-bin logic: aggregate_scores and extract_bin."""
import pandas as pd
import pytest

from aggregate_bins import aggregate_scores, extract_bin


def test_aggregate_scores_sums_per_bin():
    # Two regions fall in bin 0 (centers 8, 24), one in bin 1 (center 308).
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [0, 16, 300],
        "end":   [16, 32, 316],
        "score": [0.5, 0.3, 0.2],
    })
    agg = aggregate_scores(df, window=256).sort_values("bin").reset_index(drop=True)

    assert agg["bin"].tolist() == [0, 1]
    b0, b1 = agg.iloc[0], agg.iloc[1]
    assert b0["score"] == pytest.approx(0.8)   # 0.5 + 0.3
    assert b0["n_regions"] == 2
    assert (b0["start"], b0["end"]) == (0, 256)
    assert b1["score"] == pytest.approx(0.2)
    assert b1["n_regions"] == 1
    assert (b1["start"], b1["end"]) == (256, 512)


def test_aggregate_scores_separates_chromosomes():
    df = pd.DataFrame({
        "chrom": ["chr1", "chr2"],
        "start": [0, 0],
        "end":   [16, 16],
        "score": [0.5, 0.9],
    })
    agg = aggregate_scores(df, window=256)
    # same bin id (0) but different chromosomes -> two separate rows
    assert len(agg) == 2
    assert set(agg["chrom"]) == {"chr1", "chr2"}


def test_aggregate_scores_does_not_mutate_input():
    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [16], "score": [0.5]})
    cols_before = list(df.columns)
    aggregate_scores(df, window=256)
    assert list(df.columns) == cols_before     # no 'bin' column leaked back


def test_extract_bin_right_padding():
    chrom = "A" * 100
    out = extract_bin(chrom, bin_start=0, window=256)
    assert len(out) == 256
    assert out[:100] == "A" * 100
    assert out[100:] == "N" * 156


def test_extract_bin_no_padding_in_middle():
    chrom = "ACGT" * 50  # length 200
    out = extract_bin(chrom, bin_start=40, window=40)
    assert out == chrom[40:80]
    assert len(out) == 40
