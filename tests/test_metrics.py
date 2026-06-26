"""Tests for the hand-rolled metrics in eval_report.

These are implemented from scratch with numpy/scipy (no sklearn), so they're the
most likely place for a subtle formula bug. Checked against small examples with
known closed-form answers.
"""
import numpy as np
import pytest

from eval_report import auroc, pr_curve, roc_curve_np, regression_metrics


# ----------------------------- AUROC -----------------------------

def test_auroc_perfect_ranking():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])     # positives score highest
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_worst_ranking():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.4, 0.3, 0.2, 0.1])     # positives score lowest
    assert auroc(scores, labels) == pytest.approx(0.0)


def test_auroc_half():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.2, 0.3])     # 2 of 4 pos/neg pairs correct
    assert auroc(scores, labels) == pytest.approx(0.5)


def test_auroc_single_class_is_nan():
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([0, 0])))


# ----------------------------- PR curve -----------------------------

def test_pr_curve_perfect_average_precision():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    _, _, ap = pr_curve(scores, labels)
    assert ap == pytest.approx(1.0)


def test_pr_curve_recall_reaches_one():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.9, 0.1, 0.6])
    _, recall, _ = pr_curve(scores, labels)
    assert recall[-1] == pytest.approx(1.0)      # every positive eventually recalled


# ----------------------------- ROC curve -----------------------------

def test_roc_curve_endpoints():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    fpr, tpr = roc_curve_np(scores, labels)
    assert tpr[-1] == pytest.approx(1.0)
    assert fpr[-1] == pytest.approx(1.0)


# ----------------------------- regression -----------------------------

def test_regression_metrics_perfect():
    y = np.array([0.1, 0.5, 0.9, 0.3])
    m = regression_metrics(y.copy(), y.copy())
    assert m["pearson"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)


def test_regression_metrics_constant_offset():
    targets = np.array([0.1, 0.5, 0.9, 0.3])
    preds = targets + 0.1                          # perfectly correlated, biased
    m = regression_metrics(preds, targets)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.1)
    assert m["mae"] == pytest.approx(0.1)
    # R^2 = 1 - SS_res/SS_tot = 1 - 0.04/0.35
    assert m["r2"] == pytest.approx(1 - 0.04 / 0.35)
