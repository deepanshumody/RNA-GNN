"""Tests for the imbalance-aware metrics helper.

Requires scikit-learn, so the module is skipped where it isn't installed
(e.g. the minimal CI image).
"""
import pytest

pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

import rna_metrics  # noqa: E402


def test_perfect_ranking():
    m = rna_metrics.binding_site_metrics([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9])
    assert m["roc_auc"] == 1.0
    assert m["n_pos"] == 1 and m["n_total"] == 4
    assert m["recall_at_youden"] == 1.0
    # The single positive is top-ranked, so it sits within the top 1/4.
    assert abs(m["enrichment_top_fraction"] - 0.25) < 1e-9
    assert abs(m["first_hit_fraction"] - 0.25) < 1e-9


def test_single_class_returns_nan():
    m = rna_metrics.binding_site_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["n_pos"] == 0
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])


def test_enrichment_is_pessimistic_on_ties():
    # The positive is tied with every negative; pessimistic tie-breaking ranks it
    # last, so all candidates must be scanned to recover it.
    m = rna_metrics.binding_site_metrics([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5])
    assert abs(m["enrichment_top_fraction"] - 1.0) < 1e-9


def test_format_metrics_runs():
    m = rna_metrics.binding_site_metrics([0, 1], [0.2, 0.8])
    assert "ROC-AUC" in rna_metrics.format_metrics(m)
