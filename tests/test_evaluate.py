"""
Unit tests for src/evaluate.py
Tests evaluation functions with synthetic models and data.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import compute_aucpr, compare_models
from src.train import train_logistic_regression


# ── FIXTURES ─────────────────────────────────────────────────────
@pytest.fixture
def fitted_lr():
    """Return a fitted Logistic Regression and test data."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n)
    })
    y = pd.Series((X['f1'] + X['f2'] > 0).astype(int))
    model = train_logistic_regression(X, y)
    return model, X, y


@pytest.fixture
def sample_results():
    """Return a list of model result dicts."""
    return [
        {'model': 'XGBoost',             'dataset': 'E-commerce',
         'AUC-PR': 0.61, 'F1-Score': 0.68, 'ROC-AUC': 0.77},
        {'model': 'Logistic Regression', 'dataset': 'E-commerce',
         'AUC-PR': 0.36, 'F1-Score': 0.30, 'ROC-AUC': 0.74},
        {'model': 'XGBoost',             'dataset': 'Credit Card',
         'AUC-PR': 0.84, 'F1-Score': 0.21, 'ROC-AUC': 0.98},
        {'model': 'Logistic Regression', 'dataset': 'Credit Card',
         'AUC-PR': 0.77, 'F1-Score': 0.11, 'ROC-AUC': 0.97},
    ]


# ── TESTS: compute_aucpr ──────────────────────────────────────────
class TestComputeAucPR:

    def test_returns_float(self, fitted_lr):
        """AUC-PR should be a float."""
        model, X, y = fitted_lr
        score = compute_aucpr(model, X, y)
        assert isinstance(score, float)

    def test_between_zero_and_one(self, fitted_lr):
        """AUC-PR should be between 0 and 1."""
        model, X, y = fitted_lr
        score = compute_aucpr(model, X, y)
        assert 0.0 <= score <= 1.0

    def test_better_than_random(self, fitted_lr):
        """AUC-PR should be above the random baseline."""
        model, X, y = fitted_lr
        score        = compute_aucpr(model, X, y)
        random_baseline = y.mean()
        assert score > random_baseline


# ── TESTS: compare_models ────────────────────────────────────────
class TestCompareModels:

    def test_returns_dataframe(self, sample_results):
        """Should return a pandas DataFrame."""
        result = compare_models(sample_results)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_aucpr(self, sample_results):
        """Results should be sorted by AUC-PR descending."""
        result = compare_models(sample_results)
        aucpr_vals = result['AUC-PR'].tolist()
        assert aucpr_vals == sorted(aucpr_vals, reverse=True)

    def test_correct_number_of_rows(self, sample_results):
        """DataFrame should have same number of rows as input."""
        result = compare_models(sample_results)
        assert len(result) == len(sample_results)

    def test_required_columns_present(self, sample_results):
        """DataFrame should contain all required columns."""
        result   = compare_models(sample_results)
        required = ['model', 'dataset', 'AUC-PR', 'F1-Score', 'ROC-AUC']
        for col in required:
            assert col in result.columns

    def test_best_model_is_first(self, sample_results):
        """First row should be the model with highest AUC-PR."""
        result = compare_models(sample_results)
        assert result.iloc[0]['AUC-PR'] == 0.84
