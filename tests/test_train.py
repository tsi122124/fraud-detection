"""
Unit tests for src/train.py
Tests model training functions with synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import (
    train_logistic_regression,
    train_xgboost,
    load_model
)


# ── FIXTURES ─────────────────────────────────────────────────────
@pytest.fixture
def synthetic_data():
    """Create balanced synthetic classification data."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randint(0, 2, n).astype(float)
    })
    y = pd.Series((X['feature_1'] + X['feature_2'] > 0).astype(int))
    return X, y


# ── TESTS: train_logistic_regression ─────────────────────────────
class TestTrainLogisticRegression:

    def test_returns_fitted_model(self, synthetic_data):
        """Should return a fitted LogisticRegression model."""
        X, y = synthetic_data
        model = train_logistic_regression(X, y)
        assert hasattr(model, 'coef_')

    def test_can_predict(self, synthetic_data):
        """Fitted model should produce predictions."""
        X, y = synthetic_data
        model = train_logistic_regression(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_predictions_binary(self, synthetic_data):
        """Predictions should be 0 or 1 only."""
        X, y = synthetic_data
        model = train_logistic_regression(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, synthetic_data):
        """predict_proba should return (n_samples, 2)."""
        X, y = synthetic_data
        model = train_logistic_regression(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_proba_sums_to_one(self, synthetic_data):
        """Probabilities should sum to 1 for each sample."""
        X, y = synthetic_data
        model = train_logistic_regression(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_saves_model(self, synthetic_data):
        """Model should be saved when save_path is provided."""
        X, y = synthetic_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'lr_test.pkl')
            train_logistic_regression(X, y, save_path=save_path)
            assert os.path.exists(save_path)


# ── TESTS: train_xgboost ──────────────────────────────────────────
class TestTrainXGBoost:

    def test_returns_fitted_model(self, synthetic_data):
        """Should return a fitted XGBClassifier."""
        X, y = synthetic_data
        model = train_xgboost(X, y)
        assert hasattr(model, 'feature_importances_')

    def test_can_predict(self, synthetic_data):
        """Fitted model should produce predictions."""
        X, y = synthetic_data
        model = train_xgboost(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_predictions_binary(self, synthetic_data):
        """Predictions should be 0 or 1 only."""
        X, y = synthetic_data
        model = train_xgboost(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_feature_importances_length(self, synthetic_data):
        """Feature importances length should match number of features."""
        X, y = synthetic_data
        model = train_xgboost(X, y)
        assert len(model.feature_importances_) == X.shape[1]

    def test_feature_importances_non_negative(self, synthetic_data):
        """All feature importances should be non-negative."""
        X, y = synthetic_data
        model = train_xgboost(X, y)
        assert (model.feature_importances_ >= 0).all()

    def test_saves_model(self, synthetic_data):
        """Model should be saved when save_path is provided."""
        X, y = synthetic_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'xgb_test.pkl')
            train_xgboost(X, y, save_path=save_path)
            assert os.path.exists(save_path)

    def test_scale_pos_weight_accepted(self, synthetic_data):
        """Model should train without error when scale_pos_weight is set."""
        X, y = synthetic_data
        model = train_xgboost(X, y, scale_pos_weight=10.0)
        assert model is not None


# ── TESTS: load_model ─────────────────────────────────────────────
class TestLoadModel:

    def test_loads_saved_model(self, synthetic_data):
        """Should load a model that was previously saved."""
        X, y = synthetic_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model.pkl')
            original  = train_logistic_regression(X, y, save_path=save_path)
            loaded    = load_model(save_path)
            orig_preds   = original.predict(X)
            loaded_preds = loaded.predict(X)
            assert np.array_equal(orig_preds, loaded_preds)

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_model('/nonexistent/path/model.pkl')
