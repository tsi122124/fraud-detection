"""
Unit tests for src/preprocessing.py
Tests core preprocessing functions with synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    engineer_features,
    encode_and_split,
    scale_features,
    apply_smote
)


# ── FIXTURES ─────────────────────────────────────────────────────
@pytest.fixture
def sample_fraud_df():
    """Create a minimal synthetic e-commerce fraud DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'user_id'       : range(n),
        'signup_time'   : pd.date_range('2015-01-01', periods=n, freq='h'),
        'purchase_time' : pd.date_range('2015-01-02', periods=n, freq='h'),
        'purchase_value': np.random.randint(10, 100, n),
        'device_id'     : [f'DEV{i}' for i in range(n)],
        'source'        : np.random.choice(['SEO', 'Ads', 'Direct'], n),
        'browser'       : np.random.choice(['Chrome', 'Firefox', 'Safari'], n),
        'sex'           : np.random.choice(['M', 'F'], n),
        'age'           : np.random.randint(18, 70, n),
        'ip_address'    : np.random.uniform(1e8, 4e9, n),
        'country'       : np.random.choice(['United States', 'India', 'China'], n),
        'class'         : np.random.choice([0, 1], n, p=[0.9, 0.1])
    })


@pytest.fixture
def sample_X_y():
    """Create minimal X, y for scaling and SMOTE tests."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n) * 100,
        'feature_3': np.random.randint(0, 2, n)
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.85, 0.15]))
    return X, y


# ── TESTS: engineer_features ─────────────────────────────────────
class TestEngineerFeatures:

    def test_creates_time_since_signup(self, sample_fraud_df):
        """time_since_signup column should be created."""
        result = engineer_features(sample_fraud_df)
        assert 'time_since_signup' in result.columns

    def test_time_since_signup_positive(self, sample_fraud_df):
        """All time_since_signup values should be positive."""
        result = engineer_features(sample_fraud_df)
        assert (result['time_since_signup'] >= 0).all()

    def test_creates_hour_of_day(self, sample_fraud_df):
        """hour_of_day column should be created."""
        result = engineer_features(sample_fraud_df)
        assert 'hour_of_day' in result.columns

    def test_hour_of_day_range(self, sample_fraud_df):
        """hour_of_day should be between 0 and 23."""
        result = engineer_features(sample_fraud_df)
        assert result['hour_of_day'].between(0, 23).all()

    def test_creates_day_of_week(self, sample_fraud_df):
        """day_of_week column should be created."""
        result = engineer_features(sample_fraud_df)
        assert 'day_of_week' in result.columns

    def test_day_of_week_range(self, sample_fraud_df):
        """day_of_week should be between 0 and 6."""
        result = engineer_features(sample_fraud_df)
        assert result['day_of_week'].between(0, 6).all()

    def test_creates_transaction_count(self, sample_fraud_df):
        """user_transaction_count should be created."""
        result = engineer_features(sample_fraud_df)
        assert 'user_transaction_count' in result.columns

    def test_creates_velocity(self, sample_fraud_df):
        """user_transaction_velocity should be created."""
        result = engineer_features(sample_fraud_df)
        assert 'user_transaction_velocity' in result.columns

    def test_row_count_preserved(self, sample_fraud_df):
        """Row count should not change after feature engineering."""
        result = engineer_features(sample_fraud_df)
        assert len(result) == len(sample_fraud_df)

    def test_no_nulls_in_new_features(self, sample_fraud_df):
        """Engineered features should have no null values."""
        result = engineer_features(sample_fraud_df)
        new_cols = [
            'time_since_signup', 'hour_of_day',
            'day_of_week', 'user_transaction_count',
            'user_transaction_velocity'
        ]
        assert result[new_cols].isnull().sum().sum() == 0


# ── TESTS: encode_and_split ───────────────────────────────────────
class TestEncodeAndSplit:

    def test_returns_four_objects(self, sample_fraud_df):
        """Should return X_train, X_test, y_train, y_test."""
        df = engineer_features(sample_fraud_df)
        result = encode_and_split(df, 'class')
        assert len(result) == 4

    def test_train_test_sizes(self, sample_fraud_df):
        """Test set should be approximately 20% of data."""
        df = engineer_features(sample_fraud_df)
        X_train, X_test, y_train, y_test = encode_and_split(df, 'class')
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_drop_cols_removed(self, sample_fraud_df):
        """Identifier columns should be dropped from features."""
        df = engineer_features(sample_fraud_df)
        X_train, X_test, _, _ = encode_and_split(df, 'class')
        for col in ['user_id', 'device_id', 'ip_address']:
            assert col not in X_train.columns
            assert col not in X_test.columns

    def test_target_not_in_features(self, sample_fraud_df):
        """Target column should not appear in X_train or X_test."""
        df = engineer_features(sample_fraud_df)
        X_train, X_test, _, _ = encode_and_split(df, 'class')
        assert 'class' not in X_train.columns
        assert 'class' not in X_test.columns

    def test_feature_columns_match(self, sample_fraud_df):
        """X_train and X_test should have identical columns."""
        df = engineer_features(sample_fraud_df)
        X_train, X_test, _, _ = encode_and_split(df, 'class')
        assert list(X_train.columns) == list(X_test.columns)


# ── TESTS: scale_features ─────────────────────────────────────────
class TestScaleFeatures:

    def test_returns_three_objects(self, sample_X_y):
        """Should return X_train, X_test, scaler."""
        X, y = sample_X_y
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        result = scale_features(X_train, X_test,
                                numerical_cols=['feature_1', 'feature_2'])
        assert len(result) == 3

    def test_train_mean_near_zero(self, sample_X_y):
        """Scaled training features should have mean near 0."""
        X, y = sample_X_y
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
        X_train_s, _, _ = scale_features(
            X_train, X_test, numerical_cols=['feature_1', 'feature_2']
        )
        assert abs(X_train_s['feature_1'].mean()) < 0.1
        assert abs(X_train_s['feature_2'].mean()) < 0.1

    def test_train_std_near_one(self, sample_X_y):
        """Scaled training features should have std near 1."""
        X, y = sample_X_y
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
        X_train_s, _, _ = scale_features(
            X_train, X_test, numerical_cols=['feature_1', 'feature_2']
        )
        assert abs(X_train_s['feature_1'].std() - 1.0) < 0.1

    def test_shape_preserved(self, sample_X_y):
        """Shape should not change after scaling."""
        X, y = sample_X_y
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
        X_train_s, X_test_s, _ = scale_features(
            X_train, X_test, numerical_cols=['feature_1', 'feature_2']
        )
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape  == X_test.shape


# ── TESTS: apply_smote ────────────────────────────────────────────
class TestApplySMOTE:

    def test_balances_classes(self, sample_X_y):
        """After SMOTE fraud rate should be close to 50%."""
        X, y = sample_X_y
        X_res, y_res = apply_smote(X, y)
        assert abs(y_res.mean() - 0.5) < 0.01

    def test_majority_class_unchanged(self, sample_X_y):
        """SMOTE should not reduce the majority class count."""
        X, y = sample_X_y
        majority_before = (y == 0).sum()
        _, y_res = apply_smote(X, y)
        majority_after = (y_res == 0).sum()
        assert majority_after == majority_before

    def test_output_shapes_match(self, sample_X_y):
        """X and y output shapes should be consistent."""
        X, y = sample_X_y
        X_res, y_res = apply_smote(X, y)
        assert len(X_res) == len(y_res)

    def test_increases_minority_class(self, sample_X_y):
        """Fraud count should increase after SMOTE."""
        X, y = sample_X_y
        fraud_before = (y == 1).sum()
        _, y_res = apply_smote(X, y)
        fraud_after = (y_res == 1).sum()
        assert fraud_after > fraud_before
