"""
train.py
Reusable model training functions.
All training wrapped in try/except with joblib saving.
"""

import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.config import (
    LOGISTIC_REGRESSION_PARAMS,
    XGBOOST_PARAMS,
    MODELS_DIR
)


def train_logistic_regression(X_train, y_train, save_path=None):
    """
    Train a Logistic Regression model and optionally save it.

    Parameters:
        X_train   : training features
        y_train   : training labels
        save_path : path to save model with joblib (optional)

    Returns:
        Fitted LogisticRegression model
    """
    try:
        print("Training Logistic Regression...")
        model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        model.fit(X_train, y_train)
        print("Logistic Regression training complete!")
        if save_path:
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")
        return model
    except Exception as e:
        print(f"ERROR training Logistic Regression: {e}")
        raise


def train_xgboost(
    X_train, y_train,
    X_val=None, y_val=None,
    scale_pos_weight=None,
    save_path=None
):
    """
    Train an XGBoost classifier and optionally save it.

    Parameters:
        X_train          : training features
        y_train          : training labels
        X_val            : validation features (optional)
        y_val            : validation labels (optional)
        scale_pos_weight : imbalance ratio for credit card (optional)
        save_path        : path to save model with joblib (optional)

    Returns:
        Fitted XGBClassifier model
    """
    try:
        print("Training XGBoost...")
        params = XGBOOST_PARAMS.copy()
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight
            print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

        model = xgb.XGBClassifier(**params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        print("XGBoost training complete!")
        if save_path:
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")
        return model
    except Exception as e:
        print(f"ERROR training XGBoost: {e}")
        raise


def load_model(path):
    """
    Load a saved model from disk.

    Parameters:
        path : path to the saved .pkl file

    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model not found at {path}")
        raise
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise
