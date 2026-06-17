"""
config.py
Centralized configuration for all paths and model parameters.
All notebooks and scripts import from here.
No hardcoded paths or parameters anywhere else in the project.
"""

import os

# ── BASE PATHS ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ── RAW DATA FILES ────────────────────────────────────────────────
FRAUD_DATA_PATH = os.path.join(DATA_RAW, "Fraud_Data.csv")
IP_COUNTRY_PATH = os.path.join(DATA_RAW, "IpAddress_to_Country.csv")
CREDITCARD_PATH = os.path.join(DATA_RAW, "creditcard.csv")


# ── PROCESSED DATA FILES ──────────────────────────────────────────
FRAUD_CLEANED_PATH = os.path.join(DATA_PROCESSED, "fraud_data_cleaned.csv")
FRAUD_FEATURES_PATH = os.path.join(DATA_PROCESSED, "fraud_data_features.csv")
X_TRAIN_SMOTE_PATH = os.path.join(DATA_PROCESSED, "X_train_smote.csv")
X_TEST_PATH = os.path.join(DATA_PROCESSED, "X_test.csv")
Y_TRAIN_SMOTE_PATH = os.path.join(DATA_PROCESSED, "y_train_smote.csv")
Y_TEST_PATH = os.path.join(DATA_PROCESSED, "y_test.csv")


# ── MODEL SAVE PATHS ──────────────────────────────────────────────
XGB_ECOM_PATH = os.path.join(MODELS_DIR, "xgb_ecom.pkl")
XGB_CC_PATH = os.path.join(MODELS_DIR, "xgb_cc.pkl")
LR_ECOM_PATH = os.path.join(MODELS_DIR, "lr_ecom.pkl")
LR_CC_PATH = os.path.join(MODELS_DIR, "lr_cc.pkl")
SCALER_CC_PATH = os.path.join(MODELS_DIR, "scaler_cc.pkl")


# ── PREPROCESSING PARAMETERS ─────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_ECOM = "class"
TARGET_CC = "Class"


# ── FEATURE ENGINEERING PARAMETERS ───────────────────────────────
CATEGORICAL_COLS = ["source", "browser", "sex", "country"]
NUMERICAL_COLS = [
    "purchase_value", "age", "time_since_signup",
    "hour_of_day", "day_of_week",
    "user_transaction_count", "user_transaction_velocity"
]
DROP_COLS = [
    "user_id", "device_id", "ip_address",
    "signup_time", "purchase_time"
]


# ── MODEL PARAMETERS ─────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ── CROSS-VALIDATION PARAMETERS ──────────────────────────────────
CV_FOLDS = 5

# ── SMOTE PARAMETERS ─────────────────────────────────────────────
SMOTE_RANDOM_STATE = RANDOM_STATE
