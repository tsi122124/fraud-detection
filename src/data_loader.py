"""
data_loader.py
Reusable functions for loading raw and processed datasets.
All file I/O wrapped in try/except for robustness.
"""

import pandas as pd
import os
import sys
from src.config import (
    FRAUD_DATA_PATH, IP_COUNTRY_PATH, CREDITCARD_PATH,
    X_TRAIN_SMOTE_PATH, X_TEST_PATH,
    Y_TRAIN_SMOTE_PATH, Y_TEST_PATH
                       )

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_fraud_data():
    """Load and parse raw e-commerce fraud dataset."""
    try:
        df = pd.read_csv(FRAUD_DATA_PATH)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        print(f"Loaded Fraud_Data.csv — shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {FRAUD_DATA_PATH}")
        raise
    except Exception as e:
        print(f"ERROR loading fraud data: {e}")
        raise


def load_ip_country():
    """Load IP address to country mapping dataset."""
    try:
        df = pd.read_csv(IP_COUNTRY_PATH)
        print(f"Loaded IpAddress_to_Country.csv — shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {IP_COUNTRY_PATH}")
        raise
    except Exception as e:
        print(f"ERROR loading IP country data: {e}")
        raise


def load_creditcard_data():
    """Load raw credit card fraud dataset."""
    try:
        df = pd.read_csv(CREDITCARD_PATH)
        print(f"Loaded creditcard.csv — shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {CREDITCARD_PATH}")
        raise
    except Exception as e:
        print(f"ERROR loading credit card data: {e}")
        raise


def load_ecom_train_test():
    """Load preprocessed e-commerce SMOTE train/test splits."""
    try:
        X_train = pd.read_csv(X_TRAIN_SMOTE_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_train = pd.read_csv(Y_TRAIN_SMOTE_PATH).squeeze()
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()
        print(f"E-commerce splits loaded — X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"X_train: {X_train.shape} | fraud: {y_train.mean()*100:.1f}%")
        print(f"X_test:  {X_test.shape}  | fraud: {y_test.mean()*100:.1f}%")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run feature-engineering.ipynb first.")
        raise
    except Exception as e:
        print(f"ERROR loading e-commerce splits: {e}")
        raise
