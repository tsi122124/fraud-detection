"""
preprocessing.py
Reusable preprocessing and feature engineering functions.
Each function has try/except error handling.
"""

import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    RANDOM_STATE, TEST_SIZE, CATEGORICAL_COLS,
    NUMERICAL_COLS, DROP_COLS, SMOTE_RANDOM_STATE
)


def map_ip_to_country(df, ip_df):
    """
    Map IP addresses to countries using range-based lookup.

    Parameters:
        df    : DataFrame with ip_address column (float)
        ip_df : DataFrame with lower_bound, upper_bound, country columns

    Returns:
        DataFrame with country column added
    """
    try:
        df_sorted = df.sort_values('ip_address').reset_index(drop=True)
        ip_sorted = ip_df.sort_values(
            'lower_bound_ip_address'
        ).reset_index(drop=True)

        merged = pd.merge_asof(
            df_sorted,
            ip_sorted[[
                'lower_bound_ip_address',
                'upper_bound_ip_address',
                'country'
            ]],
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        merged['country'] = merged.apply(
            lambda row: row['country']
            if pd.notna(row['upper_bound_ip_address'])
            and row['ip_address'] <= row['upper_bound_ip_address']
            else 'Unknown',
            axis=1
        )
        merged = merged.drop(
            columns=['lower_bound_ip_address', 'upper_bound_ip_address']
        )
        mapped = (merged['country'] != 'Unknown').sum()
        unmapped = (merged['country'] == 'Unknown').sum()
        print(f"IP mapping complete — {mapped:,} mapped | {unmapped:,} unknown")
        return merged
    except Exception as e:
        print(f"ERROR in IP to country mapping: {e}")
        raise


def engineer_features(df):
    """
    Create fraud-specific features from raw e-commerce data.

    Features created:
        time_since_signup       : seconds between signup and purchase
        hour_of_day             : hour of purchase (0-23)
        day_of_week             : day of purchase (0=Mon, 6=Sun)
        user_transaction_count  : total transactions per user_id
        user_transaction_velocity: transactions per hour per user

    Parameters:
        df : cleaned DataFrame with datetime signup_time, purchase_time

    Returns:
        DataFrame with new features added
    """
    try:
        df = df.copy()

        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds()

        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek

        df['user_transaction_count'] = (
            df.groupby('user_id')['user_id'].transform('count')
        )

        user_window = df.groupby('user_id').agg(
            first_purchase=('purchase_time', 'min'),
            last_purchase=('purchase_time', 'max'),
            transaction_count=('user_id', 'count')
            ).reset_index()

        user_window['window_hours'] = (
            (user_window['last_purchase'] - user_window['first_purchase'])
            .dt.total_seconds() / 3600
        ).clip(lower=1)

        user_window['user_transaction_velocity'] = (
            user_window['transaction_count'] / user_window['window_hours']
        )

        df = df.merge(
            user_window[['user_id', 'user_transaction_velocity']],
            on='user_id', how='left'
        )

        print(f"Feature engineering complete — shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR in feature engineering: {e}")
        raise


def encode_and_split(df, target_col):
    """
    One-hot encode categoricals, drop identifiers,
    and perform stratified train/test split.

    Parameters:
        df         : fully engineered DataFrame
        target_col : name of the target column

    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        df_model = df.drop(columns=cols_to_drop)

        cat_cols = [c for c in CATEGORICAL_COLS if c in df_model.columns]
        df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        print("Split complete:")
        print(f"  X_train: {X_train.shape} | fraud: {y_train.mean()*100:.2f}%")
        print(f"  X_test:  {X_test.shape}  | fraud: {y_test.mean()*100:.2f}%")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"ERROR in encode and split: {e}")
        raise


def scale_features(X_train, X_test, numerical_cols=None):
    """
    Fit StandardScaler on training set, transform both sets.
    Prevents data leakage by fitting on train only.

    Parameters:
        X_train       : training features DataFrame
        X_test        : test features DataFrame
        numerical_cols: columns to scale (defaults to config NUMERICAL_COLS)

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    try:
        if numerical_cols is None:
            numerical_cols = [
                c for c in NUMERICAL_COLS if c in X_train.columns
            ]
        scaler  = StandardScaler()
        X_train = X_train.copy()
        X_test  = X_test.copy()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])
        print(f"Scaling applied to {len(numerical_cols)} numerical features")
        return X_train, X_test, scaler
    except Exception as e:
        print(f"ERROR in feature scaling: {e}")
        raise


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to training set only to handle class imbalance.

    Parameters:
        X_train : training features
        y_train : training labels

    Returns:
        X_resampled, y_resampled
    """
    try:
        print(f"Before SMOTE — fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
        smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE  — fraud: {y_resampled.sum():,} ({y_resampled.mean()*100:.2f}%)")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"ERROR applying SMOTE: {e}")
        raise
