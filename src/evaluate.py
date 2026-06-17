"""
evaluate.py
Reusable evaluation functions for fraud detection models.
Computes AUC-PR, F1-Score, ROC-AUC and confusion matrix.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, auc, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import CV_FOLDS, RANDOM_STATE, DATA_PROCESSED


def compute_aucpr(model, X, y):
    """
    Compute Precision-Recall AUC score.

    Parameters:
        model : fitted classifier with predict_proba
        X     : features
        y     : true labels

    Returns:
        float AUC-PR score
    """
    try:
        y_prob = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_prob)
        return auc(recall, precision)
    except Exception as e:
        print(f"ERROR computing AUC-PR: {e}")
        raise


def evaluate_model(model, X_test, y_test, model_name, dataset_name):
    """
    Full evaluation: AUC-PR, F1, ROC-AUC, confusion matrix.

    Parameters:
        model        : fitted classifier
        X_test       : test features
        y_test       : true test labels
        model_name   : string label for the model
        dataset_name : string label for the dataset

    Returns:
        dict with model, dataset, AUC-PR, F1-Score, ROC-AUC
    """
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        aucpr = compute_aucpr(model, X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'='*55}")
        print(f"  {model_name} — {dataset_name}")
        print(f"{'='*55}")
        print(f"  AUC-PR  : {aucpr:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC : {roc_auc:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=['Legitimate', 'Fraud']
        ))

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_title(
            f'Confusion Matrix\n{model_name} — {dataset_name}',
            fontweight='bold'
        )
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        save_name = (
            f"cm_{model_name.replace(' ', '_')}"
            f"_{dataset_name.replace(' ', '_')}.png"
        )
        plt.savefig(
            os.path.join(DATA_PROCESSED, save_name),
            dpi=150, bbox_inches='tight'
        )
        plt.show()

        return {
            'model': model_name,
            'dataset': dataset_name,
            'AUC-PR': round(aucpr, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        }
    except Exception as e:
        print(f"ERROR evaluating model: {e}")
        raise


def run_cross_validation(model, X, y, model_name, dataset_name):
    """
    Run Stratified K-Fold CV and report mean ± std.

    Parameters:
        model        : sklearn-compatible classifier
        X            : features
        y            : labels
        model_name   : string label
        dataset_name : string label

    Returns:
        dict of CV results
    """
    try:
        skf = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )
        cv_results = cross_validate(
            model, X, y,
            cv=skf,
            scoring=['f1', 'roc_auc'],
            n_jobs=-1
        )
        f1_mean = cv_results['test_f1'].mean()
        f1_std = cv_results['test_f1'].std()
        auc_mean = cv_results['test_roc_auc'].mean()
        auc_std = cv_results['test_roc_auc'].std()

        print(f"\n=== CV: {model_name} — {dataset_name} ===")
        print(f"  F1-Score : {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  ROC-AUC  : {auc_mean:.4f} ± {auc_std:.4f}")

        return {
            'model' : model_name,
            'dataset': dataset_name,
            'cv_f1_mean': round(f1_mean, 4),
            'cv_f1_std' : round(f1_std, 4),
            'cv_auc_mean': round(auc_mean, 4),
            'cv_auc_std': round(auc_std, 4)
        }
    except Exception as e:
        print(f"ERROR in cross-validation: {e}")
        raise


def compare_models(results_list):
    """
    Build and display side-by-side model comparison table.

    Parameters:
        results_list : list of dicts from evaluate_model()

    Returns:
        Sorted DataFrame
    """
    try:
        df = pd.DataFrame(results_list)
        df = df.sort_values('AUC-PR', ascending=False).reset_index(drop=True)
        print("\n" + "="*60)
        print("        FULL MODEL COMPARISON TABLE")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        return df
    except Exception as e:
        print(f"ERROR comparing models: {e}")
        raise
