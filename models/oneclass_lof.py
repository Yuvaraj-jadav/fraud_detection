"""Unsupervised anomaly detection stage using Local Outlier Factor and One-Class SVM."""

from typing import Any, List
import pandas as pd


def run_stage(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Run an unsupervised anomaly detection stage."""
    try:
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the LOF / One-Class SVM stage. "
            "Install it with: pip install scikit-learn"
        ) from exc

    result_df = df.copy()
    X = result_df[feature_columns].fillna(0).astype(float)

    if X.shape[0] < 5 or X.shape[1] == 0:
        result_df["lof_anomaly"] = 0
        result_df["lof_score"] = 0.0
        result_df["ocsvm_anomaly"] = 0
        result_df["ocsvm_score"] = 0.0
        result_df["anomaly_ensemble"] = 0
        return result_df

    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, X.shape[0] - 1), contamination=0.01)
        lof_pred = lof.fit_predict(X)
        result_df["lof_anomaly"] = (lof_pred == -1).astype(int)
        result_df["lof_score"] = (-lof.negative_outlier_factor_).astype(float)
    except Exception:
        result_df["lof_anomaly"] = 0
        result_df["lof_score"] = 0.0

    try:
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
        ocsvm.fit(X)
        ocsvm_pred = ocsvm.predict(X)
        result_df["ocsvm_anomaly"] = (ocsvm_pred == -1).astype(int)
        result_df["ocsvm_score"] = ocsvm.decision_function(X).astype(float)
    except Exception:
        result_df["ocsvm_anomaly"] = 0
        result_df["ocsvm_score"] = 0.0

    result_df["anomaly_ensemble"] = (
        (result_df["lof_anomaly"] == 1) | (result_df["ocsvm_anomaly"] == 1)
    ).astype(int)

    return result_df
