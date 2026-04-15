"""Isolation Forest and Logistic Regression model stage."""

from typing import Any, List

import pandas as pd


def detect_anomalies(df: Any, feature_columns: List[str]) -> Any:
    """Detect suspicious transactions using Isolation Forest."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the Isolation Forest stage. "
            "Install it with: pip install scikit-learn"
        ) from exc

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df[feature_columns])
    df = df.copy()
    df["anomaly_score"] = model.decision_function(df[feature_columns])
    df["is_suspicious"] = model.predict(df[feature_columns]) == -1
    return df


def calculate_fraud_probability(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Estimate fraud probability using Logistic Regression."""
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn and imbalanced-learn are required for the Logistic Regression stage."
        ) from exc

    # If an external training set is provided, use it instead
    if train_df is not None:
        X_train = train_df[feature_columns].copy()
        y_train = train_df["is_fraud"].astype(int)
    else:
        df_copy = df.copy()
        df_copy["is_suspicious"] = df_copy["is_suspicious"].astype(int)
        suspicious_df = df_copy[df_copy["is_suspicious"] == 1].copy()
        normal_df = df_copy[df_copy["is_suspicious"] == 0].copy()
        if suspicious_df.empty:
            df_copy["fraud_probability"] = 0.0
            return df_copy
        sample_size = min(max(len(suspicious_df) * 10, 1000), len(normal_df))
        normal_sample = normal_df.sample(n=sample_size, random_state=42)
        X_train = pd.concat([suspicious_df, normal_sample], ignore_index=True)[feature_columns].copy()
        y_train = pd.concat([suspicious_df, normal_sample], ignore_index=True)["is_suspicious"].astype(int)

    # SMOTE Balancing
    try:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train.fillna(0), y_train)
    except (ValueError, ValueError):
        X_train_res, y_train_res = X_train.fillna(0), y_train

    model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    model.fit(X_train_res, y_train_res)
    
    df_out = df.copy()
    df_out["fraud_probability"] = model.predict_proba(df_out[feature_columns].fillna(0))[:, 1]
    return df_out

def run_stage(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Run the Isolation Forest + Logistic Regression pipeline stage."""
    df = detect_anomalies(df, feature_columns)
    df = calculate_fraud_probability(df, feature_columns, train_df=train_df)
    return df

