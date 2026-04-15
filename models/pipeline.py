"""Full fraud detection model pipeline."""

from typing import Any, List

from .behavior_lstm import run_stage as run_behavior_lstm
from .isolation_logistic import run_stage as run_isolation_logistic
from .xgb_svm import run_stage as run_xgb_svm
from .oneclass_lof import run_stage as run_oneclass_lof
from .autoencoder import run_stage as run_autoencoder

def run_full_pipeline(
    df: Any,
    raw_feature_columns: List[str],
    sequence_columns: List[str],
    user_column: str,
    train_df: Any = None,
) -> Any:
    """Run the full fraud detection pipeline.

    Pipeline stages:
    1. Isolation Forest + Logistic Regression
    2. XGBoost + SVM
    3. Local Outlier Factor / One-Class SVM
    4. Autoencoder anomaly detection
    5. LSTM behavioral analysis
    """
    # 1. Isolation Forest / Logistic Stage
    df = run_isolation_logistic(df, raw_feature_columns, train_df=train_df)

    # 2. XGBoost + SVM Stage
    df = run_xgb_svm(df, raw_feature_columns, train_df=train_df)

    # 3. Unsupervised anomaly detection stage
    df = run_oneclass_lof(df, raw_feature_columns, train_df=train_df)

    # 4. Autoencoder anomaly detection stage
    try:
        df = run_autoencoder(df, raw_feature_columns, train_df=train_df)
    except Exception:
        if "autoencoder_error" not in df.columns:
            df["autoencoder_error"] = 0.0
        if "autoencoder_anomaly" not in df.columns:
            df["autoencoder_anomaly"] = 0
        if "autoencoder_threshold" not in df.columns:
            df["autoencoder_threshold"] = 0.0

    # 5. Behavioral stage
    df = run_behavior_lstm(df, sequence_columns, user_column)

    # Final decision logic: combine high-confidence supervised predictions with anomaly signals.
    is_fraud_xgb = df.get("xgb_score", 0) > 0.95
    is_fraud_svm = df.get("svm_probability", 0) > 0.60
    is_unsupervised_anomaly = (
        (df.get("anomaly_ensemble", 0) == 1)
        | (df.get("autoencoder_anomaly", 0) == 1)
        | (df.get("is_suspicious", 0) == 1)
    )
    is_behavioral_alert = df.get("behavior_prediction", False)

    df["ensemble_prediction"] = (
        (is_fraud_xgb & is_fraud_svm)
        | (is_unsupervised_anomaly & is_behavioral_alert)
    ).astype(int)
    df["final_decision"] = df["ensemble_prediction"].apply(lambda x: "Fraud" if x else "Genuine")

    return df
