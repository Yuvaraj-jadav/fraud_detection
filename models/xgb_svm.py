"""XGBoost and SVM model stage."""

from typing import Any, List

import pandas as pd


def detect_patterns(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Use XGBoost to score transactions."""
    try:
        import xgboost as xgb
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "xgboost and imbalanced-learn are required for this stage."
        ) from exc

    result_df = df.copy()
    
    # Define training data
    if train_df is not None:
        X_train = train_df[feature_columns].copy().fillna(0)
        y_train = train_df["is_fraud"].astype(int)
    else:
        # Fallback to internal labels if no train_df
        result_df["fraud_label"] = (result_df.get("fraud_probability", 0) > 0.3).astype(int)
        if result_df["fraud_label"].nunique() < 2:
            result_df["xgb_score"] = 0.0
            return result_df
        X_train = result_df[feature_columns].copy().fillna(0)
        y_train = result_df["fraud_label"]

    # SMOTE Balancing: Lighter oversampling (20% fraud) to prevent over-sensitivity
    sm = SMOTE(sampling_strategy=0.2, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # 1. XGBoost
    model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    model.fit(X_train_res, y_train_res)
    result_df["xgb_score"] = model.predict_proba(result_df[feature_columns].fillna(0))[:, 1]
    
    return result_df


def classify_svm(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Use SVM for final fraud decision."""
    try:
        from sklearn.svm import SVC
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("scikit-learn and imbalanced-learn are required for the SVM stage.") from exc

    result_df = df.copy()
    
    if train_df is not None:
        X_train = train_df[feature_columns].copy().fillna(0)
        y_train = train_df["is_fraud"].astype(int)
    else:
        result_df["svm_label"] = (result_df.get("xgb_score", 0) > 0.3).astype(int)
        if result_df["svm_label"].nunique() < 2:
            result_df["svm_prediction"] = 0
            result_df["svm_probability"] = 0.0
            return result_df
        # Sampling for SVM as it is slow on large datasets
        samp = result_df.sample(n=min(len(result_df), 10000), random_state=42)
        X_train = samp[feature_columns].copy().fillna(0)
        y_train = samp["svm_label"]

    # SMOTE Balancing (using a subset if needed for SVM performance)
    sm = SMOTE(random_state=42)
    # limit to 10k rows for SVM speed if needed
    X_train_res, y_train_res = sm.fit_resample(X_train[:10000], y_train[:10000])

    clf = SVC(probability=True, random_state=42, class_weight='balanced')
    clf.fit(X_train_res, y_train_res)
    result_df["svm_prediction"] = clf.predict(result_df[feature_columns].fillna(0))
    result_df["svm_probability"] = clf.predict_proba(result_df[feature_columns].fillna(0))[:, 1]
    return result_df


def run_stage(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Run the XGBoost + SVM pipeline stage."""
    df = detect_patterns(df, feature_columns, train_df=train_df)
    df = classify_svm(df, feature_columns, train_df=train_df)
    return df
