import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from models.isolation_logistic import run_stage as run_isolation_logistic
from models.xgb_svm import run_stage as run_xgb_svm
from models.behavior_lstm import run_stage as run_behavior_lstm
from models.oneclass_lof import run_stage as run_oneclass_lof
from models.autoencoder import run_stage as run_autoencoder
from src.processing.preprocessor import prepare_features

def get_metrics(y_true, y_pred):
    # Ensure y_true is a numeric pandas Series
    y_true = pd.to_numeric(pd.Series(y_true), errors='coerce').fillna(0).astype(int)
    # Ensure y_pred is a numeric pandas Series
    y_pred = pd.to_numeric(pd.Series(y_pred), errors='coerce').fillna(0).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return acc, rec

def compare_all():
    processed_dir = Path("/home/yuvaraj_hadoop/fraud_detection/data/processed")
    models_dir = Path("/home/yuvaraj_hadoop/fraud_detection/models")
    
    labeled_files = [
        "bank_transaction_fraud_detection_unified.csv",
        "banking_transactions_usa_unified.csv",
        "financial_fraud_detection_dataset_unified.csv"
    ]

    results = []
    
    # Load Global Model
    global_model = xgb.XGBClassifier()
    global_model_path = models_dir / "global_fraud_model.json"
    if global_model_path.exists():
        global_model.load_model(global_model_path)
    else:
        global_model = None

    for filename in labeled_files:
        path = processed_dir / filename
        if not path.exists():
            continue
            
        print(f"Evaluating algorithms on {filename}...")
        df = pd.read_csv(path, nrows=50000) # Sample 50k rows for faster evaluation
        
        # Clean labels
        label_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0, '1': 1, '0': 0, 1: 1, 0: 0}
        df["is_fraud"] = df["is_fraud"].fillna(0).astype(str).str.strip().map(lambda x: label_map.get(x, 0))
        y_true = df["is_fraud"].astype(int)

        # Feature engineering and scaling for consistent stage input
        train_df, feature_columns, encoders, scaler = prepare_features(df, is_training=True)
        df, _, _, _ = prepare_features(df, encoders=encoders, scaler=scaler)

        user_column = "customer_id"
        sequence_columns = ["transaction_amount"]

        # Stage 1: Isolation Forest
        iso_df = run_isolation_logistic(df, feature_columns, train_df=train_df)
        acc1, rec1 = get_metrics(y_true, iso_df["is_suspicious"])
        
        # Stage 2: XGBoost + SVM
        xgb_df = run_xgb_svm(iso_df, feature_columns, train_df=train_df)
        acc2, rec2 = get_metrics(y_true, xgb_df["svm_prediction"])
        
        # Stage 3: Unsupervised anomaly detection
        anomaly_df = run_oneclass_lof(xgb_df, feature_columns, train_df=train_df)
        acc_lof, rec_lof = get_metrics(y_true, anomaly_df["anomaly_ensemble"])

        # Stage 4: Autoencoder
        auto_df = run_autoencoder(anomaly_df, feature_columns, train_df=train_df)
        acc_auto, rec_auto = get_metrics(y_true, auto_df["autoencoder_anomaly"])

        # Stage 5: Behavior LSTM
        behavior_df = run_behavior_lstm(auto_df, sequence_columns, user_column)
        acc3, rec3 = get_metrics(y_true, behavior_df["behavior_prediction"])

        # Combined decision logic
        ensemble_prediction = (
            (behavior_df["behavior_prediction"] == 1)
            | (behavior_df["svm_prediction"] == 1)
            | (behavior_df["autoencoder_anomaly"] == 1)
            | (behavior_df["anomaly_ensemble"] == 1)
        ).astype(int)
        acc_e, rec_e = get_metrics(y_true, ensemble_prediction)
        
        # Global Model
        if global_model:
            df_g = df.copy()
            df_g["transaction_amount"] = pd.to_numeric(df_g["transaction_amount"], errors='coerce').fillna(0)
            if "transaction_date" in df_g.columns:
                df_g["transaction_date"] = pd.to_datetime(df_g["transaction_date"], errors='coerce')
                df_g["hour"] = df_g["transaction_date"].dt.hour.fillna(0)
            else:
                df_g["hour"] = 0
            type_map = {'debit': 0, 'credit': 1, 'transfer': 2, 'withdrawal': 3, 'payment': 4, 'deposit': 5, 'bill payment': 4, 'others': 6}
            if "transaction_type" in df_g.columns:
                df_g["type_code"] = df_g["transaction_type"].str.lower().str.strip().map(lambda x: type_map.get(x, 6)).fillna(6)
            else:
                df_g["type_code"] = 6
            X_g = df_g[["transaction_amount", "hour", "type_code"]]
            y_pred_g = global_model.predict(X_g)
            acc_g, rec_g = get_metrics(y_true, y_pred_g)
        else:
            acc_g, rec_g = 0, 0

        results.append({
            "Dataset": filename,
            "IsoForest Acc/Rec": f"{acc1:.2f}/{rec1:.2f}",
            "XGB+SVM Acc/Rec": f"{acc2:.2f}/{rec2:.2f}",
            "Behavior LSTM Acc/Rec": f"{acc3:.2f}/{rec3:.2f}",
            "Unsupervised Ensemble Acc/Rec": f"{acc_lof:.2f}/{rec_lof:.2f}",
            "Autoencoder Acc/Rec": f"{acc_auto:.2f}/{rec_auto:.2f}",
            "Combined Ensemble Acc/Rec": f"{acc_e:.2f}/{rec_e:.2f}",
            "Global Model Acc/Rec": f"{acc_g:.2f}/{rec_g:.2f}"
        })

    report_df = pd.DataFrame(results)
    print("\n=== Comprehensive Algorithm Comparison (Compare all at one) ===")
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    compare_all()
