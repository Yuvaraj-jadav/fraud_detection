import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from models.isolation_logistic import run_stage as run_isolation_logistic
from models.xgb_svm import run_stage as run_xgb_svm
from models.behavior_lstm import run_stage as run_behavior_lstm

def evaluate(y_true, y_pred, name):
    # Ensure types are consistent
    y_true = pd.to_numeric(y_true, errors='coerce').fillna(0).astype(int)
    y_pred = pd.to_numeric(y_pred, errors='coerce').fillna(0).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    print(f"{name:25} | Accuracy: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")

def compare():
    processed_dir = Path("/home/yuvaraj_hadoop/fraud_detection/data/processed")
    target_file = processed_dir / "bank_transaction_fraud_detection_unified.csv"
    
    if not target_file.exists():
        print(f"Error: {target_file} not found.")
        return

    print(f"Loading {target_file.name} for algorithm comparison...")
    df = pd.read_csv(target_file, nrows=50000) # Use 50k rows for a robust but faster evaluation
    
    # 1. Clean labels
    label_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0, '1': 1, '0': 0, 1: 1, 0: 0}
    df["is_fraud"] = df["is_fraud"].fillna(0).astype(str).str.strip().map(lambda x: label_map.get(x, 0))
    y_true = df["is_fraud"]

    from src.processing.preprocessor import prepare_features

    # Feature engineering for a stronger comparison
    train_df, feature_columns, encoders, scaler = prepare_features(df, is_training=True)
    df, _, _, _ = prepare_features(df, encoders=encoders, scaler=scaler)
    user_column = "customer_id"
    sequence_columns = ["transaction_amount"]

    print("\nRunning algorithm stages individually...")

    # Stage 1: Isolation Forest + Logistic Regression
    print("Evaluating Isolation Forest + Logistic Regression...")
    iso_df = run_isolation_logistic(df, feature_columns, train_df=train_df)
    evaluate(y_true, iso_df["is_suspicious"], "Isolation Forest (+LR)")

    # Stage 2: XGBoost + SVM
    print("Evaluating XGBoost + SVM...")
    # This stage expects 'fraud_probability' if running locally, or it calculates its own if features allow
    xgb_df = run_xgb_svm(iso_df, feature_columns, train_df=train_df)
    evaluate(y_true, xgb_df["svm_prediction"], "XGBoost + SVM")

    # Stage 3: Behavior LSTM
    print("Evaluating Behavior LSTM...")
    behavior_df = run_behavior_lstm(xgb_df, sequence_columns, user_column)
    evaluate(y_true, behavior_df["behavior_prediction"], "Behavior LSTM")

if __name__ == "__main__":
    compare()
