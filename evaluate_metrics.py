import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set project root for imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from models.pipeline import run_full_pipeline

def evaluate():
    print("Starting evaluation of each algorithm stage...")
    
    # Using the labeled unified dataset for accurate metrics
    # This file has 'is_fraud' ground truth
    input_file = "data/processed/bank_transaction_fraud_detection_unified.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading data from {input_file}...")
    full_df = pd.read_csv(input_file)
    
    # Proper 80/20 Train-Test Split for validation
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['is_fraud'])
    
    print(f"Using {len(train_df)} rows for training and {len(test_df)} for evaluation.")
    
    # Define features (simulating the ingestion process)
    from src.processing.preprocessor import prepare_features
    # Feature engineering for both, using shared encoders and scaler so train/test mappings remain aligned
    train_df, feature_cols, encoders, scaler = prepare_features(train_df, is_training=True)
    test_df, _, _, _ = prepare_features(test_df, encoders=encoders, scaler=scaler)
    
    df = test_df  # We evaluate metrics on the test set
    
    sequence_cols = ["transaction_amount"]
    user_col = "customer_id"

    # Run the full pipeline to get all stage outputs
    print("Running multi-stage pipeline with full training split...")
    results_df = run_full_pipeline(df, feature_cols, sequence_cols, user_col, train_df=train_df)
    
    y_true = results_df['is_fraud'].astype(int)
    metrics = []

    # 1. Isolation Forest / Logistic
    if 'is_suspicious' in results_df.columns:
        y_pred = results_df['is_suspicious'].astype(int)
        metrics.append({
            "Algorithm": "Isolation Forest + Logistic",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        })

    # 2. XGBoost
    if 'xgb_score' in results_df.columns:
        # Increase threshold to 0.95 for much higher precision and accuracy
        y_pred = (results_df['xgb_score'] > 0.95).astype(int)
        metrics.append({
            "Algorithm": "XGBoost",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        })

    # 3. SVM
    if 'svm_probability' in results_df.columns:
        # Using probability for better thresholding
        y_pred = (results_df['svm_probability'] > 0.7).astype(int)
        metrics.append({
            "Algorithm": "SVM",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        })

    # 4. Autoencoder anomaly detection
    if 'autoencoder_anomaly' in results_df.columns:
        y_pred = results_df['autoencoder_anomaly'].astype(int)
        metrics.append({
            "Algorithm": "Autoencoder",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        })

    # 5. Combined Ensemble (Double-Check)
    if 'ensemble_prediction' in results_df.columns:
        y_pred = results_df['ensemble_prediction'].astype(int)
        metrics.append({
            "Algorithm": "Combined Ensemble (Double-Check)",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        })

    # Summary
    metrics_df = pd.DataFrame(metrics)
    print("\nIndividual Algorithm Metrics:")
    print(metrics_df.to_string(index=False))
    
    # Save to CSV for later use
    os.makedirs("results/reports", exist_ok=True)
    metrics_df.to_csv("results/reports/algorithm_metrics.csv", index=False)
    print("\nMetrics saved to results/reports/algorithm_metrics.csv")

if __name__ == "__main__":
    evaluate()
