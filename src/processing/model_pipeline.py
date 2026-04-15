import os
import sys
from pathlib import Path
from typing import List, Any
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from models import run_full_pipeline
from src.processing.preprocessor import prepare_features

BASE_DIR = Path(__file__).resolve().parent
processed_dir = (BASE_DIR / "../../data/processed").resolve()
models_dir = (BASE_DIR / "../../data/models").resolve()
if not models_dir.exists():
    models_dir = (BASE_DIR / "../../models").resolve()

# Unified Column Names
USER_COL = "customer_id"
AMOUNT_COL = "transaction_amount"
DATE_COL = "transaction_date"
LABEL_COL = "is_fraud"
TYPE_COL = "transaction_type"

def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, on_bad_lines="skip")

def run_global_inference(df: pd.DataFrame) -> pd.DataFrame:
    model_path = models_dir / "global_fraud_model.json"
    if not model_path.exists():
        print(f"Warning: Global model not found at {model_path}. Skipping global inference.")
        return df

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Feature Engineering (same as training)
    df = df.copy()
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors='coerce').fillna(0)
    
    if "transaction_date" in df.columns:
        df["temp_date"] = pd.to_datetime(df["transaction_date"], errors='coerce')
        df["hour"] = df["temp_date"].dt.hour.fillna(0)
    else:
        df["hour"] = 0
    
    type_map = {'debit': 0, 'credit': 1, 'transfer': 2, 'withdrawal': 3, 'payment': 4, 'deposit': 5, 'bill payment': 4, 'others': 6}
    if "transaction_type" in df.columns:
        df["type_code"] = df["transaction_type"].str.lower().str.strip().map(lambda x: type_map.get(x, 6)).fillna(6)
    else:
        df["type_code"] = 6

    X = df[["transaction_amount", "hour", "type_code"]]
    df["global_fraud_probability"] = model.predict_proba(X)[:, 1]
    df["global_fraud_prediction"] = model.predict(X)
    
    # Clean up temp columns
    if "temp_date" in df.columns:
        df.drop(columns=["temp_date"], inplace=True)
    
    return df

def run_pipeline() -> None:
    output_dir = processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use unified files
    csv_files = sorted(processed_dir.glob("*_unified.csv"))
    if not csv_files:
        print(f"No unified files found in {processed_dir}. Please run unify_columns.py first.")
        return

    eligible_results = []
    for csv_file in csv_files:
        if csv_file.name == "transactions_data_unified.csv":
            print(f"Skipping {csv_file.name} (too large for standard pipeline processing).")
            continue

        print(f"Processing dataset: {csv_file.name}")
        df = load_dataset(csv_file)

        # 1. Normalize labels for supervised stages
        if "is_fraud" in df.columns:
            label_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0, '1': 1, '0': 0, 1: 1, 0: 0}
            df["is_fraud"] = df["is_fraud"].fillna(0).astype(str).str.strip().map(lambda x: label_map.get(x, 0))

        # 2. Feature engineering and scaling
        df_processed, feature_columns, encoders, scaler = prepare_features(df, is_training=True)
        df_processed, _, _, _ = prepare_features(df, encoders=encoders, scaler=scaler)

        # 3. Run the full fraud detection pipeline
        df_processed = run_full_pipeline(
            df_processed,
            feature_columns,
            sequence_columns=["transaction_amount"],
            user_column=USER_COL,
            train_df=df_processed,
        )

        # 4. Add global model predictions if available
        df_processed = run_global_inference(df_processed)

        # 5. Save individual output
        output_path = output_dir / f"fraud_pipeline_output_{csv_file.stem.replace('_unified', '')}.csv"
        df_processed.to_csv(output_path, index=False)
        print(f"  Saved output for {csv_file.name} to {output_path}")

        if "is_fraud" in df.columns:
            # For metrics_report.py to find the prediction, we must rename it to one of its guesses
            # metrics_report.py uses detect_label_columns which looks for svm_prediction, etc.
            df_processed["svm_prediction"] = df_processed.get("global_fraud_prediction", 0)
            eligible_results.append(df_processed)

    if eligible_results:
        combined_df = pd.concat(eligible_results, ignore_index=True)
        combined_path = output_dir / "fraud_pipeline_output_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined output for labeled datasets to {combined_path}")

if __name__ == "__main__":
    run_pipeline()
