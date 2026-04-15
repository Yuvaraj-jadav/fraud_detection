import pandas as pd
import xgboost as xgb
import os
from pathlib import Path
from sklearn.metrics import classification_report

# Mapping for transaction_type
TYPE_MAP = {
    'debit': 0, 'credit': 1, 'transfer': 2, 'withdrawal': 3,
    'payment': 4, 'deposit': 5, 'bill payment': 4, 'others': 6
}

# Label Mapping
LABEL_MAP = {
    'Yes': 1, 'No': 0, 'True': 1, 'False': 0,
    True: 1, False: 0, '1': 1, '0': 0, 1: 1, 0: 0
}

def preprocess_dataset(path: Path) -> pd.DataFrame:
    """Load and preprocess a dataset for training/evaluation."""
    if not path.exists():
        print(f"Skipping {path.name}: not found.")
        return None
    
    print(f"Loading and preprocessing {path.name}...")
    df = pd.read_csv(path)
    
    if "is_fraud" not in df.columns:
        print(f"  Warning: 'is_fraud' column missing in {path.name}")
        return None

    # 1. Label Mapping
    df["is_fraud"] = df["is_fraud"].fillna(0).astype(str).str.strip().map(lambda x: LABEL_MAP.get(x, 0))
    
    # 2. Amount Feature
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors='coerce').fillna(0)
    
    # 3. Date Features (Hour)
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors='coerce')
        df["hour"] = df["transaction_date"].dt.hour.fillna(0)
    else:
        df["hour"] = 0
    
    # 4. Type Encoding
    if "transaction_type" in df.columns:
        df["type_code"] = df["transaction_type"].str.lower().str.strip().map(lambda x: TYPE_MAP.get(x, 6)).fillna(6)
    else:
        df["type_code"] = 6
        
    return df[["transaction_amount", "hour", "type_code", "is_fraud"]]

def train_global_model():
    processed_dir = Path("/home/yuvaraj_hadoop/fraud_detection/data/processed")
    models_dir = Path("/home/yuvaraj_hadoop/fraud_detection/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # User requested split
    train_files = [
        "bank_transaction_fraud_detection_unified.csv",
        "banking_transactions_usa_2023_2024_unified.csv",
        "financial_fraud_detection_dataset_unified.csv"
    ]
    
    test_files = [
        "transactions_data_unified.csv",
        "bank_transactions_data_2_unified.csv"
    ]

    # Load Training Data
    train_dfs = []
    for filename in train_files:
        df = preprocess_dataset(processed_dir / filename)
        if df is not None:
            train_dfs.append(df)
    
    if not train_dfs:
        print("No training data found.")
        return
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Total training samples: {len(train_df)}")

    # Load Test Data
    test_dfs = []
    for filename in test_files:
        df = preprocess_dataset(processed_dir / filename)
        if df is not None:
            test_dfs.append(df)
            
    if not test_dfs:
        print("No test data found. Training only...")
        test_df = None
    else:
        test_df = pd.concat(test_dfs, ignore_index=True)
        print(f"Total test samples: {len(test_df)}")

    # Class imbalance handling
    counts = train_df['is_fraud'].value_counts()
    print(f"Training Fraud distribution:\n{counts}")
    scale_pos_weight = (counts[0] / counts[1]) if counts[1] > 0 else 1.0

    X = train_df[["transaction_amount", "hour", "type_code"]]
    y = train_df["is_fraud"]

    # Split training data for internal validation
    from sklearn.model_selection import train_test_split
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training Global XGBoost Model on 80% of training data (scale_pos_weight={scale_pos_weight:.2f})...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train_sub, y_train_sub)

    # Internal Validation Report
    y_val_pred = model.predict(X_val)
    print("\n--- Internal Validation Report (20% of Training Data) ---")
    print(classification_report(y_val, y_val_pred))
    print("----------------------------------------------------------\n")

    # Evaluation
    if test_df is not None:
        X_test = test_df[["transaction_amount", "hour", "type_code"]]
        y_test = test_df["is_fraud"]
        y_pred = model.predict(X_test)
        print("\nModel Evaluation on User-Specified Test Set:")
        print(classification_report(y_test, y_pred))
    else:
        print("\nSkipping evaluation as no test data was loaded.")

    model_path = models_dir / "global_fraud_model.json"
    model.save_model(model_path)
    print(f"Global model saved to {model_path}")

if __name__ == "__main__":
    train_global_model()

