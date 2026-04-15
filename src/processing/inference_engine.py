import pandas as pd
import xgboost as xgb
import os
import re
from pathlib import Path
from datetime import datetime

# Mapping for transaction_type
TYPE_MAP = {
    'debit': 0, 'credit': 1, 'transfer': 2, 'withdrawal': 3,
    'payment': 4, 'deposit': 5, 'bill payment': 4, 'others': 6
}

# Unified Target Schema with robust synonym mapping (subset for inference)
TARGET_COLUMNS = {
    'transaction_amount': ['transactionamount', 'amount', 'amt', 'value', 'price', 'transactionvalue', 'txnamt', 'txamt', 'amountusd'],
    'transaction_date': ['transactiondate', 'timestamp', 'date', 'time', 'datetime', 'txndate', 'eventtime', 'txtime', 'transactiontime', 'timestamp'],
    'transaction_type': ['transactiontype', 'type', 'paymentmethod', 'txntype', 'category', 'method', 'txtype', 'transactioncategory']
}

class InferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the XGBoost model."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}. Run training first.")
            return
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")

    def _clean_name(self, name: str) -> str:
        """Normalize column name for mapping."""
        return re.sub(r'[^a-z0-9]', '', str(name).lower())

    def _map_record(self, record: dict) -> dict:
        """Map raw record fields to our unified target schema."""
        mapped = {}
        original_keys = list(record.keys())
        cleaned_keys = [self._clean_name(k) for k in original_keys]

        for target, synonyms in TARGET_COLUMNS.items():
            for syn in synonyms:
                if syn in cleaned_keys:
                    idx = cleaned_keys.index(syn)
                    mapped[target] = record[original_keys[idx]]
                    break
        return mapped

    def predict(self, record: dict) -> str:
        """Predict whether a transaction is Fraud or Genuine."""
        if self.model is None:
            return "Unknown (Model missing)"

        try:
            # 1. Map labels to standard schema
            mapped = self._map_record(record)
            
            # 2. Extract features
            # Amount
            amount = float(mapped.get('transaction_amount', 0))
            
            # Hour
            date_val = mapped.get('transaction_date')
            try:
                # Common formats: %Y-%m-%d %H:%M:%S or ISO
                dt = pd.to_datetime(date_val, errors='coerce')
                hour = dt.hour if not pd.isna(dt) else 0
            except:
                hour = 0
            
            # Type
            txn_type = str(mapped.get('transaction_type', 'others')).lower().strip()
            type_code = TYPE_MAP.get(txn_type, 6)
            
            # 3. Build feature array
            # Feature order: [transaction_amount, hour, type_code]
            features_df = pd.DataFrame([{
                "transaction_amount": amount,
                "hour": hour,
                "type_code": type_code
            }])
            
            # 4. Predict
            prediction = self.model.predict(features_df)[0]
            probability = self.model.predict_proba(features_df)[0][1]
            
            return "Fraud" if prediction == 1 else "Genuine", float(probability)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error", 0.0

if __name__ == "__main__":
    # Test with sample record
    engine = InferenceEngine("models/global_fraud_model.json")
    test_record = {
        "Amount": 5000,
        "TimeStamp": "2023-05-07 20:21:19",
        "TransactionType": "transfer"
    }
    result, prob = engine.predict(test_record)
    print(f"Prediction: {result} (Probability: {prob:.4f})")
