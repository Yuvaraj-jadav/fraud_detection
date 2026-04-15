import pandas as pd
import os
import re
from pathlib import Path

# Unified Target Schema with robust synonym mapping
TARGET_COLUMNS = {
    'customer_id': ['userid', 'senderaccount', 'accountnumber', 'customerid', 'custid', 'senderid', 'id', 'account', 'acctid', 'accountno'],
    'transaction_amount': ['transactionamount', 'amount', 'amt', 'value', 'price', 'transactionvalue', 'txnamt', 'txamt', 'amountusd'],
    'transaction_date': ['transactiondate', 'timestamp', 'date', 'time', 'datetime', 'txndate', 'eventtime', 'txtime', 'transactiontime'],
    'is_fraud': ['isfraud', 'fraudflag', 'fraudulent', 'class', 'label', 'fraud', 'fraudclass', 'isfraudulent'],
    'transaction_type': ['transactiontype', 'type', 'paymentmethod', 'txntype', 'category', 'method', 'txtype', 'transactioncategory']
}

def clean_name(name):
    """Removes all non-alphanumeric characters and converts to lowercase."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def detect_mapping(df_cols):
    """Dynamically detects mapping from input columns to unified schema."""
    mapping = {}
    original_cols = list(df_cols)
    cleaned_cols = [clean_name(c) for c in original_cols]

    for target, synonyms in TARGET_COLUMNS.items():
        for syn in synonyms:
            if syn in cleaned_cols:
                idx = cleaned_cols.index(syn)
                mapping[original_cols[idx]] = target
                break
    return mapping

def unify_data():
    raw_dir = Path("/home/yuvaraj_hadoop/fraud_detection/data/raw")
    processed_dir = Path("/home/yuvaraj_hadoop/fraud_detection/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(raw_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found recursively in {raw_dir}")
        return

    print(f"Found {len(csv_files)} files in data/raw/ subdirectories. Starting Aggressive Smart Unification...")

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            sample_df = pd.read_csv(csv_file, nrows=1, low_memory=False, on_bad_lines='skip')
        except Exception as e:
            print(f"  Error reading header: {e}")
            continue
            
        mapping = detect_mapping(sample_df.columns)
        
        if not mapping:
            print(f"  Warning: No recognizable columns found in {csv_file.name}. skipping.")
            continue
            
        print(f"  Mapped columns: {mapping}")
        
        chunk_size = 100000
        first_chunk = True
        output_file = processed_dir / f"{csv_file.stem.lower()}_unified.csv"
        
        try:
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False, on_bad_lines='skip'):
                chunk = chunk.rename(columns=mapping)
                if first_chunk:
                    chunk.to_csv(output_file, index=False)
                    first_chunk = False
                else:
                    chunk.to_csv(output_file, index=False, mode='a', header=False)
            print(f"  Successfully saved to {output_file.name}")
        except Exception as e:
            print(f"  Error processing content: {e}")

if __name__ == "__main__":
    unify_data()
