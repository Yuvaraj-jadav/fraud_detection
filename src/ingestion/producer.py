from kafka import KafkaProducer
from kafka.errors import KafkaError
import pandas as pd
import json
import time
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "test"

# User requested test files to stream
TEST_FILES = [
    "Transactions_data.csv",
    "Bank_transactions_data_2.csv"
]

producer = KafkaProducer(
    bootstrap_servers=['127.0.0.1:9092', '127.0.0.1:9093', '127.0.0.1:9094'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',              # wait for all replicas to acknowledge
    retries=3                # retry on transient failures
)

def on_error(e):
    print(f"Send failed: {e}")

try:
    for filename in TEST_FILES:
        csv_file = RAW_DIR / filename
        if not csv_file.exists():
            print(f"Skipping {filename}: Not found in {RAW_DIR}")
            continue
            
        print(f"Streaming dataset: {csv_file.name}")
        # Efficiently read in chunks if files are large
        for df_chunk in pd.read_csv(csv_file, chunksize=1000):
            df_chunk = df_chunk.where(pd.notnull(df_chunk), None)
            for record in df_chunk.to_dict('records'):
                record['source_file'] = csv_file.name
                future = producer.send('transactions', value=record)
                future.add_errback(on_error)
                print(f"Sent from {csv_file.name}: {record}")
                time.sleep(1)
finally:
    producer.flush()
    producer.close()