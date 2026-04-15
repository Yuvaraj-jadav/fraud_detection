KAFKA_CONFIG = {
    "bootstrap_servers": "127.0.0.1:9092,127.0.0.1:9093,127.0.0.1:9094",
    "topic": "transactions",
    "group_id": "fraud_detection_group",
}

SPARK_CONFIG = {
    "app_name": "FraudDetection",
    "master": "local[*]",
}

HIVE_CONFIG = {
    "database": "fraud_db",
    "table": "transactions",
}