import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

COLUMN_SYNONYMS = {
    "customer_id": ["userid", "senderaccount", "accountnumber", "customerid", "custid", "senderid", "id", "account", "acctid", "accountno"],
    "transaction_amount": ["transactionamount", "amount", "amt", "value", "price", "transactionvalue", "txnamt", "txamt", "amountusd"],
    "transaction_date": ["transactiondate", "timestamp", "date", "time", "datetime", "txndate", "eventtime", "txtime", "transactiontime"],
    "transaction_time": ["transactiontime", "timeoftransaction", "txntime", "timestamptime", "time"],
    "transaction_type": ["transactiontype", "type", "paymentmethod", "txntype", "category", "method", "txtype", "transactioncategory"],
    "is_fraud": ["isfraud", "fraudflag", "fraudulent", "class", "label", "fraud", "fraudclass", "isfraudulent"],
    "city": ["city", "location", "merchantcity", "billingcity", "transactioncity"],
    "country": ["country", "nation", "merchantcountry", "billingcountry", "transactioncountry"],
    "ip_address": ["ipaddress", "ip_address", "ip", "ipaddr", "sourceip", "destinationip"],
    "device_id": ["deviceid", "device_id", "device", "dev_id"],
    "device_type": ["devicetype", "device_type", "devicecategory", "terminaltype"],
    "merchant_id": ["merchantid", "merchant_id", "merchant", "merchantcode", "mid"],
    "merchant_name": ["merchantname", "merchant_name", "vendor", "vendorname"],
    "payment_method": ["paymentmethod", "payment_method", "method", "paymentmode"],
    "channel": ["channel", "channeltype", "transactionchannel", "txchannel"],
    "account_balance": ["accountbalance", "balance", "acctbal", "account_balance"],
    "transaction_status": ["transactionstatus", "status", "txstatus", "paymentstatus"],
    "age": ["age", "customerage", "custage", "ageyears"],
    "gender": ["gender", "customer_gender", "sex"],
    "merchant_category": ["merchantcategory", "category", "merchant_type", "industry"],
}

CATEGORICAL_COLUMNS = [
    "gender",
    "device_type",
    "channel",
    "payment_method",
    "transaction_type",
    "transaction_status",
    "city",
    "country",
    "merchant_category",
]

NUMERIC_COLUMNS = [
    "transaction_amount",
    "age",
    "account_balance",
    "day_of_week",
    "hour",
    "timedelta",
    "txn_freq",
    "city_change",
    "country_change",
    "fast_location_change",
    "device_change",
    "merchant_change",
    "customer_txn_count",
    "merchant_txn_count",
    "merchant_unique_customers",
    "customer_unique_merchants",
    "customer_device_count",
    "merchant_device_count",
    "ip_customer_count",
    "ip_transaction_count",
    "ip_device_count",
    "ip_last_octet",
    "ip_device_mismatch",
    "amount_delta",
    "amount_ratio",
    "large_amount_jump",
    "high_value_transaction",
    "is_weekend",
    "week_of_year",
    "is_night",
]


def clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_column_mapping(columns):
    mapping = {}
    cleaned_columns = [clean_name(col) for col in columns]
    for target, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in cleaned_columns:
                mapping[columns[cleaned_columns.index(synonym)]] = target
                break
    return mapping


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = detect_column_mapping(df.columns)
    if mapping:
        df = df.rename(columns=mapping)
    return df


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "transaction_time" in df.columns:
        df["transaction_datetime"] = pd.to_datetime(
            df["transaction_date"].astype(str).str.strip() + " " + df["transaction_time"].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        df["transaction_datetime"] = pd.to_datetime(df.get("transaction_date", pd.Series([])), errors="coerce")
    return df


def _safe_label_encode(series: pd.Series, encoder: LabelEncoder = None):
    series = series.fillna("missing").astype(str)
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series)
    else:
        known = np.array(encoder.classes_, dtype=object)
        unknown = "__unknown__"
        if unknown not in known:
            encoder.classes_ = np.append(encoder.classes_, unknown)
        transformed = [x if x in encoder.classes_ else unknown for x in series]
        encoded = encoder.transform(transformed)
    return encoded, encoder


def prepare_features(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    scaler: StandardScaler | None = None,
    is_training: bool = False,
):
    """Prepare features for fraud detection models."""
    df = standardize_columns(df.copy())
    df = _ensure_datetime(df)

    if "customer_id" in df.columns and "transaction_datetime" in df.columns:
        df = df.sort_values(["customer_id", "transaction_datetime"])
        df["timedelta"] = df.groupby("customer_id")["transaction_datetime"].diff().dt.total_seconds().fillna(0)
        df["txn_freq"] = df.groupby("customer_id").cumcount()
        if "city" in df.columns:
            df["prev_city"] = df.groupby("customer_id")["city"].shift(1)
        else:
            df["prev_city"] = None
        if "country" in df.columns:
            df["prev_country"] = df.groupby("customer_id")["country"].shift(1)
        else:
            df["prev_country"] = None
        if "device_id" in df.columns:
            df["prev_device_id"] = df.groupby("customer_id")["device_id"].shift(1)
        else:
            df["prev_device_id"] = None
        if "merchant_id" in df.columns:
            df["prev_merchant_id"] = df.groupby("customer_id")["merchant_id"].shift(1)
        else:
            df["prev_merchant_id"] = None
    else:
        df["timedelta"] = 0
        df["txn_freq"] = 0
        df["prev_city"] = None
        df["prev_country"] = None
        df["prev_device_id"] = None
        df["prev_merchant_id"] = None

    df["day_of_week"] = df["transaction_datetime"].dt.dayofweek.fillna(0)
    df["hour"] = df["transaction_datetime"].dt.hour.fillna(0)
    df["week_of_year"] = df["transaction_datetime"].dt.isocalendar().week.fillna(0).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)

    df["city_change"] = ((df.get("city") != df.get("prev_city")).astype(int)).fillna(0).astype(int)
    df["country_change"] = ((df.get("country") != df.get("prev_country")).astype(int)).fillna(0).astype(int)
    df["device_change"] = ((df.get("device_id") != df.get("prev_device_id")).astype(int)).fillna(0).astype(int)
    df["merchant_change"] = ((df.get("merchant_id") != df.get("prev_merchant_id")).astype(int)).fillna(0).astype(int)
    df["fast_location_change"] = ((df["timedelta"] < 3600) & (df["city_change"] == 1)).astype(int)

    df["is_weekend"] = df["transaction_datetime"].dt.weekday.isin([5, 6]).astype(int)
    df["amount_delta"] = (df["transaction_amount"] - df.groupby("customer_id")["transaction_amount"].shift(1)).fillna(0)
    df["amount_ratio"] = df["transaction_amount"] / (df.groupby("customer_id")["transaction_amount"].shift(1).replace(0, np.nan).fillna(1))
    df["amount_ratio"] = df["amount_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
    df["large_amount_jump"] = (df["amount_ratio"] > 3).astype(int)
    df["high_value_transaction"] = (df["transaction_amount"] > df["transaction_amount"].quantile(0.95)).astype(int)

    if "ip_address" in df.columns:
        df["ip_last_octet"] = (
            df["ip_address"].astype(str)
            .str.split(".")
            .str[-1]
            .replace("nan", "0")
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("0")
            .astype(int)
        )
        df["ip_transaction_count"] = df.groupby("ip_address")["transaction_amount"].transform("count").fillna(0)
        df["ip_customer_count"] = df.groupby("ip_address")["customer_id"].transform("nunique").fillna(0)
        df["ip_device_count"] = df.groupby("ip_address")["device_id"].transform("nunique").fillna(0)
        df["ip_device_mismatch"] = ((df["device_id"].notna()) & (df["ip_address"].notna()) & (df["ip_customer_count"] > 1)).astype(int)
    else:
        df["ip_last_octet"] = 0
        df["ip_transaction_count"] = 0
        df["ip_customer_count"] = 0
        df["ip_device_count"] = 0
        df["ip_device_mismatch"] = 0

    if "merchant_id" in df.columns:
        df["merchant_txn_count"] = df.groupby("merchant_id")["transaction_amount"].transform("count").fillna(0)
        df["merchant_unique_customers"] = df.groupby("merchant_id")["customer_id"].transform("nunique").fillna(0)
        if "device_id" in df.columns:
            df["merchant_device_count"] = df.groupby("merchant_id")["device_id"].transform("nunique").fillna(0)
        else:
            df["merchant_device_count"] = 0
    else:
        df["merchant_txn_count"] = 0
        df["merchant_unique_customers"] = 0
        df["merchant_device_count"] = 0

    if "customer_id" in df.columns:
        if "merchant_id" in df.columns:
            df["customer_unique_merchants"] = df.groupby("customer_id")["merchant_id"].transform("nunique").fillna(0)
        else:
            df["customer_unique_merchants"] = 0
        if "device_id" in df.columns:
            df["customer_device_count"] = df.groupby("customer_id")["device_id"].transform("nunique").fillna(0)
        else:
            df["customer_device_count"] = 0
        df["customer_txn_count"] = df.groupby("customer_id")["transaction_amount"].transform("count").fillna(0)
    else:
        df["customer_unique_merchants"] = 0
        df["customer_device_count"] = 0
        df["customer_txn_count"] = 0

    if "transaction_amount" in df.columns:
        df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0)
    if "account_balance" in df.columns:
        df["account_balance"] = pd.to_numeric(df["account_balance"], errors="coerce").fillna(0)

    # Categorical encoding
    encoders = {} if encoders is None else encoders
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            encoded, encoders[col] = _safe_label_encode(df[col], encoders.get(col) if not is_training else None)
            df[col] = encoded

    for col in ["transaction_type", "transaction_status", "payment_method", "channel"]:
        if col in df.columns:
            encoded, encoders[col] = _safe_label_encode(df[col], encoders.get(col) if not is_training else None)
            df[col] = encoded

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    all_feature_columns = [col for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS if col in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        if all_feature_columns:
            df[all_feature_columns] = scaler.fit_transform(df[all_feature_columns])
    else:
        if all_feature_columns:
            df[all_feature_columns] = scaler.transform(df[all_feature_columns])

    return df, all_feature_columns, encoders, scaler


def apply_smote(X, y):
    """Apply SMOTE to balance the dataset."""
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
