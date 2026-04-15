"""Autoencoder-based anomaly detection stage for fraud detection."""

from typing import Any, List
import pandas as pd
import numpy as np


def run_stage(df: Any, feature_columns: List[str], train_df: Any = None) -> Any:
    """Run the autoencoder anomaly detection stage."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the autoencoder stage. "
            "Install it with: pip install tensorflow"
        ) from exc

    result_df = df.copy()
    X = result_df[feature_columns].fillna(0).astype(float)

    if X.shape[0] < 20 or X.shape[1] == 0:
        result_df["autoencoder_error"] = 0.0
        result_df["autoencoder_anomaly"] = 0
        result_df["autoencoder_threshold"] = 0.0
        return result_df

    n_features = X.shape[1]
    hidden_dim = max(8, n_features // 2)
    bottleneck_dim = max(4, n_features // 4)

    inputs = tf.keras.Input(shape=(n_features,))
    encoded = layers.Dense(hidden_dim, activation="relu")(inputs)
    encoded = layers.Dense(bottleneck_dim, activation="relu")(encoded)
    decoded = layers.Dense(hidden_dim, activation="relu")(encoded)
    outputs = layers.Dense(n_features, activation="linear")(decoded)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(X, X, epochs=20, batch_size=32, verbose=0)

    reconstruction = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.power(X - reconstruction, 2), axis=1)
    threshold = np.percentile(mse, 95)

    result_df["autoencoder_error"] = mse
    result_df["autoencoder_anomaly"] = (mse > threshold).astype(int)
    result_df["autoencoder_threshold"] = float(threshold)

    return result_df
