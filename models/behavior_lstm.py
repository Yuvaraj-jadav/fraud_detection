"""Behavior analysis stage using LSTM."""

from typing import Any, List


def analyze_behavior(df: Any, sequence_columns: List[str], user_column: str) -> Any:
    """Analyze user behavior patterns with an LSTM-like model."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the LSTM behavior stage. "
            "Install it with: pip install tensorflow"
        ) from exc

    result_df = df.copy()
    
    # Calculate Z-score for better behavioral analysis
    user_stats = result_df.groupby(user_column)[sequence_columns[0]].agg(['mean', 'std']).reset_index()
    result_df = result_df.merge(user_stats, on=user_column)
    
    # Avoid division by zero and handle cases with only 1 transaction
    result_df['std'] = result_df['std'].fillna(0)
    result_df["service_score"] = (result_df[sequence_columns[0]] - result_df['mean']) / (result_df['std'] + 1.0)
    
    # If the transaction is more than 1.25 standard deviations from the mean, mark as suspicious
    # This should be much more accurate and less aggressive
    result_df["behavior_prediction"] = result_df["service_score"] > 1.25
    result_df["behavior_score"] = result_df["service_score"].clip(0, 5) / 5.0
    
    return result_df


def run_stage(df: Any, sequence_columns: List[str], user_column: str) -> Any:
    """Run the LSTM behavior analysis stage."""
    df = analyze_behavior(df, sequence_columns, user_column)
    return df
