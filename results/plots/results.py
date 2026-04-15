import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

ACTUAL_LABEL_CANDIDATES = [
    "Fraud_Flag",
    "fraud_flag",
    "FraudFlag",
    "is_fraud",
    "Is_Fraud",
    "label",
    "actual_label",
    "is_fraudulent",
    "fraudulent",
]
PREDICTION_LABEL_CANDIDATES = [
    "svm_prediction",
    "behavior_prediction",
    "fraud_label",
    "final_decision",
    "is_suspicious",
    "prediction",
    "predicted_label",
    "is_fraud_predicted",
]
PROBABILITY_CANDIDATES = [
    "fraud_probability",
    "xgb_score",
    "svm_probability",
    "behavior_score",
]
DEFAULT_THRESHOLD = 0.5


def _normalize_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float)):
        return int(bool(value))
    lower = str(value).strip().lower()
    if lower in {"1", "true", "yes", "fraud", "fraudulent", "y", "t"}:
        return 1
    if lower in {"0", "false", "no", "genuine", "non-fraud", "non fraud", "n", "f"}:
        return 0
    try:
        return int(float(lower))
    except (ValueError, TypeError):
        return None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in lower_map:
            return lower_map[normalized]
    return None


def _get_actual_label_columns(df: pd.DataFrame) -> list[str]:
    actual_columns = []
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in ACTUAL_LABEL_CANDIDATES:
        if candidate in df.columns:
            actual_columns.append(candidate)
        elif candidate.lower() in lower_map:
            actual_columns.append(lower_map[candidate.lower()])
    return [] if not actual_columns else list(dict.fromkeys(actual_columns))


def _get_actual_labels(df: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    actual_columns = _get_actual_label_columns(df)
    if not actual_columns:
        return None, []

    combined = pd.Series(index=df.index, dtype="float64")
    for col in actual_columns:
        normalized = df[col].map(_normalize_label)
        combined = combined.combine_first(normalized)

    combined = combined[combined.notna()].astype(int)
    return combined, actual_columns


def _get_predicted_labels(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    prediction_col = _find_column(df, PREDICTION_LABEL_CANDIDATES)
    if prediction_col is not None:
        normalized = df[prediction_col].map(_normalize_label)
        non_na = normalized[normalized.notna()]
        if not non_na.empty:
            return non_na, prediction_col

    for probability_col in PROBABILITY_CANDIDATES:
        if probability_col in df.columns:
            probabilities = df[probability_col]
            if probabilities.dropna().empty:
                continue
            predicted = (probabilities.astype(float) > DEFAULT_THRESHOLD).astype(int)
            return predicted[predicted.notna()], probability_col

    return None, None


def calculate_classification_metrics(df: pd.DataFrame) -> dict[str, float | int | str] | dict:
    actual_labels, actual_columns = _get_actual_labels(df)
    predicted_labels, prediction_source = _get_predicted_labels(df)
    if actual_labels is None or predicted_labels is None:
        return {}

    aligned_index = actual_labels.index.intersection(predicted_labels.index)
    if aligned_index.empty:
        return {}

    y_true = actual_labels.loc[aligned_index]
    y_pred = predicted_labels.loc[aligned_index]
    if y_true.empty or y_pred.empty:
        return {}

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "source_file": "all",
        "actual_label_column": ", ".join(actual_columns) if actual_columns else "",
        "predicted_label_source": prediction_source or "",
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1_score": float(round(f1_score, 4)),
        "accuracy": float(round(accuracy, 4)),
        "support": int(len(y_true)),
        "fraud_support": int(int(y_true.sum())),
    }


def detect_label_columns(df: pd.DataFrame) -> dict[str, object]:
    actual_col = _find_column(df, ACTUAL_LABEL_CANDIDATES)
    prediction_col = _find_column(df, PREDICTION_LABEL_CANDIDATES)
    probability_col = next((col for col in PROBABILITY_CANDIDATES if col in df.columns), None)
    candidate_actuals = [col for col in ACTUAL_LABEL_CANDIDATES if col in df.columns]
    candidate_predictions = [col for col in PREDICTION_LABEL_CANDIDATES if col in df.columns]
    if probability_col and probability_col not in candidate_predictions:
        candidate_predictions.append(probability_col)

    return {
        "actual_label_column": actual_col,
        "prediction_source": prediction_col or probability_col,
        "candidate_actuals": candidate_actuals,
        "candidate_predictions": candidate_predictions,
        "has_actual": actual_col is not None,
        "has_prediction": (prediction_col is not None or probability_col is not None),
    }


def compute_metrics_by_group(df: pd.DataFrame, group_column: str = "source_file") -> pd.DataFrame:
    metrics = []
    overall = calculate_classification_metrics(df)
    if overall:
        metrics.append(overall)
    if group_column in df.columns:
        grouped = df.groupby(group_column)
        for source, subset in grouped:
            group_metrics = calculate_classification_metrics(subset)
            if group_metrics:
                group_metrics["source_file"] = source
                metrics.append(group_metrics)
    return pd.DataFrame(metrics)


def plot_roc_curves(df: pd.DataFrame, save_path: str = None):
    """
    Plot ROC curves for all algorithms in the dataframe.

    Args:
        df: DataFrame with fraud probabilities and true labels
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))

    # Define algorithm mappings
    algorithms = {
        'fraud_probability': 'Ensemble',
        'xgb_score': 'XGBoost',
        'svm_probability': 'SVM',
        'behavior_score': 'Isolation Forest + Logistic'
    }

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    auc_scores = {}

    # Use true fraud labels
    true_labels = df['is_fraud'] if 'is_fraud' in df.columns else df['is_suspicious'].astype(int)

    for i, (prob_col, name) in enumerate(algorithms.items()):
        if prob_col in df.columns:
            # Get probabilities and true labels (drop NaN)
            probs = df[prob_col].dropna()
            labels = true_labels.loc[probs.index].dropna()

            if len(labels) > 0 and len(probs) > 0:
                # Ensure same length
                min_len = min(len(probs), len(labels))
                probs = probs.iloc[:min_len]
                labels = labels.iloc[:min_len]

                # Check if we have both classes
                if len(labels.unique()) > 1:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(labels, probs)
                    roc_auc = auc(fpr, tpr)

                    # Plot
                    plt.plot(fpr, tpr, color=colors[i], lw=2,
                            label=f'{name} (AUC = {roc_auc:.3f})')

                    auc_scores[name] = roc_auc
                else:
                    print(f"Warning: {name} has only one class in labels, skipping ROC curve")

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Fraud Detection Algorithms\n(Using Suspicious Transactions as Positive Class)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")

    plt.show()

    # Print AUC scores
    print("\nAUC Scores:")
    for name, score in auc_scores.items():
        print(f"{name}: {score:.4f}")


def plot_accuracy_comparison(algorithms: list, accuracies: list, save_path: str = None):
    """
    Plot accuracy comparison bar chart for all algorithms.

    Args:
        algorithms: List of algorithm names
        accuracies: List of accuracy scores
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))

    # Create bars
    bars = plt.bar(algorithms, accuracies, color=['blue', 'red', 'green', 'orange', 'purple'])

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Algorithms', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison - Fraud Detection Algorithms', fontsize=14, fontweight='bold')
    plt.ylim(0.85, 0.96)  # Adjust based on your data range
    plt.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison saved to: {save_path}")

    plt.show()


def plot_all_metrics(df: pd.DataFrame = None, save_dir: str = None):
    """
    Plot both ROC curves and accuracy comparison.

    Args:
        df: DataFrame with fraud data (optional - will load if not provided)
        save_dir: Directory to save plots (optional)
    """
    if df is None:
        # Load predictions
        pred_df = pd.read_csv('/home/yuvaraj_hadoop/fraud_detection/data/processed/fraud_pipeline_output.csv')

        # Load true labels from unified data
        true_df = pd.read_csv('/home/yuvaraj_hadoop/fraud_detection/data/processed/bank_transaction_fraud_detection_unified.csv')

        # Convert TransactionID to string for merging
        pred_df['TransactionID'] = pred_df['TransactionID'].astype(str)
        true_df['Transaction_ID'] = true_df['Transaction_ID'].astype(str)

        # Merge on TransactionID to get true labels
        df = pred_df.merge(true_df[['Transaction_ID', 'is_fraud']], left_on='TransactionID', right_on='Transaction_ID', how='left')

    # Plot ROC curves
    roc_path = f"{save_dir}/roc_curves.png" if save_dir else None
    plot_roc_curves(df, roc_path)

    # Calculate accuracies for comparison
    algorithms = []
    accuracies = []

    # Get accuracy from algorithm_metrics.csv if available
    try:
        metrics_df = pd.read_csv('/home/yuvaraj_hadoop/fraud_detection/results/reports/algorithm_metrics.csv')
        algorithms = metrics_df['Algorithm'].tolist()
        accuracies = metrics_df['Accuracy'].tolist()
    except:
        # Fallback: calculate from data
        true_labels = df['is_fraud']

        alg_configs = [
            ('fraud_probability', 'Ensemble'),
            ('xgb_score', 'XGBoost'),
            ('svm_probability', 'SVM'),
            ('behavior_score', 'Isolation Forest + Logistic')
        ]

        for prob_col, name in alg_configs:
            if prob_col in df.columns and not df[prob_col].isna().all():
                probs = df[prob_col].dropna()
                labels = true_labels.loc[probs.index].dropna()

                if len(labels) > 0:
                    min_len = min(len(probs), len(labels))
                    probs = probs.iloc[:min_len]
                    labels = labels.iloc[:min_len]

                    pred = (probs > 0.5).astype(int)
                    acc = accuracy_score(labels, pred)

                    algorithms.append(name)
                    accuracies.append(acc)

    # Plot accuracy comparison
    acc_path = f"{save_dir}/accuracy_comparison.png" if save_dir else None
    plot_accuracy_comparison(algorithms, accuracies, acc_path)


def plot_algorithm_comparison():
    """
    Plot comprehensive algorithm comparison using available metrics.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Load algorithm metrics
    metrics_df = pd.read_csv('/home/yuvaraj_hadoop/fraud_detection/results/reports/algorithm_metrics.csv')

    # Set up the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fraud Detection Algorithms - Performance Comparison', fontsize=16, fontweight='bold')

    algorithms = metrics_df['Algorithm'].tolist()
    accuracies = metrics_df['Accuracy'].tolist()
    precisions = metrics_df['Precision'].tolist()
    recalls = metrics_df['Recall'].tolist()
    f1_scores = metrics_df['F1-Score'].tolist()

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Accuracy Bar Chart
    bars1 = ax1.bar(algorithms, accuracies, color=colors, alpha=0.7)
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.85, 0.96)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    # 2. Precision vs Recall Scatter
    scatter = ax2.scatter(precisions, recalls, s=200, c=colors, alpha=0.7)
    ax2.set_title('Precision vs Recall', fontweight='bold')
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.grid(True, alpha=0.3)

    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax2.annotate(alg.split()[0], (precisions[i], recalls[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 3. F1-Score Bar Chart
    bars3 = ax3.bar(algorithms, f1_scores, color=colors, alpha=0.7)
    ax3.set_title('F1-Score Comparison', fontweight='bold')
    ax3.set_ylabel('F1-Score')
    ax3.set_ylim(0, 0.06)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, f1 in zip(bars3, f1_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{f1:.4f}', ha='center', va='bottom', fontweight='bold')

    # 4. Performance Summary Table
    ax4.axis('off')
    table_data = [
        ['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        *[[alg, f'{acc:.4f}', f'{prec:.4f}', f'{rec:.4f}', f'{f1:.4f}']
          for alg, acc, prec, rec, f1 in zip(algorithms, accuracies, precisions, recalls, f1_scores)]
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Performance Metrics Summary', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('/home/yuvaraj_hadoop/fraud_detection/results/plots/algorithm_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Algorithm comparison plot saved to: results/plots/algorithm_comparison.png")

    # Print summary
    print("\n🎯 PERFORMANCE SUMMARY:")
    print("=" * 50)
    best_accuracy = max(accuracies)
    best_f1 = max(f1_scores)
    best_precision = max(precisions)

    print(f"🏆 Best Accuracy: {best_accuracy:.4f} ({algorithms[accuracies.index(best_accuracy)]})")
    print(f"🎯 Best F1-Score: {best_f1:.4f} ({algorithms[f1_scores.index(best_f1)]})")
    print(f"🔍 Best Precision: {best_precision:.4f} ({algorithms[precisions.index(best_precision)]})")

    print("\n📝 NOTES:")
    print("- High accuracy indicates good overall performance")
    print("- Low precision/recall suggests difficulty detecting fraud cases")
    print("- Isolation Forest + Logistic shows best balanced performance")
    print("- Autoencoder has highest fraud detection capability")
