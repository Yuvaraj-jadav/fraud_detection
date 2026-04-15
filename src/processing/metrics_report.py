import argparse
from pathlib import Path

import pandas as pd

from results.plots.results import compute_metrics_by_group, detect_label_columns

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def summarize_metrics(df: pd.DataFrame, path: Path) -> None:
    print(f"\n=== Metrics for {path.name} ===")
    metrics_df = compute_metrics_by_group(df)
    if metrics_df.empty:
        label_info = detect_label_columns(df)
        actual_col = label_info.get("actual_label_column")
        prediction_src = label_info.get("prediction_source")
        candidate_actuals = label_info.get("candidate_actuals", [])
        candidate_predictions = label_info.get("candidate_predictions", [])

        print("No evaluation metrics could be computed.")
        print(f"  Actual label column detected: {actual_col or 'none'}")
        print(f"  Prediction source detected: {prediction_src or 'none'}")
        print(f"  Candidate actual label columns: {candidate_actuals or 'none'}")
        print(f"  Candidate prediction columns: {candidate_predictions or 'none'}")
        return

    metrics_df = metrics_df.set_index("source_file")
    print(metrics_df[["precision", "recall", "f1_score", "accuracy", "support", "fraud_support"]].to_string())


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fraud evaluation metrics from processed output CSV files.")
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to a specific processed CSV file. If omitted, all data/processed/*.csv files are evaluated.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also compute metrics on the combined labeled output file if it exists.",
    )
    args = parser.parse_args()

    if args.path:
        paths = [args.path]
    else:
        paths = sorted(PROCESSED_DIR.glob("*.csv"))

    if not paths:
        print(f"No processed CSV files found in {PROCESSED_DIR}.")
        return

    for path in paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        df = load_csv(path)
        summarize_metrics(df, path)

    if args.combined:
        combined_file = PROCESSED_DIR / "fraud_pipeline_output_all.csv"
        if combined_file.exists():
            print("\n=== Combined labeled output ===")
            df = load_csv(combined_file)
            summarize_metrics(df, combined_file)
        else:
            print(f"\nCombined labeled output not found at {combined_file}")


if __name__ == "__main__":
    main()
