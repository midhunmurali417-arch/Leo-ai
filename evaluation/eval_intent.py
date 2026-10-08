"""
=============================================================================
Module: evaluation/eval_intent.py
Description: Evaluates intent detection (should_search_google) against a labeled
             dataset of 50 queries. Builds confusion matrix, computes metrics,
             and saves detailed error analysis.
=============================================================================
"""

import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path to import google_search WITHOUT importing app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import google_search

def evaluate_intent():
    eval_dir = Path(__file__).resolve().parent
    data_path = eval_dir / "data" / "intent_queries.csv"
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} intent queries from {data_path.name}")

    tp = 0  # Actual 1, Predicted 1
    fp = 0  # Actual 0, Predicted 1
    fn = 0  # Actual 1, Predicted 0
    tn = 0  # Actual 0, Predicted 0

    errors = []

    for idx, row in df.iterrows():
        query = str(row["query"]).strip()
        actual = int(row["label"])
        pred = 1 if google_search.should_search_google(query) else 0

        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
            errors.append({
                "query": query,
                "actual_label": actual,
                "predicted_label": pred,
                "error_type": "False Positive (Static query triggered search)"
            })
        elif actual == 1 and pred == 0:
            fn += 1
            errors.append({
                "query": query,
                "actual_label": actual,
                "predicted_label": pred,
                "error_type": "False Negative (Live query missed search trigger)"
            })
        else:
            tn += 1

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "total_queries": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    # Save metrics JSON
    json_path = results_dir / "intent_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {json_path.name}")

    # Save error report CSV
    errors_df = pd.DataFrame(errors)
    errors_csv_path = results_dir / "intent_errors.csv"
    errors_df.to_csv(errors_csv_path, index=False)
    print(f"Saved {len(errors)} misclassifications to {errors_csv_path.name}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix_data = [[tp, fn], [fp, tn]]
    cax = ax.matshow(matrix_data, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = matrix_data[i][j]
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=16, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Live (1)", "Predicted Static (0)"], fontsize=10)
    ax.set_yticklabels(["Actual Live (1)", "Actual Static (0)"], fontsize=10)
    plt.title(f"Intent Detection Confusion Matrix\nAccuracy: {accuracy:.2%} | F1: {f1:.4f}", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()

    plot_path = results_dir / "intent_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix chart to {plot_path.name}")

    print("\n--- INTENT EVALUATION SUMMARY ---")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("---------------------------------")
    return metrics

if __name__ == "__main__":
    evaluate_intent()
