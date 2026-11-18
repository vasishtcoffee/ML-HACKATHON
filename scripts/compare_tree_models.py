"""Summarize recorded runs without selecting parameters on student-test data."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def summarize():
    records = []
    for name in ("random_forest", "xgboost", "xgboost_tuned", "random_forest_tuned"):
        source = ROOT / "experiments" / name / "metrics.csv"
        if not source.is_file():
            continue
        metrics = pd.read_csv(source)
        cv = metrics.loc[metrics.phase.eq("cv")]
        test = metrics.loc[metrics.phase.eq("student_test")]
        if len(test) != 1 or len(cv) != 5:
            raise ValueError(f"Expected five folds and one student-test row: {source}")
        row = test.iloc[0]
        records.append({"run": name, "cv_top1_mean": cv.top1_accuracy.mean(),
            "cv_top1_std": cv.top1_accuracy.std(), "student_top1": row.top1_accuracy,
            "student_macro_f1": row.macro_f1, "student_log_loss": row.candidate_log_loss,
            "student_continuity": row.continuity,
            "inference_seconds": row.inference_seconds})
    return pd.DataFrame(records)

if __name__ == "__main__":
    table = summarize()
    output = ROOT / "experiments/tree_comparison.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
