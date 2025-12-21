import mlflow
import pandas as pd

import main


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiments = mlflow.search_experiments()
experiment_ids = [exp.experiment_id for exp in experiments]

print(f"Found {len(experiment_ids)} experiments")

df = mlflow.search_runs(experiment_ids=experiment_ids, output_format="pandas")

print("Full MLflow DataFrame Loaded")
print(df.shape)
print(df.columns)

selected_metrics = [
    "metrics.persona_fidelity",
    "metrics.content_accuracy",
    "metrics.instruction_following",
    "metrics.overall_eval_score",
    "metrics.semantic_similarity",
]

selected_metrics = [m for m in selected_metrics if m in df.columns]

experiment_comparison = df.groupby("experiment_id")[selected_metrics].mean()

experiment_comparison["num_runs"] = df.groupby("experiment_id").size()

print("Average Metrics Per Experiment:\n")
print(experiment_comparison)
