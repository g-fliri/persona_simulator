from dotenv import load_dotenv
import os
import json
from typing import Dict, Any, Optional, Sequence
import yaml

import numpy as np
from openai import OpenAI

import mlflow
from mlflow.entities import Run

# import main app
import main


load_dotenv()
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)

EMBEDDING_MODEL_PROVIDER = config.get("embedding_model", {}).get("provider", "openai")
EMBEDDING_MODEL_NAME = config.get("embedding_model", {}).get(
    "model_name", "text-embedding-3-large"
)

EVAL_MODEL_PROVIDER = config.get("eval_model", {}).get("provider", "openai")
EVAL_MODEL_NAME = config.get("eval_model", {}).get("model_name", "gpt-5.1")

EXPERIMENT_NAME = main.MLFLOW_EXPERIMENT_NAME
MLFLOW_TRACKING_URI = main.MLFLOW_TRACKING_URI


# Reuse the client from main if available, otherwise create a new one
client: OpenAI = getattr(main, "client", OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


# evaluation by embeddings' cosine similarity
def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    v1_arr = np.array(v1, dtype=np.float32)
    v2_arr = np.array(v2, dtype=np.float32)

    denom = np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(v1_arr, v2_arr) / denom)


def compute_embedding_similarity(expected: str, actual: str) -> float:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=[expected, actual],
    )
    e1 = resp.data[0].embedding
    e2 = resp.data[1].embedding
    return cosine_similarity(e1, e2)


# evaluation by LLM-as-judge
def judge_run(
    persona_interview: str,
    user_message: str,
    expected_answer: str,
    assistant_reply: str,
) -> Optional[Dict[str, Any]]:
    """
    Call an LLM judge to score:
      - content_accuracy (0–5)
      - persona_fidelity (0–5)
      - instruction_following (0–5)
    Returns a dict or None on failure.
    """
    prompt = f"""
You are evaluating a chatbot that impersonates a persona based on an interview.

You will be given:
- The persona interview
- The user's question
- The expected answer
- The model's answer

(Mind that the persona interview and the chatbot conversation use different naming conventions.)

Evaluate the chatbot answer on:

1. Content Accuracy (0–5): How similar is the model's answer to the expected answer in meaning, ignoring phrasing differences?
2. Persona Fidelity (0–5): How well does the model's answer match the persona's tone, personality, and worldview implied by the interview?
3. Instruction Following (0–5): Does the model answer in the first person, stay in character, and avoid mentioning that it is an AI?

Persona interview:
{persona_interview}

User question:
{user_message}

Expected answer:
{expected_answer}

Model answer:
{assistant_reply}

Respond ONLY with a JSON object of the form:
{{
  "content_accuracy": 0-5,
  "persona_fidelity": 0-5,
  "instruction_following": 0-5,
  "short_comment": "one short sentence summary"
}}
""".strip()

    resp = client.responses.create(
        model=EVAL_MODEL_NAME,
        input=[{"role": "user", "content": prompt}],
    )

    # Try to extract plain text from the response (mirrors logic in main.py)
    try:
        text = getattr(resp, "output_text", None)
        if text is None:
            text = resp.output[0].content[0].text
    except Exception:
        return None

    text = text.strip()

    # The model should return JSON, but we guard just in case.
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        # You may want to log 'text' somewhere if debugging
        return None


def run_already_evaluated(run: Run) -> bool:
    """
    Check if we've already added evaluation metrics to this run.
    You can change the condition (e.g., check for just one metric).
    """
    metrics = run.data.metrics
    return "semantic_similarity" in metrics and "content_accuracy" in metrics


# evaluatuon routine
def evaluate_all(force: bool = False) -> None:
    """
    Iterate over all runs in the configured MLflow experiment and add:
      - semantic_similarity (embedding similarity)
      - content_accuracy, persona_fidelity, instruction_following
      - overall_eval_score
    Skips runs that are already evaluated (unless force=True).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    runs_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
    )

    print(f"Found {len(runs_df)} runs in experiment '{EXPERIMENT_NAME}'.")

    for _, row in runs_df.iterrows():
        run_id = row["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params

        expected_answer = params.get("expected_answer")
        assistant_reply = params.get("assistant_reply")
        user_message = params.get("user_message", "")
        consumer_id_str = params.get("consumer_id")

        # Skip runs without test data
        if not expected_answer or not assistant_reply or not consumer_id_str:
            continue

        if run_already_evaluated(run) and not force:
            print(f"[{run_id}] Already evaluated, skipping.")
            continue

        print(f"Evaluating run {run_id} ...")

        # Load persona interview text
        try:
            consumer_id = int(consumer_id_str)
        except ValueError:
            print(
                f"[{run_id}] Invalid consumer_id/interview_id: {consumer_id_str}, skipping."
            )
            continue

        try:
            persona_interview = main.load_interview_text(consumer_id)
        except FileNotFoundError:
            print(
                f"[{run_id}] Interview file not found for consumer_id={consumer_id}, skipping."
            )
            continue

        # 1) Embedding similarity
        try:
            similarity = compute_embedding_similarity(expected_answer, assistant_reply)
        except Exception as e:
            print(f"[{run_id}] Error computing embeddings: {e}")
            similarity = None

        # 2) LLM-as-judge
        judge_scores = judge_run(
            persona_interview=persona_interview,
            user_message=user_message,
            expected_answer=expected_answer,
            assistant_reply=assistant_reply,
        )

        # Log metrics back into the same run
        with mlflow.start_run(run_id=run_id):
            if similarity is not None:
                mlflow.log_metric("semantic_similarity", similarity)

            if judge_scores is not None:
                # Make sure they're numbers
                ca = int(judge_scores.get("content_accuracy", 0))
                pf = int(judge_scores.get("persona_fidelity", 0))
                ifm = int(judge_scores.get("instruction_following", 0))

                mlflow.log_metric("content_accuracy", ca)
                mlflow.log_metric("persona_fidelity", pf)
                mlflow.log_metric("instruction_following", ifm)

                # overall score
                overall = 0.5 * ca + 0.3 * pf + 0.2 * ifm
                mlflow.log_metric("overall_eval_score", overall)

                # optional: log the judge's comment as a param or artifact
                comment = judge_scores.get("short_comment", "")
                if comment:
                    mlflow.log_param("eval_short_comment", comment)

        print(f"[{run_id}] Done. Similarity={similarity}, judge={judge_scores}")


if __name__ == "__main__":
    # Set force=True if you want to recompute metrics for all runs.
    evaluate_all(force=False)
