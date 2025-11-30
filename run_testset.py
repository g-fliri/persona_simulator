import os
from typing import Any
import yaml

import pandas as pd
from fastapi.testclient import TestClient

# import main app
import main

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)

EXCEL_PATH = config.get("paths", {}).get("test_questions_file")
SHEET_NAME: Any = "questionnaire-history"


def to_consumer_id(user_id: Any) -> int:
    # user_id is always numeric (probably this is redundant)
    if isinstance(user_id, str):
        user_id = user_id.strip()
    return int(user_id)


def main_script() -> None:
    if not EXCEL_PATH:
        raise RuntimeError("paths.test_questions_file is not set in config.yaml")

    # Load Excel
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    required_cols = {"id", "question_id", "question", "answer", "user_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in Excel: {missing}")

    # create a TestClient for main app
    client = TestClient(main.app)

    print(f"Loaded {len(df)} rows from {EXCEL_PATH}")
    print("Sending them to /chat...\n")

    for idx, row in df.iterrows():
        row_id = int(row["id"])
        question_id = str(row["question_id"])
        question = str(row["question"])
        expected_answer = str(row["answer"])
        user_id = row["user_id"]
        consumer_id = to_consumer_id(user_id)

        payload = {
            "user_message": question,
            "conversation_id": None,
            "consumer_id": consumer_id,
            "expected_answer": expected_answer,
            "test_question_id": question_id,
            "id": row_id,
        }

        resp = client.post("/chat", json=payload)

        if resp.status_code != 200:
            print(
                f"[row {idx}] ERROR {resp.status_code} for "
                f"id={row_id}, question_id={question_id}: {resp.text}"
            )
            print("-" * 80)
            continue

        data = resp.json()
        reply = data.get("reply", "")
        conv_id = data.get("conversation_id")

        print(
            f"[row {idx}] id={row_id}, question_id={question_id}, "
            f"consumer_id={consumer_id}, conversation_id={conv_id}"
        )
        print(f"  Q:          {question}")
        print(f"  A_model:    {reply}")
        print(f"  A_expected: {expected_answer}")
        print("-" * 80)


if __name__ == "__main__":
    main_script()
