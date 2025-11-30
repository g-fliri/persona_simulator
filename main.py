from dotenv import load_dotenv
import os
import time
from typing import Dict, List, Optional
import uuid
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

import mlflow


load_dotenv()
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)

# LLM config
LLM_PROVIDER = config.get("llm", {}).get("provider", "openai")
LLM_MODEL_NAME = config.get("llm", {}).get("model_name", "gpt-5.1")
# LLM_TEMPERATURE = float(config.get("llm", {}).get("temperature", 0.25))    # GPT 5.1 seems to support only reasoning={ "effort": "none | low | medium | high" } and text={ "verbosity": "low | medium | high"} parameters
# LLM_TOP_P = float(config.get("top_p", {}).get("top_p", 1.00))
LLM_MAX_TOKENS = int(config.get("llm", {}).get("max_tokens", -1))
LLM_REASONING = config.get("llm", {}).get("reasoning", None)
LLM_VERBOSITY = config.get("llm", {}).get("text", {})


# Paths
PROMPTS_DIR = config.get("paths", {}).get("prompts_dir", "prompts")
BASE_DATA_DIR = config.get("paths", {}).get("interview_data_dir", "data")
INTERVIEWS_DIR = os.path.join(BASE_DATA_DIR, "interviews")

# Prompt of choice
PROMPT_NAME = config.get("llm", {}).get("prompt_file")

# MLflow
MLFLOW_EXPERIMENT_NAME = config.get("mlflow", {}).get(
    "experiment_name", "Persona Generation PoC"
)
MLFLOW_TRACKING_URI = config.get("mlflow", {}).get("mlruns")


# Load system prompt template
def load_prompt_template_from_file(prompt_name: str) -> str:
    """Load a system prompt template from the prompts folder."""
    prompt_path = os.path.join(PROMPTS_DIR, prompt_name)
    if not os.path.exists(prompt_path):
        raise RuntimeError(f"Prompt file not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


SYSTEM_PROMPT_TEMPLATE = load_prompt_template_from_file(PROMPT_NAME)

# Env overrides
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", LLM_MODEL_NAME)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
SYSTEM_PROMPT_TEMPLATE = os.getenv("SYSTEM_PROMPT_TEMPLATE", SYSTEM_PROMPT_TEMPLATE)

if LLM_PROVIDER.lower() != "openai":
    raise RuntimeError(
        f"Only 'openai' provider is supported in this PoC, got {LLM_PROVIDER}."
    )

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def get_interview_filepath(consumer_id: int) -> str:
    base_name = f"consumer_{consumer_id}"
    interview_path = os.path.join(INTERVIEWS_DIR, base_name + ".json")

    if not os.path.exists(interview_path):
        raise FileNotFoundError(f"Interview file not found for id={consumer_id}. ")

    return interview_path


def load_interview_text(consumer_id: int) -> str:
    """
    Loads the content of an interview file.
    Interview is not parsed, it is directly injected into the system prompt.
    """
    filepath = get_interview_filepath(consumer_id)
    with open(filepath, "r") as f:
        text = f.read().strip()

    return text


def build_system_prompt(consumer_id: int) -> str:
    interview_text = load_interview_text(consumer_id)
    return SYSTEM_PROMPT_TEMPLATE.format(interview=interview_text)


# In-memory conversation store for PoC (not for production)
ConversationMessage = Dict[str, str]  # {"role": "user"/"assistant", "content": "..."}

CONVERSATIONS: Dict[str, List[ConversationMessage]] = {}


# fastAPI classes
class ChatRequest(BaseModel):
    user_message: str
    conversation_id: Optional[str] = None  # if None => start new conversation
    consumer_id: int = 1  # which interview persona to use
    test_question_id: Optional[str] = None
    expected_answer: Optional[str] = None
    id: Optional[int] = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    model: str
    consumer_id: int
    usage: Optional[Dict[str, int]] = None


def call_openai_with_history(
    user_message: str,
    history: List[ConversationMessage],
    system_prompt: str,
    model: str = OPENAI_MODEL,
) -> object:
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    request_kwargs = {
        "model": model,
        "input": messages,
    }

    if LLM_MAX_TOKENS and LLM_MAX_TOKENS > 0:
        request_kwargs["max_output_tokens"] = LLM_MAX_TOKENS

    # reasoning
    if LLM_REASONING in {"low", "medium", "high"}:
        request_kwargs["reasoning"] = {"effort": LLM_REASONING}

    # text verbosity
    if LLM_VERBOSITY in {"low", "medium", "high"}:
        request_kwargs["text"] = {"verbosity": LLM_VERBOSITY}

    response = client.responses.create(**request_kwargs)
    return response


# server instantiation + experiment tracking
app = FastAPI(title="Character Impersonation PoC API")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Single endpoint:
    - Takes user_message + (optional) conversation_id + consumer_id
    - Maintains multi-turn conversation in memory
    - Builds system prompt from interview file
    - Logs everything to MLflow as a single run per call
    """
    try:
        system_prompt = build_system_prompt(req.consumer_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create or reuse conversation_id
    conversation_id = req.conversation_id or str(uuid.uuid4())
    history = CONVERSATIONS.get(conversation_id, [])

    start_time = time.time()

    # Start MLflow run
    with mlflow.start_run(run_name=f"conversation_{conversation_id}"):
        response = call_openai_with_history(
            user_message=req.user_message,
            history=history,
            system_prompt=system_prompt,
            model=OPENAI_MODEL,
        )

        latency = time.time() - start_time

        reply_text = getattr(response, "output_text", None)
        if reply_text is None:
            try:
                reply_text = response.output[0].content[0].text
            except Exception:
                reply_text = "<no text output>"

        usage_dict = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            raw_usage = getattr(usage, "__dict__", {})
            usage_dict = {
                k: int(v) for k, v in raw_usage.items() if isinstance(v, (int, float))
            }

        params = {
            "model": OPENAI_MODEL,
            "conversation_id": conversation_id,
            "consumer_id": req.consumer_id,
            "system_prompt_length": len(system_prompt),
            "user_message": req.user_message,
            "assistant_reply": reply_text,
            "prompt_name": PROMPT_NAME,
            "reasoning_effort": LLM_REASONING or "",
            "text_verbosity": LLM_VERBOSITY or "",
        }

        if getattr(req, "test_question_id", None) is not None:
            params["test_question_id"] = req.test_question_id
        if getattr(req, "expected_answer", None) is not None:
            params["expected_answer"] = req.expected_answer
        if getattr(req, "id", None) is not None:
            params["id"] = req.id

        mlflow.log_params(params)

        # Log metrics to MLflow
        mlflow.log_metric("latency_seconds", latency)
        if usage_dict:
            for key, value in usage_dict.items():
                mlflow.log_metric(key, value)

        # Log prompt + response as text
        transcript_text = [
            "=== System prompt ===",
            system_prompt,
            "",
            "=== Previous history ===",
        ]
        for m in history:
            transcript_text.append(f"{m['role'].upper()}: {m['content']}")
        transcript_text.extend(
            [
                "",
                "=== New turn ===",
                f"USER: {req.user_message}",
                f"ASSISTANT: {reply_text}",
            ]
        )
        mlflow.log_text(
            "\n".join(transcript_text),
            artifact_file="transcript.txt",
        )

        mlflow.log_text(
            f"User message:\n{req.user_message}\n",
            artifact_file="input.txt",
        )

    # Update in-memory conversation
    history.append({"role": "user", "content": req.user_message})
    history.append({"role": "assistant", "content": reply_text})
    CONVERSATIONS[conversation_id] = history

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        model=OPENAI_MODEL,
        consumer_id=req.consumer_id,
        usage=usage_dict,
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


# going online
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8801,
        reload=True,
    )
