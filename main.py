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

# Paths
BASE_DATA_DIR = config.get("paths", {}).get("interview_data_dir", "data")
INTERVIEWS_DIR = os.path.join(BASE_DATA_DIR, "interviews")

# MLflow
MLFLOW_EXPERIMENT_NAME = config.get("mlflow", {}).get(
    "experiment_name", "Persona Generation PoC"
)
MLFLOW_TRACKING_URI = config.get("mlruns", {}).get("mlruns")

# System prompt template
SYSTEM_PROMPT_TEMPLATE = """
You are an AI that is impersonating a character based on the following interview.

INTERVIEW:
{interview}

You must:
- Answer strictly as this character, in the first person.
- Maintain a consistent tone and personality across the whole conversation.
- If the user asks something unrelated to the interview, answer as the character would.
- Never break character and never mention that you are an AI model.
""".strip()

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


def get_interview_filepath(interview_id: int) -> str:
    base_name = f"consumer_{interview_id}"
    interview_path = os.path.join(INTERVIEWS_DIR, base_name + ".json")
    print("Full path:", interview_path)

    if not os.path.exists(interview_path):
        raise FileNotFoundError(f"Interview file not found for id={interview_id}. ")

    return interview_path


def load_interview_text(interview_id: int) -> str:
    """
    Loads the content of an interview file.

    Expected format (simplified):
        INTERVIEW:
        [{<dictionary of assistant/user multi-turn conversation>}]

    Interview is not parsed, it is directly injected into the system prompt.
    """
    filepath = get_interview_filepath(interview_id)
    with open(filepath, "r") as f:
        text = f.read().strip()

    return text


def build_system_prompt(interview_id: int) -> str:
    interview_text = load_interview_text(interview_id)
    return SYSTEM_PROMPT_TEMPLATE.format(interview=interview_text)


# In-memory conversation store for PoC (not for production)
ConversationMessage = Dict[str, str]  # {"role": "user"/"assistant", "content": "..."}

CONVERSATIONS: Dict[str, List[ConversationMessage]] = {}


# fastAPI classes
class ChatRequest(BaseModel):
    user_message: str
    conversation_id: Optional[str] = None  # if None => start new conversation
    interview_id: int = 1  # which interview persona to use


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    model: str
    interview_id: int
    usage: Optional[Dict[str, int]] = None  # token usage etc.


def call_openai_with_history(
    user_message: str,
    history: List[ConversationMessage],
    system_prompt: str,
    model: str = OPENAI_MODEL,
) -> object:
    """
    Calls the OpenAI Responses API with:
    - system message containing the interview
    - previous conversation messages
    - current user message
    - parameters read from config.yaml
    """
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    # Build kwargs to avoid sending None/invalid values
    request_kwargs = {
        "model": model,
        "input": messages,
        # "temperature": LLM_TEMPERATURE,
        # "top_p": LLM_TOP_P
    }

    if LLM_MAX_TOKENS and LLM_MAX_TOKENS > 0:
        # For Responses API the parameter is max_output_tokens
        request_kwargs["max_output_tokens"] = LLM_MAX_TOKENS

    # Optional reasoning effort
    if LLM_REASONING in {"low", "medium", "high"}:
        request_kwargs["reasoning"] = {"effort": LLM_REASONING}

    response = client.responses.create(**request_kwargs)

    return response


# server instantiation + experiment tracking
app = FastAPI(title="Character Impersonation PoC API")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Single endpoint:
    - Takes user_message + (optional) conversation_id + interview_id
    - Maintains multi-turn conversation in memory
    - Builds system prompt from interview file
    - Logs everything to MLflow as a single run per call
    """
    try:
        system_prompt = build_system_prompt(req.interview_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create or reuse conversation_id
    conversation_id = req.conversation_id or str(uuid.uuid4())
    history = CONVERSATIONS.get(conversation_id, [])

    # Start timing for latency metric
    start_time = time.time()

    # Start MLflow run
    with mlflow.start_run(run_name=f"conversation_{conversation_id}"):
        # Log high-level parameters
        mlflow.log_params(
            {
                "model": OPENAI_MODEL,
                "conversation_id": conversation_id,
                "interview_id": req.interview_id,
                "system_prompt_length": len(system_prompt),
            }
        )

        # Log the raw input message
        mlflow.log_text(
            f"User message:\n{req.user_message}\n",
            artifact_file="input.txt",
        )

        # Call OpenAI API
        response = call_openai_with_history(
            user_message=req.user_message,
            history=history,
            system_prompt=system_prompt,
            model=OPENAI_MODEL,
        )

        latency = time.time() - start_time

        # Extract output text
        reply_text = getattr(response, "output_text", None)
        if reply_text is None:
            try:
                reply_text = response.output[0].content[0].text
            except Exception:
                reply_text = "<no text output>"

        # Extract usage if available
        usage_dict = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            raw_usage = getattr(usage, "__dict__", {})
            usage_dict = {
                k: int(v) for k, v in raw_usage.items() if isinstance(v, (int, float))
            }

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

    # Update in-memory conversation
    history.append({"role": "user", "content": req.user_message})
    history.append({"role": "assistant", "content": reply_text})
    CONVERSATIONS[conversation_id] = history

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        model=OPENAI_MODEL,
        interview_id=req.interview_id,
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
