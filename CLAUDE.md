# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GenAI-based digital persona system that creates realistic chatbot personas from human interview transcripts. The system enables these personas to respond to questions in a manner consistent with the original interviewee's style, tone, and content. The primary goal is to mimic real respondents as closely as possible for market research applications.

## Core Architecture

### Three-Stage Workflow

1. **Interactive Chat (main.py)**: FastAPI server with web frontend for real-time persona interaction
2. **Batch Testing (run_testset.py)**: Automated test execution against Excel dataset
3. **Evaluation (evaluate.py)**: Dual-metric evaluation system (semantic similarity + LLM-as-judge)

### Configuration-Driven Design

All system behavior is controlled via `config.yaml`:
- LLM provider settings (currently OpenAI-only)
- Model configuration (gpt-5.1 with reasoning/verbosity controls)
- Prompt selection via `prompt_file` parameter
- MLflow experiment naming and tracking
- File paths for interviews, test data, and outputs

### Key Components

**Persona Loading System** (main.py:86-110)
- Interview files stored as `data/interviews/consumer_{id}.json`
- Raw JSON content injected directly into system prompt template
- System prompt built by formatting template with interview text

**Conversation Management** (main.py:113-116)
- In-memory conversation store (PoC pattern, not production-ready)
- Multi-turn conversation support via conversation_id
- History maintained per conversation and passed to OpenAI API

**MLflow Integration**
- Every chat interaction logged as a run
- Logs: params (model config, consumer_id, prompt), metrics (latency, token usage), artifacts (full transcript)
- Evaluation metrics appended to existing runs by evaluate.py
- Database: `sqlite:///mlflow.db`

**OpenAI API Usage** (main.py:139-171)
- Uses `client.responses.create()` with new GPT-5.1 API format
- Supports reasoning effort ("low"/"medium"/"high") and text verbosity controls
- Response parsing handles both `output_text` attribute and nested content structure

### Prompt Engineering Strategy

Prompts stored in `prompts/` directory as individual files. Current production prompt (`prompt_5`):
- Instructs model to adopt persona identity completely
- Emphasizes matching interview style: sentence length, vocabulary, detail level, fillers
- Enforces plain text format (no markdown, lists, or formatting)
- Constrains response length to match interview patterns
- Allows graceful handling of topics not in interview

### Evaluation Framework

**Dual Metrics** (evaluate.py):

1. **Semantic Similarity**: Cosine similarity of embeddings (text-embedding-3-large) between expected and actual answers

2. **LLM-as-Judge**: GPT-5.1 scores responses on:
   - content_accuracy (0-5)
   - persona_fidelity (0-5)
   - instruction_following (0-5)
   - Penalizes generic AI patterns (markdown, bullet points, overly formal tone)
   - Rewards casual, human-like chat style matching interview

Overall score: `0.5 * content_accuracy + 0.4 * persona_fidelity + 0.1 * instruction_following`

**Evaluation Process**:
- Iterates through MLflow runs that have `expected_answer` parameter
- Skips already-evaluated runs (unless force=True)
- Logs metrics back to original run for unified view

## Common Commands

### Development

**Start interactive server**:
```bash
python main.py
```
Launches FastAPI server at http://127.0.0.1:8808 with auto-reload. Access web frontend at http://localhost:8808

**Run batch tests**:
```bash
python run_testset.py
```
Processes all questions in `data/test_questions_answers.xlsx` (sheet: "questionnaire-history") and logs to MLflow

**Evaluate test results**:
```bash
python evaluate.py
```
Computes semantic similarity and LLM-as-judge scores for all unevaluated MLflow runs

**View MLflow UI**:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Opens web interface for experiment tracking and metric visualization

**Compare experiments (raw summary)**:
```bash
python mlflow_exp_summarizer.py
```
Basic comparison tool for MLflow experiments

### Code Formatting

Project uses `ruff` for formatting. Format code with:
```bash
ruff format .
```

## Environment Setup

**Required**: `.env` file in root directory with:
```
OPENAI_API_KEY=<your-key-here>
```

**Optional overrides**:
- `CONFIG_PATH`: Path to config.yaml (default: "config.yaml")
- `OPENAI_MODEL`: Override model from config
- `MLFLOW_TRACKING_URI`: Override MLflow database URI
- `SYSTEM_PROMPT_TEMPLATE`: Override prompt template directly

## Data Structure

**Interview Files**: `data/interviews/consumer_{id}.json`
- Stored as plain JSON (not parsed, injected raw into prompt)
- consumer_id maps to user_id in test dataset

**Test Dataset**: `data/test_questions_answers.xlsx`
- Sheet: "questionnaire-history"
- Required columns: id, question_id, question, answer, user_id
- user_id converts to consumer_id for persona lookup

## Important Implementation Notes

### GPT-5.1 API Pattern

This codebase uses OpenAI's newer `responses.create()` API with different parameter structure:
- Input: `input` parameter instead of `messages`
- Parameters: `reasoning` (effort level), `text` (verbosity), `max_output_tokens`
- Response parsing: Check `output_text` attribute first, fallback to `output[0].content[0].text`

### Prompt Selection Workflow

To experiment with prompts:
1. Create new prompt file in `prompts/` directory
2. Update `config.yaml`: `llm.prompt_file: "your_prompt_name"`
3. Restart server or re-run tests
4. Update `mlflow.experiment_name` to track different prompt versions separately

### Test Client Pattern

`run_testset.py` uses FastAPI's `TestClient` to call the server without network:
```python
from fastapi.testclient import TestClient
client = TestClient(main.app)
resp = client.post("/chat", json=payload)
```
This pattern enables batch testing without running uvicorn server.

### Evaluation Idempotency

`evaluate.py` checks for existing metrics before re-evaluating:
```python
def run_already_evaluated(run: Run) -> bool:
    metrics = run.data.metrics
    return "semantic_similarity" in metrics and "content_accuracy" in metrics
```
Set `force=True` in `evaluate_all()` to re-evaluate all runs.
