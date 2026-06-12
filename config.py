"""
Central configuration for MTP Assistant.

To switch LLM backend:
    LLM_BACKEND=local python main.py ...

To change Anthropic model:
    ANTHROPIC_MODEL=claude-opus-4-8 python main.py ...

To use a different local model:
    LLM_BACKEND=local OLLAMA_MODEL=llama3 python main.py ...
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env from project root into os.environ before any keys are consumed

# --- LLM backend ---
# "anthropic" | "local"
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "anthropic")

# --- Anthropic backend ---
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- Local backend (Ollama) ---
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
