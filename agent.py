import os
import sqlite3
import threading
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional

from dotenv import load_dotenv
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from tools import tools

Path("data").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

ALLOWED_MODELS = {
    # Google Gemini
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    # Groq
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "deepseek-r1-distill-llama-70b",
    # Mistral
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
}

MODEL_CONFIG = {
    "gemini-2.5-flash": {"provider": "google", "max_tokens": 8192, "context": 1_000_000},
    "gemini-2.5-pro": {"provider": "google", "max_tokens": 8192, "context": 1_000_000},
    "llama-3.3-70b-versatile": {"provider": "groq", "max_tokens": 32768, "context": 128_000},
    "llama-3.1-8b-instant": {"provider": "groq", "max_tokens": 8192, "context": 128_000},
    "deepseek-r1-distill-llama-70b": {"provider": "groq", "max_tokens": 32768, "context": 128_000},
    "mistral-small-latest": {"provider": "mistral", "max_tokens": 32000, "context": 128_000},
    "mistral-medium-latest": {"provider": "mistral", "max_tokens": 32000, "context": 128_000},
    "mistral-large-latest": {"provider": "mistral", "max_tokens": 32000, "context": 128_000},
}

SYSTEM_PROMPT = """
You are AdilGPT, an advanced AI assistant. Provide accurate, helpful, and intelligent responses.

Rules:
- Be concise, accurate, and professional.
- Use Markdown formatting (headings, lists, tables, code blocks).
- For uploaded files: use RAG tool first.
- For real-time info: use Tavily Search.
- For math: use calculator tool.
- For memories: use recall/remember tools.
- Never fabricate information. State limitations clearly.
- Prioritize correctness over speed.
"""

# ─────────────────────────────────────────────────────────────
# Thread-safe Agent Cache
# ─────────────────────────────────────────────────────────────
_agent_cache: Dict[str, any] = {}
_cache_lock = threading.Lock()


def normalize_model_name(model_name: Optional[str]) -> str:
    """Validate and normalize model name."""
    if not model_name:
        return DEFAULT_MODEL
    model_name = model_name.strip()
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    return model_name


def validate_api_key_for_model(model_name: str) -> None:
    """Ensure required API key exists for selected model."""
    provider = MODEL_CONFIG.get(model_name, {}).get("provider", "")

    key_map = {
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }

    required_key = key_map.get(provider)
    if required_key and not os.getenv(required_key):
        raise RuntimeError(
            f"Model '{model_name}' requires {required_key} environment variable."
        )


def build_agent(model_name: str):
    """Build a LangGraph agent for the selected AI model."""
    import time
    start = time.time()
    """
    Build a LangGraph agent for the selected AI model.
    Supports Google Gemini, Groq, and Mistral.
    """
    selected_model = normalize_model_name(model_name)
    validate_api_key_for_model(selected_model)

    config = MODEL_CONFIG.get(selected_model, {})
    max_tokens = config.get("max_tokens", 4096)

    try:
        # Google Gemini
        if selected_model.startswith("gemini"):
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                temperature=0.3,
                max_output_tokens=max_tokens,
                streaming=True,
            )

        # Groq Models
        elif selected_model.startswith(("llama", "deepseek", "qwen")):
            llm = ChatGroq(
                model=selected_model,
                temperature=0.3,
                max_tokens=max_tokens,
                streaming=True,
            )

        # Mistral Models
        elif selected_model.startswith("mistral"):
            llm = ChatMistralAI(
                model=selected_model,
                temperature=0.3,
                max_tokens=max_tokens,
                streaming=True,
            )

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM '{selected_model}': {e}") from e

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    # Use configurable SQLite path
    db_path = os.getenv("LANGGRAPH_DB_PATH", "data/langgraph_checkpoints.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    print(f"[Agent] Built agent for {selected_model} in {time.time() - start:.2f}s")
    return workflow.compile(checkpointer=checkpointer)


def get_agent(model_name: Optional[str] = None):
    """
    Get or create a cached agent instance for the given model.
    Thread-safe singleton pattern with LRU-style caching.
    """
    selected_model = normalize_model_name(model_name)

    with _cache_lock:
        if selected_model not in _agent_cache:
            _agent_cache[selected_model] = build_agent(selected_model)
        return _agent_cache[selected_model]


def clear_agent_cache():
    """Clear all cached agents. Useful for testing or memory management."""
    with _cache_lock:
        _agent_cache.clear()