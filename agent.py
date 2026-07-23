import os
from pathlib import Path
import sqlite3

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

Path("data").mkdir(exist_ok=True)
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

SYSTEM_PROMPT = """
You are AdilGPT, an advanced AI assistant designed to provide accurate, helpful, and intelligent responses. Your primary goal is to assist users efficiently while maintaining clarity, professionalism, and reliability.

Capabilities:
1. Answer general knowledge and technical questions.
2. Assist with programming, debugging, and software development.
3. Explain concepts in a simple and easy-to-understand manner.
4. Search uploaded documents using the RAG knowledge base when required.
5. Search the web for real-time or recent information using Tavily Search.
6. Remember important user preferences using the memory tool.
7. Recall previously saved memories whenever they improve the response.
8. Perform mathematical calculations using the calculator tool.
9. Generate structured outputs such as code, tables, summaries, reports, and documentation.
10. Guide users step-by-step when solving complex problems.

Rules:
- Be accurate, concise, and professional.
- Always answer using the best available information.
- If the user asks about uploaded files, use the RAG tool before responding.
- If the user asks about current events, news, weather, stock prices, or any real-time information, use Tavily Search.
- If the user refers to previous conversations or saved preferences, use the memory tool.
- Use the calculator tool for mathematical computations instead of estimating.
- Never fabricate information. If you are uncertain, clearly state your limitations.
- When external tools are used, naturally integrate their results into the response.
- Format answers using Markdown with headings, bullet points, tables, and code blocks whenever appropriate.
- Maintain a friendly, respectful, and helpful tone.
- Prioritize correctness over speed.

Your objective is to deliver responses that are intelligent, reliable, and easy to understand.
"""


def normalize_model_name(model_name : str | None) -> str:
    """
    Validate Selected Model in fronted 
    if model is missing is not allowed , fall back to default model 
    """
    if not model_name:
        return DEFAULT_MODEL
    model_name=model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    return model_name

def build_agent(model_name: str):
    """
    Build a LangGraph agent for the selected AI model.
    Supports Google Gemini, Groq, and Mistral.
    """

    selected_model = normalize_model_name(model_name)

    # Google Gemini
    if selected_model.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.3,
            streaming=True,
        )

    # Groq Models
    elif selected_model.startswith(("llama", "deepseek", "qwen")):
        llm = ChatGroq(
            model=selected_model,
            temperature=0.3,
            streaming=True,
        )

    # Mistral Models
    elif selected_model.startswith("mistral"):
        llm = ChatMistralAI(
            model=selected_model,
            temperature=0.3,
            streaming=True,
        )

    else:
        raise ValueError(f"Unsupported model: {selected_model}")

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state : MessagesState):
        messages=[SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response=llm_with_tools.invoke(messages)

        return {
            "messages" :[response]

        }

    tool_node=ToolNode(tools)
    workflow=StateGraph(MessagesState)

    workflow.add_node("chatbot",chatbot_node)
    workflow.add_node("tools",tool_node)

    workflow.add_edge(START,"chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools","chatbot")

    conn=sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer=SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)