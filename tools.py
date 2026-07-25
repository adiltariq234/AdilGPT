"""
Tool Definitions — Thread-safe, no global state.
Each tool accepts thread_id as a parameter.
"""

import os
import math
import ast
import operator
from typing import Any, Dict, Optional
from functools import wraps
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from database import search_memory, save_memory
from rag import retrieve_from_rag


# ─────────────────────────────────────────────────────────────
# Error Boundary Decorator
# ─────────────────────────────────────────────────────────────
def tool_error_boundary(func):
    """Wrap tools to catch exceptions and return graceful errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return f"[Tool Error - {tool_name}]: {str(e)}"
    return wrapper


# ─────────────────────────────────────────────────────────────
# Safe Calculator (NO eval())
# ─────────────────────────────────────────────────────────────
class SafeCalculator:
    """Safe mathematical expression evaluator using AST."""

    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Any:
        """Safely evaluate a mathematical expression."""
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            return cls._eval_node(tree.body)
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.ALLOWED_OPS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = cls._eval_node(node.operand)
            return cls.ALLOWED_OPS[op_type](operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in cls.ALLOWED_FUNCS:
                raise ValueError(f"Unknown function: {func_name}")
            args = [cls._eval_node(arg) for arg in node.args]
            return cls.ALLOWED_FUNCS[func_name](*args)

        elif isinstance(node, ast.Name):
            if node.id not in cls.ALLOWED_FUNCS:
                raise ValueError(f"Unknown identifier: {node.id}")
            return cls.ALLOWED_FUNCS[node.id]

        elif isinstance(node, ast.Expression):
            return cls._eval_node(node.body)

        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")


# ─────────────────────────────────────────────────────────────
# Web Search Tool
# ─────────────────────────────────────────────────────────────
class TavilySearchTool:
    """Wrapper around TavilySearch with structured output."""

    def __init__(self):
        self._search = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )

    def invoke(self, query: str) -> str:
        try:
            result = self._search.invoke(query)
            return self._format_results(result)
        except Exception as e:
            return f"[Search Error]: {str(e)}"

    def _format_results(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            parts = []
            if "answer" in result and result["answer"]:
                parts.append(f"**Answer:** {result['answer']}")

            if "results" in result and result["results"]:
                parts.append("\n**Sources:**")
                for i, item in enumerate(result["results"][:5], 1):
                    title = item.get("title", "Untitled")
                    url = item.get("url", "")
                    content = item.get("content", "")[:300]
                    parts.append(f"{i}. [{title}]({url})\n   {content}")

            return "\n\n".join(parts) if parts else str(result)

        return str(result)


web_search_tool = TavilySearchTool()


# ─────────────────────────────────────────────────────────────
# Tool Definitions (Thread-safe)
# ─────────────────────────────────────────────────────────────
@tool
@tool_error_boundary
def calculator(expression: str) -> str:
    """
    Perform safe mathematical calculations.

    Supports: +, -, *, /, //, %, **, sqrt(), sin(), cos(), tan(),
    log(), log10(), exp(), abs(), round(), min(), max(), sum(),
    floor(), ceil(), pi, e
    """
    try:
        result = SafeCalculator.evaluate(expression)
        return f"Result: {result}"
    except ValueError as e:
        return f"Calculation Error: {e}"


@tool
@tool_error_boundary
def search_uploaded_documents(query: str, thread_id: str = "") -> str:
    """
    Search relevant information from uploaded documents.
    Use when the user asks about uploaded files.
    """
    if not thread_id:
        return "Error: No thread ID provided. Cannot search documents."

    return retrieve_from_rag(query=query, thread_id=thread_id)


@tool
@tool_error_boundary
def remember_memory(memory: str, thread_id: str = "") -> str:
    """Save an important user preference or fact into long-term memory."""
    if not thread_id:
        return "Error: No thread ID provided. Cannot save memory."

    if not memory or not memory.strip():
        return "Error: Empty memory cannot be saved."

    timestamp = datetime.now(timezone.utc).isoformat()
    enriched_memory = f"[{timestamp}] {memory.strip()}"

    return save_memory(thread_id=thread_id, memory=enriched_memory)


@tool
@tool_error_boundary
def recall_memory(query: str, thread_id: str = "") -> str:
    """Search previously saved memories for this conversation thread."""
    if not thread_id:
        return "Error: No thread ID provided. Cannot recall memories."

    memories = search_memory(thread_id=thread_id, query=query)

    if memories == "No saved memory found.":
        return "No memories found for this conversation."

    return f"**Relevant Memories:**\n{memories}"


@tool
@tool_error_boundary
def web_search(query: str) -> str:
    """
    Search the web for real-time or recent information.
    Use for current events, news, weather, stock prices, or recent data.
    """
    if not query or not query.strip():
        return "Error: Empty search query."

    return web_search_tool.invoke(query.strip())


# Export tools list for agent binding
tools = [
    calculator,
    search_uploaded_documents,
    remember_memory,
    recall_memory,
    web_search,
]
