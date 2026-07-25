from dotenv import load_dotenv
import os
import json
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from config import settings
from agent import get_agent, ALLOWED_MODELS
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
    get_conversation,
    delete_conversation,
    rename_conversation,
    user_exists,
    get_user_by_username
)
from rag import add_document_to_rag, delete_thread_documents
from tools import tools
from middleware import (
    RequestIDMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware
)
from auth import (
    authenticate_user,
    register_user,
    create_access_token,
    require_auth,
    optional_auth,
    hash_password
)

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
class RequestIDFilter(logging.Filter):
    """Filter that ensures request_id is always present on log records."""
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        return True

# Setup logging with safe format
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s"
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.addFilter(RequestIDFilter())

# File handler
file_handler = logging.FileHandler("data/app.log", encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.addFilter(RequestIDFilter())

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.log_level))
root_logger.handlers = []  # Clear existing handlers
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    thread_id: str = Field(default="default", min_length=1, max_length=100)
    model: str = Field(default="gemini-2.5-flash", description="LLM model name")

    @field_validator("model")
    def validate_model(cls, v):
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Invalid model. Allowed: {', '.join(sorted(ALLOWED_MODELS))}")
        return v

    @field_validator("message")
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    email: Optional[str] = Field(default=None, max_length=100)


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    role: str
    content: str


# ─────────────────────────────────────────────────────────────
# Startup Timer
# ─────────────────────────────────────────────────────────────
import time
_startup_start = time.time()

def _print_startup(msg):
    elapsed = time.time() - _startup_start
    print(f"[Startup {elapsed:.2f}s] {msg}")

_print_startup("Imports loaded")

# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 {settings.app_name} v{settings.app_version} starting up in {settings.env} mode...")
    init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info(f"🛑 {settings.app_name} shutting down...")


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready AI Chatbot with Auth, RAG, Multi-LLM, and Memory",
    docs_url="/docs" if settings.show_docs else None,
    redoc_url="/redoc" if settings.show_docs else None,
    lifespan=lifespan,
)

# Middleware Stack
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_requests_per_minute
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Directories
Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Health & Monitoring
# ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": settings.app_version}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    try:
        conversations = list_conversations()
        return {
            "status": "ready",
            "database": "connected",
            "conversations_count": len(conversations)
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# ─────────────────────────────────────────────────────────────
# Auth Endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/auth/register", tags=["Authentication"])
async def register(request: RegisterRequest):
    """Register a new user account."""
    try:
        user = register_user(
            username=request.username,
            password=request.password,
            email=request.email
        )
        logger.info(f"User registered: {user['username']}")
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"]
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login(request: LoginRequest):
    """Login and get JWT access token."""
    user = authenticate_user(request.username, request.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": str(user["id"]), "username": user["username"]})

    logger.info(f"User logged in: {user['username']}")

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "is_admin": user["is_admin"]
        }
    }


@app.get("/api/v1/auth/me", tags=["Authentication"])
async def get_me(current_user: dict = Depends(require_auth)):
    """Get current logged-in user info."""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "is_admin": current_user["is_admin"]
    }


# ─────────────────────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Pages"])
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "models": sorted(ALLOWED_MODELS)
        }
    )


# ─────────────────────────────────────────────────────────────
# Conversation API (Protected)
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/conversations", response_model=list[ConversationResponse], tags=["Conversations"])
async def get_conversations(current_user: Optional[dict] = Depends(optional_auth)):
    """List conversations for current user (or all if anonymous)."""
    try:
        user_id = current_user["id"] if current_user else None
        items = list_conversations(user_id=user_id)
        return [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else ""
            }
            for item in items
        ]
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")


@app.get("/api/v1/history/{thread_id}", response_model=list[MessageResponse], tags=["Conversations"])
async def get_history(thread_id: str, current_user: Optional[dict] = Depends(optional_auth)):
    """Get chat history for a specific thread."""
    try:
        # Verify ownership if user is logged in
        if current_user:
            conv = get_conversation(thread_id)
            if conv and conv.user_id and conv.user_id != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")

        messages = get_chat_history(thread_id)
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@app.put("/api/v1/conversations/{thread_id}/rename", tags=["Conversations"])
async def rename_thread(
    thread_id: str,
    request: RenameRequest,
    current_user: Optional[dict] = Depends(optional_auth)
):
    """Rename a conversation."""
    try:
        if current_user:
            conv = get_conversation(thread_id)
            if conv and conv.user_id and conv.user_id != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")

        success = rename_conversation(thread_id, request.title)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"success": True, "message": "Conversation renamed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rename conversation {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename conversation")


@app.delete("/api/v1/conversations/{thread_id}", tags=["Conversations"])
async def delete_conversation_endpoint(
    thread_id: str,
    current_user: Optional[dict] = Depends(optional_auth)
):
    """Delete a conversation and all associated data."""
    try:
        if current_user:
            conv = get_conversation(thread_id)
            if conv and conv.user_id and conv.user_id != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")

        deleted = delete_conversation(thread_id)
        delete_thread_documents(thread_id)

        if deleted:
            return {"success": True, "message": f"Conversation {thread_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


# ─────────────────────────────────────────────────────────────
# File Upload (Protected)
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user: Optional[dict] = Depends(optional_auth)
):
    """Upload a document for RAG processing."""
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required")

    thread_id = thread_id.strip()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    filename = file.filename
    suffix = Path(filename).suffix.lower()
    allowed_extensions = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB"
        )

    try:
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-").strip()
        if not safe_filename:
            safe_filename = f"uploaded_file{suffix}"

        file_id = str(uuid.uuid4())
        file_path = Path("uploads") / f"{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        user_id = current_user["id"] if current_user else None
        create_or_update_conversation(thread_id, f"Uploaded: {filename}", user_id=user_id)

        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id
        )

        logger.info(f"Uploaded {filename} for thread {thread_id}: {result['chunks']} chunks")

        return {
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks.",
            "file_id": file_id,
            "chunks": result["chunks"],
            "duplicates_skipped": result.get("duplicates_skipped", 0)
        }

    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}")
        if "file_path" in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Streaming Chat (Protected)
# ─────────────────────────────────────────────────────────────
def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata: Optional[dict] = None) -> bool:
    metadata = metadata or {}
    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False
    if isinstance(chunk, ToolMessage):
        return False
    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False
    if getattr(chunk, "tool_calls", None):
        return False
    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        return "".join(text_parts)
    return ""


@app.post("/api/v1/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    current_user: Optional[dict] = Depends(optional_auth)
):
    """Stream chat responses using SSE."""
    logger.info(f"Chat request: thread={request.thread_id}, model={request.model}, user={current_user['username'] if current_user else 'anonymous'}")

    try:
        agent = get_agent(request.model)
    except RuntimeError as e:
        logger.error(f"Agent initialization failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected agent error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize AI agent")

    # Verify ownership
    if current_user:
        conv = get_conversation(request.thread_id)
        if conv and conv.user_id and conv.user_id != current_user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")

    try:
        user_id = current_user["id"] if current_user else None
        create_or_update_conversation(request.thread_id, request.message, user_id=user_id)
        save_chat_message(request.thread_id, "user", request.message)
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save message")

    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        token_count = 0

        try:
            inputs = {"messages": [HumanMessage(content=request.message)]}

            import asyncio
            loop = asyncio.get_event_loop()

            def stream_agent():
                return list(agent.stream(inputs, config=config, stream_mode="messages"))

            chunks = await loop.run_in_executor(None, stream_agent)

            for chunk, metadata in chunks:
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)
                if token:
                    final_answer += token
                    token_count += 1
                    yield sse_data({"token": token})

            if final_answer.strip():
                try:
                    save_chat_message(request.thread_id, "assistant", final_answer)
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}")

            logger.info(f"Stream complete: {token_count} tokens for thread {request.thread_id}")
            yield sse_data({"done": True, "tokens": token_count})

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield sse_data({"error": "An error occurred during generation. Please try again."})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        }
    )


# ─────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": True, "detail": "Internal server error", "status_code": 500}
    )


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _print_startup(f"Starting Uvicorn server on {settings.host}:{settings.port} (reload={settings.reload})")
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        access_log=True,
        log_level=settings.log_level.lower()
    )
