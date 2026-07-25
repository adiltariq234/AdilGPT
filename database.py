"""
Database layer with User authentication support.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import List, Optional, Generator

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Index, Boolean, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool

Path("data").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/chatbot_memory.db")
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# ─────────────────────────────────────────────────────────────
# Engine with Connection Pooling
# ─────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_recycle=POOL_RECYCLE,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

Base = declarative_base()


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=True)  # NULL = anonymous
    title = Column(String(200), default="New Chat")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_conversation_user_updated", "user_id", "updated_at"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String(100), index=True, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_message_thread_time", "thread_id", "created_at"),
    )


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String(100), index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=True)
    memory = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_memory_thread_time", "thread_id", "created_at"),
    )


# ─────────────────────────────────────────────────────────────
# Database Initialization
# ─────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────────────────────
# Context Managers
# ─────────────────────────────────────────────────────────────
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# User Operations
# ─────────────────────────────────────────────────────────────
def create_user(username: str, hashed_password: str, email: Optional[str] = None) -> User:
    """Create a new user."""
    with get_db_session() as db:
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(user)
        db.flush()  # Get ID without committing
        db.refresh(user)
        return user


def get_user_by_username(username: str) -> Optional[User]:
    """Get user by username."""
    with get_db_session() as db:
        return db.query(User).filter(User.username == username).first()


def get_user_by_id(user_id: int) -> Optional[User]:
    """Get user by ID."""
    with get_db_session() as db:
        return db.query(User).filter(User.id == user_id).first()


def user_exists(username: str) -> bool:
    """Check if username already exists."""
    with get_db_session() as db:
        return db.query(User).filter(User.username == username).first() is not None


# ─────────────────────────────────────────────────────────────
# Conversation Operations (with user_id)
# ─────────────────────────────────────────────────────────────
def create_or_update_conversation(
    thread_id: str,
    first_message: Optional[str] = None,
    user_id: Optional[int] = None
) -> None:
    """Create or update conversation with user association."""
    with get_db_session() as db:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"
            if first_message:
                title = first_message.strip()[:40]
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                user_id=user_id,
                title=title,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            db.add(conversation)
        else:
            conversation.updated_at = datetime.now(timezone.utc)
            if user_id and not conversation.user_id:
                conversation.user_id = user_id


def list_conversations(
    user_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Conversation]:
    """List conversations, optionally filtered by user."""
    with get_db_session() as db:
        query = db.query(Conversation)
        if user_id is not None:
            query = query.filter(
                (Conversation.user_id == user_id) | (Conversation.user_id.is_(None))
            )
        return (
            query.order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )


def get_conversation(thread_id: str) -> Optional[Conversation]:
    """Get a single conversation by thread ID."""
    with get_db_session() as db:
        return (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )


def delete_conversation(thread_id: str) -> bool:
    """Delete a conversation and all associated data."""
    with get_db_session() as db:
        db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).delete()
        db.query(LongTermMemory).filter(LongTermMemory.thread_id == thread_id).delete()
        result = db.query(Conversation).filter(Conversation.thread_id == thread_id).delete()
        return result > 0


def rename_conversation(thread_id: str, new_title: str) -> bool:
    """Rename a conversation."""
    with get_db_session() as db:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )
        if conversation:
            conversation.title = new_title[:200]
            conversation.updated_at = datetime.now(timezone.utc)
            return True
        return False


# ─────────────────────────────────────────────────────────────
# Message Operations
# ─────────────────────────────────────────────────────────────
def save_chat_message(thread_id: str, role: str, content: str) -> None:
    """Save a chat message and update conversation timestamp."""
    with get_db_session() as db:
        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc)
        )
        db.add(msg)

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )
        if conversation:
            conversation.updated_at = datetime.now(timezone.utc)


def get_chat_history(
    thread_id: str,
    limit: int = 100,
    offset: int = 0
) -> List[ChatMessage]:
    """Get chat history for a thread with pagination."""
    with get_db_session() as db:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
            .offset(offset)
            .all()
        )


def get_message_count(thread_id: str) -> int:
    """Count messages in a thread."""
    with get_db_session() as db:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .count()
        )


def delete_chat_history(thread_id: str) -> bool:
    """Delete all messages for a thread."""
    with get_db_session() as db:
        result = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).delete()
        return result > 0


# ─────────────────────────────────────────────────────────────
# Memory Operations
# ─────────────────────────────────────────────────────────────
def save_memory(
    thread_id: str,
    memory: str,
    user_id: Optional[int] = None
) -> str:
    """Save a memory for a thread."""
    if not memory or not memory.strip():
        return "Error: Cannot save empty memory."

    with get_db_session() as db:
        item = LongTermMemory(
            thread_id=thread_id,
            user_id=user_id,
            memory=memory.strip(),
            created_at=datetime.now(timezone.utc)
        )
        db.add(item)

    return "Memory saved successfully."


def search_memory(
    thread_id: str,
    query: str,
    user_id: Optional[int] = None,
    limit: int = 20
) -> str:
    """Search memories for a thread."""
    with get_db_session() as db:
        q = db.query(LongTermMemory).filter(LongTermMemory.thread_id == thread_id)
        if user_id is not None:
            q = q.filter(LongTermMemory.user_id == user_id)

        memories = (
            q.order_by(LongTermMemory.created_at.desc())
            .limit(limit)
            .all()
        )

        if not memories:
            return "No saved memory found."

        return "\n".join([f"- {m.memory}" for m in memories])


def delete_memories(thread_id: str) -> bool:
    """Delete all memories for a thread."""
    with get_db_session() as db:
        result = db.query(LongTermMemory).filter(LongTermMemory.thread_id == thread_id).delete()
        return result > 0


# ─────────────────────────────────────────────────────────────
# Cleanup Operations
# ─────────────────────────────────────────────────────────────
def cleanup_old_conversations(days: int = 30) -> int:
    """Delete conversations older than specified days."""
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db_session() as db:
        old_threads = (
            db.query(Conversation.thread_id)
            .filter(Conversation.updated_at < cutoff)
            .all()
        )

        count = 0
        for (thread_id,) in old_threads:
            delete_conversation(thread_id)
            count += 1

        return count
