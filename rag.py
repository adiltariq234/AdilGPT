"""
RAG Pipeline with lazy loading for faster startup.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Document readers (lazy import)
try:
    import docx2txt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# LangChain imports
try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        Language,
        MarkdownHeaderTextSplitter
    )
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"[RAG] LangChain components not available: {e}")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))

# Create directories
Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────
_embeddings = None
_vectorstore = None

def get_embeddings():
    """Lazy-load embeddings model."""
    global _embeddings
    if _embeddings is None:
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available")
        print("[RAG] Loading embedding model...")
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        print("[RAG] Embedding model loaded.")
    return _embeddings

def get_vectorstore():
    """Lazy-load vector store."""
    global _vectorstore
    if _vectorstore is None:
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available")
        print("[RAG] Loading vector store...")
        _vectorstore = Chroma(
            collection_name="Agentic_Chatbot_docs",
            embedding_function=get_embeddings(),
            persist_directory="chroma_db"
        )
        print("[RAG] Vector store loaded.")
    return _vectorstore


# ─────────────────────────────────────────────────────────────
# Document Reading
# ─────────────────────────────────────────────────────────────
def read_file_text(file_path: str) -> str:
    """Read text from supported file types with error handling."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if suffix == ".pdf":
        if not PDF_AVAILABLE:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        return _read_pdf(file_path)

    elif suffix == ".docx":
        if not DOCX_AVAILABLE:
            raise ImportError("docx2txt not installed. Run: pip install docx2txt")
        return docx2txt.process(file_path)

    elif suffix in [".txt", ".py", ".md", ".csv", ".json", ".yaml", ".yml"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Allowed: .pdf, .docx, .txt, .md, .py, .csv"
    )


def _read_pdf(file_path: str) -> str:
    """Read PDF with better error handling."""
    reader = PdfReader(file_path)
    text_parts = []

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        except Exception as e:
            text_parts.append(f"[Error extracting page {i+1}: {e}]")

    return "\n".join(text_parts)


# ─────────────────────────────────────────────────────────────
# Content Hashing (Deduplication)
# ─────────────────────────────────────────────────────────────
def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_existing_hashes(thread_id: str) -> set:
    """Get hashes of existing documents for a thread."""
    try:
        results = get_vectorstore().get(
            where={"thread_id": thread_id},
            include=["metadatas"]
        )
        if results and "metadatas" in results:
            return {
                meta.get("content_hash", "")
                for meta in results["metadatas"]
                if meta
            }
    except Exception:
        pass
    return set()


# ─────────────────────────────────────────────────────────────
# Smart Chunking
# ─────────────────────────────────────────────────────────────
def get_splitter(file_path: str):
    """Get appropriate text splitter based on file type."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".py":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    elif suffix == ".md":
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
    elif suffix == ".csv":
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", ","]
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )


# ─────────────────────────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────────────────────────
def add_document_to_rag(file_path: str, thread_id: str) -> Dict[str, Any]:
    """
    Read a document, split into chunks, deduplicate, and store in ChromaDB.
    """
    path = Path(file_path)

    # Read text
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    # Get existing hashes for deduplication
    existing_hashes = get_existing_hashes(thread_id)

    # Split text
    splitter = get_splitter(file_path)

    if isinstance(splitter, MarkdownHeaderTextSplitter):
        raw_chunks = splitter.split_text(text)
        chunks = [chunk.page_content for chunk in raw_chunks]
    else:
        chunks = splitter.split_text(text)

    # Build documents with deduplication
    docs: List[Document] = []
    duplicates = 0

    for i, chunk in enumerate(chunks):
        content_hash = compute_content_hash(chunk)

        if content_hash in existing_hashes:
            duplicates += 1
            continue

        doc = Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": path.name,
                "chunk_index": i,
                "content_hash": content_hash,
                "file_type": path.suffix.lower()
            }
        )
        docs.append(doc)
        existing_hashes.add(content_hash)

    if not docs:
        return {
            "filename": path.name,
            "chunks": 0,
            "duplicates_skipped": duplicates,
            "status": "all_duplicates"
        }

    # Add to vector store
    try:
        get_vectorstore().add_documents(docs)
    except Exception as e:
        raise RuntimeError(f"Failed to store documents in vector DB: {e}")

    return {
        "filename": path.name,
        "chunks": len(docs),
        "duplicates_skipped": duplicates,
        "status": "success"
    }


# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────
def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4,
    score_threshold: Optional[float] = None
) -> str:
    """
    Retrieve relevant documents from RAG with similarity scoring.
    """
    threshold = score_threshold or SIMILARITY_THRESHOLD

    try:
        # Try with scores first
        results = get_vectorstore().similarity_search_with_relevance_scores(
            query=query,
            k=k * 2,  # Fetch more to filter
            filter={"thread_id": thread_id}
        )

        # Filter by threshold and take top k
        filtered = [
            (doc, score)
            for doc, score in results
            if score >= threshold
        ][:k]

        if not filtered:
            return "No relevant documents found for this query."

        formatted = []
        for i, (doc, score) in enumerate(filtered, start=1):
            source = doc.metadata.get("source", "uploaded document")
            formatted.append(
                f"[Source {i}: {source} | Relevance: {score:.2f}]\n{doc.page_content}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        # Fallback to regular search without scores
        docs = get_vectorstore().similarity_search(
            query=query,
            k=k,
            filter={"thread_id": thread_id}
        )

        if not docs:
            return "No relevant documents uploaded for this thread."

        formatted = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "uploaded document")
            formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n".join(formatted)


def delete_thread_documents(thread_id: str) -> bool:
    """Delete all documents for a thread from ChromaDB."""
    try:
        get_vectorstore().delete(filter={"thread_id": thread_id})
        return True
    except Exception as e:
        print(f"Error deleting documents for thread {thread_id}: {e}")
        return False


def get_thread_document_count(thread_id: str) -> int:
    """Count documents stored for a thread."""
    try:
        results = get_vectorstore().get(where={"thread_id": thread_id})
        if results and "ids" in results:
            return len(results["ids"])
    except Exception:
        pass
    return 0
