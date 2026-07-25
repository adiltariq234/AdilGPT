# ─────────────────────────────────────────────────────────────
# AdilGPT - Production Dockerfile
# ─────────────────────────────────────────────────────────────

# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     libpq-dev     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends     libpq5     && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appgroup . .

# Create necessary directories
RUN mkdir -p data uploads chroma_db templates     && chown -R appuser:appgroup data uploads chroma_db

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
