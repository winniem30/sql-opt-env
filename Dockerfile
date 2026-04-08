# ─────────────────────────────────────────────────────────────────────────────
# SQL Query Optimization OpenEnv — Dockerfile
# Person B deliverable
#
# Build:   docker build -t sql-opt-env .
# Run:     docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN sql-opt-env
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="SQL Query Optimization OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv RL environment for SQL query optimization"
LABEL org.opencontainers.image.version="1.0.0"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first (layer cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY sql_opt_env/ ./sql_opt_env/
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# Copy dashboard if present (Person C's output)
RUN mkdir -p ./dashboard/ || true
COPY dashboard/ ./dashboard/

# Copy setup files
COPY setup.py .
COPY README.md .

# Install the env package
RUN pip install --no-cache-dir -e . --no-deps

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root
USER appuser

# HF Spaces uses port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

# Health check — OpenEnv validator will ping /reset
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H 'Content-Type: application/json' \
        -d '{"task_id":"select_star_cleanup"}' \
        | grep -q '"task_id"' || exit 1

# Start server
CMD ["python", "server.py"]
