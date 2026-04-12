FROM python:3.11-slim

LABEL org.opencontainers.image.title="SQL Query Optimization OpenEnv"
LABEL org.opencontainers.image.version="2.0.0"

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY sql_opt_env/ ./sql_opt_env/
COPY server/ ./server/
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY setup.py .
COPY README.md .

RUN mkdir -p ./dashboard/
COPY dashboard/ ./dashboard/

RUN pip install --no-cache-dir -e . --no-deps

# Create __init__.py to make server a proper Python package
RUN echo "from .app import app, main\n__all__ = ['app', 'main']" > /app/server/__init__.py

RUN chown -R appuser:appuser /app
USER appuser

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H 'Content-Type: application/json' \
        -d '{"task_id":"select_star_cleanup"}' \
        | grep -q '"task_id"' || exit 1

# Run server.py directly — it has main() and all endpoints
CMD ["uvicorn", "server.app:main", "--host", "0.0.0.0", "--port", "7860"]