# ORACLE backend — slim image for Render / any Docker host.
# Single-stage: numpy/scipy/bcrypt all ship wheels, no compiler needed.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends sqlite3 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Render/free hosts have an ephemeral filesystem: keep the SQLite demo DB in /tmp
# and re-seed it on every boot.
ENV PYTHONPATH=/app \
    ORACLE_DB_PATH=/tmp/oracle.db \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
