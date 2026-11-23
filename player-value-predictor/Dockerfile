# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PVP_ARTIFACT_DIR=/app/artifacts/release \
    PVP_LLM_ENABLED=false \
    PVP_LLM_USAGE_DB=/tmp/player-value/llm_usage.sqlite3 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    PORT=8000

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --home-dir /app app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt && pip check
COPY src/ ./src/
COPY artifacts/release/ ./artifacts/release/
COPY licenses/ ./licenses/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=3)"
CMD ["python", "src/server.py"]
