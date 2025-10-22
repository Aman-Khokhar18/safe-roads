FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PREFECT_LOGGING_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml* setup.cfg* requirements.txt* ./
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    apt-get purge -y --auto-remove build-essential git && \
    rm -rf /var/lib/apt/lists/*


COPY . .
RUN pip install -e .


