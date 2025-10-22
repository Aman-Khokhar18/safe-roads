FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PREFECT_LOGGING_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage docker layer cache for deps
COPY pyproject.toml* setup.cfg* requirements.txt* ./
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Copy project and install
COPY . .
RUN pip install -e . && pip install "prefect>=3,<4"
