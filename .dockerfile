
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml* setup.cfg* requirements.txt* ./
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi


COPY . .
RUN pip install -e . && pip install "prefect>=3,<4"


ENV PREFECT_LOGGING_LEVEL=INFO
