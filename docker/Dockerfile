# docker/Dockerfile
FROM python:3.12-slim

ENV TZ=Europe/London PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . .

# <<< ADD THIS so /app/src is on the import path
ENV PYTHONPATH=/app/src

# Run your flow
CMD ["python", "-m", "safe_roads.deploy.deploy"]
