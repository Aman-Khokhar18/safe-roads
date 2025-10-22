# docker/Dockerfile
FROM python:3.12-slim

# Set container timezone to Europe/London so any localtime ops match the schedule
ENV TZ=Europe/London \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r deploy_requirements.txt

# Copy your project
COPY . .

CMD ["python", "src/safe_roads/deploy/deploy.py"]
