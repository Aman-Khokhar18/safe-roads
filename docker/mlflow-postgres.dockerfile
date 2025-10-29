# mlflow-postgres.Dockerfile
FROM ghcr.io/mlflow/mlflow:v3.4.0
RUN pip install --no-cache-dir psycopg2-binary
