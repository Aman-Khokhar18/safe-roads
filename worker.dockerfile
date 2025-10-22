FROM prefecthq/prefect:3-latest

# Avoid pip-as-root warnings inside the image
ENV PIP_ROOT_USER_ACTION=ignore

# Install Docker worker support once (cleaner than doing it every boot)
RUN pip install --no-cache-dir -U prefect-docker docker

# On start: ensure pool exists, then start worker
CMD ["bash","-lc", "\
(prefect work-pool inspect docker-safe-roads >/dev/null 2>&1) || \
prefect work-pool create -t docker docker-safe-roads && \
prefect worker start --type docker --pool docker-safe-roads \
"]
