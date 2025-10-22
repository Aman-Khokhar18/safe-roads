FROM prefecthq/prefect:3-latest
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --no-cache-dir -U prefect-docker docker
CMD ["bash","-lc", "\
(prefect work-pool inspect docker-safe-roads >/dev/null 2>&1) || \
prefect work-pool create -t docker docker-safe-roads && \
prefect worker start --type docker --pool docker-safe-roads \
"]
