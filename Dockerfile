# Base: AWS Lambda Python 3.11 (also fine to run on EC2/containers)
FROM public.ecr.aws/lambda/python:3.11

# Timezone
ENV TZ=Europe/London
RUN yum install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone && \
    yum clean all

# Runtime/cache dirs on Lambda's writable /tmp
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
    METEOSTAT_CACHE_DIR=/tmp/meteostat \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY . ${LAMBDA_TASK_ROOT}

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}:${LAMBDA_TASK_ROOT}/src

# Disable Lambda entrypoint so we can run a normal command
ENTRYPOINT []

# Run your deploy command (adjust if your module entry point differs)
CMD ["python", "-m", "safe_roads.deploy.deploy"]
