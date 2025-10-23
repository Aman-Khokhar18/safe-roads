# Base: AWS Lambda Python 3.11 (fine to run on EC2 too)
FROM public.ecr.aws/lambda/python:3.11

# Timezone
ENV TZ=Europe/London
RUN yum install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone && \
    yum clean all


ENV HOME=/tmp XDG_CACHE_HOME=/tmp TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
    PREFECT_HOME=/tmp/prefect JOBLIB_TEMP_FOLDER=/tmp/joblib \
    METEOSTAT_CACHE_DIR=/tmp/meteostat MPLCONFIGDIR=/tmp/mpl \
    PREFECT_API_ENABLE_EPHEMERAL_SERVER=false \
    PREFECT_LOGGING_TO_API_ENABLED=true \
    PREFECT_RESULTS_PERSIST_BY_DEFAULT=false


WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"


COPY . ${LAMBDA_TASK_ROOT}

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}:${LAMBDA_TASK_ROOT}/src

# Disable Lambda entrypoint so we can run a normal command
ENTRYPOINT []

# Run your deploy command (adjust if your module entry point differs)
CMD ["python", "-m", "safe_roads.deploy.deploy"]
