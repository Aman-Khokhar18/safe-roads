FROM public.ecr.aws/lambda/python:3.11

ENV TZ=Europe/London
RUN yum install -y tzdata && yum clean all

# Lambda write-to-/tmp hygiene
ENV HOME=/tmp XDG_CACHE_HOME=/tmp TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
    PREFECT_HOME=/tmp/prefect JOBLIB_TEMP_FOLDER=/tmp/joblib \
    METEOSTAT_CACHE_DIR=/tmp/meteostat MPLCONFIGDIR=/tmp/mpl \
    PREFECT_API_ENABLE_EPHEMERAL_SERVER=false \
    PREFECT_LOGGING_TO_API_ENABLED=true \
    PREFECT_RESULTS_PERSIST_BY_DEFAULT=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY . ${LAMBDA_TASK_ROOT}
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src

CMD ["safe_roads.deploy.deploy.handler"]
