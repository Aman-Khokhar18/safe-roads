FROM postgis/postgis:17-3.4

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      postgresql-17-h3 \
 && rm -rf /var/lib/apt/lists/*
