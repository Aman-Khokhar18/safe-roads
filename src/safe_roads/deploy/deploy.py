import os
from dotenv import load_dotenv
from prefect import flow
from prefect.client.schemas.schedules import CronSchedule

from safe_roads.deploy.fetch_weather_hourly import get_hourly_weather
from safe_roads.deploy.predict import predict
from safe_roads.deploy.transform_live_data import transform_data

load_dotenv()

@flow(name="safe-roads-hourly-pipeline")
def hourly_pipeline():
    f = get_hourly_weather.submit()
    t = transform_data.submit(wait_for=[f])
    p = predict.submit(wait_for=[t])
    p.result()

if __name__ == "__main__":
    RUN_ENV = {
        "PGHOST": os.getenv("PGHOST"),
        "PGPORT": os.getenv("PGPORT", "5432"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB"),
        "POSTGRES_SSLMODE": os.getenv("POSTGRES_SSLMODE", "require"),
        "HUGGINGFACE_HUB_TOKEN": os.getenv("HUGGINGFACE_HUB_TOKEN"),
        "HF_MODEL_REPO": os.getenv("HF_MODEL_REPO"),
        "HF_MODEL_FILE": os.getenv("HF_MODEL_FILE"),
    }

    hourly_pipeline.deploy(
        name="hourly-eu-london",
        work_pool_name="docker-safe-roads",
        schedule=CronSchedule(cron="0 * * * *", timezone="Europe/London"),
        job_variables={
            "image": "amank/saferoads:latest",
            "env": RUN_ENV,
        },
    )
