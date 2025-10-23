# src/safe_roads/deploy/deploy.py
from dotenv import load_dotenv
from prefect import flow

from safe_roads.deploy.fetch_weather_hourly import get_hourly_weather
from safe_roads.deploy.transform_live_data import transform_data
from safe_roads.deploy.predict import predict


load_dotenv()

@flow(name="safe-roads-hourly-pipeline")
def hourly_pipeline():
    get_hourly_weather()
    transform_data ()
    predict.submit()

# ---- Lambda entrypoint ----
def handler(event, context):
    result = hourly_pipeline()
    return {"ok": True, "result": result}

if __name__ == "__main__":
    hourly_pipeline()
