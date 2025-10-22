from prefect import flow
from prefect.deployments import Deployment
from prefect.client.schemas.schedules import CronSchedule

# tasks:
from safe_roads.deploy.fetch_weather_hourly import get_hourly_weather
from safe_roads.deploy.predict import predict
from safe_roads.deploy.transform_live_data import transform_data

TIMEZONE = "Europe/London"

@flow(name="safe-roads-hourly-pipeline")
def hourly_pipeline():
    f = get_hourly_weather.submit()
    t = transform_data.submit(wait_for=[f])
    p = predict.submit(wait_for=[t])
    p.result()  # optional: blocks the flow run until done

if __name__ == "__main__":
    Deployment.build_from_flow(
        flow=hourly_pipeline,
        name="hourly-eu-london",
        schedule=CronSchedule(cron="0 * * * *", timezone=TIMEZONE), 
        work_pool_name="default-agent-pool",  
        parameters={},  
        tags=["hourly", "prod"],
    ).apply()
