from dotenv import load_dotenv
from prefect import flow

from safe_roads.deploy.fetch_weather_hourly import get_hourly_weather
from safe_roads.deploy.transform_live_data import transform_data
from safe_roads.deploy.predict import predict

load_dotenv()

@flow(name="safe-roads-hourly-pipeline")
def hourly_pipeline():
    f = get_hourly_weather.submit()
    t = transform_data.submit(wait_for=[f])  
    p = predict.submit(wait_for=[t])         
    return p.result()             

           
if __name__ == "__main__":
    hourly_pipeline()
