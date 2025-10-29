from dotenv import load_dotenv
from safe_roads.utils.data import free_memory
from safe_roads.deploy.transform_live_data import transform_data
from safe_roads.deploy.predict import predict
from safe_roads.deploy.fetch_weather_hourly import get_hourly_weather
from safe_roads.deploy.commit_to_git import commit_to_git

load_dotenv()

def hourly_pipeline():
    try:
        get_hourly_weather()
    finally:
        free_memory()

    try:
        transform_data()
    finally:
        free_memory()

    try:
        predict()
    finally:
        free_memory()

    try:
        commit_to_git()
    finally:
        free_memory()


def handler(event, context):
    result = hourly_pipeline()
    return {"ok": True, "result": str(result)}

if __name__ == "__main__":
    hourly_pipeline()
