from prefect import task, get_run_logger
from pathlib import Path
from safe_roads.tasks.download_collision_file import download_collision_file
from safe_roads.utils.config import load_config




@task(name="Download Collision Dataset")
def download_collision_dataset(start_year, end_year):
    
    log = get_run_logger()
    config = load_config()

    collision_data_types = config['COLLISION_DATA_TYPES']
    outdir = config['COLLISION_DATA_OUTDIR']

    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year})")
    
    for year in range(start_year, end_year+1):
        for file_type in collision_data_types:
            download_collision_file(year=year,file_type=file_type,outdir=outdir)

            log.info(f"Downloaded File for: {year} {file_type}")
