from pathlib import Path
from typing import List

from prefect import flow, task, get_run_logger

from safe_roads.tasks.download_collision_data import download_collision_dataset
from safe_roads.tasks.load_csv_to_postgres import load_csv_to_pg
from safe_roads.utils.config import load_config, get_pg_url, year_month


@task(name='Search CSV files')
def find_csvs(dirpath: Path, pattern: str = "*.csv") -> List[Path]:
    files = sorted(p for p in dirpath.glob(pattern) if p.is_file())
    return files


@flow(name="ingest-collision-data")
def ingest_collision_data(
    force_download: bool = False,
    pattern: str = "*.csv",
    schema: str = "public",
    if_exists: str = "append",
):

    logger = get_run_logger()
    config = load_config()


    outdir = config['COLLISION_DATA_OUTDIR']
    outdir = Path(outdir)

    start_date = config['START_DATE']
    end_date = config['END_DATE']

    start_year, _ = year_month(start_date)
    end_year, _ = year_month(end_date)

    logger.info("Loading collisiondata")
    csvs = find_csvs.submit(outdir, pattern).result()
    if force_download or not csvs:
        logger.info(
            "No CSVs found at %s (pattern=%s) or force_download=True. Downloading dataset...",
            outdir, pattern,
        )

        download_collision_dataset(start_year=start_year, end_year=end_year)
        csvs = find_csvs.submit(outdir, pattern).result()


    logger.info("Found %d CSV file(s) in %s", len(csvs), outdir)

    db_url = get_pg_url()
    summaries = []

    for csv_path in csvs:
        fut = load_csv_to_pg.submit(
            csv_path=csv_path,
            db_url=db_url
        )
        summaries.append(fut)

    results = [s.result() for s in summaries]
    logger.info("collisiondata loaded to database")
    return results


if __name__ == "__main__":
    ingest_collision_data()
