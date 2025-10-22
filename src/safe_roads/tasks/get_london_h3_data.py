import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Polygon
from prefect import task, get_run_logger

from safe_roads.utils.config import load_config, get_pg_url
from safe_roads.utils.data import df_to_pg

def get_borough_data() -> gpd.GeoDataFrame:

    config = load_config()
    url = config["LONDON_BOROUGH_URL"]

    gdf = gpd.read_file(url)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def borough_to_h3(df, res: int) -> pd.DataFrame:

    rows = []
    df['district'] = (
    df['district']
      .str.lower()
      .str.replace(r'[^a-z0-9]+', '', regex=True) 
      .replace({
          'cityandcountyofthecityoflondon': 'cityoflondon',
          'cityofwestminster': 'westminster'
      })
    )

    for _, row in df[['district', 'geometry']].iterrows():
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue
        cells = h3.geo_to_cells(geom, res)
        rows.extend((row['district'].strip().lower(), h) for h in cells)

    return pd.DataFrame(rows, columns=['district', 'h3'])


@task(name="Load London Borough Data")
def load_borough_h3_data():

    url = get_pg_url()
    logger = get_run_logger()

    logger.info("Downloading Borough boundaries")
    gdf = get_borough_data()

    logger.info("Converting to H3 cells")
    df = borough_to_h3(gdf, 10)

    logger.info("Loading table into database")
    df_to_pg(df, 'london_h3', url, if_exists='replace')
    logger.info("Loaded to psql database")


if __name__ == '__main__':
    load_borough_h3_data()