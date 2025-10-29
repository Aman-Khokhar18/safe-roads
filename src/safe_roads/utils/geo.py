from pyproj import Transformer
import numpy as np
import pandas as pd
import h3
from tqdm.auto import tqdm

import warnings
from dataclasses import dataclass
from typing import Iterable, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.ops import unary_union


def uk_to_lat_lon(easting, northing):
    """
    Convert UK grid coordinates (Easting/Northing) to latitude/longitude
    """
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326")
    lat, lon = transformer.transform(easting, northing)
    return lat, lon


def convert_df_to_h3(df: pd.DataFrame, RES: int) -> pd.DataFrame:
    """
    Converts British National Grid (EPSG:27700) Easting/Northing -> H3 cell (v4) at RES.
    """
    if not {"Easting", "Northing"}.issubset(df.columns):
        raise KeyError("DataFrame must contain 'Easting' and 'Northing' columns")

    df = df.copy()

    # Vectorized projection: (E,N) -> (lon,lat)
    transformer = Transformer.from_crs(27700, 4326, always_xy=True)
    E = df["Easting"].to_numpy()
    N = df["Northing"].to_numpy()
    lons, lats = transformer.transform(E, N)

    # Fast Python-level loop for H3 (no vectorized API available)
    cells = [h3.latlng_to_cell(lat, lon, RES) for lat, lon in zip(lats, lons)]

    df["h3"] = cells
    return df


def closest_h3_station(df: pd.DataFrame, parent_res: int = 4) -> pd.DataFrame:
    if "h3" in df.columns:
        df = df.dropna(subset=["h3"])
        df['parent_h3'] = [h3.cell_to_parent(h, parent_res) for h in df["h3"]]    
    else:
        return print("DataFrame doesnot contain H3 Column")
    return df


