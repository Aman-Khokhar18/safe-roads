import pandas as pd
from sqlalchemy import text, create_engine
from sklearn.model_selection import train_test_split

from safe_roads.utils.config import get_pg_url


def data_loader(table, chunksize: int | None = None):
    url = get_pg_url()
    engine = create_engine(url)
    
    query = f"""
        SELECT
            h3,
            neighbor_1,
            neighbor_2,
            neighbor_3,
            neighbor_4,
            neighbor_5,
            neighbor_6,
           parent_h3,
           highway,
           name,
           lanes,
           width,
           surface,
           smoothness,
           oneway,
           junction,
           traffic_signals,
           traffic_calming,
           crossing,
           sidewalk,
           cycleway,
           bicycle,
           lit,
           access,
           vehicle,
           hgv,
           psv,
           bus,
           overtaking,
           bridge,
           tunnel,
           layer,
           incline,
           barrier,
           amenity,
           year,
           month,
           day,
           hour,
           minute,
           is_weekend,
           road_class3,
           borough,
           maxspeed_mph,
           traffic_lights,
           crossings,
           cycleways,
           motorways_other,
           is_junction,
           is_turn,
           junction_degree,
           temp,
           dwpt,
           rhum,
           prcp,
           snow,
           wdir,
           wspd,
           wpgt,
           pres,
           tsun,
           coco
        FROM {table}
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, chunksize=chunksize)

    return df


def combine_data(pos_df: pd.DataFrame,
                    neg_df: pd.DataFrame,
                    label_col: str = "target",
                    pos_label: int = 1,
                    neg_label: int = 0,
                    shuffle: bool = True,
                    random_state: int = 42) -> pd.DataFrame:
    pos = pos_df.copy()
    neg = neg_df.copy()
    pos[label_col] = pos_label
    neg[label_col] = neg_label

    df = pd.concat([pos, neg], axis=0, ignore_index=True)

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def split_train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42, stratify=None):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=(y if stratify else None)
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state,
        stratify=(y_temp if stratify else None)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test



