"""Contains functions for hdb_resale specific data functions."""
import logging

import geopy
import numpy as np
import pandas as pd
import sqlalchemy
from geopy.extra.rate_limiter import RateLimiter

from hdb_resale import sql, utils

# Setup logger
logger = logging.getLogger(__name__)


def geocode_data(data_df, hash_key_columns, engine, metadata, schema_name, table_name, api_key):
    """Geocoded address.

    Get full address, lat and lng for each unique address.
    Only need to run when new addresses get added.
    Uses Bing Map API to geocode addresses.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data with addresses to be geocoded.
    hash_key_columns : List[str]
        List of column names to be used in creating unique hash ID for each row.
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).
    schema_name : str
        Schema name to insert new geocoded address.
    table_name : str
        Table name to insert new geocoded address.
    api_key : str
        API key for Bing Map API.

    Returns
    -------
    add_df : pd.DataFrame
        Data with geocoded latitude and longitude.
    """
    add_df = data_df[["block", "street_name", "town"]].drop_duplicates().reset_index(drop=True)
    add_df["country"] = "Singapore"
    add_df["address"] = add_df["block"] + " " + add_df["street_name"] + ", " + add_df["town"] + ", " + add_df["country"]

    # Setup geocode object
    geolocator = geopy.geocoders.Bing(api_key=api_key)
    rateltd_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.1, max_retries=3)

    # Extract location information
    def get_locinfo(add_str, geo_obj):
        loc_obj = geo_obj(add_str, include_country_code=True)

        if loc_obj:
            lat = loc_obj.raw["point"]["coordinates"][0]
            lng = loc_obj.raw["point"]["coordinates"][1]
            add = loc_obj.raw["address"]["formattedAddress"]
        else:
            lat = np.nan
            lng = np.nan
            add = ""

        return (lat, lng, add)

    add_df["lat"] = 0.0
    add_df["lng"] = 0.0
    add_df["full_address"] = ""
    for i in range(add_df.shape[0]):
        per_completed = int(100 * (i / add_df.shape[0]))
        # Log every 10% completed
        if divmod(per_completed, 10) == (i, 0):
            logger.info(f"{per_completed}: Processing address number {i} out of {add_df.shape[0]}......")
        add_tuple = get_locinfo(add_df["address"][i], rateltd_geocode)

        add_df.loc[i, "lat"] = add_tuple[0]
        add_df.loc[i, "lng"] = add_tuple[1]
        add_df.loc[i, "full_address"] = add_tuple[2]

    logger.info("Done all geocoding.")

    # Drop columns for processing use
    add_df = add_df.drop(columns=["address"])

    # Record the source of geocoded information
    add_df["source_name"] = "bing_map_api"

    # Create row hash identifier using key columns
    add_df["_row_hash_id"] = add_df[hash_key_columns].apply(utils.create_hash, axis=1)

    # Remove data based on primary key
    # Needs to be done before data insertion to prevent database duplicated errors
    # No data will be removed if the primary key does not exist
    sql.delete_data_primary_key(
        engine=engine,
        metadata=metadata,
        schema_table_name=f"{schema_name}.{table_name}",
        primary_key=add_df["_row_hash_id"].to_list(),
    )

    # Write data to database
    with engine.connect() as con:
        add_df.to_sql(name=table_name, schema=schema_name, con=con, if_exists="append", index=False)

    return add_df


def get_training_data(cfg, engine, metadata):
    """Get data for ML model training.

    Retrieve data from database.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configs in OmegaConf format.
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).

    Returns
    -------
    X: pd.DataFrame, array-like, shape (n_samples, n_features)
        Data with feature values.
    y: pd.Series, array-like, shape (n_samples,)
        Data with target values.
    """
    MasterData = metadata.tables[cfg.database.master_data.schema_table_name]
    TownData = metadata.tables[cfg.database.town_data.schema_table_name]

    master_query = sqlalchemy.select(
        MasterData.c.date,
        MasterData.c.town,
        MasterData.c.flat_type,
        MasterData.c.floor_area_sqm,
        MasterData.c.remaining_lease,
        MasterData.c.resale_price,
        MasterData.c.min_storey_range,
        MasterData.c.lat,
        MasterData.c.lng,
    )
    master_data = pd.read_sql(master_query, con=engine)

    town_query = sqlalchemy.select(
        TownData.c.town,
        TownData.c.pct_population_below24,
        TownData.c.pct_population_2559,
        TownData.c.pct_population_above60,
        TownData.c.pct_hdb,
        TownData.c.pct_condo,
        TownData.c.pct_landed,
        TownData.c.pct_income_below4k,
        TownData.c.pct_income_4k8k,
        TownData.c.pct_income_8k12k,
        TownData.c.pct_income_above12k,
        TownData.c.pct_household_withdisability,
    )
    town_data = pd.read_sql(town_query, con=engine)

    data = master_data.merge(town_data, how="left", on="town")

    # Perform data type conversion
    data_type_conversion = {
        "date": "datetime64[ns]",
        "town": "string",
        "flat_type": "string",
        "floor_area_sqm": "int",
        "remaining_lease": "int",
        "resale_price": "float",
        "min_storey_range": "int",
        "lat": "float",
        "lng": "float",
        "pct_population_below24": "float",
        "pct_population_2559": "float",
        "pct_population_above60": "float",
        "pct_hdb": "float",
        "pct_condo": "float",
        "pct_landed": "float",
        "pct_income_below4k": "float",
        "pct_income_4k8k": "float",
        "pct_income_8k12k": "float",
        "pct_income_above12k": "float",
        "pct_household_withdisability": "float",
    }
    data = data.astype(data_type_conversion)

    # Engineer features needed
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    data["flat_type_value"] = data["flat_type"].map(cfg.model.flat_type_mapping, na_action="ignore")

    # Split into features (x) and target (y)
    X = data.loc[:, cfg.model.colnames_selection.x]
    y = data.loc[:, cfg.model.colnames_selection.y]

    # Enforce that all column names have string type
    # Otherwise sklearn model.fit will error out
    X = X.rename(columns=str)

    return X, y
