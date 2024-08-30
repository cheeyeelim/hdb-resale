"""DAG task to process HDB resale data."""
import datetime
import logging
import os
import re

import numpy as np
import pandas as pd
import sqlalchemy
from airflow_submodule.hdb_resale import data, sql
from dateutil.relativedelta import relativedelta
from dateutil.rrule import MONTHLY, rrule
from dotenv import load_dotenv

# Setup logger
logger = logging.getLogger(__name__)

# Retrieve environment variables
load_dotenv()

BING_MAP_API_KEY = os.environ.get("BING_MAP_API_KEY")
POSTGRESQL_DASH_USER = os.environ.get("POSTGRESQL_DASH_USER")
POSTGRESQL_DASH_PASSWORD = os.environ.get("POSTGRESQL_DASH_PASSWORD")
POSTGRESQL_DASH_DATABASE = os.environ.get("POSTGRESQL_DASH_DATABASE")
POSTGRESQL_HOST = os.environ.get("POSTGRESQL_HOST")
POSTGRESQL_PORT = os.environ.get("POSTGRESQL_PORT")


def run(cfg):
    """Main DAG task to process HDB resale data.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configs in OmegaConf format.

    Returns
    -------
    None
    """
    engine, metadata = sql.setup_database(
        postgresql_dash_user=POSTGRESQL_DASH_USER,
        postgresql_dash_password=POSTGRESQL_DASH_PASSWORD,
        postgresql_dash_database=POSTGRESQL_DASH_DATABASE,
        postgresql_host=POSTGRESQL_HOST,
        postgresql_port=POSTGRESQL_PORT,
    )
    _process_source_data(cfg=cfg, engine=engine, metadata=metadata)


def _process_source_data(cfg, engine, metadata, chunksize=10000):
    """Process source data into master data.

    Reads in and store data out to Postgresql database.

    Based on run_mode, will either process all data or last N month of data.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configs in OmegaConf format.
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).
    chunksize : int
        Number of rows of data to be processed at a time per chunk.
        Defaults to 10000.

    Returns
    -------
    None
    """
    SourceData = metadata.tables[cfg.database.source_data.schema_table_name]
    AddressData = metadata.tables[cfg.database.address_data.schema_table_name]

    if cfg.run.run_mode == "full":
        data_query = sqlalchemy.select(
            SourceData.c._row_hash_id,
            SourceData.c.month,
            SourceData.c.town,
            SourceData.c.flat_type,
            SourceData.c.block,
            SourceData.c.street_name,
            SourceData.c.storey_range,
            SourceData.c.floor_area_sqm,
            SourceData.c.flat_model,
            SourceData.c.lease_commence_date,
            SourceData.c.remaining_lease,
            SourceData.c.resale_price,
        )
    elif cfg.run.run_mode == "delta":
        end_month = datetime.date.today().replace(day=1)
        start_month = end_month - relativedelta(months=cfg.run.past_n_mth - 1)
        # Create a range of months, in specific string formats
        range_month = [dt.strftime("%Y-%m") for dt in rrule(MONTHLY, dtstart=start_month, until=end_month)]

        data_query = sqlalchemy.select(
            SourceData.c._row_hash_id,
            SourceData.c.month,
            SourceData.c.town,
            SourceData.c.flat_type,
            SourceData.c.block,
            SourceData.c.street_name,
            SourceData.c.storey_range,
            SourceData.c.floor_area_sqm,
            SourceData.c.flat_model,
            SourceData.c.lease_commence_date,
            SourceData.c.remaining_lease,
            SourceData.c.resale_price,
        ).where(SourceData.c.month.in_(range_month))
    else:
        raise Exception(f"run_mode {cfg.run.run_mode} not implemented.")

    # Read geocoded data
    logger.info("Loading geocoded address data......")

    add_query = sqlalchemy.select(
        AddressData.c.block,
        AddressData.c.street_name,
        AddressData.c.town,
        AddressData.c.country,
        AddressData.c.full_address,
        AddressData.c.lat,
        AddressData.c.lng,
    )

    add_data = pd.read_sql(add_query, con=engine)

    # NOTE stream_results and chunksize are both needed for data chunking to be handled server side
    with engine.connect().execution_options(stream_results=True) as con:
        data_iter = pd.read_sql(data_query, con, chunksize=chunksize)

        for data_ind, data_chunk in enumerate(data_iter):
            logger.info(f"Loading source data chunk {data_ind}......")

            # Process data
            data_chunk = data_chunk.rename(columns={"month": "date"}).sort_values("date")

            # Convert date format
            data_chunk["date"] = pd.to_datetime(data_chunk["date"])

            # Convert floor area into integers
            data_chunk["floor_area_sqm"] = np.floor(data_chunk["floor_area_sqm"].astype(float)).astype(int)

            # Recalculate remaining time from start_year
            def estimate_remaining_time(start_year, cur_date, ori_lease_year=99):
                # Function to estimate remaining lease
                start_year = int(start_year)
                current_year = int(cur_date.year)

                remaining_year = ori_lease_year - (current_year - start_year)

                return remaining_year

            data_chunk["est_remaining_lease"] = data_chunk[["date", "lease_commence_date"]].apply(
                lambda row: estimate_remaining_time(row["lease_commence_date"], row["date"]), axis=1
            )

            # Take calculated remaining years if available
            data_chunk.loc[data_chunk["remaining_lease"].isna(), "remaining_lease"] = data_chunk.loc[
                data_chunk["remaining_lease"].isna(), "est_remaining_lease"
            ]

            # Clean up remaining years format
            def get_year(lease_str):
                try:
                    nyr = int(lease_str)
                except ValueError:  # int conversion failed
                    yr_obj = re.match(r"(\d{2}) year", lease_str)
                    if yr_obj:
                        nyr = int(yr_obj[1])
                    else:
                        nyr = 0

                return nyr

            data_chunk["remaining_lease"] = data_chunk["remaining_lease"].apply(get_year)

            # Convert storey range into numeric format
            def get_min_storey_range(x):
                x_list = x.split(" ")
                y = int(x_list[0])
                return y

            def get_max_storey_range(x):
                x_list = x.split(" ")
                y = int(x_list[2])
                return y

            data_chunk["min_storey_range"] = data_chunk["storey_range"].apply(get_min_storey_range)
            data_chunk["max_storey_range"] = data_chunk["storey_range"].apply(get_max_storey_range)

            # Get geocoded data by combining with stored data
            data_chunk = data_chunk.merge(
                add_data[["block", "street_name", "town", "full_address", "lat", "lng"]],
                how="left",
                on=["block", "street_name", "town"],
            )

            # Geocode any data with missing information
            missing_row_bool = data_chunk["lat"].isna() & data_chunk["lng"].isna()
            missing_add = data_chunk.loc[missing_row_bool, ["block", "street_name", "town"]]
            if missing_add.shape[0] > 0:
                logger.info("Regeocoding data......")
                logger.info("Note that this takes a long time to run.")
                geocoded_add = data.geocode_data(
                    data_df=missing_add,
                    hash_key_columns=cfg.database.address_data.hash_key_columns,
                    engine=engine,
                    metadata=metadata,
                    schema_name=cfg.database.address_data.schema_name,
                    table_name=cfg.database.address_data.table_name,
                    api_key=BING_MAP_API_KEY,
                )
                geocoded_add = geocoded_add.rename(
                    columns={"full_address": "new_full_address", "lat": "new_lat", "lng": "new_lng"}
                )

                data_chunk = data_chunk.merge(
                    geocoded_add[["block", "street_name", "town", "new_full_address", "new_lat", "new_lng"]],
                    how="left",
                    on=["block", "street_name", "town"],
                )

                data_chunk.loc[missing_row_bool, "full_address"] = data_chunk.loc[missing_row_bool, "new_full_address"]
                data_chunk.loc[missing_row_bool, "lat"] = data_chunk.loc[missing_row_bool, "new_lat"]
                data_chunk.loc[missing_row_bool, "lng"] = data_chunk.loc[missing_row_bool, "new_lng"]

                data_chunk = data_chunk.drop(columns=["new_full_address", "new_lat", "new_lng"])

            # Create tooltip text for map markers
            data_chunk["tooltip_text"] = data_chunk.apply(
                lambda row: f"""
                <p>Registration Date : {row["date"]}</p>
                <p>Address : {row["full_address"]}</p>
                <p>Remaining Lease : {row["remaining_lease"]}</p>
                <p>Floor Area (SQM) : {row["floor_area_sqm"]}</p>
                <p>Flat Type : {row["flat_type"]}</p>
                <p>Flat Model : {row["flat_model"]}</p>
                <p>Resale Price : SGD {row["resale_price"]}</p>
            """.strip(),
                axis=1,
            )

            # Drop columns for processing use
            data_chunk = data_chunk.drop(columns=["est_remaining_lease"])

            # Remove data based on primary key
            # Needs to be done before data insertion to prevent database duplicated errors
            # No data will be removed if the primary key does not exist
            sql.delete_data_primary_key(
                engine=engine,
                metadata=metadata,
                schema_table_name=cfg.database.master_data.schema_table_name,
                primary_key=data_chunk["_row_hash_id"].to_list(),
            )

            # Insert data into database
            with engine.connect() as con:
                data_chunk.to_sql(
                    name=cfg.database.master_data.table_name,
                    schema=cfg.database.master_data.schema_name,
                    con=con,
                    if_exists="append",
                    index=False,
                    chunksize=chunksize,
                )

            logger.info(
                f"Loaded {data_chunk.shape[0]} rows of data into table {cfg.database.master_data.schema_table_name}."
            )
