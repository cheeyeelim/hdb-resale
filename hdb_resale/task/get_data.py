"""DAG task to download HDB resale data from SG Gov API."""

import datetime
import logging
import os

import pandas as pd
from dateutil.relativedelta import relativedelta
from dateutil.rrule import MONTHLY, rrule
from dotenv import load_dotenv

from hdb_resale import api, sql, utils

# Setup logger
logger = logging.getLogger(__name__)

# Retrieve environment variables
load_dotenv()

POSTGRESQL_DASH_USER = os.environ.get("POSTGRESQL_DASH_USER")
POSTGRESQL_DASH_PASSWORD = os.environ.get("POSTGRESQL_DASH_PASSWORD")
POSTGRESQL_DASH_DATABASE = os.environ.get("POSTGRESQL_DASH_DATABASE")
POSTGRESQL_HOST = os.environ.get("POSTGRESQL_HOST")
POSTGRESQL_PORT = os.environ.get("POSTGRESQL_PORT")


def run(cfg):
    """Main DAG task to download HDB resale data from SG Gov API.

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
    _retrieve_data_api(cfg=cfg, engine=engine, metadata=metadata)


def _retrieve_data_api(cfg, engine, metadata):
    """Call API to get data and store into database.

    Reads in postgresql authentication details from env file.

    Based on run_mode, will either download all data or last N month of data.

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
    None
    """
    end_month = datetime.date.today().replace(day=1)  # today
    if cfg.run.run_mode == "full":
        start_month = datetime.date(2017, 1, 1)  # start of data
    elif cfg.run.run_mode == "delta":
        start_month = end_month - relativedelta(months=cfg.run.past_n_mth - 1)
    else:
        raise Exception(f"run_mode {cfg.run.run_mode} not implemented.")
    # Create a range of months, in datetime date formats
    range_month = [dt.date() for dt in rrule(MONTHLY, dtstart=start_month, until=end_month)]

    start_offset = 0  # starting offset value

    for cur_month in range_month:  # loop through all months

        cur_offset = start_offset
        cur_row_retrieved = 0
        exp_row_retrieved = 0
        first_while_loop = True
        res = pd.DataFrame({'A': [1]})  # placeholder dataframe to initiate while loop

        while first_while_loop or not res.empty:  # loop through all rows
            # Get the next final formatted URL with base and query strings
            data_query = f'{{"month":"{cur_month.strftime("%Y-%m")}"}}'
            api_final_url = f"{cfg.api.formatted_url}&q={data_query}&limit={cfg.api.limit}&offset={cur_offset}"

            logger.info(f"Retrieving data from endpoint with query - {api_final_url}")

            res = api.get_sggov_hdb_data(cfg=cfg, api_url=api_final_url)

            # Break while loop when there is no longer any data
            if res.empty:
                assert int(cur_row_retrieved) == int(
                    exp_row_retrieved
                )  # Should have retrieved same number of rows as reported by API endpoint
                break

            # Rename original column names
            res = res.rename(columns={"rank month": "rank_month"})

            # Create row hash identifier using key columns
            res["_row_hash_id"] = res[cfg.database.source_data.hash_key_columns].apply(utils.create_hash, axis=1)

            # Remove data based on primary key
            # Needs to be done before data insertion to prevent database duplicated errors
            # No data will be removed if the primary key does not exist
            sql.delete_data_primary_key(
                engine=engine,
                metadata=metadata,
                schema_table_name=cfg.database.source_data.schema_table_name,
                primary_key=res["_row_hash_id"].to_list(),
            )

            # Insert data into database
            with engine.connect() as con:
                res.to_sql(
                    name=cfg.database.source_data.table_name,
                    schema=cfg.database.source_data.schema_name,
                    con=con,
                    if_exists="append",
                    index=False,
                    chunksize=10000,
                )

            logger.info(f"Loaded {res.shape[0]} rows of data into table {cfg.database.source_data.schema_table_name}.")

            cur_offset += cfg.api.limit
            cur_row_retrieved += res.shape[0]
            exp_row_retrieved = res["_full_count"][0]  # This is the total row counts reported by API endpoint
            first_while_loop = False
