"""Contains functions for hdb_resale specific SQL operations."""
import logging

from sqlalchemy import TIMESTAMP, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, func

# Setup logger
logger = logging.getLogger(__name__)


def setup_database(
    postgresql_dash_user, postgresql_dash_password, postgresql_dash_database, postgresql_host, postgresql_port
):
    """Setup required schema and database in postgresql database.

    Reads in postgresql authentication details from env file.

    Parameters
    ----------
    postgresql_dash_user : str
        Username for dash-related usage in Postgresql.
    postgresql_dash_password : str
        Password for dash-related usage in Postgresql.
    postgresql_dash_database : str
        Database for dash-related usage in Postgresql.
    postgresql_host : str
        Host for Postgresql.
    postgresql_port : str
        Port for Postgresql.

    Returns
    -------
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).
    """
    # Create the connection string
    connection_string = (
        f"postgresql+psycopg2://{postgresql_dash_user}:{postgresql_dash_password}@"
        f"{postgresql_host}:{postgresql_port}/{postgresql_dash_database}"
    )

    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)

    # Define schema & tables in database using SQLAlchemy
    metadata = define_all_schema_table()

    # Create schema & tables if not exist
    create_all_table(engine=engine, metadata=metadata)

    return engine, metadata


def define_all_schema_table():
    """Define all required schema and tables.

    This uses SQLAlchemy generalisation wrapper regardless of underlying SQL dialect used.

    NOTE updated_at will only work with a custom trigger implemented,
    https://dba.stackexchange.com/questions/156980/create-a-trigger-on-all-the-last-modified-columns-in-postgresql.

    Parameters
    ----------
    None

    Returns
    -------
    metadata_obj : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).
    """
    # Holds a collection of multiple definitions for a database
    metadata_obj = MetaData(schema="hdb_resale")

    # Source data table
    Table(
        "source_data",
        metadata_obj,
        Column("_row_hash_id", String(50), primary_key=True),
        Column("_id", String(50), nullable=False),
        Column("month", String(10)),
        Column("town", String(50)),
        Column("flat_type", String(50)),
        Column("block", String(50)),
        Column("street_name", String),
        Column("storey_range", String(50)),
        Column("floor_area_sqm", String(50)),
        Column("flat_model", String(50)),
        Column("lease_commence_date", String(10)),
        Column("remaining_lease", String(50)),
        Column("resale_price", String),
        Column("rank_month", Float),
        Column("_full_count", Integer),
        Column("created_at", TIMESTAMP, server_default=func.now()),
        Column("updated_at", TIMESTAMP, onupdate=func.now()),
    )

    # Processed master data table
    Table(
        "master_data",
        metadata_obj,
        Column("_row_hash_id", String(50), primary_key=True),
        Column("date", DateTime),
        Column("town", String(50)),
        Column("flat_type", String(50)),
        Column("block", String(50)),
        Column("street_name", String),
        Column("storey_range", String(50)),
        Column("floor_area_sqm", String(50)),
        Column("flat_model", String(50)),
        Column("lease_commence_date", String(10)),
        Column("remaining_lease", String(50)),
        Column("resale_price", String),
        Column("min_storey_range", Integer),
        Column("max_storey_range", Integer),
        Column("full_address", String),
        Column("lat", Float),
        Column("lng", Float),
        Column("tooltip_text", String),
        Column("created_at", TIMESTAMP, server_default=func.now()),
        Column("updated_at", TIMESTAMP, onupdate=func.now()),
    )

    # Geocoded address table
    Table(
        "address_data",
        metadata_obj,
        Column("_row_hash_id", String(50), primary_key=True),
        Column("source_name", String(50)),
        Column("block", String(50)),
        Column("street_name", String),
        Column("town", String(50)),
        Column("country", String(50)),
        Column("full_address", String),
        Column("lat", Float),
        Column("lng", Float),
        Column("created_at", TIMESTAMP, server_default=func.now()),
        Column("updated_at", TIMESTAMP, onupdate=func.now()),
    )

    # Town demographic table
    # Data obtained manually from https://www.singstat.gov.sg/-/media/files/publications/cop2020/sr2/cop2020sr2.ashx
    Table(
        "town_data",
        metadata_obj,
        Column("_row_hash_id", String(50), primary_key=True),
        Column("town", String(50)),
        Column("census_mapping", String),
        Column("source_name", String(50)),
        Column("total_population", Integer),
        Column("population_below24", Integer),
        Column("population_2559", Integer),
        Column("population_above60", Integer),
        Column("num_hdb", Integer),
        Column("num_condo", Integer),
        Column("num_landed", Integer),
        Column("num_others", Integer),
        Column("income_below4k", Integer),
        Column("income_4k8k", Integer),
        Column("income_8k12k", Integer),
        Column("income_above12k", Integer),
        Column("household_wodisability", Integer),
        Column("household_withdisability", Integer),
        Column("pct_population_below24", Float),
        Column("pct_population_2559", Float),
        Column("pct_population_above60", Float),
        Column("pct_hdb", Float),
        Column("pct_condo", Float),
        Column("pct_landed", Float),
        Column("pct_income_below4k", Float),
        Column("pct_income_4k8k", Float),
        Column("pct_income_8k12k", Float),
        Column("pct_income_above12k", Float),
        Column("pct_household_wodisability", Float),
        Column("pct_household_withdisability", Float),
        Column("lat", Float),
        Column("lng", Float),
        Column("created_at", TIMESTAMP, server_default=func.now()),
        Column("updated_at", TIMESTAMP, onupdate=func.now()),
    )

    logger.info("Defined all schema and tables.")

    return metadata_obj


def create_all_table(engine, metadata):
    """Create all tables as defined.

    It will not recreate tables if exist.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).

    Returns
    -------
    None
    """
    metadata.create_all(engine, checkfirst=True)

    logger.info("Created all schema and tables.")


def drop_all_table(engine, metadata):
    """Drop all tables as defined.

    It will not dropped schema if there are other tables in existence
    that are not part of the definitions.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).

    Returns
    -------
    None
    """
    metadata.drop_all(engine, checkfirst=True)

    logger.info("Dropped all schema and tables.")


def delete_data_primary_key(engine, metadata, schema_table_name, primary_key=None):
    """Delete data based on primary keys.

    Example use case is before the loading of new data into database that may have the same primary key.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Engine with specified connection details to a database.
    metadata : sqlalchemy.schema.MetaData
        A collection of multiple definitions for a database (e.g. schema, table).
    schema_table_name : str
        Table name with schema name.
    primary_key : List
        List of primary keys which data will be deleted.
        Default to None.

    Returns
    -------
    None
    """
    if primary_key:
        selected_table = metadata.tables[schema_table_name]

        # Specify SQL statement in SQLAlchemy wrapper syntax
        stmt = selected_table.delete().where(selected_table.c._row_hash_id.in_(primary_key))

        with engine.connect() as con:
            con.execute(stmt)

        logger.info(f"Deleted data from {schema_table_name} matching the supplied list of primary keys.")
