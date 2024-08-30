"""Contains functions for hdb_resale specific supporting functions."""
import hashlib
import logging

# Setup logger
logger = logging.getLogger(__name__)


def create_hash(row):
    """Convert all selected columns into strings, combine them into one and calculate hash.

    Function to be used inside pandas' apply function over rows of dataframe.

    Parameters
    ----------
    row : pd.DataFrame
        A row of data from a dataframe.

    Returns
    -------
    row_id : str
        Unique hash ID for the given row.
    """
    row_id = '_'.join(row.values.astype(str)).encode("utf-8")
    row_id = hashlib.sha1(row_id).hexdigest()

    return row_id
