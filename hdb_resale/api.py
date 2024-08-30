"""Contains functions for hdb_resale specific API operations."""
import logging
import time

import pandas as pd
import requests

# Setup logger
logger = logging.getLogger(__name__)


def get_sggov_hdb_data(cfg, api_url):
    """Get data from API endpoint and return a data in a dataframe.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configs in OmegaConf format.
    api_url : str
        API endpoint with formatted queries.

    Returns
    -------
    result : pd.DataFrame
        Retrieved data.
    """
    time.sleep(cfg.api.query_sleep_offset)

    response = requests.get(api_url)
    assert response.status_code == 200  # Success request

    # Convert json into dataframe
    result = pd.DataFrame.from_records(response.json()["result"]["records"])

    return result
