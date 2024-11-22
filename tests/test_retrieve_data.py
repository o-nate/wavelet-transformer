"""Test data retrieval functions"""

# %%
import logging
import sys

import numpy as np
import pandas as pd

from src.utils.logging_helpers import define_other_module_log_level
from constants import ids
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("info")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)

NOMINAL_VALUE = 100
CPI_T = 200
CPI_CONSTANT = 100
CONSTANT_DOLLAR_DATE = "2017-12-01"

# %%
logger.info("Testing get_fed_data, cleaned data")
data = retrieve_data.get_fed_data(ids.US_CPI, freq="m")
assert isinstance(data, list)

# %%
logger.info("Testing get_fed_data, no headers")
data = retrieve_data.get_fed_data(ids.US_CPI, no_headers=False)
assert isinstance(data, dict)
# %%
logger.info("Testing get_fed_data, cleaned data")
data = retrieve_data.get_fed_data(ids.US_CPI, freq="m")
df, clean_t, clean_y = retrieve_data.clean_fed_data(data)
assert isinstance(clean_t[1], np.datetime64)

logger.info("Testing nominal to real value conversion")
REAL_VALUE = retrieve_data.convert_to_real_value(NOMINAL_VALUE, CPI_T, CPI_CONSTANT)
assert REAL_VALUE == 50

logger.info("Testing convert column to real values")
df = pd.DataFrame(
    {
        "date": [
            pd.Timestamp("2016-12-01"),
            pd.Timestamp("2017-12-01"),
            pd.Timestamp("2018-12-01"),
        ],
        "cpi": [50, 100, 200],
        "nominal": [100, 100, 100],
    }
)

df["real"] = retrieve_data.convert_column_to_real_value(
    data=df, column="nominal", cpi_column="cpi", constant_date=CONSTANT_DOLLAR_DATE
)
assert (
    df[df["date"] == pd.Timestamp("2018-12-01")]["real"].iat[0]
    < df[df["date"] == pd.Timestamp("2016-12-01")]["real"].iat[0]
)
# %%
logger.info("Testing get_insee_data")
data = retrieve_data.get_insee_data("000857180")
assert isinstance(data, list)

logger.info("Testing clean_insee_data")
df, clean_t, clean_y = retrieve_data.clean_insee_data(data)
assert isinstance(clean_y, np.ndarray)
assert isinstance(clean_t, np.ndarray)

logger.info("Data retrieval functions testing complete.")
