"""Test CWT functions"""

from __future__ import division
import logging
import sys

import numpy as np

from src.utils.logging_helpers import define_other_module_log_level
from src import retrieve_data
from src.utils import wavelet_helpers
from src import xwt

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Preprocess data for testing
MEASURE_1 = "000857180"
MEASURE_2 = "000857181"
raw_data = retrieve_data.get_insee_data(MEASURE_1)
df1, _, _ = retrieve_data.clean_insee_data(raw_data)
raw_data = retrieve_data.get_insee_data(MEASURE_2)
df2, _, _ = retrieve_data.clean_insee_data(raw_data)

# * Pre-process data: Align time series temporally
dfcombo = df1.merge(df2, how="left", on="date", suffixes=("_1", "_2"))
dfcombo.dropna(inplace=True)

# * Pre-process data: Standardize and detrend
y1 = dfcombo["value_1"].to_numpy()
y2 = dfcombo["value_2"].to_numpy()
t_test1 = np.linspace(1, y1.size + 1, y1.size)
t_test2 = np.linspace(1, y2.size + 1, y2.size)
y1 = wavelet_helpers.standardize_series(y1, detrend=False, remove_mean=True)
y2 = wavelet_helpers.standardize_series(y2, detrend=False, remove_mean=True)


logger.info("Testing DataForXWT class initialization")
mother_xwt = xwt.MOTHER_DICT[xwt.MOTHER]
xwt_data = xwt.DataForXWT(
    y1,
    y2,
    mother_xwt,
    xwt.DT,
    xwt.DJ,
    xwt.S0,
    xwt.LEVELS,
)
assert len(xwt_data.t_values) == min(len(t_test1), len(t_test2))
logger.info("Test passed")

logger.info("Testing ResultsFromXWT class initialization")
results_from_xwt = xwt.run_xwt(xwt_data, ignore_strong_trends=False)
assert len(results_from_xwt.power) < len(xwt_data.y1_values)
logger.info("Test passed")

logger.info("Tests complete")
