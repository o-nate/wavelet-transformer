"""Statistical tests for time series analysis"""

import logging
from pathlib import Path
import sys

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from src import descriptive_stats
from src.utils.logging_helpers import define_other_module_log_level
from constants import ids
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("info")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("Testing Shapiro-Wilk")

date = np.linspace(1, 300, 300)
x = np.random.normal(size=300)
y = np.random.uniform(size=300)

df = pd.DataFrame({"date": date, "x": x, "y": y})

results = descriptive_stats.test_normality(
    "Shapiro-Wilk", data=df, date_column="date", add_pvalue_stars=True
)
assert list(results.keys()) == ["x", "y"]
assert "*" not in results["x"] and "*" in results["y"]

logger.info("Testing Jarque-Bera")

date = np.linspace(1, 3000, 3000)
x = np.random.normal(size=3000)
y = np.random.uniform(size=3000)

df = pd.DataFrame({"date": date, "x": x, "y": y})

results = descriptive_stats.test_normality(
    "Jarque-Bera", data=df, date_column="date", add_pvalue_stars=True
)
assert list(results.keys()) == ["x", "y"]
assert "*" not in results["x"] and "*" in results["y"]

logger.info("Test complete")
