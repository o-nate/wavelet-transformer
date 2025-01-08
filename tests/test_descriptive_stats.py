"""Statistical tests for time series analysis"""

import numpy as np
import pandas as pd

from src import descriptive_stats

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

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
