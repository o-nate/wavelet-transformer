"""Script to generate data files"""

# %%
from pathlib import Path

from constants import ids

from src import retrieve_data

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Export settings
parent_directory = Path(__file__).parents[1]
export_directory = parent_directory / "data"

DATA_REQUESTED = ids.US_SAVINGS_RATE
FILE_NAME = ids.SAVINGS_RATE

# %% [markdown]
# Retrieve data via API

raw_data = retrieve_data.get_fed_data(DATA_REQUESTED)
us_data, t, y = retrieve_data.clean_fed_data(raw_data)
us_data.head()

# %%
us_data.to_csv(f"{export_directory}/{FILE_NAME}.csv", sep=",", index=False)
print(f"Data exported to {export_directory}")
