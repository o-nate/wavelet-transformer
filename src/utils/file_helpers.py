"""Helper functions for handling data"""

import logging
import sys
from typing import Type

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def convert_to_dataframe(file: Type[UploadedFile]) -> pd.DataFrame:
    """Read file based on type

    Args:
        file (Type[UploadedFile]): Uploaded file object

    Returns:
        pd.DataFrame: DataFrame with columns `date` and `value`
    """
    if file.type == "text/csv":
        return pd.read_csv(file, sep=None)
    return pd.read_excel(file)
