"""Helper functions for handling data"""

import logging
import sys
from typing import Type

import pandas as pd
import openpyxl
from streamlit.runtime.uploaded_file_manager import UploadedFile

from logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


def convert_to_dataframe(uploaded_file: Type[UploadedFile]) -> pd.DataFrame:
    """Read file based on type

    Args:
        uploaded_file (Type[UploadedFile]): Uploaded file object

    Returns:
        pd.DataFrame: DataFrame with columns `date` and `value`
    """
    if uploaded_file.type == "text/csv":
        logger.info("Successfully loaded CSV file: %s", uploaded_file.name)
        return pd.read_csv(uploaded_file, sep=None)
    logger.info("Successfully loaded Excel file: %s", uploaded_file.name)
    return pd.read_excel(uploaded_file)
