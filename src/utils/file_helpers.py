"""Helper functions for handling data"""

from typing import Union, Type
from pathlib import Path
import os

import pandas as pd
import openpyxl
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.utils.config import INDEX_COLUMN_NAME
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

SAMPLE_DATA_PATH = Path(__file__).parents[2] / "sample_data"


def validate_datetime_index_of(data: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates if a pandas DataFrame has a datetime index.

    Parameters:
    data (pandas.DataFrame): DataFrame to validate

    Returns:
    tuple: (bool, str) - (True if index is datetime, explanation message)
    """
    # Check if the DataFrame has an index
    if data.index.empty:
        return False, "DataFrame index is empty"

    # Check if index is already datetime
    if pd.api.types.is_datetime64_any_dtype(data.index):
        return True, "Index is datetime type"

    # Check if index can be converted to datetime
    try:
        pd.to_datetime(data.index)
        return True, "Index can be converted to datetime"
    except (ValueError, TypeError):
        return False, "Index cannot be converted to datetime"


def validate_first_column_numeric(data: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates if the first column of a DataFrame contains numerical values.

    Parameters:
    data (pandas.DataFrame): DataFrame to validate

    Returns:
    tuple: (bool, str) - (True if first column is numeric, explanation message)
    """
    # Check if DataFrame is empty
    if data.empty:
        return False, "DataFrame is empty"

    # Check if DataFrame has any columns
    if len(data.columns) == 0:
        return False, "DataFrame has no columns"

    # Get the first column
    first_col = data.iloc[:, 0]

    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(first_col):
        return True, "First column is numeric"

    # Check if the column can be converted to numeric
    try:
        pd.to_numeric(first_col)
        return True, "First column can be converted to numeric"
    except (ValueError, TypeError):
        return False, "First column cannot be converted to numeric"


def standardize_columns_with_file_name_for(
    data: pd.DataFrame, file_name: str
) -> pd.DataFrame:
    """
    Renames the first column of a DataFrame using the provided file name.

    Parameters:
        data (pd.DataFrame): Input DataFrame whose first column needs to be renamed.
        file_name (str): The new name to be assigned to the first column.

    Returns:
        pd.DataFrame: DataFrame with the first column renamed to the file name.

    Example:
        >>> df = pd.DataFrame({'old_name': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        >>> file_name = 'my_csv.csv'
        >>> standardize_columns_with_file_name_for(df, file_name)
           my_csv  col2
        0          1     a
        1          2     b
        2          3     c
    """
    clean_file_name = file_name.split(".")[0]
    data = data.rename(columns={data.columns[0]: clean_file_name})
    return data


def convert_to_dataframe(file_input: Union[UploadedFile, Path, str]) -> pd.DataFrame:
    """Read file based on type

    Args:
        file_input: Can be a Streamlit UploadedFile, a Path object, or a string file path

    Returns:
        pd.DataFrame: DataFrame with columns `date` and `value`
    """
    # Determine file type and path
    if hasattr(file_input, "type") and hasattr(
        file_input, "name"
    ):  # Streamlit UploadedFile
        file_name = file_input.name
        file_type = file_input.type
        file_obj = file_input  # The file object itself
    else:  # Path object or string
        file_path = SAMPLE_DATA_PATH / file_input
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        # Determine file type based on extension
        if file_ext == ".csv":
            file_type = "text/csv"
        elif file_ext in [".xlsx", ".xls"]:
            file_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")

        file_obj = file_path  # The file path as string or Path object

    # Read file based on determined type
    if "csv" in file_type:
        logger.info("Successfully loaded CSV file: %s", file_name)
        return pd.read_csv(
            file_obj,
            sep=None,
            parse_dates=[0],
            index_col=0,
        )
    logger.info("Successfully loaded Excel file: %s", file_name)
    return pd.read_excel(
        file_obj,
        parse_dates=[0],
        index_col=0,
    )


def load_file(file_to_load: Union[UploadedFile, Path, str]) -> pd.DataFrame:
    """Load data from uploaded file to dataframe with error handling and logging

    This function handles different file types:
    - streamlit.runtime.uploaded_file_manager.UploadedFile objects
    - pathlib.Path objects
    - string file paths

    Args:
        file_to_load: Can be a Streamlit UploadedFile, a Path object, or a string file path

    Raises:
        ValueError: Invalid file format

    Returns:
        pd.DataFrame: Columns: 'date' and 'value'
    """
    try:
        # Get file name based on type
        if hasattr(file_to_load, "name"):  # Streamlit UploadedFile
            file_name = file_to_load.name
        elif isinstance(file_to_load, (str, Path)):  # Path object or string
            file_name = os.path.basename(str(file_to_load))
        else:
            raise TypeError(f"Unsupported file type: {type(file_to_load)}")

        df_from_file = convert_to_dataframe(file_to_load)

        # * Set index to specific name to make all conform
        df_from_file.index.name = INDEX_COLUMN_NAME

        # Validate DataFrame to make sure columns contain appropriate values
        validate_datetime, datetime_result_message = validate_datetime_index_of(
            df_from_file
        )
        if not validate_datetime:
            logger.error(datetime_result_message)
            st.error("The left-hand column must contain the date and/or timestamps")
            raise ValueError(datetime_result_message)

        validate_numeric, numeric_result_message = validate_first_column_numeric(
            df_from_file
        )
        if not validate_numeric:
            logger.error(numeric_result_message)
            st.error("The right-hand column must contain numerical values")
            raise ValueError(numeric_result_message)

        df_from_file = standardize_columns_with_file_name_for(
            df_from_file, file_name=file_name
        )

        return df_from_file

    except ValueError as ve:
        if hasattr(file_to_load, "name"):
            file_name = file_to_load.name
        else:
            file_name = str(file_to_load)
        logger.error("Validation error with file %s: %s", file_name, ve)
        st.error(f"Invalid file format: {ve}")
        return None

    except Exception as e:
        if hasattr(file_to_load, "name"):
            file_name = file_to_load.name
        else:
            file_name = str(file_to_load)
        logger.exception("Error loading file %s: %s", file_name, e)
        st.error(f"An error occurred while loading the file: {e}")
        return None
