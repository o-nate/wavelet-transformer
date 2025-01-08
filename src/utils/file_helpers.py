"""Helper functions for handling data"""

from typing import Type

import pandas as pd
import openpyxl
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


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


def convert_to_dataframe(uploaded_file: Type[UploadedFile]) -> pd.DataFrame:
    """Read file based on type

    Args:
        uploaded_file (Type[UploadedFile]): Uploaded file object

    Returns:
        pd.DataFrame: DataFrame with columns `date` and `value`
    """
    if uploaded_file.type == "text/csv":
        logger.info("Successfully loaded CSV file: %s", uploaded_file.name)
        return pd.read_csv(
            uploaded_file,
            sep=None,
            parse_dates=[0],
            index_col=0,
        )
    logger.info("Successfully loaded Excel file: %s", uploaded_file.name)
    return pd.read_excel(
        uploaded_file,
        parse_dates=[0],
        index_col=0,
    )


def load_file(file_to_load: Type[UploadedFile]) -> pd.DataFrame:
    """Load data from uploaded file to dataframe with error handling and logging

    Args:
        file_to_load (Type[UploadedFile]):

    Raises:
        ValueError: Invalid file format

    Returns:
        pd.DataFrame: Columns: 'date' and 'value'
    """
    try:
        df_from_file = convert_to_dataframe(file_to_load)

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
            df_from_file, file_name=file_to_load.name
        )
        logger.debug("df dtypes: %s", df_from_file.dtypes)

        return df_from_file

    except ValueError as ve:
        logger.error("Validation error with file %s: %s", file_to_load.name, ve)
        st.error(f"Invalid file format: {ve}")
        return None

    except Exception as e:
        logger.exception("Error loading file %s: %s", file_to_load.name, e)
        st.error(f"An error occurred while loading the file: {e}")
        return None
