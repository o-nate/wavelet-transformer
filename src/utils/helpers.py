"""Cross-project helper functions"""

from functools import reduce
from typing import Dict, Generator, List

import numpy as np
import pandas as pd


def nested_dict_values(nested_dict: Dict) -> Generator[any, any, any]:
    """Extract nested dict values"""
    for v in nested_dict.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def nested_list_values(nested_list: List[List[str]]) -> Generator[any, any, any]:
    """Extract nested list values"""
    for v in nested_list:
        if isinstance(v, list):
            yield from nested_list_values(v)
        else:
            yield v


def convert_to_real_value(
    nominal_value: float, cpi_t: float, cpi_constant: float
) -> pd.DataFrame:
    """Adjust values to constant dollar amount based on the CPI measure and year defined"""
    return (nominal_value * cpi_constant) / cpi_t


def convert_column_to_real_value(
    data: pd.DataFrame, column: str, cpi_column: str, constant_date: int
) -> pd.DataFrame:
    """Apply real value conversion to column with constant year's CPI as base"""
    cpi_constant = data[data["date"] == pd.Timestamp(f"{constant_date}")][
        cpi_column
    ].iat[0]
    return data.apply(
        lambda x: convert_to_real_value(x[column], x[cpi_column], cpi_constant), axis=1
    )


def add_real_value_columns(
    data: pd.DataFrame, nominal_columns: List[str], **kwargs
) -> pd.DataFrame:
    """Convert nominal to real values for each column in list"""
    for col in nominal_columns:
        data[f"real_{col}"] = convert_column_to_real_value(
            data=data, column=col, **kwargs
        )
    return data


def calculate_diff_in_log(
    data: pd.DataFrame, columns: List[str], new_columns: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling difference of log(values), adding as new column if defined
    """
    for col in columns:
        if new_columns:
            col_name = f"diff_log_{col}"
        else:
            col_name = col
        data[col_name] = 100 * np.log(data[col]).diff()
    return data


def combine_series(dataframes: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Merge dataframes from a list

    Args:
        dataframes List[pd.DataFrame]: List of dataframes to merge

    Kwargs:
        on List[str]: Column(s) to merge on
        how {str}: 'left', 'right', 'inner', 'outer'

    Returns:
        pd.DataFrame: Combined dataframe
    """
    return reduce(lambda left, right: pd.merge(left, right, **kwargs), dataframes)
