"""Streamlit dashboard app"""

import logging
import sys
from typing import Type

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import results_configs
from logging_config import get_logger

from src import dwt

from src.utils.file_helpers import convert_to_dataframe
from src.utils.helpers import combine_series
from src.utils.transform_helpers import create_dwt_dict, create_dwt_results_dict

# * Logging settings
logger = get_logger(__name__)


def load_file(uploaded_file: Type[UploadedFile]) -> pd.DataFrame:
    """Load data from uploaded file with error handling and logging

    Args:
        uploaded_file (Type[UploadedFile]):

    Raises:
        ValueError: Invalid file format

    Returns:
        pd.DataFrame: Columns: 'date' and 'value'
    """
    try:
        df = convert_to_dataframe(uploaded_file)

        # Validate DataFrame
        if not {"date", "value"}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'date' and 'value' columns")

        return df

    except ValueError as ve:
        logger.error("Validation error with file %s: %s", uploaded_file.name, ve)
        st.error(f"Invalid file format: {ve}")
        return None

    except Exception as e:
        logger.exception("Error loading file %s: %s", uploaded_file.name, e)
        st.error(f"An error occurred while loading the file: {e}")
        return None


st.title("Wavelet Analysis Interactive Dashboard")

# File uploader
uploaded_files = st.file_uploader(
    "Choose Excel or CSV files", type=["csv", "xlsx"], accept_multiple_files=True
)

# Create dict to store file names with file objects
file_dict = {uploaded_file.name: uploaded_file for uploaded_file in uploaded_files}

# Create plot
if uploaded_files:
    fig = go.Figure()

    dict_of_combined_dataframes = {
        uploaded_file.name: load_file(uploaded_file) for uploaded_file in uploaded_files
    }

    combined_dfs = combine_series(
        dict_of_combined_dataframes.values(), how="left", on="date"
    )

    # dwt_dict = create_dwt_dict(
    #     combined_dfs.dropna(),
    #     dwt_measures,
    #     mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    # )

    # # * Run DWTs
    # dwt_results_dict = create_dwt_results_dict(dwt_dict, dwt_measures)

    for uploaded_file in uploaded_files:
        # Add trace to plot
        df = dict_of_combined_dataframes[uploaded_file.name]
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["value"], mode="lines", name=uploaded_file.name
            )
        )

    # # * DWT components
    # fig = dwt.plot_components(
    #     label=uploaded_file.name,
    #     coeffs=dwt_results_dict[ids.EXPECTATIONS].coeffs,
    #     time=t,
    #     levels=dwt_results_dict[ids.EXPECTATIONS].levels,
    #     wavelet=results_configs.DWT_MOTHER_WAVELET,
    #     figsize=(15, 20),
    #     sharex=True,
    # )
    # plt.legend("", frameon=False)

    # ## Figure 6 - Smoothing of expectations
    # dwt_results_dict[ids.EXPECTATIONS].smooth_signal(
    #     y_values=dwt_dict[ids.EXPECTATIONS].y_values,
    #     mother_wavelet=dwt_dict[ids.EXPECTATIONS].mother_wavelet,
    # )

    # fig = dwt.plot_smoothing(
    #     dwt_results_dict[ids.EXPECTATIONS].smoothed_signal_dict,
    #     t,
    #     dwt_dict[ids.EXPECTATIONS].y_values,
    #     ascending=True,
    #     figsize=(15, 20),
    #     sharex=True,
    # )
    # plt.legend("", frameon=False)

    # Update layout
    fig.update_layout(
        title=f"Data Visualization of: {' '.join(file_dict.keys())}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend={
            "orientation": "h",
            "entrywidthmode": "fraction",
            "entrywidth": 0.8,
            "yanchor": "top",
            "y": 1.15,
            "xanchor": "right",
            "x": 1,
        },
    )

    # Display plot
    st.plotly_chart(fig)

else:
    st.write("Upload a spreadsheet file to begin.")
    st.write(
        """Please, ensure that each file has only two columns: 
        the **date column on the left** and the **value column on the right**."""
    )
