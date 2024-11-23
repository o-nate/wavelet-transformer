"""Streamlit dashboard app"""

from typing import Type

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import results_configs
from utils.logging_config import get_logger

from src import dwt

from src.utils.file_helpers import load_file
from src.utils.helpers import combine_series
from src.utils.transform_helpers import create_dwt_dict, create_dwt_results_dict

# * Logging settings
logger = get_logger(__name__)

st.title("Wavelet Analysis Interactive Dashboard")

# File uploader
uploaded_files = st.file_uploader(
    "Choose Excel or CSV files", type=["csv", "xlsx"], accept_multiple_files=True
)

# Create dict to store file names with file objects
file_dict = {
    uploaded_file.name.split(".")[0]: uploaded_file for uploaded_file in uploaded_files
}

# Create list of column names based on file name to differentiate after merging the dataframes
column_names = list(file_dict)
logger.debug("column name: %s", column_names)

# Create plot
if uploaded_files:
    fig = go.Figure()

    dict_of_combined_dataframes = {
        column_name: load_file(uploaded_file)
        for column_name, uploaded_file in zip(column_names, uploaded_files)
    }
    logger.debug(
        "dict_of_combined_dataframes keys: %s", list(dict_of_combined_dataframes)
    )

    combined_dfs = combine_series(
        dict_of_combined_dataframes.values(),
        how="left",
        on="date",
        # suffixes=tuple(dict_of_combined_dataframes),
    )

    logger.debug("combined df columns: %s", combined_dfs.columns.to_list())

    dwt_dict = create_dwt_dict(
        combined_dfs.dropna(),
        column_names,
        mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    )

    # * Run DWTs
    dwt_results_dict = create_dwt_results_dict(dwt_dict, column_names)

    st.write(f"NOT showing DWT of: {', '.join(dwt_dict.keys())}")

    for column_name, df in dict_of_combined_dataframes.items():
        # Add trace to plot
        fig.add_trace(
            go.Scatter(x=df.index, y=df[column_name], mode="lines", name=column_name)
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
