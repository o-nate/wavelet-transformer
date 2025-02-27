"""Streamlit dashboard app"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from plotly import tools

from constants import ids
from utils.logging_config import get_logger

from src import wavelet_plots

from src.utils.config import INDEX_COLUMN_NAME
from src.utils.file_helpers import load_file
from src.utils.helpers import adjust_sidebar, calculate_diff_in_log, combine_series

# * Logging settings
logger = get_logger(__name__)

st.title("Wavelet Transformer")
st.text(
    "This app allows you conduct wavelet analysis by applying the wavelet transform to data!"
)

# Add sidebar for controlling parameters
transform_selection = st.sidebar.selectbox(
    "**Select a wavelet transform**",
    (ids.CWT, ids.DWT, ids.XWT),
    key="wavelet_selection",
)
dwt_plot_selection = adjust_sidebar(transform_selection)
dwt_smooth_plot_order = adjust_sidebar(dwt_plot_selection)

# Sample datasets
selected_data = st.sidebar.multiselect(
    "**Select a dataset**", ids.SAMPLE_DATA + ["I have my own!🤓"], max_selections=2
)

logger.debug("selected data: %s", selected_data)

# Initialize uploaded_files and file_dict
uploaded_files = []
file_dict = {}

# Check if any sample data is selected
if any(data in ids.SAMPLE_DATA for data in selected_data):
    # Filter for only the sample data items
    sample_data_items = [data for data in selected_data if data in ids.SAMPLE_DATA]
    for dataset in sample_data_items:
        file_path = ids.API_DICT[dataset]
        # Handle both string paths and file objects
        if isinstance(file_path, str):
            # For string paths, use the filename as the key
            key = os.path.basename(file_path).split(".")[0]
            file_dict[key] = file_path
        else:
            # For file objects with a name attribute
            key = file_path.name.split(".")[0]
            file_dict[key] = file_path
        uploaded_files.append(file_path)

# File uploader - append to uploaded_files and file_dict if user has own data
if "I have my own!🤓" in selected_data:
    user_files = st.sidebar.file_uploader(
        "Choose Excel or CSV files", type=["csv", "xlsx"], accept_multiple_files=True
    )
    if user_files:
        for file in user_files:
            key = file.name.split(".")[0]
            file_dict[key] = file
            uploaded_files.append(file)

logger.debug("uploaded_files: %s", uploaded_files)
logger.debug("file_dict keys: %s", list(file_dict.keys()))

# Create plot
if file_dict:
    # Create list of column names based on file name to differentiate after merging the dataframes
    column_names = list(file_dict.keys())
    logger.debug("column names: %s", column_names)

    # Load each file into a dataframe
    dict_of_combined_dataframes = {
        column_name: load_file(file_path)
        for column_name, file_path in file_dict.items()
    }
    logger.debug(
        "dict_of_combined_dataframes keys: %s", list(dict_of_combined_dataframes)
    )

    combined_dfs = combine_series(
        dict_of_combined_dataframes.values(),
        how="left",
        on=INDEX_COLUMN_NAME,
    )

    logger.debug("combined df columns: %s", combined_dfs.columns.to_list())

    if transform_selection in (ids.CWT, ids.XWT) and all(
        data in selected_data for data in [ids.INFLATION, ids.EXPECTATIONS]
    ):
        st.warning(
            "Converting to diff in log of CPI inflation to avoid AR(1) upper-bound error."
        )
        df_cpi = load_file(ids.API_DICT[ids.CPI])
        df_cpi = calculate_diff_in_log(df_cpi, [ids.CPI])
        combined_dfs = combine_series(
            [dict_of_combined_dataframes[ids.EXPECTATIONS], df_cpi],
            how="left",
            on=INDEX_COLUMN_NAME,
        )
        logger.debug("Columns: %s", combined_dfs.columns)
        wavelet_plots.plot_xwt(combined_dfs, [ids.EXPECTATIONS, ids.DIFF_LOG_CPI])

    elif transform_selection == ids.DWT:
        wavelet_plots.plot_dwt(
            combined_dfs, column_names, dwt_plot_selection, dwt_smooth_plot_order
        )

    elif transform_selection == ids.CWT and len(column_names) == 1:
        wavelet_plots.plot_cwt(combined_dfs, column_names)

    elif transform_selection == ids.CWT and len(column_names) == 2:
        st.write("Looks like you're looking for the _cross-wavelet transform_")
        wavelet_plots.plot_xwt(combined_dfs, column_names)

    elif transform_selection == ids.XWT and len(column_names) == 2:
        wavelet_plots.plot_xwt(combined_dfs, column_names)

    elif transform_selection == ids.XWT and len(column_names) < 2:
        st.write("Please supply a second series.")

    plt.legend("", frameon=False)

else:
    st.subheader("How it works")
    st.text("1.) Navigate to the left-hand sidebar.")
    st.text("2.) Select a wavelet transform.")
    st.text("3.) Select a sample dataset or upload your own!.")

st.markdown(
    """*Please note that this tool has been tested on a limited number of datasets so far. 
        Please, [contact me](mailto:nathaniel@nathaniellawrence.com) if yours isn't working!*"""
)
