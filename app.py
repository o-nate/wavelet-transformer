"""Streamlit dashboard app"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from plotly import tools

from constants import ids, results_configs
from utils.logging_config import get_logger

from src import dwt, wavelet_plots

from src.utils.file_helpers import load_file
from src.utils.helpers import adjust_sidebar, combine_series
from src.utils.plot_helpers import plot_dwt_decomposition_for, plot_dwt_smoothing_for
from src.utils.transform_helpers import create_dwt_dict, create_dwt_results_dict

# * Logging settings
logger = get_logger(__name__)

st.title("Wavelet Analysis Interactive Dashboard")

# Add sidebar for controlling parameters
transform_selection = st.sidebar.selectbox(
    "**Select a wavelet transform**",
    (ids.CWT, ids.DWT, ids.XWT),
    key="wavelet_selection",
)
dwt_plot_selection = adjust_sidebar(transform_selection)
dwt_smooth_plot_order = adjust_sidebar(dwt_plot_selection)

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
    # TODO change to list now that file name is stored as column name
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
    )

    logger.debug("combined df columns: %s", combined_dfs.columns.to_list())

    if transform_selection == ids.DWT:
        wavelet_plots.plot_dwt(
            combined_dfs, column_names, dwt_plot_selection, dwt_smooth_plot_order
        )

    if transform_selection == ids.CWT and len(column_names) == 1:
        wavelet_plots.plot_cwt(combined_dfs, column_names)

    if transform_selection == ids.CWT and len(column_names) == 2:
        st.write("Looks like you're looking for the _cross-wavelet transform_")
        wavelet_plots.plot_xwt(combined_dfs, column_names)

    if transform_selection == ids.XWT and len(column_names) == 2:
        wavelet_plots.plot_xwt(combined_dfs, column_names)

    if transform_selection == ids.XWT and len(column_names) < 2:
        st.write("Please supply a second series.")

    plt.legend("", frameon=False)

else:
    st.write("Upload a spreadsheet file to begin.")
    st.write(
        """Please, ensure that each file has only two columns: 
        the **date column on the left** and the **value column on the right**."""
    )
