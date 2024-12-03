"""Streamlit dashboard app"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from plotly import tools

from constants import ids, results_configs
from utils.logging_config import get_logger

from src import dwt

from src.utils.file_helpers import load_file
from src.utils.helpers import adjust_sidebar, combine_series
from src.utils.plot_helpers import plot_dwt_decomposition_for, plot_dwt_smoothing_for
from src.utils.transform_helpers import create_dwt_dict, create_dwt_results_dict

# * Logging settings
logger = get_logger(__name__)

st.title("Wavelet Analysis Interactive Dashboard")

# Add sidebar for controlling parameters
add_selectbox = st.sidebar.selectbox(
    "**Select a wavelet transform**",
    (ids.DWT, ids.CWT, ids.XWT),
    key="wavelet_selection",
)
logger.debug(type(add_selectbox))
dwt_plot_selection = adjust_sidebar(add_selectbox)
logger.debug(dwt_plot_selection)
dwt_smooth_plot_order = adjust_sidebar(dwt_plot_selection)
logger.debug(dwt_smooth_plot_order)

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

    dwt_dict = create_dwt_dict(
        combined_dfs.dropna(),
        column_names,
        mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    )

    # * Run DWTs
    dwt_results_dict = create_dwt_results_dict(dwt_dict, column_names)

    st.write(f"Showing DWT of: {', '.join(dwt_dict.keys())}")

    # Frequency decomposition plot
    t = combined_dfs.dropna().index
    if dwt_plot_selection == ids.DECOMPOSE:
        fig = plot_dwt_decomposition_for(
            dwt_results_dict,
            t,
            wavelet=results_configs.DWT_MOTHER_WAVELET,
            figsize=(15, 20),
            sharex=True,
        )
        plt.legend("", frameon=False)

        # Display plot
        st.plotly_chart(fig)

    if dwt_plot_selection == ids.SMOOTH:
        if dwt_smooth_plot_order == ids.ASCEND:
            ASCENDING = True
        else:
            ASCENDING = False
        if len(dwt_results_dict) == 1:
            fig = plot_dwt_smoothing_for(
                dwt_dict,
                dwt_results_dict,
                t,
                ascending=ASCENDING,
                figsize=(15, 20),
                sharex=True,
            )
            plt.legend("", frameon=False)

            # Display plot
            st.plotly_chart(fig)
        elif len(dwt_results_dict) == 2:
            col1, col2 = st.columns(2)
            with col1:
                dwt_results_dict[column_names[0]].smooth_signal(
                    y_values=dwt_dict[column_names[0]].y_values,
                    mother_wavelet=dwt_dict[column_names[0]].mother_wavelet,
                )
                fig1 = dwt.plot_smoothing(
                    dwt_results_dict[column_names[0]].smoothed_signal_dict,
                    t,
                    dwt_dict[column_names[0]].y_values,
                    ascending=ASCENDING,
                    figsize=(15, 20),
                    sharex=True,
                )

                # Display plot
                # fig1.layout.update(width=600, height=600)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                dwt_results_dict[column_names[1]].smooth_signal(
                    y_values=dwt_dict[column_names[1]].y_values,
                    mother_wavelet=dwt_dict[column_names[1]].mother_wavelet,
                )
                fig2 = dwt.plot_smoothing(
                    dwt_results_dict[column_names[1]].smoothed_signal_dict,
                    t,
                    dwt_dict[column_names[1]].y_values,
                    ascending=ASCENDING,
                    figsize=(15, 20),
                    sharex=True,
                )

                # Display plot
                # fig2.layout.update(width=600, height=600)
                st.plotly_chart(fig2, use_container_width=True)
        plt.legend("", frameon=False)

else:
    st.write("Upload a spreadsheet file to begin.")
    st.write(
        """Please, ensure that each file has only two columns: 
        the **date column on the left** and the **value column on the right**."""
    )
