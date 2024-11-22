"""Streamlit dashboard app"""

import logging
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils.file_helpers import convert_to_dataframe
from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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

    for uploaded_file in uploaded_files:
        df = convert_to_dataframe(uploaded_file)
        logger.debug("df columns: %s", df.columns)

        # Add trace to plot
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["value"], mode="lines", name=uploaded_file.name
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Data Visualization of: {' '.join(file_dict.keys())}",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    # Display plot
    st.plotly_chart(fig)
else:
    st.write("Upload a file to begin")
