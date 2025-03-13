"""Streamlit dashboard app"""

import os

import streamlit as st

from constants import ids

from src import wavelet_plots
from src.utils.helpers import adjust_sidebar

from utils.logging_config import get_logger

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
    "**Select a dataset**",
    [f"US {ids.DISPLAY_NAMES[name]}" for name in ids.SAMPLE_DATA]
    + ["I have my own!🤓"],
    max_selections=2,
)

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
            key = os.path.basename(file_path).split(".")[0]
            # Define key based on naming convention
            name = ids.DISPLAY_NAMES[key]
            file_dict[name] = file_path
        else:
            # For file objects with a name attribute
            key = file_path.name.split(".")[0]
            # Define key based on naming convention
            name = ids.DISPLAY_NAMES[key]
            file_dict[name] = file_path
        uploaded_files.append(file_path)

# File uploader - append to uploaded_files and file_dict if user has own data
if "I have my own!🤓" in selected_data:
    user_files = st.sidebar.file_uploader(
        "Choose Excel or CSV files", type=["csv", "xlsx"], accept_multiple_files=True
    )
    if user_files:
        for file in user_files:
            # Define key based on file name
            key = file.name.split(".")[0]
            file_dict[key] = file
            uploaded_files.append(file)


# Create plot
if file_dict:
    wavelet_plots.generate_plot(
        file_dict=file_dict,
        transform_selection=transform_selection,
        selected_data=selected_data,
        dwt_plot_selection=dwt_plot_selection,
        dwt_smooth_plot_order=dwt_smooth_plot_order,
    )

else:
    st.subheader("How it works")
    st.text("1.) 👈 Navigate to the left-hand sidebar.")
    st.text("2.) 🌊Select a wavelet transform.")
    st.text("3.) 📈Select a sample dataset or upload your own!")

st.markdown(
    """*Please note that this tool has been tested on a limited number of datasets so far. 
        Please, [contact me](mailto:nathaniel@nathaniellawrence.com) if yours isn't working!*"""
)
