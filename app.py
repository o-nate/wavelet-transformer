"""Streamlit dashboard app"""

import os

import streamlit as st
import pandas as pd
import numpy as np

from constants import ids

from src import wavelet_plots
from src.utils.helpers import adjust_sidebar
from src import statistical_analysis as stats
from src.utils.file_helpers import load_file
from src.utils.config import INDEX_COLUMN_NAME
from src.utils.transform_helpers import create_dwt_dict
from src import dwt, regression

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

st.title("Wavelet Transformer")

# Add sidebar for controlling parameters
transform_selection = st.sidebar.selectbox(
    "**Select a wavelet transform**",
    (ids.CWT, ids.DWT, ids.WCT),
    key="wavelet_selection",
)

# Add checkbox for CWT/WCT significance testing
calculate_significance = False
significance_level = 95

# Sample datasets
selected_data = st.sidebar.multiselect(
    "**Select a dataset**",
    [ids.DISPLAY_NAMES[name] for name in ids.SAMPLE_DATA] + ["I have my own!🤓"],
    max_selections=2,
)

dwt_plot_selection = adjust_sidebar(transform_selection)
dwt_smooth_plot_order = adjust_sidebar(dwt_plot_selection)

# Initialize uploaded_files and file_dict
uploaded_files = []
file_dict = {}

# Check if any sample data is selected
if any(data in ids.DISPLAY_NAMES.values() for data in selected_data):
    # Filter for only the sample data items
    sample_data_items = [
        data for data in selected_data if data in ids.DISPLAY_NAMES.values()
    ]
    for dataset in sample_data_items:
        file_path = ids.DATA_SCHEMA[dataset]["file_path"]
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

    # Create plot and descriptive stats tabs when files exist
if file_dict:
    if transform_selection == ids.DWT and len(file_dict) == 2:
        tab_plot, tab_stats, tab_regression = st.tabs(
            ["Plot", "Descriptive statistics", "Time-scale regression"]
        )
    else:
        tab_plot, tab_stats = st.tabs(["Plot", "Descriptive statistics"])

    with tab_plot:
        if transform_selection == ids.DWT and len(file_dict) == 2:
            wavelet_plots.generate_plot(
                file_dict=file_dict,
                transform_selection=transform_selection,
                selected_data=selected_data,
                dwt_plot_selection=dwt_plot_selection,
                dwt_smooth_plot_order=dwt_smooth_plot_order,
            )
        else:
            wavelet_plots.generate_plot(
                file_dict=file_dict,
                transform_selection=transform_selection,
                selected_data=selected_data,
                dwt_plot_selection=dwt_plot_selection,
                dwt_smooth_plot_order=dwt_smooth_plot_order,
                calculate_significance=calculate_significance,
                significance_level=significance_level,
            )

    if transform_selection == ids.CWT and len(file_dict) == 1:
        calculate_significance = st.sidebar.checkbox(
            "Calculate statistical significance", value=True
        )
        if calculate_significance:
            significance_level = st.sidebar.number_input(
                "Significance level", min_value=0, max_value=100, value=95
            )
    elif transform_selection == ids.WCT or (
        transform_selection == ids.CWT and len(file_dict) > 1
    ):
        calculate_significance = st.sidebar.checkbox(
            "Calculate statistical significance"
        )
        if calculate_significance:
            significance_level = st.sidebar.number_input(
                "Significance level", min_value=0, max_value=100, value=95
            )

    with tab_stats:
        plural = "s" if len(file_dict) > 1 else ""
        st.subheader(f"Descriptive statistics for the dataset{plural} loaded")
        parts = []
        errors = []
        for name, path_or_file in file_dict.items():
            try:
                df = load_file(path_or_file)
                if df is None:
                    errors.append(f"{name}: could not load file into DataFrame")
                    continue

                # If loader stored dates in the index (INDEX_COLUMN_NAME), move it to a 'date' column
                if (
                    df.index.name == INDEX_COLUMN_NAME
                    or INDEX_COLUMN_NAME in df.columns
                ):
                    df = df.reset_index()
                    if INDEX_COLUMN_NAME in df.columns:
                        df = df.rename(columns={INDEX_COLUMN_NAME: "date"})

                # Ensure there's a 'date' column and it's datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # pick the first non-date column as the value column and rename it to the display name
                value_cols = [c for c in df.columns if c.lower() != "date"]
                if not value_cols:
                    errors.append(f"{name}: no value column found")
                    continue
                part = df[["date", value_cols[0]]].rename(columns={value_cols[0]: name})
                parts.append(part)
            except FileNotFoundError as fe:
                errors.append(f"{name}: {fe}")
            except Exception as e:
                errors.append(f"{name}: {e}")

        if parts:
            merged = parts[0]
            for p in parts[1:]:
                merged = merged.merge(p, on="date", how="outer")
            merged = merged.sort_values("date").reset_index(drop=True)

            # Call descriptive stats generator on the merged dataframe (expects a 'date' column)
            results_df = stats.generate_descriptive_statistics(
                merged, stats.DESCRIPTIVE_STATS
            )
            st.markdown(
                f"Start date: `{merged['date'].min().date()}` | End date: `{merged['date'].max().date()}`"
            )
            st.dataframe(results_df)
        else:
            st.info("No valid datasets available to compute descriptive statistics.")

        for err in errors:
            st.error(f"Could not compute descriptive stats for {err}")

    if transform_selection == ids.DWT and len(file_dict) == 2:
        with tab_regression:
            st.markdown("### Time-scale Regression Analysis", unsafe_allow_html=True)

            # Load and process the data
            data_frames = []
            for name, path_or_file in file_dict.items():
                df = load_file(path_or_file)
                if df is not None:
                    # Process datetime index if needed
                    if (
                        df.index.name == INDEX_COLUMN_NAME
                        or INDEX_COLUMN_NAME in df.columns
                    ):
                        df = df.reset_index()
                        if INDEX_COLUMN_NAME in df.columns:
                            df = df.rename(columns={INDEX_COLUMN_NAME: "date"})

                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")

                    value_cols = [c for c in df.columns if c.lower() != "date"]
                    if value_cols:
                        df_renamed = df[["date", value_cols[0]]].copy()
                        df_renamed.columns = ["date", name]
                        data_frames.append(df_renamed)

            if len(data_frames) == 2:
                # Merge the data frames
                merged_df = data_frames[0].merge(data_frames[1], on="date", how="inner")
                merged_df = merged_df.sort_values("date").reset_index(drop=True)

                if not merged_df.empty:
                    # Get the column names for X and Y variables
                    var_names = list(merged_df.columns[1:])  # Skip date column
                    x_var = var_names[0]
                    y_var = var_names[1]

                    # Prepare data and validate
                    level = 6  # Use a fixed level of 6 for decomposition
                    x_data = merged_df[x_var].values
                    y_data = merged_df[y_var].values

                    # Calculate time-scale regression
                    try:
                        # Check for invalid values
                        if not (
                            np.isfinite(x_data).all() and np.isfinite(y_data).all()
                        ):
                            st.error(
                                "Data contains invalid values (NaN or infinite). Please check your input data."
                            )
                        else:
                            results = regression.time_scale_regression(
                                x_data, y_data, level, "db4", add_constant=True
                            )
                            # Display results
                            st.markdown(f"Independent variable (x1): `{x_var}`")
                            st.markdown(f"Dependent variable (y): `{y_var}`")
                            st.markdown(f"Number of decomposition levels: `{level}`")

                            # Convert regression results to HTML and display
                            st.write(results.as_html(), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during regression analysis: {str(e)}")
                else:
                    st.error("No overlapping dates found between the two datasets.")
            else:
                st.error(
                    "Could not properly load both datasets for regression analysis."
                )

else:
    st.text(
        "This app allows you conduct wavelet analysis by applying the wavelet transform to data!"
    )
    st.subheader("How it works")
    st.text("1.) 👈 Navigate to the left-hand sidebar.")
    st.text("2.) 🌊Select a wavelet transform.")
    st.text("3.) 📈Select a sample dataset or upload your own!")

    st.markdown(
        """\n*Please note that this tool has been tested on a limited number of datasets so far. 
        Please, [contact me](mailto:nathaniel@nathaniellawrence.com) if yours isn't working!*"""
    )
