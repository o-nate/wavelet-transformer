"""Orchestrator module."""

import streamlit as st
from utils.logging_config import get_logger

from constants import ids
from src import statistical_analysis as stats

from app.ui import build_sidebar
from app.plotting import generate_plot_ui
from app.input_output import load_dataset
from app.data_processing import sanitize_df, merge_parts
from app.regression_ui import render_regression_tab

logger = get_logger(__name__)


def main():
    selection = build_sidebar()
    file_dict = selection.file_dict

    if file_dict:
        if selection.transform == ids.DWT and len(file_dict) == 2:
            tab_plot, tab_stats, tab_regression = st.tabs(
                ["Plot", "Descriptive statistics", "Time-scale regression"]
            )
        else:
            tab_plot, tab_stats = st.tabs(["Plot", "Descriptive statistics"])

        with tab_plot:
            generate_plot_ui(selection)

        # Significance options in sidebar for CWT/WCT handled by the plotting adapter or UI

        with tab_stats:
            plural = "s" if len(file_dict) > 1 else ""
            st.subheader(f"Descriptive statistics for the dataset{plural} loaded")
            parts = []
            errors = []
            for name, path_or_file in file_dict.items():
                try:
                    df = load_dataset(path_or_file)
                    if df is None:
                        errors.append(f"{name}: could not load file into DataFrame")
                        continue

                    # Sanitize and pick a single value column per dataset
                    part = sanitize_df(df, name)
                    parts.append(part)
                except FileNotFoundError as fe:
                    errors.append(f"{name}: {fe}")
                except Exception as e:
                    errors.append(f"{name}: {e}")

            if parts:
                merged = merge_parts(parts)
                # Call descriptive stats generator on the merged dataframe (expects a 'date' column)
                results_df = stats.generate_descriptive_statistics(
                    merged, stats.DESCRIPTIVE_STATS
                )
                st.markdown(
                    f"Start date: `{merged['date'].min().date()}` | End date: `{merged['date'].max().date()}`"
                )
                st.dataframe(results_df)
            else:
                st.info(
                    "No valid datasets available to compute descriptive statistics."
                )

            for err in errors:
                st.error(f"Could not compute descriptive stats for {err}")

        if selection.transform == ids.DWT and len(file_dict) == 2 and tab_regression:
            with tab_regression:
                render_regression_tab(file_dict)
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


if __name__ == "__main__":
    main()
