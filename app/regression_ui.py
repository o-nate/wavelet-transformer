"""Regression tab for DWT time-scale regression."""

from typing import Dict, Any

import numpy as np
import streamlit as st

from app.input_output import load_dataset
from src import regression as _regression
from utils.logging_config import get_logger

logger = get_logger(__name__)


def render_regression_tab(file_dict: Dict[str, Any]) -> None:
    """Render the regression tab given a file_dict mapping names to files/paths."""
    st.markdown("### Time-scale Regression Analysis", unsafe_allow_html=True)

    data_frames = []
    for name, path_or_file in file_dict.items():
        df = load_dataset(path_or_file)
        if df is not None:
            # Ensure date column present and parsed
            if "date" in df.columns:
                df["date"] = df["date"]

            value_cols = [c for c in df.columns if c.lower() != "date"]
            if value_cols:
                df_renamed = df[["date", value_cols[0]]].copy()
                df_renamed.columns = ["date", name]
                data_frames.append(df_renamed)

    if len(data_frames) == 2:
        merged_df = data_frames[0].merge(data_frames[1], on="date", how="inner")
        merged_df = merged_df.sort_values("date").reset_index(drop=True)

        if not merged_df.empty:
            var_names = list(merged_df.columns[1:])
            x_var = var_names[0]
            y_var = var_names[1]

            level = 6
            x_data = merged_df[x_var].values
            y_data = merged_df[y_var].values

            try:
                if not (np.isfinite(x_data).all() and np.isfinite(y_data).all()):
                    st.error(
                        "Data contains invalid values (NaN or infinite). Please check your input data."
                    )
                else:
                    results = _regression.time_scale_regression(
                        x_data, y_data, level, "db4", add_constant=True
                    )
                    st.markdown(f"Independent variable (x1): `{x_var}`")
                    st.markdown(f"Dependent variable (y): `{y_var}`")
                    st.markdown(f"Number of decomposition levels: `{level}`")
                    st.write(results.as_html(), unsafe_allow_html=True)
            except Exception as e:
                logger.exception("Error during regression")
                st.error(f"Error during regression analysis: {str(e)}")
        else:
            st.error("No overlapping dates found between the two datasets.")
    else:
        st.error("Could not properly load both datasets for regression analysis.")
