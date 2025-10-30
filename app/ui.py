"""Builds the sidebar and returns a Selection dataclass describing
the user's choices plus a prepared `file_dict` mapping display names to
either file paths or uploaded file-like objects.
"""

from typing import Dict, List
import os

import streamlit as st

from constants import ids
from app.types import Selection


def adjust_sidebar(selection: str) -> str | None:
    """Add second option to sidebar if DWT selected

    Args:
        selection (str): _description_

    Returns:
        str: _description_
    """
    if selection == ids.DWT:
        return st.sidebar.selectbox(
            "**Select a DWT plot**",
            (ids.SMOOTH, ids.DECOMPOSE),
            key="dwt_selection",
        )
    if selection == ids.SMOOTH:
        return st.sidebar.radio(
            "**Select plot order**",
            (ids.DESCEND, ids.ASCEND),
        )
    return None


def build_sidebar() -> Selection:
    """Builds the left-hand sidebar and returns a Selection object.

    This centralizes all UI state related to the sidebar so the main
    `app.py` can be a thin orchestrator.
    """
    st.title("Wavelet Transformer")

    transform_selection = st.sidebar.selectbox(
        "**Select a wavelet transform**",
        (ids.CWT, ids.DWT, ids.WCT),
        key="wavelet_selection",
    )

    # Significance testing defaults
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
    # normalize to an integer when possible
    try:
        dwt_smooth_plot_order = int(dwt_smooth_plot_order)
    except Exception:
        dwt_smooth_plot_order = None

    uploaded_files: List = []
    file_dict: Dict[str, object] = {}

    if transform_selection in [ids.CWT, ids.WCT]:
        calculate_significance = st.sidebar.checkbox(
            "Calculate statistical significance"
        )
        if calculate_significance:
            significance_level = st.sidebar.number_input(
                "Significance level", min_value=0, max_value=100, value=95
            )

    # Handle selected sample data
    if any(data in ids.DISPLAY_NAMES.values() for data in selected_data):
        sample_data_items = [
            data for data in selected_data if data in ids.DISPLAY_NAMES.values()
        ]
        for dataset in sample_data_items:
            file_path = ids.DATA_SCHEMA[dataset]["file_path"]
            if isinstance(file_path, str):
                key = file_path.split(os.path.sep)[-1].split(".")[0]
                name = ids.DISPLAY_NAMES.get(key, key)
                file_dict[name] = file_path
            else:
                key = getattr(file_path, "name", "uploaded").split(".")[0]
                name = ids.DISPLAY_NAMES.get(key, key)
                file_dict[name] = file_path
            uploaded_files.append(file_path)

    # File uploader - append to uploaded_files and file_dict if user has own data
    if "I have my own!🤓" in selected_data:
        user_files = st.sidebar.file_uploader(
            "Choose Excel or CSV files",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
        )
        if user_files:
            for file in user_files:
                key = file.name.split(".")[0]
                file_dict[key] = file
                uploaded_files.append(file)

    return Selection(
        transform=transform_selection,
        selected_data=selected_data,
        calculate_significance=calculate_significance,
        significance_level=significance_level,
        dwt_plot_selection=dwt_plot_selection,
        dwt_smooth_plot_order=dwt_smooth_plot_order,
        file_dict=file_dict,
        uploaded_files=uploaded_files,
    )
