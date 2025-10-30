"""File loading and caching helpers"""

from typing import Optional

import pandas as pd
import streamlit as st

from src.utils.file_helpers import load_file as _orig_load_file
from src.utils.config import INDEX_COLUMN_NAME
from utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data(show_spinner=False)
def load_dataset(path_or_file) -> Optional[pd.DataFrame]:
    """Load a file (path or uploaded file) and normalize index/date column.

    Returns None on failure.
    """
    try:
        df = _orig_load_file(path_or_file)
    except Exception as _:
        logger.exception(
            "Failed to load file: %s", getattr(path_or_file, "name", str(path_or_file))
        )
        return None

    if df is None:
        return None

    try:
        # If loader stored dates in the index (INDEX_COLUMN_NAME), move it to a 'date' column
        if (
            getattr(df.index, "name", None) == INDEX_COLUMN_NAME
            or INDEX_COLUMN_NAME in df.columns
        ):
            df = df.reset_index()
            if INDEX_COLUMN_NAME in df.columns:
                df = df.rename(columns={INDEX_COLUMN_NAME: "date"})

        # Ensure there's a 'date' column and it's datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df
    except Exception:
        logger.exception("Failed to normalize dataset")
        return None
