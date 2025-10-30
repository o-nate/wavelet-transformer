"""Helpers to sanitize and merge DataFrames for plotting/stats/regression."""

from typing import List

import pandas as pd


def sanitize_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Select the first non-date column and rename it to the provided name.

    Raises ValueError if no value column found.
    """
    value_cols = [c for c in df.columns if c.lower() != "date"]
    if not value_cols:
        raise ValueError(f"No value column found for {name}")
    part = df[["date", value_cols[0]]].rename(columns={value_cols[0]: name})
    return part


def merge_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of small dataframes on 'date' and sort the result."""
    if not parts:
        return pd.DataFrame()
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged
