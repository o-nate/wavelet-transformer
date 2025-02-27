"""Produce wavelet transform plots"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from matplotlib.figure import Figure
from plotly import tools

from constants import ids, results_configs
from utils.logging_config import get_logger

from src import cwt, dwt, xwt

from src.utils.file_helpers import load_file
from src.utils.helpers import adjust_sidebar, combine_series
from src.utils.plot_helpers import (
    plot_dwt_decomposition_for,
    plot_dwt_smoothing_for,
    set_x_ticks,
)
from src.utils.transform_helpers import (
    create_cwt_dict,
    create_cwt_results_dict,
    create_dwt_dict,
    create_dwt_results_dict,
)
from src.utils.wavelet_helpers import standardize_series

# * Logging settings
logger = get_logger(__name__)


def plot_cwt(data: pd.DataFrame, series_names: list[str]) -> Figure:
    """
    Performs Continuous Wavelet Transform (CWT) analysis on a time series and creates a visualization.

    This function processes the input time series by standardizing and optionally detrending it,
    computes the CWT using specified parameters, and generates a heatmap visualization of the
    wavelet power spectrum.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing the time series data.
        The index should represent time points.
        Must contain the column specified in series_names.

    series_names : list[str]
        List of column names in the DataFrame to analyze.
        Currently, only the first series name is used.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the CWT plot.
        Note: The figure is also displayed using streamlit's st.pyplot().

    Technical Details
    ----------------
    - Preprocessing:
        * Removes NaN values from the input series
        * Standardizes the series using standardize_series()

    - CWT Parameters (from results_configs):
        * Mother Wavelet: Set in CWT_MOTHER
        * Time step (delta_t): Set in DT
        * Scale resolution (delta_j): Set in DJ
        * Initial scale (s0): Set in S0
        * Number of scales: Set in LEVELS

    - Plot Configuration:
        * Figure properties are set in results_configs.CWT_FIG_PROPS
        * Plot properties are set in results_configs.CWT_PLOT_PROPS
        * Y-axis is labeled in years

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range(start='2000-01-01', periods=1000, freq='D')
    >>> values = np.sin(2 * np.pi * dates.dayofyear / 365)  # Annual cycle
    >>> df = pd.DataFrame({'temperature': values}, index=dates)
    >>>
    >>> # Plot CWT
    >>> fig = plot_cwt(df, ['temperature'])

    Notes
    -----
    - The function currently only processes the first series in series_names
    - Logging is implemented for debugging purposes
    - The plot is automatically displayed using streamlit
    - The y-axis (period) is displayed in years

    See Also
    --------
    standardize_series : Function used for preprocessing the time series
    cwt.DataForCWT : Class handling CWT input data
    cwt.run_cwt : Function performing the CWT computation
    cwt.plot_cwt : Function creating the CWT visualization

    Warnings
    --------
    - Input data should have a regular time step for meaningful results
    - Large datasets may require significant computational resources
    - NaN values in the input series are automatically removed
    """
    logger.debug("series_name: %s", series_names)
    series_name = series_names[0]
    logger.debug(series_name)

    # * Pre-process data: Standardize and detrend
    logger.debug("nans: %s", data[f"{series_name}"].isna().sum())
    t = data.dropna().index.to_numpy()
    y1 = data[f"{series_name}"].dropna().to_numpy()
    y1 = standardize_series(y1)  # , detrend=False, remove_mean=True)

    data_for_cwt = cwt.DataForCWT(
        t,
        y1,
        mother_wavelet=results_configs.CWT_MOTHER,
        delta_t=results_configs.DT,
        delta_j=results_configs.DJ,
        initial_scale=results_configs.S0,
        levels=results_configs.LEVELS,
    )

    results_from_cwt = cwt.run_cwt(data_for_cwt, standardize=True)

    # * Plot results
    plt.close("all")
    fig, ax = plt.subplots(1, 1, **results_configs.CWT_FIG_PROPS)
    cwt.plot_cwt(ax, data_for_cwt, results_from_cwt, **results_configs.CWT_PLOT_PROPS)

    # * Set labels/title
    ax.set_xlabel("")
    ax.set_ylabel("Period (years)")
    ax.set_title(series_name)

    st.pyplot(fig)


def plot_dwt(
    data: pd.DataFrame,
    series_names: list[str],
    plot_selection: str,
    plot_order: str = ids.ASCEND,
) -> Figure:
    """
    Performs Discrete Wavelet Transform (DWT) analysis on time series data and
    creates visualizations.

    This function supports both decomposition and smoothing visualization modes for
    either one or two time series. When processing two series, smoothing plots
    are displayed side by side using Streamlit columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing the time series data.
        The index should represent time points.
        Must contain the columns specified in series_names.

    series_names : list[str]
        List of column names in the DataFrame to analyze.
        Supports either one or two series names.

    plot_selection : str
        Type of plot to generate. Must be one of:
        - ids.DECOMPOSE: Shows wavelet decomposition at different levels
        - ids.SMOOTH: Shows progressive signal reconstruction

    plot_order : str, optional
        Order of plotting for smoothing visualization. Default is ids.ASCEND.
        - ids.ASCEND: Plot from finest to coarsest scales
        - ids.DESCEND: Plot from coarsest to finest scales

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the DWT plot.
        Note: The figure is also displayed using streamlit's plotly_chart().

    Technical Details
    ----------------
    - Preprocessing:
        * Removes NaN values from the input series
        * Creates DWT dictionary using create_dwt_dict()
        * Processes results using create_dwt_results_dict()

    - DWT Parameters (from results_configs):
        * Mother Wavelet: Set in DWT_MOTHER_WAVELET

    - Visualization Modes:
        1. Decomposition (plot_selection == ids.DECOMPOSE):
            * Shows the original signal and its wavelet components
            * All plots share the same x-axis
            * Figure size is set to (15, 20)

        2. Smoothing (plot_selection == ids.SMOOTH):
            * For single series: Shows progressive signal reconstruction
            * For two series: Creates side-by-side plots using Streamlit columns
            * Plots are ordered according to plot_order parameter

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range(start='2000-01-01', periods=1000, freq='D')
    >>> values1 = np.sin(2 * np.pi * dates.dayofyear / 365)  # Annual cycle
    >>> values2 = np.sin(2 * np.pi * dates.dayofyear / 180)  # Semi-annual cycle
    >>> df = pd.DataFrame({
    ...     'series1': values1,
    ...     'series2': values2
    ... }, index=dates)
    >>>
    >>> # Plot DWT decomposition
    >>> fig1 = plot_dwt(df, ['series1'], plot_selection=ids.DECOMPOSE)
    >>>
    >>> # Plot DWT smoothing for two series
    >>> fig2 = plot_dwt(
    ...     df,
    ...     ['series1', 'series2'],
    ...     plot_selection=ids.SMOOTH,
    ...     plot_order=ids.ASCEND
    ... )

    Notes
    -----
    - The function supports one or two time series analysis
    - For two series in smoothing mode, plots are displayed side by side
    - Uses Streamlit for interactive display
    - Legends are disabled in the current implementation

    See Also
    --------
    create_dwt_dict : Function creating DWT input dictionary
    create_dwt_results_dict : Function processing DWT results
    plot_dwt_decomposition_for : Function for decomposition visualization
    plot_dwt_smoothing_for : Function for smoothing visualization
    dwt.plot_smoothing : Base function for smoothing plots

    Warnings
    --------
    - Input data should have regular time steps
    - NaN values are automatically removed
    - When using two series, both should have similar scale properties
    - Memory usage increases with data length and decomposition levels
    """

    dwt_dict = create_dwt_dict(
        data.dropna(),
        series_names,
        mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    )
    dwt_results_dict = create_dwt_results_dict(dwt_dict, series_names)

    st.write(f"Showing DWT of: {', '.join(dwt_dict.keys())}")

    t = data.dropna().index
    if plot_selection == ids.DECOMPOSE:
        fig = plot_dwt_decomposition_for(
            dwt_results_dict,
            t,
            wavelet=results_configs.DWT_MOTHER_WAVELET,
            figsize=(15, 20),
            sharex=True,
        )
        # plt.legend("", frameon=False)

        st.pyplot(fig)

    if plot_selection == ids.SMOOTH:
        ascending_order = bool(plot_order == ids.ASCEND)
        if len(dwt_results_dict) == 1:
            fig = plot_dwt_smoothing_for(
                dwt_dict,
                dwt_results_dict,
                t,
                ascending=ascending_order,
                figsize=(15, 20),
                sharex=True,
            )
            fig.suptitle(f"Smoothing of {series_names[0]}", fontsize=24)

            st.pyplot(fig)

        elif len(dwt_results_dict) == 2:
            col1, col2 = st.columns(2)
            with col1:
                dwt_results_dict[series_names[0]].smooth_signal(
                    y_values=dwt_dict[series_names[0]].y_values,
                    mother_wavelet=dwt_dict[series_names[0]].mother_wavelet,
                )
                fig1 = dwt.plot_smoothing(
                    dwt_results_dict[series_names[0]].smoothed_signal_dict,
                    t,
                    dwt_dict[series_names[0]].y_values,
                    ascending=ascending_order,
                    figsize=(15, 20),
                    sharex=True,
                )
                fig1.suptitle(f"Smoothing of {series_names[0]}", fontsize=36)

                st.pyplot(fig1, use_container_width=True)

            with col2:
                dwt_results_dict[series_names[1]].smooth_signal(
                    y_values=dwt_dict[series_names[1]].y_values,
                    mother_wavelet=dwt_dict[series_names[1]].mother_wavelet,
                )
                fig2 = dwt.plot_smoothing(
                    dwt_results_dict[series_names[1]].smoothed_signal_dict,
                    t,
                    dwt_dict[series_names[1]].y_values,
                    ascending=ascending_order,
                    figsize=(15, 20),
                    sharex=True,
                )
                fig2.suptitle(f"Smoothing of {series_names[1]}", fontsize=36)

                st.pyplot(fig2, use_container_width=True)


def plot_xwt(data: pd.DataFrame, series_names: list[str]) -> Figure:
    """
    Performs Cross Wavelet Transform (XWT) analysis between two time series and creates a visualization.

    This function computes the cross wavelet transform between two time series, showing their
    common power and relative phase in time-frequency space. It includes significance testing,
    cone of influence, and phase difference visualization.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing both time series data.
        The index should represent time points.
        Must contain both columns specified in series_names.

    series_names : list[str]
        List containing exactly two column names for the series to analyze.
        Order matters: series_names[0] is treated as y1, series_names[1] as y2.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the XWT plot.
        Note: The figure is also displayed using streamlit's st.pyplot().

    Technical Details
    ----------------
    - Preprocessing:
        * Removes NaN values from both series
        * For first series (y1):
            - Standardizes and removes mean
            - No detrending
        * For second series (y2):
            - Standardizes and detrends
            - Keeps mean

    - XWT Parameters (from results_configs):
        * Mother Wavelet: Set in XWT_MOTHER_DICT[XWT_MOTHER]
        * Time step (delta_t): Set in XWT_DT
        * Scale resolution (delta_j): Set in XWT_DJ
        * Initial scale (s0): Set in XWT_S0
        * Number of scales: Set in LEVELS

    - Plot Features:
        * Power spectrum with significance contours
        * Cone of influence showing edge effects
        * Phase difference arrows
        * Period displayed in logarithmic scale (base 2)
        * Custom x-axis tick formatting
        * Figure size: 10x8 inches

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data with two related signals
    >>> dates = pd.date_range(start='2000-01-01', periods=1000, freq='D')
    >>> values1 = np.sin(2 * np.pi * dates.dayofyear / 365)  # Annual cycle
    >>> values2 = np.sin(2 * np.pi * dates.dayofyear / 365 + np.pi/4)  # Phase-shifted
    >>> df = pd.DataFrame({
    ...     'temperature': values1,
    ...     'precipitation': values2
    ... }, index=dates)
    >>>
    >>> # Plot XWT
    >>> fig = plot_xwt(df, ['temperature', 'precipitation'])

    Plot Interpretation
    ------------------
    - Heatmap: Shows common power between signals
        * Brighter colors indicate stronger common power
        * Black contours indicate statistical significance
    - Arrows: Indicate phase relationship
        * Right: In-phase
        * Left: Anti-phase
        * Up: Second series leads by 90°
        * Down: First series leads by 90°
    - Cone of Influence: Region outside is subject to edge effects

    Notes
    -----
    - Y-axis (period) is displayed in years
    - Requires exactly two input series
    - Plot includes statistical significance testing
    - Edge effects are indicated by the cone of influence

    See Also
    --------
    standardize_series : Function used for preprocessing
    xwt.DataForXWT : Class handling XWT input data
    xwt.run_xwt : Function performing the XWT computation
    xwt.plot_xwt : Function creating the XWT visualization
    set_x_ticks : Function formatting x-axis ticks

    Warnings
    --------
    - Input series should have matching timestamps
    - Series should be sufficiently long for meaningful analysis
    - Edge effects become significant at larger scales
    - Interpretation of phase differences requires careful consideration
        of physical relationships between variables
    """

    # * Pre-process data: Standardize and detrend
    data_no_nans = data.dropna()
    t = data_no_nans.index.to_list()
    y1 = data_no_nans[series_names[0]].to_numpy()
    y2 = data_no_nans[series_names[1]].to_numpy()
    y1 = standardize_series(y1, detrend=True, remove_mean=False)
    y2 = standardize_series(y2, detrend=True, remove_mean=False)

    xwt_data = xwt.DataForXWT(
        y1_values=y1,
        y2_values=y2,
        mother_wavelet=results_configs.XWT_MOTHER_DICT[results_configs.XWT_MOTHER],
        delta_t=results_configs.XWT_DT,
        delta_j=results_configs.XWT_DJ,
        initial_scale=results_configs.XWT_S0,
        levels=results_configs.LEVELS,
    )

    results_from_xwt = xwt.run_xwt(xwt_data)

    # * Plot XWT power spectrum
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    xwt.plot_xwt(
        axs,
        xwt_data,
        results_from_xwt,
        include_significance=True,
        include_cone_of_influence=True,
        include_phase_difference=True,
        **results_configs.XWT_PLOT_PROPS,
    )

    axs.set_ylim(axs.get_ylim()[::-1])

    # * Set y axis tick labels
    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(results_from_xwt.period.min())),
        np.ceil(np.log2(results_from_xwt.period.max())),
    )
    axs.set_yticks(np.log2(y_ticks))
    axs.set_yticklabels(y_ticks, size=12)

    # * Set x axis tick labels
    x_tick_positions, x_ticks = set_x_ticks(t)
    axs.set_xticks(x_tick_positions)
    axs.set_xticklabels(x_ticks, size=12)

    axs.set_title(f"{series_names[0]} X {series_names[1]}", size=16)
    axs.set_ylabel("Period (years)", size=14)

    st.pyplot(fig)
