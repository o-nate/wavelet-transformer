"""Produce wavelet transform plots"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from matplotlib.figure import Figure

from constants import ids, results_configs
from utils.logging_config import get_logger

from src import cwt, dwt, wct, xwt

from src.utils.config import INDEX_COLUMN_NAME
from src.utils.file_helpers import load_file
from src.utils.helpers import adjust_series_for_ar1_bound, combine_series
from src.utils.plot_helpers import (
    plot_dwt_decomposition_for,
    plot_dwt_smoothing_for,
    set_x_ticks,
)
from src.utils.transform_helpers import create_dwt_dict, create_dwt_results_dict
from src.utils.wavelet_helpers import standardize_series

# * Logging settings
logger = get_logger(__name__)


def plot_cwt(
    data: pd.DataFrame,
    series_names: list[str],
    calculate_significance: bool = True,
    significance_level: int = 95,
) -> Figure:
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

    calculate_significance : bool, optional
        Whether to calculate statistical significance, by default True

    significance_level : int, optional
        The significance level as an integer (e.g., 95 for 95%), by default 95

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
    series_name = series_names[0]

    # * Pre-process data: Standardize and detrend
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

    results_from_cwt = cwt.run_cwt(
        data_for_cwt,
        standardize=True,
        calculate_significance=calculate_significance,
        significance_level=significance_level / 100,
    )

    # * Plot results
    plt.close("all")
    fig, ax = plt.subplots(1, 1, **results_configs.CWT_FIG_PROPS)
    cwt.plot_cwt(
        ax,
        data_for_cwt,
        results_from_cwt,
        include_significance=calculate_significance,
        **results_configs.CWT_PLOT_PROPS,
    )

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


def plot_wct(
    data: pd.DataFrame,
    series_names: list[str],
    calculate_significance: bool = False,
    significance_level: int = 95,
) -> Figure:
    """
    Performs Wavelet Coherence Transform (WCT) analysis between two time series and creates a visualization.

    This function computes the wavelet coherence transform between two time series, showing their
    coherence magnitude and relative phase in time-frequency space. It includes significance testing,
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

    calculate_significance : bool, optional
        Whether to calculate statistical significance, by default False

    significance_level : int, optional
        The significance level as an integer (e.g., 95 for 95%), by default 95

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the WCT plot.
        Note: The figure is also displayed using streamlit's st.pyplot().

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

    - WCT Parameters (from results_configs):
        * Mother Wavelet: Set in XWT_MOTHER_DICT[XWT_MOTHER]
        * Time step (delta_t): Set in XWT_DT
        * Scale resolution (delta_j): Set in XWT_DJ
        * Initial scale (s0): Set in XWT_S0
        * Number of scales: Set in LEVELS

    - Plot Features:
        * Coherence magnitude with significance contours
        * Cone of influence showing edge effects
        * Phase difference arrows
        * Period displayed in logarithmic scale (base 2)
        * Custom x-axis tick formatting
        * Figure size: 10x8 inches
    """

    # * Pre-process data: Standardize and detrend
    data_no_nans = data.dropna()
    t = data_no_nans.index.to_list()
    y1 = data_no_nans[series_names[0]].to_numpy()
    y2 = data_no_nans[series_names[1]].to_numpy()
    y1 = standardize_series(y1, detrend=False, remove_mean=True)
    y2 = standardize_series(y2, detrend=True, remove_mean=False)

    wct_data = wct.DataForWCT(
        y1_values=y1,
        y2_values=y2,
        mother_wavelet=results_configs.XWT_MOTHER_DICT[results_configs.XWT_MOTHER],
        delta_t=results_configs.XWT_DT,
        delta_j=results_configs.XWT_DJ,
        initial_scale=results_configs.XWT_S0,
        levels=results_configs.LEVELS,
    )

    results_from_wct = wct.run_wct(
        wct_data,
        calculate_signficance=calculate_significance,
        significance_level=significance_level / 100,
    )

    # * Plot WCT coherence spectrum
    fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    wct.plot_wct(
        axs,
        wct_data,
        results_from_wct,
        include_significance=calculate_significance,
        include_cone_of_influence=True,
        include_phase_difference=True,
        **results_configs.XWT_PLOT_PROPS,
    )

    axs.set_ylim(axs.get_ylim()[::-1])

    # * Set y axis tick labels
    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(results_from_wct.period.min())),
        np.ceil(np.log2(results_from_wct.period.max())),
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


def generate_plot_without_regression(
    combined_dfs: pd.DataFrame,
    column_names: list[str],
    dwt_plot_selection: str,
    dwt_smooth_plot_order: str,
) -> None:
    """Generate DWT plot without regression analysis"""
    dwt_dict = create_dwt_dict(
        combined_dfs.dropna(),
        column_names,
        mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    )
    if dwt_plot_selection == ids.DECOMPOSE:
        fig = plot_dwt_decomposition_for(
            dwt_dict,
            combined_dfs.dropna().index,
            wavelet=results_configs.DWT_MOTHER_WAVELET,
            figsize=(15, 20),
            sharex=True,
        )
        st.pyplot(fig)
    elif dwt_plot_selection == ids.SMOOTH:
        ascending_order = bool(dwt_smooth_plot_order == ids.ASCEND)
        if len(dwt_dict) == 1:
            fig = plot_dwt_smoothing_for(
                dwt_dict,
                combined_dfs.dropna().index,
                ascending=ascending_order,
                figsize=(15, 20),
                sharex=True,
            )
            st.pyplot(fig)
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = plot_dwt_smoothing_for(
                    {list(dwt_dict.keys())[0]: dwt_dict[list(dwt_dict.keys())[0]]},
                    combined_dfs.dropna().index,
                    ascending=ascending_order,
                    figsize=(15, 20),
                    sharex=True,
                )
                st.pyplot(fig1)
            with col2:
                fig2 = plot_dwt_smoothing_for(
                    {list(dwt_dict.keys())[1]: dwt_dict[list(dwt_dict.keys())[1]]},
                    combined_dfs.dropna().index,
                    ascending=ascending_order,
                    figsize=(15, 20),
                    sharex=True,
                )
                st.pyplot(fig2)


def generate_plot(
    file_dict: dict[str, Path],
    transform_selection: str,
    selected_data: list[str],
    dwt_plot_selection: str = None,
    dwt_smooth_plot_order: str = None,
    calculate_significance: bool = False,
    significance_level: int = 95,
) -> None:
    """Generate plot for wavelet transform

    Args:
        file_dict (dict[str, Path]): Dictionary with files and file names
        transform_selection (str): User-selected transform
        selected_data (list[str]): Name of dataset to transform
        dwt_plot_selection (str, optional): DWT plot type selected. Defaults to None.
        dwt_smooth_plot_order (str, optional): DWT smoothing plot order. Defaults to None.
        calculate_significance (bool, optional): Whether to calculate statistical significance for WCT. Defaults to False.
        significance_level (int, optional): The significance level for WCT as an integer. Defaults to 95.
    """

    with st.spinner(f"Applying {transform_selection}🧮"):

        # Create list of column names based on file name to differentiate after merging the dataframes
        column_names = list(file_dict.keys())

        # Load each file into a dataframe
        dict_of_combined_dataframes = {
            column_name: load_file(file_path)
            for column_name, file_path in file_dict.items()
        }

        combined_dfs = combine_series(
            list(dict_of_combined_dataframes.values()),
            how="left",
            on=INDEX_COLUMN_NAME,
        )

        new_column_names = dict(zip(combined_dfs.columns.to_list(), file_dict.keys()))

        combined_dfs = combined_dfs.rename(columns=new_column_names)

        if transform_selection == ids.DWT:
            plot_dwt(
                combined_dfs, column_names, dwt_plot_selection, dwt_smooth_plot_order
            )

        elif transform_selection in (ids.CWT, ids.WCT):
            # First, try to plot normally using the loaded/combined data.
            try:
                if transform_selection == ids.CWT:
                    if len(column_names) == 1:
                        plot_cwt(
                            combined_dfs,
                            column_names,
                            calculate_significance=calculate_significance,
                            significance_level=significance_level,
                        )
                    else:
                        st.toast(
                            "Looks like you're looking for the _wavelet coherence transform_"
                        )
                        plot_wct(
                            combined_dfs,
                            column_names,
                            calculate_significance=calculate_significance,
                            significance_level=significance_level,
                        )

                else:  # ids.WCT
                    if len(column_names) == 2:
                        plot_wct(
                            combined_dfs,
                            column_names,
                            calculate_significance=calculate_significance,
                            significance_level=significance_level,
                        )
                    else:
                        st.warning("Please supply a second series.")

            except Exception as e:
                # If normal plotting fails (e.g. AR(1) upper-bound error), try the AR(1) adjustment
                logger.debug("Initial CWT/WCT plotting failed: %s", e)

                try:
                    st.toast(
                        "Converting to diff in log of CPI inflation to avoid AR(1) upper-bound error."
                    )

                    # Choose which series to keep when replacing CPI: prefer EXPECTATIONS, then SAVINGS_RATE,
                    # otherwise pick the first non-CPI series, or the first series available.
                    preferred = None
                    for pref in (ids.EXPECTATIONS, ids.SAVINGS_RATE):
                        disp = ids.DISPLAY_NAMES[pref]
                        if disp in column_names:
                            preferred = disp
                            break

                    if preferred is None:
                        preferred = next(
                            (
                                c
                                for c in column_names
                                if c != ids.DISPLAY_NAMES[ids.CPI]
                            ),
                            column_names[0] if column_names else None,
                        )

                    if preferred is None:
                        raise ValueError(
                            "No series available to keep for AR(1) adjustment."
                        )

                    combined_dfs, column_names = adjust_series_for_ar1_bound(
                        dict_of_combined_dataframes,
                        series_to_keep=preferred,
                        replacement_series=ids.DISPLAY_NAMES[ids.CPI],
                        diff_in_log=True,
                    )

                    # Retry plotting after adjustment
                    if transform_selection == ids.CWT:
                        if len(column_names) == 1:
                            plot_cwt(
                                combined_dfs,
                                column_names,
                                calculate_significance=calculate_significance,
                                significance_level=significance_level,
                            )
                        else:
                            st.toast(
                                "Looks like you're looking for the _wavelet coherence transform_"
                            )
                            plot_wct(
                                combined_dfs,
                                column_names,
                                calculate_significance=calculate_significance,
                                significance_level=significance_level,
                            )

                    else:  # ids.WCT
                        if len(column_names) == 2:
                            plot_wct(
                                combined_dfs,
                                column_names,
                                calculate_significance=calculate_significance,
                                significance_level=significance_level,
                            )
                        else:
                            st.warning("Please supply a second series.")

                except Exception as exc:
                    logger.debug(
                        "adjust_series_for_ar1_bound skipped or failed: %s", exc
                    )
                    st.error(
                        "Plotting failed after applying AR(1) adjustment. See logs for details."
                    )

        elif transform_selection == ids.CWT and len(column_names) == 1:
            plot_cwt(
                combined_dfs,
                column_names,
                calculate_significance=calculate_significance,
                significance_level=significance_level,
            )

        elif transform_selection == ids.CWT and len(column_names) == 2:
            st.toast("Looks like you're looking for the _wavelet coherence transform_")
            plot_wct(
                combined_dfs,
                column_names,
                calculate_significance=calculate_significance,
                significance_level=significance_level,
            )

        elif transform_selection == ids.WCT and len(column_names) == 2:
            plot_wct(
                combined_dfs,
                column_names,
                calculate_significance=calculate_significance,
                significance_level=significance_level,
            )

        elif transform_selection == ids.WCT and len(column_names) < 2:
            st.warning("Please supply a second series.")

    plt.legend("", frameon=False)
