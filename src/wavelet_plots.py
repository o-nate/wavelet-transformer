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
from src.utils.plot_helpers import plot_dwt_decomposition_for, plot_dwt_smoothing_for
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

    results_from_cwt = cwt.run_cwt(data_for_cwt, normalize=True)

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
    dwt_dict = create_dwt_dict(
        data.dropna(),
        series_names,
        mother_wavelet=results_configs.DWT_MOTHER_WAVELET,
    )
    dwt_results_dict = create_dwt_results_dict(dwt_dict, series_names)

    st.write(f"####Showing DWT of: {', '.join(dwt_dict.keys())}")

    t = data.dropna().index
    if plot_selection == ids.DECOMPOSE:
        fig = plot_dwt_decomposition_for(
            dwt_results_dict,
            t,
            wavelet=results_configs.DWT_MOTHER_WAVELET,
            figsize=(15, 20),
            sharex=True,
        )
        plt.legend("", frameon=False)

        st.plotly_chart(fig)

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
            plt.legend("", frameon=False)

            st.plotly_chart(fig)

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

                st.plotly_chart(fig1, use_container_width=True)

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

                st.plotly_chart(fig2, use_container_width=True)


def plot_xwt(data: pd.DataFrame, series_names: list[str]) -> Figure:

    # * Pre-process data: Standardize and detrend
    data_no_nans = data.dropna()
    t = data_no_nans.index.to_list()
    y1 = data_no_nans[series_names[0]].to_numpy()
    y2 = data_no_nans[series_names[1]].to_numpy()
    y1 = standardize_series(y1, detrend=False, remove_mean=True)
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

    results_from_xwt = xwt.run_xwt(xwt_data, ignore_strong_trends=True)

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
    x_dates = [t[0]] + [t[i + 99] for i in range(0, 500, 100)]
    x_ticks = [str(date.year) for date in x_dates]
    x_tick_positions = [i for i in range(0, 600, 100)]
    logger.debug("dates %s", x_ticks)
    logger.debug("dates %s", x_tick_positions)
    logger.debug("dates len %s", len(x_ticks))
    axs.set_xticks(x_tick_positions)
    axs.set_xticklabels(x_ticks, size=12)

    axs.set_title(f"{series_names[0]} X {series_names[1]}", size=16)
    axs.set_ylabel("Period (years)", size=14)

    st.pyplot(fig)
