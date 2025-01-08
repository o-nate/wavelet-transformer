"""MODWT"""

import logging
import sys
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
import pdb
import pywt
from scipy.ndimage import convolve1d
import statsmodels.api as sm
import statsmodels.iolib.summary2

from constants import ids, results_configs
from src import dwt, retrieve_data
from src.utils import helpers

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


# ! Define mother wavelet
MOTHER = pywt.Wavelet("db4")

SERIES_COMPARISONS = [
    (ids.DIFF_LOG_CPI, ids.EXPECTATIONS),
    (ids.EXPECTATIONS, ids.NONDURABLES_CHG),
    (ids.EXPECTATIONS, ids.DURABLES_CHG),
    (ids.EXPECTATIONS, ids.SAVINGS_RATE),
    (ids.DIFF_LOG_CPI, ids.DIFF_LOG_EXPECTATIONS),
    (ids.DIFF_LOG_CPI, ids.DIFF_LOG_REAL_NONDURABLES),
    (ids.DIFF_LOG_CPI, ids.DIFF_LOG_REAL_DURABLES),
    (ids.DIFF_LOG_CPI, ids.DIFF_LOG_REAL_SAVINGS),
    (ids.EXPECTATIONS, ids.DIFF_LOG_NONDURABLES),
    (ids.EXPECTATIONS, ids.DIFF_LOG_DURABLES),
    (ids.EXPECTATIONS, ids.DIFF_LOG_SAVINGS),
    (ids.EXPECTATIONS, ids.DIFF_LOG_REAL_NONDURABLES),
    (ids.EXPECTATIONS, ids.DIFF_LOG_REAL_DURABLES),
    (ids.EXPECTATIONS, ids.DIFF_LOG_REAL_SAVINGS),
    (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_NONDURABLES),
    (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_DURABLES),
    (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_SAVINGS),
    (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_NONDURABLES),
    (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_DURABLES),
    (ids.DIFF_LOG_EXPECTATIONS, ids.DIFF_LOG_REAL_SAVINGS),
]


def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def circular_convolve_mra(h_j_o, w_j):
    """calculate the mra D_j"""
    return convolve1d(w_j, np.flip(h_j_o), mode="wrap", origin=(len(h_j_o) - 1) // 2)


def circular_convolve_d(h_t, v_j_1, j):
    """
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    """
    N = len(v_j_1)
    w_j = np.zeros(N)
    ker = np.zeros(len(h_t) * 2 ** (j - 1))

    # make kernel
    for i, h in enumerate(h_t):
        ker[i * 2 ** (j - 1)] = h

    w_j = convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)
    return w_j


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    """
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    """
    n = len(v_j)

    h_ker = np.zeros(len(h_t) * 2 ** (j - 1))
    g_ker = np.zeros(len(g_t) * 2 ** (j - 1))

    for i, (h, g) in enumerate(zip(h_t, g_t)):
        h_ker[i * 2 ** (j - 1)] = h
        g_ker[i * 2 ** (j - 1)] = g

    v_j_1 = np.zeros(n)

    v_j_1 = convolve1d(w_j, np.flip(h_ker), mode="wrap", origin=(len(h_ker) - 1) // 2)
    v_j_1 += convolve1d(v_j, np.flip(g_ker), mode="wrap", origin=(len(g_ker) - 1) // 2)
    return v_j_1


def modwt(x, filters, level):
    """
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    """
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


def imodwt(w, filters):
    """inverse modwt"""
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def modwtmra(w, filters):
    """Multiresolution analysis based on MODWT"""
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.0))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)


def time_scale_regression(
    input_coeffs: npt.NDArray,
    output_coeffs: npt.NDArray,
    levels: int,
    add_constant: bool = True,
) -> Type[statsmodels.iolib.summary2.Summary]:
    """Regresses output on  input for each component vector S_J, D_J, ..., D_1,
    where J=levels"""
    regressions_dict = {}
    for j in range(levels + 1):
        if j == 0:
            vector_name = f"S_{levels}"
        else:
            vector_name = f"D_{levels - j + 1}"
        print(f"Regressing on component vector {vector_name}")
        # * Reconstruct each component vector indiviually
        logger.debug("lengths, %s, %s", len(input_coeffs[j]), len(output_coeffs[j]))
        input_j = input_coeffs[j]
        output_j = output_coeffs[j]
        logger.debug("lengths output, %s, %s", len(input_j), len(output_j))

        # * Run regression
        if add_constant:
            input_j = sm.add_constant(input_j)
        model = sm.OLS(output_j, input_j)
        regressions_dict[vector_name] = model.fit()
        # print(regressions_dict[vector_name].summary())
    results = statsmodels.iolib.summary2.summary_col(
        list(regressions_dict.values()),
        stars=True,
        model_names=list(regressions_dict),
    )
    return results


def smooth_signal(
    modwt_coeffs: npt.NDArray, mother_wavelet: str, levels: int
) -> dict[int, dict[str, npt.NDArray]]:
    """Generate smoothed signals based off wavelet coefficients for each pre-defined level"""
    ## Initialize dict for reconstructed signals
    signals_dict = {}

    ## Loop through levels and remove detail level component(s)
    # ! Note: signal_dict[l] provides the signal with levels <= l removed
    for l in range(levels, 0, -1):
        print(f"s_{l} stored with key {l}")
        smooth_coeffs = modwt_coeffs.copy()
        signals_dict[l] = {}
        ## Set remaining detail coefficients to zero
        for coeff in range(l):
            smooth_coeffs[coeff] = np.zeros_like(smooth_coeffs[coeff])
        signals_dict[l]["coeffs"] = smooth_coeffs
        # Reconstruct the signal using only the approximation coefficients
        signals_dict[l]["signal"] = imodwt(smooth_coeffs, mother_wavelet)
    return signals_dict


def plot_smoothing(
    smooth_signals: dict,
    original_t: npt.NDArray,
    original_y: npt.NDArray,
    ascending: bool = False,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, str]:
    """Graph series of smoothed signals with original signal"""
    fig, axs = plt.subplots(len(smooth_signals), 1, **kwargs)
    # * Loop through levels and add detail level components
    if ascending:
        order = reversed(list(smooth_signals.items()))
    else:
        order = list(smooth_signals.items())
    for i, (level, signal) in enumerate(order, 1):
        logger.debug("length of smooth_signal %s", len(smooth_signals))
        logger.debug("i, level: %s, %s", i, level)
        logger.debug("signal[signal]: %s", signal["signal"].shape)
        smooth_level = len(smooth_signals) - level
        ## Subplot for each smooth signal
        # plt.subplot(len(smooth_signals), 1, i)
        axs[i - 1].plot(original_t, original_y, label="Original")
        axs[i - 1].plot(original_t, signal["signal"])
        axs[i - 1].set_title(rf"Approximation: $S_{{j-{smooth_level}}}$", size=15)
        if i - 1 == 0:
            axs[i - 1].legend(loc="upper right")
        else:
            axs[i - 1].legend("", frameon=False)
    return fig


if __name__ == "__main__":
    # * CPI
    raw_data = retrieve_data.get_fed_data(ids.US_CPI)
    cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
    cpi.rename(columns={"value": ids.CPI}, inplace=True)

    # * Inflation rate
    raw_data = retrieve_data.get_fed_data(ids.US_CPI, units="pc1", freq="m")
    measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    measured_inf.rename(columns={"value": ids.INFLATION}, inplace=True)

    # * Inflation expectations
    raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

    # * Non-durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_NONDURABLES_CONSUMPTION)
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": ids.NONDURABLES}, inplace=True)

    # * Durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_DURABLES_CONSUMPTION)
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": ids.DURABLES}, inplace=True)

    # * Non-durables consumption change, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_NONDURABLES_CONSUMPTION, units="pc1", freq="m"
    )
    nondur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump_chg.rename(columns={"value": ids.NONDURABLES_CHG}, inplace=True)

    # * Durables consumption change, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_DURABLES_CONSUMPTION, units="pc1", freq="m"
    )
    dur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump_chg.rename(columns={"value": ids.DURABLES_CHG}, inplace=True)

    # * Personal savings
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS)
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": ids.SAVINGS}, inplace=True)

    # * Personal savings change
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS, units="pc1", freq="m")
    save_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_chg.rename(columns={"value": ids.SAVINGS_CHG}, inplace=True)

    # * Personal savings rate
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS_RATE)
    save_rate, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_rate.rename(columns={"value": ids.SAVINGS_RATE}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    dataframes = [
        cpi,
        measured_inf,
        inf_exp,
        nondur_consump,
        nondur_consump_chg,
        dur_consump,
        dur_consump_chg,
        save,
        save_chg,
        save_rate,
    ]
    us_data = helpers.combine_series(dataframes, on=[ids.DATE], how="left")

    # # * Remove rows without data for all measures
    us_data.dropna(inplace=True)

    # * Add real value columns
    logger.info(
        "Using constant dollars from %s, CPI: %s",
        results_configs.CONSTANT_DOLLAR_DATE,
        us_data[
            us_data[ids.DATE] == pd.Timestamp(results_configs.CONSTANT_DOLLAR_DATE)
        ][ids.CPI].iat[0],
    )
    us_data = helpers.add_real_value_columns(
        data=us_data,
        nominal_columns=[ids.NONDURABLES, ids.DURABLES, ids.SAVINGS],
        cpi_column=ids.CPI,
        constant_date=results_configs.CONSTANT_DOLLAR_DATE,
    )
    us_data = helpers.calculate_diff_in_log(
        data=us_data,
        columns=[
            ids.CPI,
            ids.EXPECTATIONS,
            ids.NONDURABLES,
            ids.DURABLES,
            ids.SAVINGS,
            ids.REAL_NONDURABLES,
            ids.REAL_DURABLES,
            ids.REAL_SAVINGS,
        ],
    )

    print(
        f"""Inflation expectations observations: {len(inf_exp)}, \n
        Non-durables consumption observations: {len(nondur_consump)}, \n
        Durables consumption observations: {len(dur_consump)}, \n
        Savings observation {len(save)}.\nNew dataframe lengths: {len(us_data)}"""
    )
    print(us_data.head(), "\n", us_data.tail())
    print("--------------------------Descriptive stats--------------------------\n")
    print(us_data.describe())

    # sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

    # * Wavelet decomposition
    ## Create dwt dict

    # plt.plot(y1)

    FILTER = "db4"
    LEVELS = 6

    # w = modwt(y1, "db4", LEVELS)
    # fig, ax = plt.subplots(LEVELS + 1, 1, sharex=True)
    # for k in range(LEVELS + 1):
    #     ax[k].plot(w[k])

    # plt.show()
    for comp in (
        SERIES_COMPARISONS[14:17] + SERIES_COMPARISONS[1:4]
    ):  # + SERIES_COMPARISONS[17:]

        # ## Shorter series
        # data = us_data[
        #     (us_data[ids.DATE] >= pd.to_datetime("1979-05-31"))
        #     & (us_data[ids.DATE] <= pd.to_datetime("2024-05-31"))
        # ].dropna()
        # t = data[ids.DATE].to_numpy()
        # logger.info("Date range: %s to %s", t.min(), t.max())

        # signal_1 = data[comp[0]].to_numpy()
        # signal_2 = data[comp[1]].to_numpy()
        # coeffs_1 = modwt(signal_1, FILTER, LEVELS)

        # coeffs_2 = modwt(signal_2, FILTER, LEVELS)
        # wtmra1 = modwtmra(coeffs_1, FILTER)
        # wtmra2 = modwtmra(coeffs_2, FILTER)

        # smooth_1 = smooth_signal(coeffs_1, FILTER, LEVELS)
        # logger.debug("smooth dict signal: %s", len(smooth_1))
        # smooth_2 = smooth_signal(coeffs_2, FILTER, LEVELS)

        # _ = plot_smoothing(
        #     smooth_1,
        #     t,
        #     signal_1,
        #     figsize=(10, 10),
        #     sharex=True,
        # )
        # _ = plot_smoothing(
        #     smooth_2,
        #     t,
        #     signal_2,
        #     figsize=(10, 10),
        #     sharex=True,
        # )

        # time_scale_results = time_scale_regression(
        #     input_coeffs=wtmra1,
        #     output_coeffs=wtmra2,
        #     levels=LEVELS,
        # )
        # print(f"\nRegressing {comp[1]} on {comp[0]}")
        # print(time_scale_results.as_text())

        # Longer series
        data = us_data[(us_data[ids.DATE] <= results_configs.END_DATE)].dropna()
        t = data[ids.DATE].to_numpy()
        logger.info("Date range: %s to %s", t.min(), t.max())

        signal_1_longer = data[comp[0]].to_numpy()
        signal_2_longer = data[comp[1]].to_numpy()
        coeffs_1_longer = modwt(signal_1_longer, FILTER, LEVELS)
        coeffs_2_longer = modwt(signal_2_longer, FILTER, LEVELS)
        wtmra1_longer = modwtmra(coeffs_1_longer, FILTER)
        wtmra2_longer = modwtmra(coeffs_2_longer, FILTER)

        time_scale_results = time_scale_regression(
            input_coeffs=wtmra1_longer,
            output_coeffs=wtmra2_longer,
            levels=LEVELS,
        )
        print(f"\nRegressing {comp[1]} on {comp[0]}")
        print(time_scale_results.as_text())

        # fig, ax = plt.subplots(LEVELS + 1, 1, sharex=True)
        # for k in range(LEVELS + 1):
        #     ax[k].plot(wtmra1[k])
        #     ax[k].plot(wtmra2[k])
        #     ax[k].plot(wtmra1_longer[k])
        #     ax[k].plot(wtmra2_longer[k])

        smooth_1_longer = smooth_signal(coeffs_1_longer, FILTER, LEVELS)
        smooth_2_longer = smooth_signal(coeffs_2_longer, FILTER, LEVELS)

        # _ = plot_smoothing(
        #     smooth_1_longer,
        #     t,
        #     signal_1_longer,
        #     figsize=(10, 10),
        #     sharex=True,
        # )
        # _ = plot_smoothing(
        #     smooth_2_longer,
        #     t,
        #     signal_2_longer,
        #     figsize=(10, 10),
        #     sharex=True,
        # )

    plt.show()
