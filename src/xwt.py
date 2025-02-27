"""Cross wavelet transformation"""

from __future__ import division
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import pycwt as wavelet

from constants import ids, results_configs

from src import retrieve_data
from src.utils import helpers, pycwt_patches, wavelet_helpers

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Constants
DT = 1 / 12  # Delta t
DJ = 1 / 8  # Delta j
S0 = 2 * DT  # Initial scale
MOTHER = "morlet"  # Morlet wavelet with :math:`\omega_0=6`.
MOTHER_DICT = {
    "morlet": wavelet.Morlet(6),
    "paul": wavelet.Paul(),
    "DOG": wavelet.DOG(),
    "mexicanhat": wavelet.MexicanHat(),
}
LEVELS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]


XWT_PLOT_PROPS = {
    "cmap": "jet",
    "sig_colors": "k",
    "sig_linewidths": 2,
    "coi_color": "k",
    "coi_alpha": 0.3,
    "coi_hatch": "--",
    "phase_diff_units": "width",
    "phase_diff_angles": "uv",
    "phase_diff_pivot": "mid",
    "phase_diff_linewidth": 0.5,
    "phase_diff_edgecolor": "k",
    "phase_diff_alpha": 0.7,
}


@dataclass
class DataForXWT:
    """Holds data for XWT"""

    t_values: npt.NDArray = field(init=False)
    y1_values: npt.NDArray
    y2_values: npt.NDArray
    mother_wavelet: Type
    delta_t: float
    delta_j: float
    initial_scale: float
    levels: List[float]

    def __post_init__(self):
        self.t_values = np.linspace(1, self.y1_values.size + 1, self.y1_values.size)


@dataclass
class ResultsFromXWT:
    """Holds results from Cross-Wavelet Transform"""

    power: npt.NDArray
    period: npt.NDArray
    significance_levels: npt.NDArray
    coi: npt.NDArray
    phase_diff_u: npt.NDArray
    phase_diff_v: npt.NDArray


def run_xwt(
    cross_wavelet_transform: Type[DataForXWT],
    # ignore_strong_trends: bool = False,
    normalize: bool = True,
) -> Type[ResultsFromXWT]:
    """Conduct Cross-Wavelet Transformation on two series.\n
    Returns cross-wavelet power, period, significance levels, cone of influence,
    and phase"""

    # * Perform cross wavelet transform
    xwt_result, coi, freqs, signif = wavelet.xwt(
        y1=cross_wavelet_transform.y1_values,
        y2=cross_wavelet_transform.y2_values,
        dt=cross_wavelet_transform.delta_t,
        dj=cross_wavelet_transform.delta_j,
        s0=cross_wavelet_transform.initial_scale,
        wavelet=cross_wavelet_transform.mother_wavelet,
        # ignore_strong_trends=ignore_strong_trends,
    )

    if normalize:
        # * Normalize results
        signal_size = cross_wavelet_transform.y1_values.size
        period, power, sig95, coi_plot = wavelet_helpers.normalize_xwt_results(
            signal_size,
            xwt_result,
            coi,
            np.log2(cross_wavelet_transform.levels[2]),
            freqs,
            signif,
        )
    else:
        period = 1 / freqs
        power = xwt_result
        sig95 = np.ones([1, signal_size]) * signif[:, None]
        sig95 = power / sig95  ## Want where power / sig95 > 1
        coi_plot = coi

    # * Caclulate wavelet coherence
    _, phase, _, _, _ = wavelet.wct(
        cross_wavelet_transform.y1_values,
        cross_wavelet_transform.y2_values,
        cross_wavelet_transform.delta_t,
        delta_j=cross_wavelet_transform.delta_t,
        s0=-1,
        J=-1,
        sig=False,  #! To save time
        # significance_level=0.8646,
        wavelet=cross_wavelet_transform.mother_wavelet,
        normalize=True,
        cache=True,
    )

    # * Calculate phase difference
    phase_diff_u, phase_diff_v = calculate_phase_difference(phase)

    return ResultsFromXWT(power, period, sig95, coi_plot, phase_diff_u, phase_diff_v)


def calculate_phase_difference(
    xwt_phase: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculates the phase between both time series.\n
    Returns u and v phase difference vectors \n
    Description:\n
    These phase arrows follow the convention of Torrence and Webster (1999).
    In-phase points north, and anti-phase point south. If y1 leads y2, the arrows
    point east, and if y2 leads y1, the arrows point west."""

    angle = 0.5 * np.pi - xwt_phase
    xwt_u, xwt_v = np.cos(angle), np.sin(angle)
    logger.debug("Comparing length of phase arrays: %s, %s", len(xwt_u), len(xwt_v))
    return xwt_u, xwt_v


def plot_xwt(
    xwt_ax: plt.Axes,
    xwt_data: Type[DataForXWT],
    xwt_results: Type[ResultsFromXWT],
    include_significance: bool = True,
    include_cone_of_influence: bool = True,
    include_phase_difference: bool = True,
    **kwargs,
) -> None:
    """Plot Cross-Wavelet Power Spectrum of two series\n
    **kwargs**\n
    For power sprectrum: `cmap="jet"`\n
    For significance levels: `sig_colors="k"`, `sig_linewidths=2`\n
    For cone of influence: `coi_color="k"`, `coi_alpha=0.3`, `coi_hatch="--"`\n
    For phase difference: `phase_diff_units="width"`, `phase_diff_angles="uv"`,
    `phase_diff_pivot="mid"`,`phase_diff_linewidth=0.5`, `phase_diff_edgecolor="k"`,
    `phase_diff_alpha=0.7`"""
    extent = [
        min(xwt_data.t_values),
        max(xwt_data.t_values),
        min(xwt_results.coi),
        max(xwt_results.period),
    ]

    # * Normalized XWT power spectrum
    xwt_ax.contourf(
        xwt_data.t_values,
        np.log2(xwt_results.period),
        np.log2(xwt_results.power),
        np.log2(xwt_data.levels),
        extend="both",
        cmap=kwargs["cmap"],
        extent=extent,
    )

    # * Add XWT power spectrum features
    if include_significance:
        ## Plot significance level contours
        wavelet_helpers.plot_signficance_levels(
            xwt_ax,
            xwt_results.significance_levels,
            xwt_data.t_values,
            xwt_results.period,
            **kwargs,
        )
    if include_cone_of_influence:
        ## Plot cone of influence
        wavelet_helpers.plot_cone_of_influence(
            xwt_ax,
            xwt_results.coi,
            xwt_data.t_values,
            xwt_data.levels,
            xwt_results.period,
            xwt_data.delta_t,
            tranform_type="xwt",
            **kwargs,
        )
    if include_phase_difference:
        ## Plot phase difference indicator vector arrows
        plot_phase_difference(
            xwt_ax,
            xwt_data.t_values,
            xwt_results.period,
            xwt_results.phase_diff_u,
            xwt_results.phase_diff_v,
            **kwargs,
        )


def plot_phase_difference(
    xwt_ax: plt.Axes,
    t_values: npt.NDArray,
    period: npt.NDArray,
    phase_diff_u: npt.NDArray,
    phase_diff_v: npt.NDArray,
    **kwargs,
) -> None:
    """Plot shaded area for cone of influence, where edge effects may occur\n
    **kwargs**\n
    `phase_diff_units=`: "width"\n
    `phase_diff_angles=`: "uv"\n
    `phase_diff_pivot=`: "mid"\n
    `phase_diff_linewidth=`: 0.5\n
    `phase_diff_edgecolor=`: "k"\n
    `phase_diff_alpha=`: 0.7"""
    xwt_ax.quiver(
        t_values[::12],
        np.log2(period[::8]),
        phase_diff_u[::12, ::12],
        phase_diff_v[::12, ::12],
        units=kwargs["phase_diff_units"],
        angles=kwargs["phase_diff_angles"],
        pivot=kwargs["phase_diff_pivot"],
        linewidth=kwargs["phase_diff_linewidth"],
        edgecolor=kwargs["phase_diff_edgecolor"],
        alpha=kwargs["phase_diff_alpha"],
    )


def main() -> None:
    """Run script"""

    series_comparisons = [(ids.DIFF_LOG_CPI, ids.EXPECTATIONS)]

    series_titles = {
        ids.INFLATION: "CPI inflation",
        ids.EXPECTATIONS: "Inflation expectations",
        ids.SAVINGS_RATE: "Savings rate",
        ids.NONDURABLES_CHG: "Nondurables consumption (% chg)",
        ids.DURABLES_CHG: "Durables consumption (% chg)",
        ids.SAVINGS_CHG: "Savings (% chg)",
        ids.DIFF_LOG_CPI: "CPI inflation (diff in log)",
        ids.DIFF_LOG_EXPECTATIONS: "Inflation expectations (diff in log)",
        ids.DIFF_LOG_NONDURABLES: "Nondurables consumption (diff in log)",
        ids.DIFF_LOG_DURABLES: "Durables consumption (diff in log)",
        ids.DIFF_LOG_SAVINGS: "Savings (diff in log)",
        ids.DIFF_LOG_REAL_NONDURABLES: "Real nondurables consumption (diff in log)",
        ids.DIFF_LOG_REAL_DURABLES: "Real durables consumption (diff in log)",
        ids.DIFF_LOG_REAL_SAVINGS: "Real savings (diff in log)",
    }
    ## Pre-process data
    # US data
    # * CPI
    raw_data = retrieve_data.get_fed_data(
        ids.US_CPI,
    )
    cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
    cpi.rename(columns={"value": ids.CPI}, inplace=True)

    # * Inflation rate
    raw_data = retrieve_data.get_fed_data(
        ids.US_CPI,
        units="pc1",
        freq="m",
    )
    measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    measured_inf.rename(columns={"value": ids.INFLATION}, inplace=True)

    # * Inflation expectations
    raw_data = retrieve_data.get_fed_data(
        ids.US_INF_EXPECTATIONS,
    )
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    dataframes = [cpi, measured_inf, inf_exp]
    us_data = helpers.combine_series(dataframes, on=[ids.DATE], how="left")
    us_data = us_data[us_data["date"] <= pd.to_datetime(results_configs.END_DATE)]

    # * Remove rows without data for all measures
    us_data.dropna(inplace=True)

    # * Add real value columns
    logger.info(
        "Using constant dollars from %s, CPI: %s",
        results_configs.CONSTANT_DOLLAR_DATE,
        us_data[
            us_data[ids.DATE] == pd.Timestamp(results_configs.CONSTANT_DOLLAR_DATE)
        ][ids.CPI].iat[0],
    )

    us_data = helpers.calculate_diff_in_log(
        data=us_data,
        columns=[ids.CPI, ids.EXPECTATIONS],
    )

    measured_inf = helpers.calculate_diff_in_log(
        data=measured_inf,
        columns=[ids.INFLATION],
    )
    logger.debug("df shape %s", measured_inf.shape)

    for (
        comp
    ) in series_comparisons:  # series_comparisons[1:4] + series_comparisons[14:]:

        logger.debug(comp)

        # * Pre-process data: Standardize and detrend

        ## Create a copy of dataframe to drop NaNs
        data = us_data.copy()
        logger.debug("NaNs: %s", data.isna().sum())
        data = data.dropna(subset=[comp[0], comp[1]])
        logger.debug(data.shape)

        y1 = data[comp[0]].to_numpy()
        y2 = data[comp[1]].to_numpy()
        y1 = wavelet_helpers.standardize_series(y1, detrend=False, remove_mean=True)
        y2 = wavelet_helpers.standardize_series(y2, detrend=True, remove_mean=False)

        mother_xwt = MOTHER_DICT[MOTHER]

        xwt_data = DataForXWT(
            y1_values=y1,
            y2_values=y2,
            mother_wavelet=mother_xwt,
            delta_t=DT,
            delta_j=DJ,
            initial_scale=S0,
            levels=LEVELS,
        )

        results_from_xwt = run_xwt(xwt_data)

        # * Plot XWT power spectrum
        _, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

        plot_xwt(
            axs,
            xwt_data,
            results_from_xwt,
            include_significance=True,
            include_cone_of_influence=True,
            include_phase_difference=True,
            **XWT_PLOT_PROPS,
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
        dates = us_data["date"].to_list()
        x_dates = [dates[0]] + [dates[i + 99] for i in range(0, 500, 100)]
        x_ticks = [str(date.year) for date in x_dates]
        x_tick_positions = [i for i in range(0, 600, 100)]
        logger.debug("dates %s", x_ticks)
        logger.debug("dates %s", x_tick_positions)
        logger.debug("dates len %s", len(x_ticks))
        axs.set_xticks(x_tick_positions)
        axs.set_xticklabels(x_ticks, size=12)

        axs.set_title(f"{series_titles[comp[0]]} X {series_titles[comp[1]]}", size=16)
        axs.set_ylabel("Period (years)", size=14)

        plt.show()


if __name__ == "__main__":
    main()
