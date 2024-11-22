"""
Continuous wavelet transform of signals
based off: https://pycwt.reaDThedocs.io/en/latest/tutorial.html
"""

from __future__ import division
import logging
from pathlib import Path
import sys
from dataclasses import dataclass, field

from typing import List, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import pycwt as wavelet

from constants import ids, results_configs

from src import retrieve_data
from src.utils import helpers, wavelet_helpers
from src.utils.logging_helpers import define_other_module_log_level
from src.utils.wavelet_helpers import (
    plot_cone_of_influence,
    plot_signficance_levels,
    standardize_series,
)

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Define title and labels for plots
UNITS = "%"

NORMALIZE = True  # Define normalization
DT = 1 / 12  # In years
S0 = 2 * DT  # Starting scale
DJ = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / DJ  # Seven powers of two with DJ sub-octaves
MOTHER = wavelet.Morlet(f0=6)
LEVELS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]  # Period scale is logarithmic


@dataclass
class DataForCWT:
    """Holds data for continuous wavelet transform"""

    t_values: npt.NDArray
    y_values: npt.NDArray
    mother_wavelet: Type
    delta_t: float
    delta_j: float
    initial_scale: float
    levels: List[float]
    time_range: npt.NDArray = field(init=False)

    def __post_init__(self):
        self.time_range = self.time_range(self)

    def time_range(self) -> npt.NDArray:
        """Takes first date and creates array with date based on defined dt"""
        # Define starting time and time step
        t0 = min(self.t_values)
        logger.debug("t0 type %s", type(t0))
        t0 = t0.astype("datetime64[Y]").astype(int) + 1970
        num_observations = self.t_values.size
        self.time_range = np.arange(1, num_observations + 1) * self.delta_t + t0
        return np.arange(1, num_observations + 1) * self.delta_t + t0


@dataclass
class ResultsFromCWT:
    """Holds results from continuous wavelet transform"""

    power: npt.NDArray
    period: npt.NDArray
    significance_levels: npt.NDArray
    coi: npt.NDArray


# * Functions
def run_cwt(
    cwt_data: Type[DataForCWT],
    normalize: bool = True,
    standardize: bool = False,
    **kwargs,
) -> Type[ResultsFromCWT]:
    """Conducts Continuous Wavelet Transform\n
    Returns power spectrum, period, cone of influence, and significance levels (95%)"""
    # p = np.polyfit(t - t0, dat, 1)
    # dat_notrend = dat - np.polyval(p, t - t0)
    std = cwt_data.y_values.std()  #! dat_notrend.std()  # Standard deviation

    if normalize:
        dat_norm = cwt_data.y_values / std  #! dat_notrend / std  # Normalized dataset
    if standardize:
        dat_norm = standardize_series(cwt_data.y_values, **kwargs)
    else:
        dat_norm = cwt_data.y_values

    alpha, _, _ = wavelet.ar1(cwt_data.y_values)  # Lag-1 autocorrelation for red noise

    # * Conduct transformations
    # Wavelet transform
    wave, scales, freqs, cwt_coi, _, _ = wavelet.cwt(
        dat_norm, DT, DJ, S0, J, cwt_data.mother_wavelet
    )
    # Normalized wavelet power spectrum
    cwt_power = (np.abs(wave)) ** 2
    # Normalized Fourier equivalent periods
    cwt_period = 1 / freqs

    # * Statistical significance
    # where the ratio ``cwt_power / sig95 > 1``.
    num_observations = len(cwt_data.t_values)
    signif, _ = wavelet.significance(
        1.0,
        DT,
        scales,
        0,
        alpha,
        significance_level=0.95,
        wavelet=cwt_data.mother_wavelet,
    )
    cwt_sig95 = np.ones([1, num_observations]) * signif[:, None]
    cwt_sig95 = cwt_power / cwt_sig95

    return ResultsFromCWT(cwt_power, cwt_period, cwt_sig95, cwt_coi)


def plot_cwt(
    cwt_ax: plt.Axes,
    cwt_data: Type[DataForCWT],
    cwt_results: Type[ResultsFromCWT],
    include_significance: bool = True,
    include_cone_of_influence: bool = True,
    **kwargs,
) -> None:
    """Plot Power Spectrum for Continuous Wavelet Transform"""
    _ = cwt_ax.contourf(
        cwt_data.time_range,
        np.log2(cwt_results.period),
        np.log2(cwt_results.power),
        np.log2(cwt_data.levels),
        extend="both",
        cmap=kwargs["cmap"],
    )

    if include_significance:
        plot_signficance_levels(
            cwt_ax,
            cwt_results.significance_levels,
            cwt_data.time_range,
            cwt_results.period,
            **kwargs,
        )

    if include_cone_of_influence:
        plot_cone_of_influence(
            cwt_ax,
            cwt_results.coi,
            cwt_data.time_range,
            cwt_data.levels,
            cwt_results.period,
            cwt_data.delta_t,
            tranform_type="cwt",
            **kwargs,
        )

    # * Invert y axis
    cwt_ax.set_ylim(cwt_ax.get_ylim()[::-1])

    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(cwt_results.period.min())),
        np.ceil(np.log2(cwt_results.period.max())),
    )
    cwt_ax.set_yticks(np.log2(y_ticks))
    cwt_ax.set_yticklabels(y_ticks, size=15)


def main() -> None:
    """Run script"""
    cwt_measures = {
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
        # # # # ids.NONDURABLES,
        # # # # ids.DURABLES,
        # # # ids.SAVINGS,
        # # # ids.REAL_NONDURABLES,
        # # # ids.REAL_DURABLES,
        # # # ids.REAL_SAVINGS,
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

    # * Non-durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_NONDURABLES_CONSUMPTION,
    )
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": ids.NONDURABLES}, inplace=True)

    # * Durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_DURABLES_CONSUMPTION,
    )
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": ids.DURABLES}, inplace=True)

    # * Non-durables consumption change, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_NONDURABLES_CONSUMPTION,
        units="pc1",
        freq="m",
    )
    nondur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump_chg.rename(columns={"value": ids.NONDURABLES_CHG}, inplace=True)

    # * Durables consumption change, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_DURABLES_CONSUMPTION,
        units="pc1",
        freq="m",
    )
    dur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump_chg.rename(columns={"value": ids.DURABLES_CHG}, inplace=True)

    # * Personal savings
    raw_data = retrieve_data.get_fed_data(
        ids.US_SAVINGS,
    )
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": ids.SAVINGS}, inplace=True)

    # * Personal savings change
    raw_data = retrieve_data.get_fed_data(
        ids.US_SAVINGS,
        units="pc1",
        freq="m",
    )
    save_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_chg.rename(columns={"value": ids.SAVINGS_CHG}, inplace=True)

    # * Personal savings rate
    raw_data = retrieve_data.get_fed_data(
        ids.US_SAVINGS_RATE,
    )
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

    measured_inf = helpers.calculate_diff_in_log(
        data=measured_inf,
        columns=[ids.INFLATION],
    )
    logger.debug("df shape %s", measured_inf.shape)

    for measure, label in cwt_measures.items():
        logger.debug(measure)

        # * If INFLATION, use full series
        if measure == ids.INFLATION:
            data = measured_inf.copy()

        else:
            data = us_data.copy()

        # * Pre-process data: Standardize and detrend
        logger.debug("nans: %s", data[f"{measure}"].isna().sum())
        data = data.dropna()
        y1 = data[f"{measure}"].to_numpy()
        y1 = wavelet_helpers.standardize_series(
            y1
        )  # , detrend=False, remove_mean=True)
        logger.debug(
            "length of y1: %s. length of date: %s",
            len(y1),
            len(data["date"].to_numpy()),
        )

        data_for_cwt = DataForCWT(
            data["date"].to_numpy(),
            y1,
            MOTHER,
            DT,
            DJ,
            S0,
            LEVELS,
        )

        results_from_cwt = run_cwt(data_for_cwt, normalize=True)

        # * Plot results
        plt.close("all")
        # plt.ioff()
        figprops = {"figsize": (20, 10), "dpi": 72}
        _, ax = plt.subplots(1, 1, **figprops)

        # * Add plot features
        cwt_plot_props = {
            "cmap": "jet",
            "sig_colors": "k",
            "sig_linewidths": 2,
            "coi_color": "k",
            "coi_alpha": 0.3,
            "coi_hatch": "--",
        }
        plot_cwt(ax, data_for_cwt, results_from_cwt, **cwt_plot_props)

        # * Set labels/title
        ax.set_xlabel("", size=20)
        ax.set_ylabel("Period (years)", size=20)
        ax.set_title(label, size=20)

        # # ! Export plot
        # parent_dir = Path(__file__).parents[1]
        # export_file = parent_dir / "results" / f"cwt_module_{measure}.png"
        # plt.savefig(export_file, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    main()
