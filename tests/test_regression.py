"""Test regression module"""

import logging
import sys

import pywt
import matplotlib.figure

from src.utils.logging_helpers import define_other_module_log_level
from src import retrieve_data, dwt, regression

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ! Define mother wavelet
MOTHER = "db4"
mother_wavelet = pywt.Wavelet(MOTHER)

# * Measured inflation
raw_data = retrieve_data.get_fed_data("CPIAUCSL", units="pc1", freq="m")
measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
measured_inf.rename(columns={"value": "inflation"}, inplace=True)
print("Descriptive stats for measured inflation")
print(measured_inf.describe())

# * Inflation expectations
raw_data = retrieve_data.get_fed_data("MICH")
inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
inf_exp.rename(columns={"value": "expectation"}, inplace=True)
print("Descriptive stats for inflation expectations")
print(inf_exp.describe())

# * Inflation expectations (percent change)
raw_data = retrieve_data.get_fed_data("MICH", units="pc1")
inf_exp_perc, _, _ = retrieve_data.clean_fed_data(raw_data)
inf_exp_perc.rename(columns={"value": "expectation_%_chg"}, inplace=True)
print("Descriptive stats for inflation expectations (percent change)")
print(inf_exp_perc.describe())

# * Non-durables consumption, monthly
raw_data = retrieve_data.get_fed_data("PCEND", units="pc1")
nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

# * Merge dataframes to align dates and remove extras
us_data = measured_inf.merge(inf_exp, how="left")
us_data = us_data.merge(inf_exp_perc, how="left")
us_data = us_data.merge(nondur_consump, how="left")

logger.info("Testing align_series")
exp = us_data["expectation"].to_numpy()[:101]  ## Setting to odd-numbered length
nondur = us_data["nondurable"].to_numpy()[:101]  ## Setting to odd-numbered length
t = us_data["date"].to_numpy()[:101]  ## Setting to odd-numbered length
logger.debug(
    "Lengths of exp: %s, nondurables: %s, and t: %s", len(exp), len(nondur), len(t)
)
# * Create data objects for each measure
exp_for_dwt = dwt.DataForDWT(exp, mother_wavelet)
nondur_for_dwt = dwt.DataForDWT(nondur, mother_wavelet)
# * Run DWTs and extract smooth signals
results_exp_dwt = dwt.smooth_signal(exp_for_dwt)
results_nondur_dwt = dwt.smooth_signal(nondur_for_dwt)

smooth_component_exp = dwt.reconstruct_signal_component(
    results_exp_dwt.coeffs, mother_wavelet, 0
)
smooth_component_nondur = dwt.reconstruct_signal_component(
    results_nondur_dwt.coeffs, mother_wavelet, 0
)
logger.debug(
    "Lengths of exp: %s, nondurables: %s, and t: %s",
    len(smooth_component_exp),
    len(smooth_component_nondur),
    len(t),
)

assert len(t) != len(smooth_component_exp) and len(t) != len(smooth_component_nondur)

smooth_component_exp_test = regression.align_series(t, smooth_component_exp)
smooth_component_nondur_test = regression.align_series(t, smooth_component_nondur)

assert len(t) == len(smooth_component_exp_test) and len(t) == len(
    smooth_component_nondur_test
)

logger.info(
    "Testing plot_compare_components -- that t array matches length of signal arrays"
)
fig = regression.plot_compare_components(
    a_label="expectation",
    b_label="nondurable",
    smooth_a_coeffs=results_exp_dwt.coeffs,
    smooth_b_coeffs=results_nondur_dwt.coeffs,
    time=t,
    levels=results_exp_dwt.levels,
    wavelet=MOTHER,
    figsize=(15, 10),
)

assert isinstance(fig, matplotlib.figure.Figure)

logger.info("Testing wavelet_approximation function")
# * Remove NaN rows
us_data.dropna(inplace=True)

# * Create data objects for each measure
exp_for_dwt = dwt.DataForDWT(us_data["expectation"].to_numpy(), mother_wavelet)
nondur_for_dwt = dwt.DataForDWT(us_data["nondurable"].to_numpy(), mother_wavelet)

# * Run DWTs and extract smooth signals
results_exp_dwt = dwt.smooth_signal(exp_for_dwt)
results_nondur_dwt = dwt.smooth_signal(nondur_for_dwt)
approximations = regression.wavelet_approximation(
    smooth_t_dict=results_exp_dwt.smoothed_signal_dict,
    original_y=nondur_for_dwt.y_values,
    levels=results_exp_dwt.levels,
)

assert len(approximations) == results_exp_dwt.levels

logger.info("Test complete")
