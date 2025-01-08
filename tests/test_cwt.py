"""Test CWT functions"""

from __future__ import division

import matplotlib.pyplot as plt

from constants import ids
from src.utils import wavelet_helpers
from src import retrieve_data, cwt

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
df_exp, t_date, dat = retrieve_data.clean_fed_data(raw_data)
df_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)
data_for_cwt = cwt.DataForCWT(
    t_date, dat, cwt.MOTHER, cwt.DT, cwt.DJ, cwt.S0, cwt.LEVELS
)

raw_data2 = retrieve_data.get_fed_data(ids.US_CPI, units="pc1", freqs="m")
df_inf, t_date2, dat2 = retrieve_data.clean_fed_data(raw_data)
df_inf.rename(columns={"value": ids.INFLATION}, inplace=True)


logger.info("Test set_time_range")
t_test = data_for_cwt.time_range
assert len(t_test) == len(t_date)

logger.info("Test run_cwt")
results_from_cwt = cwt.run_cwt(data_for_cwt)
assert len(results_from_cwt.power) == len(results_from_cwt.period)

logger.info("Testing standardization of short series")
## Use merged dataframe with shortened CPI inflation series
df_combo = df_exp.merge(df_inf, how="left")
logger.debug("df_combo shape: %s", df_combo.shape)
assert len(df_exp) == len(df_combo)

y2 = wavelet_helpers.standardize_series(
    df_combo[ids.INFLATION].to_numpy(), detrend=True, remove_mean=False
)

data_for_cwt2 = cwt.DataForCWT(
    df_combo[ids.DATE].to_numpy(),
    y2,
    cwt.MOTHER,
    cwt.DT,
    cwt.DJ,
    cwt.S0,
    cwt.LEVELS,
)
logger.debug("Length of cpi series: %s", len(df_combo[ids.INFLATION].to_numpy()))

results_for_cwt2 = cwt.run_cwt(data_for_cwt2)

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
cwt.plot_cwt(ax, data_for_cwt2, results_for_cwt2, **cwt_plot_props)

# * Set labels/title
ax.set_xlabel("")
ax.set_ylabel("Period (years)")
ax.set_title(ids.INFLATION)

logger.info("CWT test complete!")
