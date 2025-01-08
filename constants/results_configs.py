"""Constants for results module"""

import pywt
import pycwt as wavelet

END_DATE = "2024-07-31"

# * Define constant currency years
CONSTANT_DOLLAR_DATE = "2017-12-01"

DECIMAL_PLACES = 2

# * Define statistical tests to run on data
STATISTICS_TESTS = [
    "count",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "Jarque-Bera",
    "Shapiro-Wilk",
    "Ljung-Box",
]
HYPOTHESIS_THRESHOLD = [0.1, 0.05, 0.001]

# * Define DWT configs
DWT_MOTHER = "db4"
DWT_MOTHER_WAVELET = pywt.Wavelet(DWT_MOTHER)

# * Define CWT configs
CWT_MOTHER = wavelet.Morlet(f0=6)
NORMALIZE = True  # Define normalization
DT = 1 / 12  # In years
S0 = 2 * DT  # Starting scale
DJ = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / DJ  # Seven powers of two with DJ sub-octaves
LEVELS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]  # Period scale is logarithmic
CWT_FIG_PROPS = {"figsize": (20, 10), "dpi": 72}
CWT_PLOT_PROPS = {
    "cmap": "jet",
    "sig_colors": "k",
    "sig_linewidths": 2,
    "coi_color": "k",
    "coi_alpha": 0.3,
    "coi_hatch": "--",
}

# * Define XWT configs
XWT_DT = 1 / 12  # Delta t
XWT_DJ = 1 / 8  # Delta j
XWT_S0 = 2 * DT  # Initial scale
XWT_MOTHER = "morlet"  # Morlet wavelet with :math:`\omega_0=6`.
XWT_MOTHER_DICT = {
    "morlet": wavelet.Morlet(6),
    "paul": wavelet.Paul(),
    "DOG": wavelet.DOG(),
    "mexicanhat": wavelet.MexicanHat(),
}

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
