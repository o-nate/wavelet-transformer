"""IDs for datasets"""

from utils.logging_config import get_logger

logger = get_logger(__name__)

# * Wavelet transform labels
CWT = "Continuous (CWT)"
DWT = "Discrete (DWT)"
WCT = "Wavelet Coherence (WCT)"
DECOMPOSE = "Decomposition"
SMOOTH = "Smoothing"
ASCEND = "Ascending"
DESCEND = "Descending"

# * US data (FRED API)
US_CPI = "CPIAUCNS"
US_CPI_SEASONAL_ADJ = "CPIAUCSL"
US_CPI_NONDURABLES = "CUUR0000SAN"
US_CPI_DURABLES = "CUUR0000SAD"
US_INF_EXPECTATIONS = "MICH"
US_NONDURABLES_CONSUMPTION = "PCEND"
US_DURABLES_CONSUMPTION = "PCEDG"
US_SAVINGS = "PMSAVE"
US_SAVINGS_RATE = "PSAVERT"

# * French data (FRED API)
FR_CPI = "FRACPIALLMINMEI"

# * French data (INSEE API)
FR_INF_EXPECTATIONS = "000857180"
FR_INF_PERCEPTIONS = "000857179"
FR_FOOD_CONSUMPTION = "011794482"
FR_GOODS_CONSUMPTION = "011794487"
FR_DURABLES_CONSUMPTION = "011794493"

# * Columns in dataframes
DATE = "date"
CPI = "cpi"
INFLATION = "inflation"
EXPECTATIONS = "expectation"
NONDURABLES = "nondurable"
DURABLES = "durable"
NONDURABLES_CHG = "nondurable_chg"
DURABLES_CHG = "durable_chg"
SAVINGS = "savings"
SAVINGS_CHG = "savings_chg"
SAVINGS_RATE = "savings_rate"
REAL_NONDURABLES = f"real_{NONDURABLES}"
REAL_DURABLES = f"real_{DURABLES}"
REAL_SAVINGS = f"real_{SAVINGS}"
DIFF_LOG_CPI = f"diff_log_{CPI}"
DIFF_LOG_EXPECTATIONS = f"diff_log_{EXPECTATIONS}"
DIFF_LOG_NONDURABLES = f"diff_log_{NONDURABLES}"
DIFF_LOG_DURABLES = f"diff_log_{DURABLES}"
DIFF_LOG_SAVINGS = f"diff_log_{SAVINGS}"
DIFF_LOG_REAL_NONDURABLES = f"diff_log_{REAL_NONDURABLES}"
DIFF_LOG_REAL_DURABLES = f"diff_log_{REAL_DURABLES}"
DIFF_LOG_REAL_SAVINGS = f"diff_log_{REAL_SAVINGS}"

SAMPLE_DATA = [
    INFLATION,
    EXPECTATIONS,
    NONDURABLES_CHG,
    DURABLES_CHG,
    SAVINGS_RATE,
]

# * Display names
DISPLAY_NAMES = {
    INFLATION: "US Inflation rate (%)",
    EXPECTATIONS: "US Expected inflation rate (%)",
    SAVINGS_RATE: "US Savings rate (%)",
    NONDURABLES_CHG: "US Nondurables consumption (% chg)",
    DURABLES_CHG: "US Durables consumption (% chg)",
    SAVINGS_CHG: "US Savings (% chg)",
    CPI: "US Consumer Price Index (CPI)",
    DIFF_LOG_CPI: "US CPI (Diff in log)",
}


DATA_SCHEMA = {
    DISPLAY_NAMES[name]: {"original_name": name, "file_path": f"{name}.csv"}
    for name in SAMPLE_DATA
} | {DISPLAY_NAMES[CPI]: {"original_name": CPI, "file_path": f"{CPI}.csv"}}
