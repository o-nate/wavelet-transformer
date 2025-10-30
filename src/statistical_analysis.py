"""Statistical tests for time series analysis"""

import sys

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.graphics.tsaplots
import statsmodels.stats.diagnostic

# Ensure project root is on sys.path so top-level packages like `constants` resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import ids
from src.utils.helpers import add_real_value_columns, calculate_diff_in_log
from src import retrieve_data

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Define constant currency years
CONSTANT_DOLLAR_DATE = "2017-12-01"

DESCRIPTIVE_STATS = [
    "count",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "Jarque-Bera",
    "Shapiro-Wilk",
    "Ljung-Box",
]
NORMALITY_TESTS = {"Jarque-Bera": stats.jarque_bera, "Shapiro-Wilk": stats.shapiro}
PANDAS_METHODS = ["count", "mean", "std", "skewness", "kurtosis"]
HYPOTHESIS_THRESHOLD = [0.1, 0.05, 0.001]
LJUNG_BOX_LAGS = [40]


def include_statistic(
    statistic: str,
    statistic_data: dict[str, float],
    results_dict: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Add skewness data for each measure"""
    for measure, result_dict in results_dict.items():
        result_dict[statistic] = statistic_data[measure]
    return results_dict


def add_p_value_stars(
    test_statistic: int | float,
    p_value: float,
    hypothesis_threshold: list[float],
    decimals_places: int = 2,
) -> str:
    """Add stars (*) for each p value threshold that the test statistic falls below"""
    star_test_statistic = str(f"%.{decimals_places}f" % test_statistic)
    for p_threshold in sorted(hypothesis_threshold):
        star_test_statistic += "*" if p_value <= p_threshold else ""
    return star_test_statistic


def test_normality(
    normality_test: str,
    data: pd.DataFrame,
    date_column: str = "date",
    add_pvalue_stars: bool = False,
) -> dict[str, str | float]:
    """Generate dictionary with Jarque-Bera test results for each dataset"""
    results_dict = {}
    cols_to_test = data.drop(date_column, axis=1).columns.to_list()
    for col in cols_to_test:
        x = data[col].dropna().to_numpy()
        test_stat, p_value = NORMALITY_TESTS[normality_test](x)
        if add_pvalue_stars:
            results_dict[col] = add_p_value_stars(
                test_stat, p_value, HYPOTHESIS_THRESHOLD
            )
        results_dict[col] = test_stat
    return results_dict


def conduct_ljung_box(
    data: pd.DataFrame,
    lags: list[int],
    date_column: str = "date",
    add_pvalue_stars: bool = False,
) -> dict[str, str | float]:
    """Generate dictionary with Ljung-Box test results for each dataset"""
    results_dict = {}
    cols_to_test = data.drop(date_column, axis=1).columns.to_list()
    for col in cols_to_test:
        test_results = statsmodels.stats.diagnostic.acorr_ljungbox(
            data[col].dropna(), lags=lags
        )
        test_stat, p_value = (
            test_results["lb_stat"].iat[0],
            test_results["lb_pvalue"].iat[0],
        )
        if add_pvalue_stars:
            result = add_p_value_stars(test_stat, p_value, HYPOTHESIS_THRESHOLD)
        results_dict[col] = test_stat
    return results_dict


def correlation_matrix_pvalues(
    data: pd.DataFrame,
    hypothesis_threshold: list[float],
    decimals: int = 2,
    display: bool = False,
    export_table: bool = False,
) -> pd.DataFrame:
    """Calculate pearson correlation and p-values and add asterisks
    to relevant values in table"""
    rho = data.corr(numeric_only=True)
    pval = data.corr(
        method=lambda x, y: stats.pearsonr(x, y)[1], numeric_only=True
    ) - np.eye(*rho.shape)
    p = pval.applymap(
        lambda x: "".join(["*" for threshold in hypothesis_threshold if x <= threshold])
    )
    corr_matrix = rho.round(decimals).astype(str) + p
    if display:
        cols = data.columns.to_list()
        print(f"P-values benchmarks: {hypothesis_threshold}")
        for c in cols:
            print(c)
            print(f"{c} p-values: \n{pval[c]}")
    if export_table:
        # * Get results directory
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "correlation_matrix.html"
        corr_matrix.to_html(export_file)
    return corr_matrix


def complete_summary_results_dict(
    initial_results_dict: dict[str, dict[str, int | float]],
    data_dict: dict[str, dict[str, str]],
) -> dict[str, dict[str, float | str]]:
    """Combine statistical test results in a single dict"""
    for stat_test, result in data_dict.items():
        initial_results_dict = include_statistic(
            stat_test, result, initial_results_dict
        )
    return initial_results_dict


def create_summary_table(
    data_dict: dict[str, dict[str, float | str]], export_table: bool = False
) -> pd.DataFrame:
    """Create table with descriptive statistics for all datasets with option to export"""
    df = pd.DataFrame(data_dict)
    if export_table:
        # * Get results directory
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "descriptive_stats.html"

        df.to_html(export_file)
    return df


def generate_descriptive_statistics(
    data: pd.DataFrame, stats_test: list[str], **kwargs
) -> pd.DataFrame:
    """Produce descriptive statistics test results"""
    ## Initialize dict to store each test
    stats_test_dict = {}
    ## Initialize dict to store final results
    results_dict = {measure: {} for measure in data.columns if "date" not in measure}
    for test in stats_test:
        if test in PANDAS_METHODS and test == "skewness":
            if test == "skewness":
                stats_test_dict[test] = data.skew(numeric_only=True).to_dict()
        elif test in PANDAS_METHODS and test != "skewness":
            stats_test_dict[test] = getattr(data, test)(numeric_only=True).to_dict()
        elif test == "Ljung-Box":
            stats_test_dict[test] = conduct_ljung_box(
                data=data,
                lags=LJUNG_BOX_LAGS,
                date_column="date",
                add_pvalue_stars=True,
            )
        else:
            stats_test_dict[test] = test_normality(
                normality_test=test,
                data=data,
                date_column="date",
                add_pvalue_stars=True,
            )
    results_dict = complete_summary_results_dict(results_dict, stats_test_dict)
    results_df = create_summary_table(results_dict, **kwargs)
    return results_df


def main() -> None:
    """Run script"""
    # * Retrieve data

    ## CPI
    raw_data = retrieve_data.get_fed_data(ids.US_CPI)
    cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
    cpi.rename(columns={"value": "cpi"}, inplace=True)

    ## Inflation
    raw_data = retrieve_data.get_fed_data(ids.US_CPI, units="pc1")
    inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf.rename(columns={"value": "inflation"}, inplace=True)

    ## Inflation expectations
    raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": "expectation"}, inplace=True)

    ## Non-durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_NONDURABLES_CONSUMPTION)
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

    ## Durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_DURABLES_CONSUMPTION)
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": "durable"}, inplace=True)

    ## Personal savings rate
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS)
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": "savings"}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    us_data = cpi.merge(inf, how="left")
    us_data = us_data.merge(inf_exp, how="left")
    us_data = us_data.merge(nondur_consump, how="left")
    us_data = us_data.merge(dur_consump, how="left")
    us_data = us_data.merge(save, how="left")

    # * Drop NaNs
    us_data.dropna(inplace=True)

    # * Add real value columns
    logger.info(
        "Using constant dollars from %s, CPI: %s",
        CONSTANT_DOLLAR_DATE,
        us_data[us_data["date"] == pd.Timestamp(CONSTANT_DOLLAR_DATE)]["cpi"].iat[0],
    )
    us_data = add_real_value_columns(
        data=us_data,
        nominal_columns=["nondurable", "durable", "savings"],
        cpi_column="cpi",
        constant_date=CONSTANT_DOLLAR_DATE,
    )
    us_data = calculate_diff_in_log(
        data=us_data, columns=["cpi", "real_nondurable", "real_durable", "real_savings"]
    )

    print(us_data.head())

    results = generate_descriptive_statistics(
        us_data, DESCRIPTIVE_STATS, export_table=False
    )
    print(results)

    us_corr = correlation_matrix_pvalues(
        data=us_data,
        hypothesis_threshold=HYPOTHESIS_THRESHOLD,
        decimals=2,
        display=False,
        export_table=False,
    )
    print(us_corr)

    _, axs = plt.subplots(len(us_data.drop("date", axis=1).columns.to_list()))
    for ax, c in zip(axs, us_data.drop("date", axis=1).columns.to_list()):
        statsmodels.graphics.tsaplots.plot_acf(us_data[c], lags=36, ax=ax)
        ax.title.set_text(c)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
