"""Tests for CAMME data processing"""

import pandas as pd

from constants.camme import IGNORE_HOUSING, IGNORE_HOUSING_YEARS, IGNORE_SUPPLEMENTS
from src import process_camme

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

test_dir = process_camme.data_dir


def test_filter_files() -> None:
    """Test that only selects standard CAMME csv files"""
    logger.info("Retrieving folders")
    camme_csv_folders = process_camme.retrieve_folders(test_dir)
    logger.info("Retrieving CSV files")
    camme_csv_folders = process_camme.retrieve_csv_files(camme_csv_folders)
    logger.debug(camme_csv_folders)
    for year, files in camme_csv_folders.items():
        for file in files["csv"]:
            assert any(
                supplemental not in file.name for supplemental in IGNORE_SUPPLEMENTS
            )

    for year, files in camme_csv_folders.items():
        for file in files["csv"]:
            if year in IGNORE_HOUSING_YEARS:
                assert IGNORE_HOUSING not in file.name


def test_columns() -> None:
    """Test that columns match"""
    logger.info("Retrieving folders")
    camme_csv_folders = process_camme.retrieve_folders(test_dir)
    logger.info("Retrieving CSV files")
    camme_csv_folders = process_camme.retrieve_csv_files(camme_csv_folders)
    for year, files in camme_csv_folders.items():
        year_cols, _ = process_camme.define_year_columns(year)
        logger.debug(year_cols)
        for table in files["csv"]:
            df = pd.read_csv(table, delimiter=";", encoding="latin-1")
            # Set columns as lowercase since some apparently are read as having
            # different cases than in their csv file
            df.columns = df.columns.str.lower()
            assert all(item in df.columns.to_list() for item in year_cols)


def test_nans() -> None:
    """Test that DataFrames are not getting improperly converted from CSV"""
    camme_csv_folders = process_camme.retrieve_folders(test_dir)
    camme_csv_folders = process_camme.retrieve_csv_files(camme_csv_folders)
    for year, files in camme_csv_folders.items():
        year_cols, _ = process_camme.define_year_columns(year)
        for table in files["csv"]:
            df = pd.read_csv(table, delimiter=";", encoding="latin-1")
            # Set columns as lowercase since some apparently are read as having
            # different cases than in their csv file
            df.columns = df.columns.str.lower()
            assert df[year_cols[0]].isnull().all() is False


def main() -> None:
    """Run test script"""
    logger.info("Testing file filtering")
    test_filter_files()
    logger.info("Testing column selection")
    test_columns()
    logger.info("Testing DataFrame generation")
    test_nans()
    logger.info("CAMME test complete")


if __name__ == "__main__":
    main()
