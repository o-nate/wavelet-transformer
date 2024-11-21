"""Script to combine and standardize yearly CAMME responses from separate csv files"""

import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union

import pandas as pd

from src.utils.logging_helpers import define_other_module_log_level
from constants.camme import (
    IGNORE_HOUSING,
    IGNORE_HOUSING_YEARS,
    IGNORE_SUPPLEMENTS,
    VARS_DICT,
)

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)

# * Get data directory folder
parent_dir = Path(__file__).parents[1]
camme_dir = parent_dir / "data" / "camme"


def retrieve_folders(path: Union[str, os.PathLike]) -> Dict[
    str,
    Dict[Dict[str, Union[str, os.PathLike]], Dict[str, List[Union[str, os.PathLike]]]],
]:
    """Loop through the csv subdirectories"""
    dir_dict = {}
    for sub_dir in path.rglob("Csv"):
        dir_dict[sub_dir.parents[0].name] = {"path": sub_dir, "csv": []}
    return dir_dict


def retrieve_csv_files(
    dir_dict: Dict[str, Union[str, os.PathLike]]
) -> Dict[str, Union[str, os.PathLike]]:
    """Add only standard questionnaire csv file paths to dictionary"""
    for year in dir_dict:
        total_files = sum(len(files) for _, _, files in os.walk(dir_dict[year]["path"]))
        logging.debug("There are %s total files for year %s", total_files, year)
        for file in dir_dict[year]["path"].rglob("*"):
            if any(supp in file.name for supp in IGNORE_SUPPLEMENTS):
                pass
            elif (
                any(y in file.parents[1].name for y in IGNORE_HOUSING_YEARS)
                and IGNORE_HOUSING in file.name
            ):
                pass
            else:
                dir_dict[year]["csv"].append(file)
        files_kept_count = len(dir_dict[year]["csv"])
        logging.debug(
            "%s of %s files maintained for year %s", files_kept_count, total_files, year
        )
    return dir_dict


def define_year_columns(year_df: str) -> Tuple[List[str], Dict[str, str]]:
    """Extract appropriate columns depending on year and corresponding names"""
    var_name_year = sorted([int(y) for y in VARS_DICT["inf_exp_qual"]])
    # Define standardized names for columns to keep
    final_cols = [c for c in VARS_DICT]
    if int(year_df) < var_name_year[1]:
        key_year = str(var_name_year[0])
    elif var_name_year[1] <= int(year_df) < var_name_year[2]:
        key_year = str(var_name_year[1])
    else:
        key_year = str(var_name_year[2])
    year_cols = [
        VARS_DICT[c][key_year].lower()
        for c in final_cols
        if VARS_DICT[c][key_year] != ""
    ]
    new_cols_dict = {VARS_DICT[new][key_year].lower(): new for new in VARS_DICT}
    return year_cols, new_cols_dict


def convert_to_year_dataframe(
    dir_dict: Dict[str, Union[str, os.PathLike]]
) -> Dict[str, pd.DataFrame]:
    """Combine all dataframes into one for all years"""
    df_dict = {}
    for year in dir_dict:
        # Empty list for DataFrames for complete year's data
        df_complete = []
        cols, new_cols = define_year_columns(year)
        logging.debug("Columns to extract for %s: %s", year, cols)
        for table in dir_dict[year]["csv"]:
            df = pd.read_csv(table, delimiter=";", encoding="latin-1")
            # * Set columns as lowercase since some apparently are read as
            # * having different cases than in their csv file
            df.columns = df.columns.str.lower()
            df = df[cols]
            df["file_name"] = Path(table).name
            df_complete.append(df)
        df_dict[year] = pd.concat(df_complete, axis=0, ignore_index=True)

        # Rename to standard columns names
        logging.debug("Renaming columns %s", new_cols)
        df_dict[year].rename(
            columns=new_cols,
            inplace=True,
        )
        df_dict[year]["year"] = int(year)
        logging.debug(
            "# of NaNs for %s: %s", year, df_dict[year]["month"].isnull().sum()
        )
        logging.debug("DataFrame shape: %s", df_dict[year].shape)
    return df_dict


def create_complete_dataframe(dir_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine dataframes for each year"""
    # Empty list for DataFrames to fill with all years
    df_complete = []
    for df in dir_dict.values():
        df_complete.append(df)
    return pd.concat(df_complete, axis=0, ignore_index=True)


def preprocess(
    data_dir: Union[str, os.PathLike]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create DataFrame with all years' data"""
    logging.info("Retrieving folders")
    camme_csv_folders = retrieve_folders(data_dir)
    logging.info("Retrieving CSV files")
    camme_csv_folders = retrieve_csv_files(camme_csv_folders)
    logging.info("Converting to DataFrames")
    start = time.time()
    dfs = convert_to_year_dataframe(camme_csv_folders)
    end = time.time()
    dtime = end - start
    ## Elapsed time before change: 11.67s
    print(dfs["1989"].head())
    print(dfs["1991"].head())
    print(dfs["2004"].head())
    print(dfs["2021"].head())
    logging.info("Elapsed time to convert to DataFrames: %s", dtime)
    return dfs, create_complete_dataframe(dfs)


def main() -> None:
    """Run script"""
    _, df_final = preprocess(camme_dir)
    print(df_final[df_final["year"] == 2021].head())
    print(df_final.tail())
    print(df_final.info())
    print(df_final.describe().T)


if __name__ == "__main__":
    main()
