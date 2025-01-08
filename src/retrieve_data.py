"""Retrieve data for analysis via API from statistics agencies and central banks"""

import json
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import requests
import xmltodict

from constants import ids
from src.utils.helpers import convert_column_to_real_value

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Retrieve API credentials
load_dotenv()
BDF_KEY = os.getenv("BDF_KEY")
FED_KEY = os.getenv("FED_KEY")
INSEE_AUTH = os.getenv("INSEE_AUTH")
TIMEOUT = 5

END_DATE = "2024-07-31"

# * Define constant currency years
CONSTANT_DOLLAR_DATE = "2017-12-01"


def get_fed_data(series: str, no_headers: bool = True, **kwargs) -> str:
    """Retrieve data series from FRED database and convert to time series if desired
    Some series codes:
    - French CPI (`"FRACPIALLMINMEI", units="pc1", freq="m"`)
    - Michigan Expected Inflation (`"MICH"`)
    - 1-Year Expected Inflation (`"EXPINF1YR"`)
    - US CPI Not seasonally adjusted(`"CPIAUCNS", units="pc1", freq="m"`)
    - US CPI Seasonally adjusted (`"CPIAUCSL", units="pc1", freq="m"`)
    - US CPI Nondurables in U.S. City Average (`"CUUR0000SAN", units="pc1", freq="m"`)
    - US CPI Durables in U.S. City Average (`"CUUR0000SAD", units="pc1", freq="m"`)
    - Personal Savings Rate (`"PSAVERT"`)
    - Personal Consumption Expenditure (`"PCE", units="pc1"`)
    - Personal Durables Consumption (`"PCEDG"`, units="pc1")
    - Personal Non-durables Consumption (`"PCEND"`, units="pc1")
    - Personal Consumption Expenditures, Not seasonal, Quarterly (`"NA000349Q"`)
    - Real Personal Consumption Expenditures, Not seasonal, Quarterly (`"ND000349Q"`)
    - Real Personal Consumption Expenditures: Durable Goods, Not seasonal, Quarterly (`"ND000346Q"`)
    - Real Personal Consumption Expenditures: Nondurable Goods, Not seasonal, Quarterly (`'PCNDGC96'`)
    """

    ## API GET request
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    units = kwargs.get("units", None)
    freq = kwargs.get("freq", None)
    realtime_end = kwargs.get("realtime_end", None)

    ## Request parameters
    params = {
        "api_key": FED_KEY,
        "series_id": series,
        "units": units,
        "freq": freq,
        "realtime_start": realtime_end,
        "realtime_end": realtime_end,
        "file_type": "json",
    }

    ## Remove parameters with None
    params = {k: v for k, v in params.items() if v is not None}

    ## Create final url for request
    final_url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"

    ## Make request
    try:
        print(f"Requesting {series}")
        r = requests.get(final_url, timeout=TIMEOUT)
        r.raise_for_status()  # Raise an exception for 4XX and 5XX HTTP status codes
        resource = r.json()["observations"] if no_headers is True else r.json()
        print(f"Retrieved {series}")
        return resource
    except requests.Timeout:
        print("Timeout error: The request took too long to complete.")
        return None
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None


def clean_fed_data(
    json_data: str, show_info: bool = False
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    """Convert Fed data to time and endogenous variables (t, y)"""

    ## Convert to dataframe
    df = pd.DataFrame(json_data)
    if show_info:
        print(df.info(verbose=True), "\n")

    ## Convert dtypes
    df.replace(".", np.nan, inplace=True)
    df.dropna(inplace=True)
    df["value"] = pd.to_numeric(df["value"])
    df["date"] = pd.to_datetime(df["date"])

    ## Drop extra columns
    df = df[["date", "value"]]

    t = df["date"].to_numpy()
    y = df["value"].to_numpy()

    return df, t, y


def catalog_camme() -> tuple[str, str, str]:
    """Get info on CAMME indicators from INSEE"""
    url = "https://api.insee.fr/series/BDM/V1/data/ENQ-CONJ-MENAGES/"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {INSEE_AUTH}",
    }

    response = requests.get(url, headers=headers, timeout=TIMEOUT)
    decoded_response = response.content.decode("utf-8")
    response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
    for i in response_json["message:StructureSpecificData"]["message:DataSet"][
        "Series"
    ]:
        if i["@SERIE_ARRETEE"] == "FALSE" and i["@CORRECTION"] == "BRUT":
            print(
                f"""{i['@INDICATEUR']} : {i['@TITLE_FR']}\n\
            {i['@IDBANK']}\n"""
            )


def get_insee_data(series_id: str) -> list:
    """
    Retrieve data (Series_BDM) from INSEE API

    Some series codes:
    - 'Expected inflation' `"000857180"`,
    - 'Perceived inflation' `"000857179"`,
    - 'Expected savings' `"000857186"`,
    - 'Expected consumption' `"000857181"`,
    - 'Household food consumption' `'011794482'`,
    - 'Household goods consumptions' `'011794487'`,
    - 'Household durables consumption' `'011794493'`,

    """
    url = f"https://api.insee.fr/series/BDM/V1/data/SERIES_BDM/{series_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {INSEE_AUTH}",
    }
    response = requests.get(url, headers=headers, timeout=TIMEOUT)
    decoded_response = response.content.decode("utf-8")
    response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
    series_title = response_json["message:StructureSpecificData"]["message:DataSet"][
        "Series"
    ]["@TITLE_FR"]
    response_data = response_json["message:StructureSpecificData"]["message:DataSet"][
        "Series"
    ]["Obs"]
    print(f"Retrieved {series_title}. \n{len(response_data)} observations\n")

    return response_data


def clean_insee_data(
    data: list, ascending: bool = True
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    """Convert INSEE data to time and endogenous variables (t, y)"""
    df = pd.DataFrame(data)
    # Convert data types
    df["@TIME_PERIOD"] = pd.to_datetime(df["@TIME_PERIOD"])
    df["@OBS_VALUE"] = df["@OBS_VALUE"].astype(float)
    if ascending is True and df["@TIME_PERIOD"].iloc[-1] < df["@TIME_PERIOD"].iloc[0]:
        df = df[::-1]
    elif (
        ascending is False and df["@TIME_PERIOD"].iloc[-1] > df["@TIME_PERIOD"].iloc[0]
    ):
        df = df[::-1]
    else:
        pass
    df.rename(columns={"@TIME_PERIOD": "date", "@OBS_VALUE": "value"}, inplace=True)
    t = df["date"].to_numpy()
    y = df["value"].to_numpy()

    return df[["date", "value"]], t, y


def get_bdf_data(series_key: str, dataset: str = "ICP", **kwargs) -> str:
    """Retrieve data from Banque de France API
    Measured inflation: `'ICP.M.FR.N.000000.4.ANR'`
    """

    ## API GET request
    data_type = kwargs.get("data_type", "data")
    req_format = kwargs.get("format", "json")
    headers = {"accept": "application/json"}

    base_url = "https://api.webstat.banque-france.fr/webstat-fr/v1/"

    params = {
        "data_type": data_type,
        "dataset": dataset,
        "series_key": series_key,
    }

    ## Remove parameters with None
    params = {k: v for k, v in params.items() if v is not None}

    ## Create final url for request
    final_url = f"{base_url}{'/'.join([f'{v}' for k, v in params.items()])}?client_id={BDF_KEY}&format={req_format}"

    print(f"Requesting {series_key}")
    r = requests.get(final_url, headers=headers, timeout=TIMEOUT)
    print(r)
    response = r.json()
    response = response["seriesObs"][0]["ObservationsSerie"]["observations"]

    return response


def clean_bdf_data(
    data: list, ascending: bool = True
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    """Convert list of dicts data from Banque de France to lists for t and y"""
    ## Dictionary of observations
    dict_obs = {
        "periodId": [],
        "periodFirstDate": [],
        "periodName": [],
        "value": [],
    }
    for i in data:
        obs = i["ObservationPeriod"]
        for k, v in dict_obs.items():
            v.append(obs[k])

    ## Convert to df
    df = pd.DataFrame(dict_obs)
    if (
        ascending is True
        and df["periodFirstDate"].iloc[-1] < df["periodFirstDate"].iloc[0]
    ):
        df = df[::-1]
    elif (
        ascending is False
        and df["periodFirstDate"].iloc[-1] > df["periodFirstDate"].iloc[0]
    ):
        df = df[::-1]
    else:
        pass
    df["periodFirstDate"] = pd.to_datetime(df["periodFirstDate"], dayfirst=True)
    df.rename(columns={"periodFirstDate": "date"}, inplace=True)
    t = df["date"].to_numpy()
    y = df["value"].to_numpy()

    return df, t, y


def get_world_bank_data(series_id: str, country: str) -> str:
    """Retrieve inflation data by country
    Inflation rate (annual %): 'FP.CPI.TOTL.ZG'
    CPI (2010 = 100): 'FP.CPI.TOTL'
    France: 'FR'
    United States: 'US'
    """
    base_url = f"https://api.worldbank.org/v2/indicator/{series_id}?locations={country}?format=json"
    response = requests.get(base_url, timeout=TIMEOUT)
    print(response)
    return response.json()


def data_to_time_series(df, index_column, measure=None):
    """Convert dataframe to time series"""
    if measure is not None:
        df = df[[index_column, "value"]][df["measure"] == measure]
    else:
        df = df[[index_column, "value"]]
    ## Set date as index
    df.set_index(index_column, inplace=True)
    df = df.astype(float)
    return df


def main() -> None:
    """Run script"""
    # * Retrieve data

    ## CPI
    raw_data = get_fed_data(ids.US_CPI, realtime_end=END_DATE)
    cpi, _, _ = clean_fed_data(raw_data)
    cpi.rename(columns={"value": "cpi"}, inplace=True)

    ## Inflation
    raw_data = get_fed_data(ids.US_CPI, units="pc1")
    inf, _, _ = clean_fed_data(raw_data)
    inf.rename(columns={"value": "inflation"}, inplace=True)

    ## Inflation expectations
    raw_data = get_fed_data(ids.US_INF_EXPECTATIONS)
    inf_exp, _, _ = clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": "expectation"}, inplace=True)

    ## Non-durables consumption, monthly
    raw_data = get_fed_data(ids.US_NONDURABLES_CONSUMPTION)
    nondur_consump, _, _ = clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

    ## Durables consumption, monthly
    raw_data = get_fed_data(ids.US_DURABLES_CONSUMPTION)
    dur_consump, _, _ = clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": "durable"}, inplace=True)

    ## Personal savings rate
    raw_data = get_fed_data(ids.US_SAVINGS)
    save, _, _ = clean_fed_data(raw_data)
    save.rename(columns={"value": "savings"}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    us_data = cpi.merge(inf, how="left")
    us_data = us_data.merge(inf_exp, how="left")
    us_data = us_data.merge(nondur_consump, how="left")
    us_data = us_data.merge(dur_consump, how="left")
    us_data = us_data.merge(save, how="left")

    # # * Drop NaNs
    # us_data.dropna(inplace=True)

    print(us_data.tail())

    # * Convert to constant dollars
    us_data["real_nondurable"] = convert_column_to_real_value(
        data=us_data,
        column="nondurable",
        cpi_column="cpi",
        constant_date=CONSTANT_DOLLAR_DATE,
    )
    print(us_data.head())

    us_melt = pd.melt(us_data[["date", "nondurable", "real_nondurable"]], "date")
    print(us_melt.head())
    print(us_melt.tail())

    sns.lineplot(us_melt, x="date", y="value", hue="variable")
    plt.show()


if __name__ == "__main__":
    main()
