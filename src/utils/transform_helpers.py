"""Helper functions to organize results of transforms"""

import pandas as pd

from constants import ids, results_configs

from src import cwt, dwt, xwt

from src.cwt import DataForCWT, ResultsFromCWT
from src.dwt import DataForDWT, ResultsFromDWT
from src.utils import wavelet_helpers
from src.xwt import DataForXWT, ResultsFromXWT

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


def create_dwt_dict(
    data_for_dwt: pd.DataFrame,
    measures_list: list[str],
    **kwargs,
) -> dict[str, DataForDWT]:
    """Create dict of discrete wavelet transform dataclass objects from DataFrame

    Args:
        data_for_dwt (pd.DataFrame): Data
        measures_list (list[str]): List of names of each time series to transform

    Returns:
        dict[str, Type[Any]]: Dict of DWT dataclasses
    """
    transform_dict = {}
    logger.debug("df shape: %s", data_for_dwt.shape)
    for measure in measures_list:
        transform_dict[measure] = dwt.DataForDWT(
            y_values=data_for_dwt[measure].to_numpy(), **kwargs
        )
    return transform_dict


def create_cwt_dict(
    data_for_cwt: pd.DataFrame,
    measures_list: list[str],
    **kwargs,
) -> dict[str, DataForCWT]:
    """Create dict of continuous wavelet transform objects from DataFrame"""
    transform_dict = {}
    for measure in measures_list:
        t_values = data_for_cwt[data_for_cwt[measure].notna()][ids.DATE].to_numpy()
        y_values = data_for_cwt[data_for_cwt[measure].notna()][measure].to_numpy()
        y_values = wavelet_helpers.standardize_series(y_values)
        transform_dict[measure] = cwt.DataForCWT(
            t_values=t_values, y_values=y_values, **kwargs
        )
    return transform_dict


def create_xwt_dict(
    data_for_xwt: pd.DataFrame, xwt_list: list[tuple[str, str]], **kwargs
) -> dict[tuple[str, str], DataForXWT]:
    """Create dict of cross-wavelet transform objects from DataFrame"""
    transform_dict = {}
    for comparison in xwt_list:
        y1 = data_for_xwt.dropna()[comparison[0]].to_numpy()
        y2 = data_for_xwt.dropna()[comparison[1]].to_numpy()
        y1 = wavelet_helpers.standardize_series(y1, **kwargs)
        y2 = wavelet_helpers.standardize_series(y2, **kwargs)

        transform_dict[comparison] = xwt.DataForXWT(
            y1_values=y1,
            y2_values=y2,
            mother_wavelet=results_configs.XWT_MOTHER_DICT[results_configs.XWT_MOTHER],
            delta_t=results_configs.XWT_DT,
            delta_j=results_configs.XWT_DJ,
            initial_scale=results_configs.XWT_S0,
            levels=results_configs.LEVELS,
        )
    return transform_dict


def create_dwt_results_dict(
    dwt_data_dict: dict[str, DataForDWT], measures_list: list[str], **kwargs
) -> dict[str, ResultsFromDWT]:
    """Create dict of DWT results instances"""
    results_dict = {}
    for measure in measures_list:
        results_dict[measure] = dwt.run_dwt(dwt_data_dict[measure], **kwargs)
    return results_dict


def create_cwt_results_dict(
    cwt_data_dict: dict[str, DataForCWT], measures_list: list[str], **kwargs
) -> dict[str, ResultsFromCWT]:
    """Create dict of CWT results instances"""
    results_dict = {}
    for measure in measures_list:
        results_dict[measure] = cwt.run_cwt(cwt_data_dict[measure], **kwargs)
    return results_dict


def create_xwt_results_dict(
    xwt_data_dict: dict[str, DataForXWT],
    xwt_list: list[tuple[str, str]],
    **kwargs,
) -> ResultsFromXWT:
    """Create dict of XWT results instances"""
    results_dict = {}
    for comparison in xwt_list:
        results_dict[comparison] = xwt.run_xwt(xwt_data_dict[comparison], **kwargs)
    return results_dict
