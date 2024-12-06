"""For plotting transforms"""

import math

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from matplotlib.figure import Figure
from pandas._libs.tslibs.timestamps import Timestamp

from src import cwt, dwt, regression, xwt

from src.cwt import DataForCWT, ResultsFromCWT
from src.dwt import DataForDWT, ResultsFromDWT
from src.utils import wavelet_helpers
from src.xwt import DataForXWT, ResultsFromXWT

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


def plot_dwt_decomposition_for(
    transform_results_dict: dict[str, ResultsFromDWT], time_array: npt.NDArray, **kwargs
) -> Figure:
    """Determines whether to create single or combined plots

    Args:
        transform_results_dict (dict[str, ResultsFromDWT]): _description_
        time_array (npt.NDArray): _description_

    Raises:
        Exception: _description_

    Returns:
        Figure: _description_
    """
    comparison = list(transform_results_dict)
    if len(transform_results_dict) == 2:
        return regression.plot_compare_components(
            a_label=comparison[0],
            b_label=comparison[1],
            a_coeffs=transform_results_dict[comparison[0]].coeffs,
            b_coeffs=transform_results_dict[comparison[1]].coeffs,
            time=time_array,
            levels=transform_results_dict[comparison[0]].levels,
            **kwargs
        )
    elif len(transform_results_dict) == 1:
        return dwt.plot_components(
            label=comparison[0],
            coeffs=transform_results_dict[comparison[0]].coeffs,
            time=time_array,
            levels=transform_results_dict[comparison[0]].levels,
            **kwargs
        )
    else:
        logger.error("Too many series provided (maximum of two permitted)")
        raise Exception("Too many series provided (maximum of two permitted)")


def plot_dwt_smoothing_for(
    transform_dict: dict[str, DataForDWT],
    transform_results_dict: dict[str, ResultsFromDWT],
    time_array: npt.NDArray,
    ascending: bool = True,
    **kwargs
) -> Figure:
    comparison = list(transform_results_dict.keys())
    if len(transform_results_dict) == 1:
        transform_results_dict[comparison[0]].smooth_signal(
            y_values=transform_dict[comparison[0]].y_values,
            mother_wavelet=transform_dict[comparison[0]].mother_wavelet,
        )
        return dwt.plot_smoothing(
            transform_results_dict[comparison[0]].smoothed_signal_dict,
            time_array,
            transform_dict[comparison[0]].y_values,
            ascending=ascending,
            **kwargs
        )


def round_down(x: float | int, n: int) -> int:
    """Round down to nearest value at same magnitude

    Args:
        x (float | int): Value to round down
        n (int): Magnitude (10**n)

    Returns:
        int: Value rounded down
    """
    return x if x % 10**n == 0 else x - x % 10**n


def round_up(x: float | int, n: int) -> int:
    """Round up to nearest value at same magnitude

    Args:
        x (float | int): Value to round up
        n (int): Magnitude (10**n)

    Returns:
        int: Value rounded up
    """
    return x if x % 10**n == 0 else x + 10**n - x % 10**n


def set_x_ticks(data: list[Timestamp]) -> tuple[list[int], list[str]]:
    """Generate ticks and tick positions for x axis of XWT power spectrum

    Args:
        data (list[Timestamp]): Time data

    Returns:
        tuple[list[int], list[str]]: Tuple of lists of positions and labels
    """
    magnitude = np.log10(len(data))
    magnitude = int(math.floor(magnitude))
    divisor = 10**magnitude
    divisor = int(divisor)
    adjust_series_length = round_down(len(data), magnitude)
    max_adjust_series_length = round_up(len(data), magnitude)

    x_dates = [data[0]] + [
        data[i + 99] for i in range(0, adjust_series_length, divisor)
    ]
    x_ticks = [str(date.year) for date in x_dates]

    x_tick_positions = list(range(0, max_adjust_series_length, divisor))

    return x_tick_positions, x_ticks
