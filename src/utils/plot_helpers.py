"""For plotting transforms"""

import numpy.typing as npt

from matplotlib.figure import Figure
from pandas._libs.tslibs.timestamps import Timestamp

from src import dwt, regression

from src.dwt import DataForDWT, ResultsFromDWT

from src.utils.config import XWT_X_TICK_NUMBERS

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
            **kwargs,
        )
    elif len(transform_results_dict) == 1:
        return dwt.plot_components(
            label=comparison[0],
            coeffs=transform_results_dict[comparison[0]].coeffs,
            time=time_array,
            levels=transform_results_dict[comparison[0]].levels,
            **kwargs,
        )
    else:
        logger.error("Too many series provided (maximum of two permitted)")
        raise Exception("Too many series provided (maximum of two permitted)")


def plot_dwt_smoothing_for(
    transform_dict: dict[str, DataForDWT],
    transform_results_dict: dict[str, ResultsFromDWT],
    time_array: npt.NDArray,
    ascending: bool = True,
    **kwargs,
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
            **kwargs,
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
    """Generate ticks and tick positions for x axis"""
    if len(data) <= XWT_X_TICK_NUMBERS:
        return list(range(len(data))), [f"{date.month}/{date.year}" for date in data]

    # Calculate step size to get around pre-defined number of ticks
    step = max(1, len(data) // XWT_X_TICK_NUMBERS)

    x_dates = [data[0]] + [
        data[i] for i in range(step, len(data), step) if i < len(data)
    ]

    x_ticks = [f"{date.month}/{date.year}" for date in x_dates]
    x_tick_positions = [data.index(date) for date in x_dates]

    return x_tick_positions, x_ticks
