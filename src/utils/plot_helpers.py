"""For plotting transforms"""

from typing import Any

import matplotlib.pyplot as plt
import numpy.typing as npt

from matplotlib.figure import Figure

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
    # if len(transform_results_dict) == 2:
    #     logger.debug("len == 2")
    #     pass
    #     fig_size = kwargs["figsize"]
    #     fig, axs = plt.subplots(2, 1, figsize=fig_size)
    #     for dwt_data, dwt_result, ax in zip(
    #         transform_dict.values(), transform_results_dict.values(), axs
    #     ):
    #         dwt.plot_smoothing(
    #             dwt_result.smoothed_signal_dict,
    #             time_array,
    #             dwt_data.y_values,
    #             ascending=ascending,
    #             ax=ax,
    #             sharex=True,
    #         )
    #     return fig
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
            figsize=(15, 20),
            sharex=True,
        )
