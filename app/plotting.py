"""Plot orchestration wrapper that adapts selections to src.wavelet_plots API."""

from typing import Any

from src import wavelet_plots


def generate_plot_ui(selection: Any) -> None:
    """Call into the existing plotting code using the Selection object."""
    file_dict = selection.file_dict
    transform_selection = selection.transform
    dwt_plot_selection = selection.dwt_plot_selection
    dwt_smooth_plot_order = selection.dwt_smooth_plot_order
    calculate_significance = selection.calculate_significance
    significance_level = selection.significance_level

    if transform_selection == "DWT" and len(file_dict) == 2:
        wavelet_plots.generate_plot(
            file_dict=file_dict,
            transform_selection=transform_selection,
            selected_data=selection.selected_data,
            dwt_plot_selection=dwt_plot_selection,
            dwt_smooth_plot_order=dwt_smooth_plot_order,
        )
    else:
        wavelet_plots.generate_plot(
            file_dict=file_dict,
            transform_selection=transform_selection,
            selected_data=selection.selected_data,
            dwt_plot_selection=dwt_plot_selection,
            dwt_smooth_plot_order=dwt_smooth_plot_order,
            calculate_significance=calculate_significance,
            significance_level=significance_level,
        )
