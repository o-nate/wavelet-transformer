"""
Selection dataclass describing the user's choices plus a prepared `file_dict` mapping
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class Selection:
    """Holds user selection from sidebar UI"""

    transform: str
    selected_data: List[str]
    calculate_significance: bool
    significance_level: int
    dwt_plot_selection: Optional[str]
    dwt_smooth_plot_order: Optional[int]
    file_dict: Dict[str, Any]
    uploaded_files: List[Any]
