"""
Selection and masking management for NDXplorer data operations.

Handles user selections, masks for Inf/NaN values, and combined filtering.
"""

from typing import List, Dict, Any
import numpy as np
from ...logging_config import logging


class SelectionManager:
    """
    Manages data selections and masks.
    
    Tracks:
    - User selections (regions, points)
    - Inf/NaN masking flags
    - Combined filtering masks
    """
    
    def __init__(self):
        self.selections: List[Dict[str, Any]] = []
        self.mask_inf: bool = False
        self.mask_nan: bool = False
        
    def add_selection(self, selection: Dict[str, Any]) -> None:
        """
        Add a new selection.
        
        Args:
            selection: Selection dictionary with criteria
        """
        self.selections.append(selection)
        logging.debug(f"SelectionManager: Added selection (total: {len(self.selections)})")
        
    def clear_selections(self) -> None:
        """Clear all selections."""
        self.selections.clear()
        logging.debug("SelectionManager: Cleared all selections")
        
    def set_mask_inf(self, mask: bool) -> None:
        """
        Set whether to mask Inf values.
        
        Args:
            mask: True to mask Inf values
        """
        self.mask_inf = mask
        logging.debug(f"SelectionManager: mask_inf = {mask}")
        
    def set_mask_nan(self, mask: bool) -> None:
        """
        Set whether to mask NaN values.
        
        Args:
            mask: True to mask NaN values
        """
        self.mask_nan = mask
        logging.debug(f"SelectionManager: mask_nan = {mask}")
        
    def get_combined_mask(self, values: np.ndarray) -> np.ndarray:
        """
        Compute combined mask from all active selections.
        
        Args:
            values: Data array (n_params, n_points)
            
        Returns:
            Boolean mask (n_points,) where True means excluded
        """
        if values.size == 0:
            return np.array([], dtype=bool)
            
        n_points = values.shape[1] if values.ndim > 1 else len(values)
        mask = np.zeros(n_points, dtype=bool)
        
        # Apply Inf/NaN masks
        if self.mask_inf or self.mask_nan:
            if self.mask_inf:
                mask |= np.any(np.isinf(values), axis=0)
            if self.mask_nan:
                mask |= np.any(np.isnan(values), axis=0)
                
        # Apply user selections (if any)
        for selection in self.selections:
            # Apply selection logic here if needed
            pass
            
        return mask
        
    def get_state(self) -> dict:
        """
        Get current selection state.
        
        Returns:
            Dictionary with selection state
        """
        return {
            'num_selections': len(self.selections),
            'mask_inf': self.mask_inf,
            'mask_nan': self.mask_nan,
            'selections': self.selections
        }
