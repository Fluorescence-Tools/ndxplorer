"""
Central data management for NDXplorer.

Separates data state and operations from UI concerns, providing:
- Single source of truth for data
- Unified cache management
- Clean interfaces for data access
- Testability without Qt dependencies
"""

from typing import Optional, Tuple, Dict, List
import numpy as np

from ...logging_config import logging
from ..data_source import DataSource
from .cache_coordinator import CacheCoordinator
from .selection_manager import SelectionManager


class DataManager:
    """
    Central data manager for NDXplorer.
    
    Manages:
    - Data source and computed columns
    - Value caching and retrieval
    - Axis value extraction
    - Cache coordination
    - Selection/masking
    """
    
    def __init__(self):
        self._default_data_source = self._create_default_data_source()
        self._data_source = DataSource()  # Start empty, will use default if needed
        
        self.cache = CacheCoordinator()
        self.selection = SelectionManager()
        
        self.constants = {}  # type: Dict[str, float]
        self.equations = []  # type: List[Dict[str, str]]
        
    def _create_default_data_source(self) -> DataSource:
        """Create default demo data source."""
        return DataSource(
            ["Tau (green)", "Proximity ratio", "r Experimental (green)"],
            np.vstack([
                np.random.multivariate_normal(
                    [4.1, 0.0, 0.05], 
                    [[0.1, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], 
                    size=500
                ),
                np.random.multivariate_normal(
                    [2.0, 0.5, 0.15], 
                    [[0.1, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], 
                    size=500
                )
            ])
        )
        
    @property
    def data_source(self) -> DataSource:
        """Get current data source (or default if empty)."""
        if self._data_source.empty:
            logging.debug("DataManager: Using default data source")
            return self._default_data_source
        return self._data_source
        
    @data_source.setter
    def data_source(self, v: DataSource) -> None:
        """Set data source and invalidate caches."""
        logging.info(f"DataManager: Setting data source with {v.values.shape[1] if not v.empty else 0} data points")
        self._data_source = v
        self.cache.invalidate_all()
        # Skip re-computation if already finalized in background
        if not getattr(v, 'is_computed', False):
            self._data_source.compute_columns(
                constants=self.constants,
                equations=self.equations
            )
        else:
            logging.info("DataManager: Skipping re-computation, data source is already computed.")
        
    @property
    def is_empty(self) -> bool:
        """Check if data source is empty (considering default fallback)."""
        # If actual data source is not empty, we have data
        if not self._data_source.empty:
            return False
        # If actual is empty, check if we have default
        return self._default_data_source is None or self._default_data_source.empty
        
    def invalidate_caches(self) -> None:
        """Invalidate all caches."""
        self.cache.invalidate_all()
        
    def compute_columns(self) -> None:
        """Recompute columns with current equations and constants."""
        logging.debug("DataManager: Computing columns with equations and constants")
        self._data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )
        
    def get_raw_values(self) -> np.ndarray:
        """
        Get raw data values without any filtering.
        
        Returns:
            2D array (n_params, n_points)
        """
        return self.data_source.values
        
    def get_filtered_values(self, 
                           mask_inf: Optional[bool] = None,
                           mask_nan: Optional[bool] = None) -> np.ndarray:
        """
        Get filtered values applying Inf/NaN masks.
        
        Uses cache when possible.
        
        Args:
            mask_inf: Override selection.mask_inf if provided
            mask_nan: Override selection.mask_nan if provided
            
        Returns:
            2D array (n_params, n_points) with masked values
        """
        if mask_inf is None:
            mask_inf = self.selection.mask_inf
        if mask_nan is None:
            mask_nan = self.selection.mask_nan
            
        # Check cache
        cached_inf = self.cache.get_cache_value('values_mask_inf')
        cached_nan = self.cache.get_cache_value('values_mask_nan')
        cached_values = self.cache.get_cache_value('filtered_values')
        
        if (cached_values is not None and 
            cached_inf == mask_inf and 
            cached_nan == mask_nan):
            logging.debug("DataManager: Using cached filtered values")
            return cached_values
            
        # Compute fresh
        logging.debug(f"DataManager: Computing filtered values (mask_inf={mask_inf}, mask_nan={mask_nan})")
        values = self.get_raw_values()
        
        if mask_inf or mask_nan:
            mask = np.zeros(values.shape[1], dtype=bool)
            if mask_inf:
                mask |= np.any(np.isinf(values), axis=0)
            if mask_nan:
                mask |= np.any(np.isnan(values), axis=0)
            values = values[:, ~mask]
            
        # Cache result
        self.cache.set_cache_value('filtered_values', values)
        self.cache.set_cache_value('values_mask_inf', mask_inf)
        self.cache.set_cache_value('values_mask_nan', mask_nan)
        
        return values
        
    def get_axis_values(self, 
                       axis: str, 
                       param_idx: int,
                       use_filtered: bool = True) -> np.ndarray:
        """
        Get values for specific axis (x, y, or z).
        
        Uses cache when possible.
        
        Args:
            axis: 'x', 'y', or 'z'
            param_idx: Parameter index to extract
            use_filtered: Whether to use filtered values (apply masks)
            
        Returns:
            1D array of values for the specified axis
        """
        axis = axis.lower()
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"Invalid axis: {axis}")
            
        # Check cache
        cached_values = self.cache.get_cache_value(f'{axis}_values')
        cached_idx = self.cache.get_cache_value(f'{axis}_param_idx')
        mask_id = self.cache.get_cache_value('values_mask_id')
        
        if (cached_values is not None and 
            cached_idx == param_idx and 
            mask_id is not None):
            logging.debug(f"DataManager: Using cached {axis} values")
            return cached_values
            
        # Compute fresh
        logging.debug(f"DataManager: Computing {axis} values for param_idx={param_idx}")
        if use_filtered:
            values = self.get_filtered_values()
        else:
            values = self.get_raw_values()
            
        axis_values = values[param_idx]
        
        # Cache result
        self.cache.set_cache_value(f'{axis}_values', axis_values)
        self.cache.set_cache_value(f'{axis}_param_idx', param_idx)
        
        return axis_values
        
    def get_value_mask(self) -> np.ndarray:
        """
        Compute combined mask from all active selections.
        
        Returns:
            Boolean mask (n_points,) where True means excluded
        """
        values = self.get_filtered_values()
        return self.selection.get_combined_mask(values)
        
    def get_stats(self) -> dict:
        """Get data and cache statistics."""
        raw_values = self.get_raw_values()
        filtered_values = self.get_filtered_values()
        
        return {
            'raw_points': raw_values.shape[1] if raw_values.ndim > 1 else len(raw_values),
            'filtered_points': filtered_values.shape[1] if filtered_values.ndim > 1 else len(filtered_values),
            'num_parameters': raw_values.shape[0] if raw_values.ndim > 1 else 1,
            'num_selections': len(self.selection.selections),
            'cache_stats': self.cache.get_stats(),
            'selection_state': self.selection.get_state(),
        }
