"""
Histogram helpers for ndxplorer.

Provides utilities for histogram computation, caching, and parameter management.
"""

from __future__ import annotations

import hashlib
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..logging_config import logging
from ..utils.performance_optimizations import (
    compute_histogram2d_adaptive,
    compute_histogram1d_adaptive,
    downsample_for_display,
    get_performance_monitor,
)

if False:  # pragma: no cover - type checking hints without runtime import
    from ..core.plot_main import NDXplorer


def get_bins(plot_control, arange, scale, n_1d, n_2d):
    """Return 1D/2D bin edges for a given axis."""
    logging.debug("get_bins: arange=%s scale=%s n_1d=%s n_2d=%s", arange, scale, n_1d, n_2d)
    xmin, xmax = arange
    if scale == "log":
        if xmin <= 0:
            xmin = 1e-6
        if xmax <= 0:
            xmax = 1e-6
        x_func = np.logspace
        x_start = np.log10(xmin)
        x_stop = np.log10(xmax)
    else:
        x_func = np.linspace
        x_start = xmin
        x_stop = xmax
    x_bins_1d = x_func(x_start, x_stop, n_1d)
    x_bins_2d = x_func(x_start, x_stop, n_2d)
    return x_bins_1d, x_bins_2d


def are_bins_valid(bins) -> bool:
    """Return True when bins is a strictly increasing 1D array."""
    try:
        if bins is None:
            return False
        arr = np.asarray(bins)
        if arr.ndim != 1 or arr.size < 2:
            return False
        return np.all(np.diff(arr) > 0)
    except Exception:
        return False


def sanitize_bins(
    bins,
    data: np.ndarray,
    default_count: int = 50,
    scale: str = "linear",
) -> np.ndarray:
    """Ensure bins are a strictly increasing 1D array.
    
    Parameters
    ----------
    bins : array-like
        Bin edges to validate
    data : np.ndarray
        Data array to generate fallback bins from
    default_count : int
        Number of bins to generate in fallback
    scale : str
        'linear' or 'log' - determines spacing of fallback bins
    """
    try:
        if are_bins_valid(bins):
            return np.asarray(bins)
        if data is not None and len(data) > 0 and np.isfinite(data).any():
            fd = data[np.isfinite(data)]
            if fd.size == 0:
                return np.array([0.0, 1.0])
            vmin = np.min(fd)
            vmax = np.max(fd)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return np.array([0.0, 1.0])
            if vmax == vmin:
                eps = 1e-9 if vmin == 0 else abs(vmin) * 1e-9
                vmin -= eps
                vmax += eps
            n = int(default_count) if default_count and default_count > 0 else 50
            
            # Generate log-spaced or linear-spaced bins based on scale
            if scale == "log":
                if vmin <= 0:
                    vmin = 1e-6
                if vmax <= 0:
                    vmax = 1e-6
                return np.logspace(np.log10(vmin), np.log10(vmax), n + 1)
            else:
                return np.linspace(vmin, vmax, n + 1)
        return np.array([0.0, 1.0])
    except Exception:
        return np.array([0.0, 1.0])


def is_data_ready(ndxplorer: "NDXplorer") -> bool:
    """Return True if data and axis selections are ready for histogram computation."""
    try:
        # Use data_source property instead of _data_source
        data_source = ndxplorer.data_source
        if data_source is None or data_source.empty:
            return False
        values = data_source.values
        if values is None or not hasattr(values, "shape"):
            return False
        if values.shape[0] < 3:
            return False
        p1 = getattr(ndxplorer.plot_control, "p1", (0, ""))[0]
        p2 = getattr(ndxplorer.plot_control, "p2", (1, ""))[0]
        p3 = getattr(ndxplorer.plot_control, "p3", (2, ""))[0]
        nrows = values.shape[0]
        if not (0 <= p1 < nrows and 0 <= p2 < nrows and 0 <= p3 < nrows):
            return False
        _ = ndxplorer.x_values
        _ = ndxplorer.y_values
        _ = ndxplorer.z_values
        return True
    except Exception:
        return False


def extract_histogram_params(ndxplorer: "NDXplorer") -> HistogramParams:
    """Collect parameters that determine histogram cache validity."""
    from .histogram_manager import HistogramParams
    
    # Get parameter names from data source
    x_param = ndxplorer.plot_control.x_label
    y_param = ndxplorer.plot_control.y_label
    z_param = ndxplorer.plot_control.z_label if hasattr(ndxplorer.plot_control, 'z_label') else ""
    
    # Get parameter indices for compatibility with background histogram system
    p1_idx = getattr(ndxplorer.plot_control, 'p1', (0, ""))[0]
    p2_idx = getattr(ndxplorer.plot_control, 'p2', (1, ""))[0]
    p3_idx = getattr(ndxplorer.plot_control, 'p3', (2, ""))[0]
    
    use_weights = hasattr(ndxplorer, "checkBoxWeight") and ndxplorer.checkBoxWeight.isChecked()
    weight_param = (
        ndxplorer.comboBoxWeight.currentText()
        if use_weights and hasattr(ndxplorer, "comboBoxWeight")
        else ""
    )
    z_enabled = hasattr(ndxplorer, "groupBox_3") and ndxplorer.groupBox_3.isChecked()
    mask_id = getattr(ndxplorer, "_cached_values_mask_id", None)
    
    # Get density settings
    normed_x = getattr(ndxplorer.plot_control, 'normed_hist_x', False)
    normed_y = getattr(ndxplorer.plot_control, 'normed_hist_y', False)
    normed_z = getattr(ndxplorer.plot_control, 'normed_hist_z', False)
    
    # Get bins as arrays (preserve actual values for computation)
    x_bins_1d_arr, x_bins_2d_arr = ndxplorer.get_x_bins()
    y_bins_1d_arr, y_bins_2d_arr = ndxplorer.get_y_bins()
    z_bins_1d_arr, _ = ndxplorer.get_z_bins()
    
    # Convert bins to strings only for cache key consistency
    x_bins_1d = str(x_bins_1d_arr)
    y_bins_1d = str(y_bins_1d_arr)
    z_bins_1d = str(z_bins_1d_arr) if z_enabled else None
    x_bins_2d = str(x_bins_2d_arr)
    y_bins_2d = str(y_bins_2d_arr)
    
    # Compute data hash for cache invalidation
    data_hash = None
    try:
        if hasattr(ndxplorer, 'values') and ndxplorer.values is not None:
            # Hash the data shape and first/last few elements for quick change detection
            data = ndxplorer.values
            if len(data) > 0:
                sample_data = np.concatenate([
                    data[:min(5, len(data))],
                    data[-min(5, len(data)):]
                ])
                data_hash = hashlib.md5(f"{data.shape}_{sample_data.tobytes()}".encode()).hexdigest()[:8]
    except Exception:
        pass
    
    params = HistogramParams(
        x_param=x_param,
        y_param=y_param,
        z_param=z_param if z_enabled else None,
        x_bins_1d=x_bins_1d,
        y_bins_1d=y_bins_1d,
        z_bins_1d=z_bins_1d,
        x_bins_2d=x_bins_2d,
        y_bins_2d=y_bins_2d,
        use_weights=use_weights,
        weight_param=weight_param,
        z_enabled=z_enabled,
        mask_id=mask_id,
        data_hash=data_hash,
        normed_x=normed_x,
        normed_y=normed_y,
        normed_z=normed_z
    )
    
    # Add index parameters for background histogram compatibility
    params_dict = params.to_dict()
    params_dict['x_idx'] = p1_idx
    params_dict['y_idx'] = p2_idx
    params_dict['z_idx'] = p3_idx
    
    # Add actual bin arrays for computation
    params_dict['x_bins_1d_arr'] = x_bins_1d_arr
    params_dict['x_bins_2d_arr'] = x_bins_2d_arr
    params_dict['y_bins_1d_arr'] = y_bins_1d_arr
    params_dict['y_bins_2d_arr'] = y_bins_2d_arr
    params_dict['z_bins_1d_arr'] = z_bins_1d_arr
    
    # Add old-style bin parameters for background histogram compatibility
    try:
        # Convert string bins back to integers for background system
        x_bins_1d_int = int(x_bins_1d) if x_bins_1d.isdigit() else 50
        y_bins_1d_int = int(y_bins_1d) if y_bins_1d.isdigit() else 50
        x_bins_2d_int = int(x_bins_2d) if x_bins_2d.isdigit() else 50
        y_bins_2d_int = int(y_bins_2d) if y_bins_2d.isdigit() else 50
        
        params_dict['x_bins'] = x_bins_1d_int
        params_dict['y_bins'] = y_bins_1d_int
        params_dict['z_bins'] = int(z_bins_1d) if z_bins_1d and z_bins_1d.isdigit() else 50
        params_dict['x_bins_2d'] = x_bins_2d_int
        params_dict['y_bins_2d'] = y_bins_2d_int
        
        # Add range parameters
        if hasattr(ndxplorer.plot_control, 'xmin') and hasattr(ndxplorer.plot_control, 'xmax'):
            params_dict['x_range'] = (ndxplorer.plot_control.xmin, ndxplorer.plot_control.xmax)
        else:
            params_dict['x_range'] = (0, 256)
            
        if hasattr(ndxplorer.plot_control, 'ymin') and hasattr(ndxplorer.plot_control, 'ymax'):
            params_dict['y_range'] = (ndxplorer.plot_control.ymin, ndxplorer.plot_control.ymax)
        else:
            params_dict['y_range'] = (0, 256)
            
        params_dict['z_range'] = (0, 50)  # Default range for Z
        
    except Exception as e:
        logging.warning(f"Error setting old-style parameters: {e}")
        # Fallback values
        params_dict['x_bins'] = 50
        params_dict['y_bins'] = 50
        params_dict['z_bins'] = 50
        params_dict['x_bins_2d'] = 50
        params_dict['y_bins_2d'] = 50
        params_dict['x_range'] = (0, 256)
        params_dict['y_range'] = (0, 256)
        params_dict['z_range'] = (0, 50)
    
    return params, params_dict


def should_recompute(ndxplorer: "NDXplorer", params) -> bool:
    """Return True when cached histograms no longer match current parameters."""
    from .histogram_manager import HistogramParams
    
    cached = getattr(ndxplorer, "_cached_hist_params", None)
    if cached is None:
        return True
    
    # Convert to dict for comparison
    current_dict = params.to_dict() if hasattr(params, 'to_dict') else params.__dict__
    cached_dict = cached if isinstance(cached, dict) else cached.__dict__
    
    # Compare key parameters
    keys_to_check = ['x_param', 'y_param', 'z_param', 'use_weights', 'weight_param', 
                    'z_enabled', 'mask_id', 'x_bins_1d', 'y_bins_1d', 'z_bins_1d',
                    'x_bins_2d', 'y_bins_2d', 'data_hash', 'normed_x', 'normed_y', 'normed_z']
    
    for key in keys_to_check:
        if current_dict.get(key) != cached_dict.get(key):
            return True
    
    return False


def save_cache(ndxplorer: "NDXplorer", params) -> None:
    """Store histogram parameters for cache comparisons."""
    # Store as dict for easy comparison
    if hasattr(params, 'to_dict'):
        ndxplorer._cached_hist_params = params.to_dict()
    else:
        ndxplorer._cached_hist_params = params.__dict__.copy()


def resolve_weights(ndxplorer: "NDXplorer", use_weights: bool, d1) -> Optional[np.ndarray]:
    """Return weight array matching d1 length or None (float32 for memory efficiency)."""
    if not use_weights:
        return None
    # Use the same data source as x_values and y_values to ensure consistency
    data_source = ndxplorer.data_source
    if data_source is not None and hasattr(data_source, "parameter_names"):
        param_names = data_source.parameter_names
        if weight_param in param_names:
            weight_idx = param_names.index(weight_param)
    
    if weight_idx < 0:
        logging.warning("Weight parameter '%s' not found in data source. Disabling weights.", weight_param)
        return None
    
    # Use the same data source as x_values and y_values to ensure consistency
    if hasattr(ndxplorer, 'data_manager'):
        # Use data_manager to get filtered weight values (same as x_values/y_values)
        try:
            weight_values = ndxplorer.data_manager.get_axis_values('x', weight_idx, use_filtered=True)
            weight_values = weight_values.astype(np.float32)
        except Exception as e:
            logging.warning("Failed to get filtered weights from data manager: %s", e)
            return None
    else:
        # Fallback: use ndxplorer.values (should be same filtered data)
        weight_values = ndxplorer.values[weight_idx].astype(np.float32)
    
    if len(weight_values) != len(d1):
        logging.warning(
            "Weights array shape (%d) doesn't match data array shape (%d). Disabling weights.",
            len(weight_values),
            len(d1),
        )
        return None
    return weight_values




