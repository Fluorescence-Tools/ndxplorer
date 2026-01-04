"""Histogram plotting functionality extracted from plot_main.py.

This module provides 2D/3D histogram plotting capabilities for NDXplorer,
including weighted histograms, normalization, and caching optimizations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

from ..logging_config import logging
from ..core.histograms import Histogram1D, Histogram2D
from ..utils.histogram_manager import HistogramParams
from ..utils.histogram_helpers import (
    get_bins,
    is_data_ready,
    extract_histogram_params,
    should_recompute,
    save_cache,
    resolve_weights,
)

if False:  # pragma: no cover - type checking hints without runtime import
    from ..core.plot_main import NDXplorer


def plot_histogram(
    ndxplorer: "NDXplorer",
    dimension: str = "2d",
    weights: Optional[np.ndarray] = None,
    normed: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Plot histogram data for specified dimension.
    
    Args:
        ndxplorer: NDXplorer instance
        dimension: One of 'x', 'y', 'z', '2d'
        weights: Optional weight array
        normed: Whether to normalize the histogram
        **kwargs: Additional plotting options
        
    Returns:
        Tuple of (histogram_data, bin_edges_or_tuple)
        
    Raises:
        ValueError: If dimension is not supported
    """
    if not is_data_ready(ndxplorer):
        logging.warning("Data not ready for histogram plotting")
        return np.array([0]), np.array([0, 1])
    
    if dimension not in ndxplorer._histogram:
        logging.warning(f"No histogram data available for dimension '{dimension}'")
        return np.array([0]), np.array([0, 1])
    
    hist_data = ndxplorer._histogram[dimension]
    
    if dimension == "2d":
        # 2D histogram returns Histogram2D object
        if isinstance(hist_data, Histogram2D):
            return hist_data.H, (hist_data.x_edges, hist_data.y_edges)
        else:
            logging.error("Invalid 2D histogram data format")
            return np.array([[0]]), (np.array([0, 1]), np.array([0, 1]))
    else:
        # 1D histogram returns Histogram1D object
        if isinstance(hist_data, Histogram1D):
            return hist_data.edges, hist_data.counts
        else:
            logging.error(f"Invalid {dimension} histogram data format")
            return np.array([0, 1]), np.array([0])


def compute_2d_histogram(
    ndxplorer: "NDXplorer",
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_bins: Union[int, np.ndarray],
    y_bins: Union[int, np.ndarray],
    weights: Optional[np.ndarray] = None,
    density: bool = False
) -> Histogram2D:
    """
    Compute 2D histogram using clean histogram manager.
    
    Args:
        ndxplorer: NDXplorer instance
        x_data: X-axis data
        y_data: Y-axis data  
        x_bins: Number of bins or bin edges for X axis
        y_bins: Number of bins or bin edges for Y axis
        weights: Optional weight array
        density: Whether to compute density histogram
        
    Returns:
        Histogram2D object with clean histogram data
    """
    from ..utils.histogram_manager import get_histogram_manager
    
    manager = get_histogram_manager()
    
    try:
        return manager.compute_histogram_2d(
            x_data, y_data, x_bins, y_bins, weights=weights
        )
    except Exception as e:
        logging.error(f"Failed to compute 2D histogram: {e}")
        # Fallback: create minimal 2x2 histogram
        return Histogram2D(
            H=np.array([[0, 0], [0, 0]]),
            x_edges=np.array([0, 1]),
            y_edges=np.array([0, 1])
        )


def compute_1d_histogram(
    ndxplorer: "NDXplorer",
    data: np.ndarray,
    bins: Union[int, np.ndarray],
    weights: Optional[np.ndarray] = None,
    normed: bool = False
) -> Histogram1D:
    """
    Compute 1D histogram using clean histogram manager.
    
    Args:
        ndxplorer: NDXplorer instance
        data: Input data array
        bins: Number of bins or bin edges
        weights: Optional weight array
        normed: Whether to normalize the histogram
        
    Returns:
        Histogram1D object with clean histogram data
    """
    from ..utils.histogram_manager import get_histogram_manager
    
    manager = get_histogram_manager()
    
    try:
        return manager.compute_histogram_1d(
            data, bins, weights=weights, density=normed
        )
    except Exception as e:
        logging.error(f"Failed to compute 1D histogram: {e}")
        # Fallback: create minimal histogram
        return Histogram1D(
            edges=np.array([0.0, 1.0]),
            counts=np.array([0.0])
        )


def update_histogram_display(ndxplorer: "NDXplorer") -> None:
    """
    Update histogram plots in the UI after data changes.
    
    This function should be called whenever the underlying data
    or histogram parameters change.
    """
    if not is_data_ready(ndxplorer):
        logging.debug("Skipping histogram update: data not ready")
        return
    
    # Import here to avoid circular import issues
    from ..utils.histogram_helpers import extract_histogram_params
    
    try:
        params, _ = extract_histogram_params(ndxplorer)
    except (ValueError, TypeError) as e:
        logging.error(f"Failed to extract histogram parameters: {e}")
        return
    
    if not should_recompute(ndxplorer, params):
        logging.debug("Using cached histogram data")
        return
    
    # Update the count display
    try:
        ndxplorer.lineEditCountCurrent.setText(str(len(ndxplorer.x_values)))
    except Exception as e:
        logging.warning(f"Failed to update count display: {e}")
    
    # Trigger histogram computation through existing helpers
    from ..plotting.plot_update_helpers import update_histograms
    update_histograms(ndxplorer)


def get_histogram_statistics(
    ndxplorer: "NDXplorer",
    dimension: str = "2d"
) -> dict:
    """
    Compute basic statistics for histogram data.
    
    Args:
        ndxplorer: NDXplorer instance
        dimension: Histogram dimension ('x', 'y', 'z', '2d')
        
    Returns:
        Dictionary containing statistics (count, mean, std, min, max)
    """
    if dimension not in ndxplorer._histogram:
        return {}
    
    hist_data = ndxplorer._histogram[dimension]
    
    if dimension == "2d":
        if isinstance(hist_data, Histogram2D):
            return {
                "count": np.sum(hist_data.H),
                "mean": np.mean(hist_data.H),
                "std": np.std(hist_data.H),
                "min": np.min(hist_data.H),
                "max": np.max(hist_data.H),
                "shape": hist_data.shape
            }
    else:
        if isinstance(hist_data, Histogram1D):
            return {
                "count": np.sum(hist_data.counts),
                "mean": np.mean(hist_data.counts),
                "std": np.std(hist_data.counts),
                "min": np.min(hist_data.counts),
                "max": np.max(hist_data.counts),
                "bins": hist_data.n_bins
            }
    
    return {}
