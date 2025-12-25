"""Histogram plotting functionality extracted from plot_main.py.

This module provides 2D/3D histogram plotting capabilities for NDXplorer,
including weighted histograms, normalization, and caching optimizations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

from ..logging_config import logging
from ..utils.histogram_helpers import (
    HistogramParams,
    get_bins,
    histogram_with_fallback,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot histogram data for specified dimension.
    
    Args:
        ndxplorer: NDXplorer instance
        dimension: One of 'x', 'y', 'z', '2d'
        weights: Optional weight array
        normed: Whether to normalize the histogram
        **kwargs: Additional plotting options
        
    Returns:
        Tuple of (histogram_data, bin_edges)
        
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
        # 2D histogram returns (H, x_edges, y_edges)
        if isinstance(hist_data, tuple) and len(hist_data) == 3:
            H, x_edges, y_edges = hist_data
            return H, (x_edges, y_edges)
        else:
            logging.error("Invalid 2D histogram data format")
            return np.array([[0]]), (np.array([0, 1]), np.array([0, 1]))
    else:
        # 1D histogram returns (counts, bin_edges)
        if isinstance(hist_data, tuple) and len(hist_data) == 2:
            counts, bin_edges = hist_data
            return counts, bin_edges
        else:
            logging.error(f"Invalid {dimension} histogram data format")
            return np.array([0]), np.array([0, 1])


def compute_2d_histogram(
    ndxplorer: "NDXplorer",
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_bins: Union[int, np.ndarray],
    y_bins: Union[int, np.ndarray],
    weights: Optional[np.ndarray] = None,
    density: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D histogram with proper error handling and fallbacks.
    
    Args:
        ndxplorer: NDXplorer instance
        x_data: X-axis data
        y_data: Y-axis data  
        x_bins: Number of bins or bin edges for X axis
        y_bins: Number of bins or bin edges for Y axis
        weights: Optional weight array
        density: Whether to compute density histogram
        
    Returns:
        Tuple of (histogram_2d, x_edges, y_edges)
    """
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            H, x_edges, y_edges = np.histogram2d(
                x_data, y_data, bins=[x_bins, y_bins], 
                weights=weights, density=density
            )
        return H, x_edges, y_edges
    except Exception as e:
        logging.error(f"Failed to compute 2D histogram: {e}")
        # Fallback: create minimal 2x2 histogram
        return np.array([[0, 0], [0, 0]]), np.array([0, 1]), np.array([0, 1])


def compute_1d_histogram(
    ndxplorer: "NDXplorer",
    data: np.ndarray,
    bins: Union[int, np.ndarray],
    weights: Optional[np.ndarray] = None,
    normed: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D histogram with automatic fallback bin generation.
    
    Args:
        ndxplorer: NDXplorer instance
        data: Input data array
        bins: Number of bins or bin edges
        weights: Optional weight array
        normed: Whether to normalize the histogram
        
    Returns:
        Tuple of (counts, bin_edges)
    """
    return histogram_with_fallback(data, bins, normed, weights)


def update_histogram_display(ndxplorer: "NDXplorer") -> None:
    """
    Update histogram plots in the UI after data changes.
    
    This function should be called whenever the underlying data
    or histogram parameters change.
    """
    if not is_data_ready(ndxplorer):
        logging.debug("Skipping histogram update: data not ready")
        return
    
    params = extract_histogram_params(ndxplorer)
    if not should_recompute(ndxplorer, params):
        logging.debug("Using cached histogram data")
        return
    
    # Update the count display
    try:
        ndxplorer.lineEditCountCurrent.setText(str(len(ndxplorer.x_values)))
    except Exception as e:
        logging.warning(f"Failed to update count display: {e}")
    
    # Trigger histogram computation through existing helpers
    from ..utils.histogram_helpers import update_histograms
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
        if isinstance(hist_data, tuple) and len(hist_data) == 3:
            H, _, _ = hist_data
            return {
                "count": np.sum(H),
                "mean": np.mean(H),
                "std": np.std(H),
                "min": np.min(H),
                "max": np.max(H),
                "shape": H.shape
            }
    else:
        if isinstance(hist_data, tuple) and len(hist_data) == 2:
            counts, _ = hist_data
            return {
                "count": np.sum(counts),
                "mean": np.mean(counts),
                "std": np.std(counts),
                "min": np.min(counts),
                "max": np.max(counts),
                "bins": len(counts)
            }
    
    return {}
