"""Helper functions for histogram computation and caching.

This module provides synchronous histogram computation functions that work
with both the background worker and immediate computation paths.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import numpy as np

from ..logging_config import logging


def compute_histograms_sync(
    data_source,
    histogram_params: Dict[str, Any],
    weights: Optional[np.ndarray] = None
) -> Dict[str, Tuple]:
    """
    Compute histograms synchronously (for immediate computation or background worker).
    
    Args:
        data_source: Data source containing values array
        histogram_params: Dictionary containing histogram computation parameters
        weights: Optional weight array
        
    Returns:
        Dictionary containing computed histogram data
    """
    import time
    start_time = time.time()
    
    try:
        # Extract data arrays
        x_data = data_source.values[histogram_params['x_idx'], :]
        y_data = data_source.values[histogram_params['y_idx'], :]
        z_data = data_source.values[histogram_params['z_idx'], :] if histogram_params.get('z_idx') is not None else None

        valid_indices = histogram_params.get('valid_indices')
        if valid_indices is not None:
            x_data = x_data[valid_indices]
            y_data = y_data[valid_indices]
            if z_data is not None:
                z_data = z_data[valid_indices]
            if weights is not None:
                weights = weights[valid_indices]
        
        result = {}
        
        # Check performance configuration for histogram backend
        from .performance_config import get_performance_config
        config = get_performance_config()
        
        if config.use_boost_histogram:
            # Use boost-histogram for maximum performance
            try:
                import boost_histogram as bh
                
                # Compute 1D histograms using boost-histogram
                x_hist = bh.Histogram(bh.axis.Regular(histogram_params['x_bins'], *histogram_params['x_range']))
                if weights is not None:
                    x_hist.fill(x_data, weight=weights)
                else:
                    x_hist.fill(x_data)
                result['x'] = (x_hist.axes[0].edges, x_hist.values())
                
                y_hist = bh.Histogram(bh.axis.Regular(histogram_params['y_bins'], *histogram_params['y_range']))
                if weights is not None:
                    y_hist.fill(y_data, weight=weights)
                else:
                    y_hist.fill(y_data)
                result['y'] = (y_hist.axes[0].edges, y_hist.values())
                
                # Compute Z histogram if available
                if z_data is not None:
                    z_hist = bh.Histogram(bh.axis.Regular(histogram_params['z_bins'], *histogram_params['z_range']))
                    if weights is not None:
                        z_hist.fill(z_data, weight=weights)
                    else:
                        z_hist.fill(z_data)
                    result['z'] = (z_hist.axes[0].edges, z_hist.values())
                
                # Compute 2D histogram using boost-histogram
                h2d = bh.Histogram(
                    bh.axis.Regular(histogram_params['x_bins_2d'], *histogram_params['x_range']),
                    bh.axis.Regular(histogram_params['y_bins_2d'], *histogram_params['y_range'])
                )
                if weights is not None:
                    h2d.fill(x_data, y_data, weight=weights)
                else:
                    h2d.fill(x_data, y_data)
                result['2d'] = (h2d.values(), h2d.axes[0].edges, h2d.axes[1].edges)
                
            except ImportError:
                logging.warning("boost-histogram not available, falling back to numpy")
                config.use_boost_histogram = False
        
        if not config.use_boost_histogram:
            # Use numpy histogram (fallback/default)
            logging.debug("Using numpy histogram computation")
            
            # Compute 1D histograms using numpy
            x_range = histogram_params['x_range']
            y_range = histogram_params['y_range']
            
            # Handle zero range data for 1D histograms
            if x_range[0] == x_range[1]:
                x_range = (x_range[0] - 0.5, x_range[1] + 0.5)
                logging.debug(f"Zero range in X for 1D histogram, expanding to {x_range}")
            
            if y_range[0] == y_range[1]:
                y_range = (y_range[0] - 0.5, y_range[1] + 0.5)
                logging.debug(f"Zero range in Y for 1D histogram, expanding to {y_range}")
            
            x_hist, x_edges = np.histogram(x_data, bins=histogram_params['x_bins'], range=x_range, weights=weights)
            result['x'] = (x_edges, x_hist)  # Store as (edges, counts)
            
            y_hist, y_edges = np.histogram(y_data, bins=histogram_params['y_bins'], range=y_range, weights=weights)
            result['y'] = (y_edges, y_hist)  # Store as (edges, counts)
            
            # Compute Z histogram if available
            if z_data is not None:
                z_range = histogram_params['z_range']
                if z_range[0] == z_range[1]:
                    z_range = (z_range[0] - 0.5, z_range[1] + 0.5)
                    logging.debug(f"Zero range in Z for 1D histogram, expanding to {z_range}")
                
                z_hist, z_edges = np.histogram(z_data, bins=histogram_params['z_bins'], range=z_range, weights=weights)
                result['z'] = (z_edges, z_hist)  # Store as (edges, counts)
            
            # Compute 2D histogram using numpy
            x_range = histogram_params['x_range']
            y_range = histogram_params['y_range']
            
            # Handle zero range data
            if x_range[0] == x_range[1]:
                x_range = (x_range[0] - 0.5, x_range[1] + 0.5)
                logging.debug(f"Zero range in X for background histogram, expanding to {x_range}")
            
            if y_range[0] == y_range[1]:
                y_range = (y_range[0] - 0.5, y_range[1] + 0.5)
                logging.debug(f"Zero range in Y for background histogram, expanding to {y_range}")
            
            H, x_edges, y_edges = np.histogram2d(
                x_data, y_data, 
                bins=[histogram_params['x_bins_2d'], histogram_params['y_bins_2d']],
                range=[x_range, y_range],
                weights=weights
            )
            # CRITICAL: np.histogram2d returns H with shape (nx, ny) but Histogram2D expects (ny, nx)
            result['2d'] = (H.T, x_edges, y_edges)
        
        # Add metadata
        result['_count'] = len(x_data)
        result['_computation_time'] = time.time() - start_time
        params_copy = histogram_params.copy()
        # Do not persist large index arrays inside cache metadata
        params_copy.pop('valid_indices', None)
        result['_params'] = params_copy
        if weights is not None:
            result['_has_weights'] = True
        
        logging.debug(f"Synchronous histogram computation completed in {result['_computation_time']:.3f}s")
        return result
        
    except Exception as e:
        logging.error(f"Histogram computation failed: {e}")
        raise


def extract_histogram_params_from_plot_control(plot_control) -> Dict[str, Any]:
    """
    Extract histogram computation parameters from plot control widget.
    
    Args:
        plot_control: SurfacePlotWidget instance
        
    Returns:
        Dictionary of histogram parameters
    """
    try:
        params = {
            'x_idx': plot_control.p1[0],
            'y_idx': plot_control.p2[0],
            'z_idx': plot_control.p3[0] if plot_control.p3[0] >= 0 else None,
            'x_bins': plot_control.n_xhist_1d,
            'y_bins': plot_control.n_yhist_1d,
            'z_bins': plot_control.n_zhist_1d,
            'x_bins_2d': plot_control.n_xhist_2d,
            'y_bins_2d': plot_control.n_yhist_2d,
            'x_range': (plot_control.xmin, plot_control.xmax),
            'y_range': (plot_control.ymin, plot_control.ymax),
            'z_range': (plot_control.zmin, plot_control.zmax),
        }
        
        # Add weight information if enabled
        if plot_control.weight_enabled:
            params['weight_parameter'] = plot_control.weight_parameter
            params['has_weights'] = True
        
        return params
    except Exception as e:
        logging.error(f"Failed to extract histogram params: {e}")
        return {}


def should_recompute_histograms(plot_control, current_params: Dict[str, Any]) -> bool:
    """
    Check if histograms need to be recomputed based on parameter changes.
    
    Args:
        plot_control: SurfacePlotWidget instance
        current_params: Current histogram parameters
        
    Returns:
        True if recomputation is needed
    """
    try:
        # Check if we have cached data with matching parameters
        if hasattr(plot_control.parent, '_histogram') and plot_control.parent._histogram:
            cached_params = plot_control.parent._histogram.get('_params', {})
            
            # Compare key parameters
            key_params = [
                'x_idx', 'y_idx', 'z_idx',
                'x_bins', 'y_bins', 'z_bins',
                'x_bins_2d', 'y_bins_2d',
                'x_range', 'y_range', 'z_range',
                # Filtering/selections (may be injected by caller)
                'valid_idx_hash', 'valid_idx_count', 'selection_hash',
            ]
            
            for param in key_params:
                if cached_params.get(param) != current_params.get(param):
                    logging.debug(f"Parameter {param} changed: {cached_params.get(param)} -> {current_params.get(param)}")
                    return True
            
            # Check weight changes
            if plot_control.weight_enabled != cached_params.get('has_weights', False):
                logging.debug("Weight setting changed")
                return True
            
            if (plot_control.weight_enabled and 
                plot_control.weight_parameter != cached_params.get('weight_parameter')):
                logging.debug("Weight parameter changed")
                return True
            
            return False
        
        return True  # No cached data, need to compute
        
    except Exception as e:
        logging.error(f"Error checking recompute condition: {e}")
        return True  # Safer to recompute


def resolve_weights(plot_control, data_source) -> Optional[np.ndarray]:
    """
    Resolve weight array based on plot control settings.
    
    Args:
        plot_control: SurfacePlotWidget instance
        data_source: Data source containing parameter values
        
    Returns:
        Weight array or None if weighting is disabled
    """
    try:
        if not plot_control.weight_enabled:
            return None
        
        weight_param_name = plot_control.weight_parameter
        if not weight_param_name or weight_param_name == "None":
            return None
        
        # Find weight parameter index
        param_names = data_source.parameter_names
        if weight_param_name not in param_names:
            logging.warning(f"Weight parameter '{weight_param_name}' not found")
            return None
        
        weight_idx = param_names.index(weight_param_name)
        weights = data_source.values[weight_idx, :]
        
        # Handle negative or zero weights
        weights = np.where(weights <= 0, 0, weights)
        
        logging.debug(f"Using weights from parameter '{weight_param_name}'")
        return weights
        
    except Exception as e:
        logging.error(f"Failed to resolve weights: {e}")
        return None
