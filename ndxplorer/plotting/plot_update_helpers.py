"""Helper routines for updating histograms/plots/spinboxes in NDXplorer."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy import QtWidgets

try:
    from qwt.plot import QwtPlot
    QWT_AVAILABLE = True
except ImportError:
    QWT_AVAILABLE = False
    QwtPlot = None

from ..logging_config import logging

try:  # Optional dependency
    import hdbscan as _hdbscan  # type: ignore
except ImportError:  # pragma: no cover - optional dep
    _hdbscan = None

hdbscan = _hdbscan

from ..utils.performance import compute_percentile_range_optimized

def _compute_selection_hash(selections):
    """Compute a hash of current selections for cache validation."""
    if not selections:
        return hash(None)
    
    # Create a tuple of selection properties that affect histogram computation
    selection_data = []
    for sel in selections:
        sel_type = type(sel).__name__
        if hasattr(sel, 'parameter_idx'):
            param_idx = sel.parameter_idx
        else:
            param_idx = None
        enabled = getattr(sel, 'enabled', True)
        
        # For different selection types, gather relevant properties
        if sel_type == 'RectangularDataSelection':
            data = (sel_type, param_idx, getattr(sel, 'lower', None), getattr(sel, 'upper', None), 
                   getattr(sel, 'invert', False), enabled)
        elif sel_type == 'Gauss2DSelection':
            data = (sel_type, getattr(sel, 'idx1', None), getattr(sel, 'idx2', None),
                   getattr(sel, 'sigma', None), getattr(sel, 'invert', False), enabled)
        elif sel_type == 'MaskDataSelection':
            # Use selection_id for mask selections
            data = (sel_type, getattr(sel, 'selection_id', None), enabled)
        else:
            # Generic fallback
            data = (sel_type, str(sel), enabled)
        
        selection_data.append(data)
    
    return hash(tuple(selection_data))


def update_histograms(ndxplorer) -> None:
    logging.info("update_histograms called")
    
    # Enhanced data loading check with multiple flags
    loading_flags = [
        getattr(ndxplorer.plot_control, '_loading_data', False),
        getattr(ndxplorer, '_loading_data', False),
        getattr(ndxplorer.plot_control, '_background_computation_pending', False)
    ]
    
    if any(loading_flags):
        logging.info("Skipping update_histograms: data loading or computation in progress (flags=%s)", loading_flags)
        return
    
    data_ready = ndxplorer.is_data_ready()
    logging.info(f"update_histograms: is_data_ready={data_ready}")
    if not data_ready:
        # Log why data isn't ready
        try:
            p1 = ndxplorer.plot_control.p1
            p2 = ndxplorer.plot_control.p2
            logging.info(f"  p1 (X axis): {p1}")
            logging.info(f"  p2 (Y axis): {p2}")
        except Exception as exc:
            logging.info(f"  Could not get axis info: {exc}")
        logging.info("Skipping update_histograms: data/axes not ready")
        return

    # Use the new background computation system if available
    has_bg_method = hasattr(ndxplorer.plot_control, 'compute_histograms_background')
    has_bg_enabled_flag = hasattr(ndxplorer.plot_control, '_background_computation_enabled')
    bg_enabled = getattr(ndxplorer.plot_control, '_background_computation_enabled', False)
    logging.info(f"Background computation check: has_method={has_bg_method}, has_flag={has_bg_enabled_flag}, enabled={bg_enabled}")
    
    if (hasattr(ndxplorer.plot_control, 'compute_histograms_background') and 
        hasattr(ndxplorer.plot_control, '_background_computation_enabled') and 
        ndxplorer.plot_control._background_computation_enabled):
        
        try:
            logging.debug("Attempting to use background histogram computation system")
            # Import the helper functions
            from ..utils.histogram_helpers import (
                extract_histogram_params,
                is_data_ready,
                resolve_weights,
                sanitize_bins,
            )
            
            # Extract histogram parameters
            params, histogram_params = extract_histogram_params(ndxplorer)
            logging.debug(f"Extracted histogram params: x_bins={histogram_params.get('x_bins')}, y_bins={histogram_params.get('y_bins')}")

            # Ensure background histograms respect current filtering (NaN/Inf masking, selections, clustering, frames)
            try:
                mask = ndxplorer.value_mask
                if mask is not None:
                    valid_indices = np.flatnonzero(~mask)
                    histogram_params['valid_indices'] = valid_indices
                    histogram_params['valid_idx_count'] = int(valid_indices.size)
                    histogram_params['valid_idx_hash'] = hash(valid_indices.tobytes())
            except Exception as exc:
                logging.debug("Could not compute valid_indices for background histograms: %s", exc)

            try:
                selections = ndxplorer.plot_control.get_selections()
                histogram_params['selection_hash'] = _compute_selection_hash(selections)
            except Exception:
                histogram_params['selection_hash'] = None
            
            # Always compute live histograms (no caching)
            
            # Resolve weights
            weights = resolve_weights(ndxplorer, params.use_weights, ndxplorer.x_values)
            
            # Use background computation
            ndxplorer.plot_control.compute_histograms_background(histogram_params, weights)
            logging.info("Successfully scheduled background histogram computation")
            return
            
        except Exception as e:
            import traceback
            logging.warning(f"Background histogram system failed, falling back to immediate computation: {e}")
            logging.debug(f"Background computation error traceback:\n{traceback.format_exc()}")
            # Fall back to the original immediate computation
    
    # Original immediate computation as fallback
    # Only run if background computation is not pending
    if not getattr(ndxplorer.plot_control, '_background_computation_pending', False):
        _update_histograms_immediate(ndxplorer)
    else:
        logging.debug("Skipping immediate computation - background computation pending")


def _update_histograms_immediate(ndxplorer) -> None:
    """Immediate histogram computation using clean histogram manager."""
    from ..utils.histogram_manager import get_histogram_manager, HistogramParams
    from ..core.histograms import Histogram1D, Histogram2D
    from ..utils.histogram_helpers import (
        extract_histogram_params,
        is_data_ready,
        resolve_weights,
        sanitize_bins,
    )
    
    # Extract parameters using clean helper
    params, params_dict = extract_histogram_params(ndxplorer)
    manager = get_histogram_manager()
    
    # Always compute live histograms (no caching)
    
    # Compute new histograms
    if not is_data_ready(ndxplorer):
        logging.info("Skipping _update_histograms_immediate: data/axes not ready")
        return
    
    d1 = ndxplorer.x_values
    d2 = ndxplorer.y_values
    d3 = ndxplorer.z_values
    ndxplorer.lineEditCountCurrent.setText(str(len(d1)))
    
    # Get bins
    x_bins_1d, x_bins_2d = ndxplorer.get_x_bins()
    y_bins_1d, y_bins_2d = ndxplorer.get_y_bins()
    z_bins_1d, _ = ndxplorer.get_z_bins()
    
    # Get scale settings
    x_scale = getattr(ndxplorer.plot_control, "scale_x", "linear")
    y_scale = getattr(ndxplorer.plot_control, "scale_y", "linear")
    z_scale = getattr(ndxplorer.plot_control, "scale_z", "linear")
    
    # Sanitize bins
    x_bins_1d = sanitize_bins(x_bins_1d, d1, default_count=getattr(ndxplorer.plot_control, "n_xhist_1d", 50), scale=x_scale)
    y_bins_1d = sanitize_bins(y_bins_1d, d2, default_count=getattr(ndxplorer.plot_control, "n_yhist_1d", 50), scale=y_scale)
    z_bins_1d = sanitize_bins(z_bins_1d, d3, default_count=getattr(ndxplorer.plot_control, "n_zhist_1d", 50), scale=z_scale)
    x_bins_2d = sanitize_bins(x_bins_2d, d1, default_count=getattr(ndxplorer.plot_control, "n_xhist_2d", 50), scale=x_scale)
    y_bins_2d = sanitize_bins(y_bins_2d, d2, default_count=getattr(ndxplorer.plot_control, "n_yhist_2d", 50), scale=y_scale)
    
    # Resolve weights
    weights = resolve_weights(ndxplorer, params.use_weights, d1)
    
    try:
        # Debug: Check density settings
        density_x = ndxplorer.plot_control.normed_hist_x
        density_y = ndxplorer.plot_control.normed_hist_y
        density_z = ndxplorer.plot_control.normed_hist_z
        logging.info(f"[HELPERS] Density settings - X: {density_x}, Y: {density_y}, Z: {density_z}")
        
        # Compute 1D histograms using clean manager
        hist_x = manager.compute_histogram_1d(
            d1, x_bins_1d, weights=weights,
            density=density_x,
            params=params
        )
        
        hist_y = manager.compute_histogram_1d(
            d2, y_bins_1d, weights=weights,
            density=density_y,
            params=params
        )
        
        # Store clean histograms
        ndxplorer._histogram["x"] = hist_x
        ndxplorer._histogram["y"] = hist_y
        
        # Z histogram if enabled
        if params.z_enabled:
            z_weights = None
            if weights is not None:
                weight_param = ndxplorer.comboBoxWeight.currentText()
                z_param = ndxplorer.plot_control.z_label
                if weight_param != z_param:
                    z_weights = weights
            
            logging.info(f"[HELPERS] Computing Z histogram with density: {density_z}")
            hist_z = manager.compute_histogram_1d(
                d3, z_bins_1d, weights=z_weights,
                density=density_z,
                params=params
            )
            ndxplorer._histogram["z"] = hist_z
        else:
            ndxplorer._histogram.pop("z", None)
        
        # Compute 2D histogram using clean manager
        hist_2d = manager.compute_histogram_2d(
            d1, d2, x_bins_2d, y_bins_2d, weights=weights, params=params
        )
        
        logging.info(f"[HELPERS] Freshly computed 2D histogram:")
        logging.info(f"[HELPERS]   H shape: {hist_2d.H.shape}")
        logging.info(f"[HELPERS]   x_edges: {len(hist_2d.x_edges)}, y_edges: {len(hist_2d.y_edges)}")
        logging.info(f"[HELPERS]   Shape check: H[0]={hist_2d.H.shape[0]} vs y-1={len(hist_2d.y_edges)-1}, H[1]={hist_2d.H.shape[1]} vs x-1={len(hist_2d.x_edges)-1}")
        logging.info(f"[HELPERS]   Validation: {hist_2d.validate()}")
        
        # Store clean histogram
        ndxplorer._histogram["2d"] = hist_2d
        
        logging.info(f"[HELPERS] Stored histogram in _histogram['2d']: shape {hist_2d.shape}")
        
        # Update marginal plots after histogram computation
        _update_marginal_plots_from_cache(ndxplorer)
        
        # Update 2D plot
        if hasattr(ndxplorer, 'update_2d_plot'):
            ndxplorer.update_2d_plot()
        if hasattr(ndxplorer, 'g_2dplot'):
            ndxplorer.g_2dplot.replot()
        
    except Exception as e:
        logging.error(f"Clean histogram computation failed: {e}")
        # Set empty histograms
        ndxplorer._histogram["x"] = Histogram1D(edges=np.array([0.0, 1.0]), counts=np.array([0]))
        ndxplorer._histogram["y"] = Histogram1D(edges=np.array([0.0, 1.0]), counts=np.array([0]))
        ndxplorer._histogram["2d"] = Histogram2D(H=np.zeros((1, 1)), x_edges=np.array([0.0, 1.0]), y_edges=np.array([0.0, 1.0]))
        ndxplorer._histogram.pop("z", None)


def _update_marginal_plots_from_cache(ndxplorer) -> None:
    """Update marginal distribution plots from cached histogram data."""
    logging.info("Starting marginal plot update")
    
    # Check if deferred init is done and plot objects exist
    if not getattr(ndxplorer, "_deferred_init_done", False):
        logging.info("Skipping marginal plot update: deferred init not done yet")
        # Try to trigger deferred init if not done
        if hasattr(ndxplorer, '_deferred_init'):
            try:
                ndxplorer._deferred_init()
                logging.info("Triggered deferred init from marginal plot update")
                # Retry the update after init
                if getattr(ndxplorer, "_deferred_init_done", False):
                    logging.info("Deferred init completed, retrying marginal plot update")
                    return _update_marginal_plots_from_cache(ndxplorer)
            except Exception as e:
                logging.warning(f"Failed to trigger deferred init: {e}")
        return
    
    # Check if marginal plot objects exist
    has_x_plot = hasattr(ndxplorer, 'g_xplot') and ndxplorer.g_xplot is not None
    has_y_plot = hasattr(ndxplorer, 'g_yplot') and ndxplorer.g_yplot is not None
    has_z_plot = hasattr(ndxplorer, 'g_zplot') and ndxplorer.g_zplot is not None
    has_x_hist = hasattr(ndxplorer, 'g_xhist_m') and ndxplorer.g_xhist_m is not None
    has_y_hist = hasattr(ndxplorer, 'g_yhist_m') and ndxplorer.g_yhist_m is not None
    has_z_hist = hasattr(ndxplorer, 'g_zhist_m') and ndxplorer.g_zhist_m is not None
    
    logging.info(f"Marginal plot objects: x_plot={has_x_plot}, y_plot={has_y_plot}, z_plot={has_z_plot}")
    logging.info(f"Marginal hist objects: x_hist={has_x_hist}, y_hist={has_y_hist}, z_hist={has_z_hist}")
    
    if not has_x_hist or not has_y_hist:
        logging.warning("Missing marginal histogram objects - cannot update marginals")
        return
    
    try:
        # Update X marginal
        x_hist = ndxplorer._histogram.get("x")
        if x_hist is not None:
            x_bin_edges = x_hist.edges
            x_counts = x_hist.counts
            
            # Ensure arrays have same length to prevent IndexError
            x_edges_for_plot = x_bin_edges[1:]
            if len(x_edges_for_plot) != len(x_counts):
                min_len = min(len(x_edges_for_plot), len(x_counts))
                x_edges_for_plot = x_edges_for_plot[:min_len]
                x_counts = x_counts[:min_len]
            
            # Additional validation for Steps curve style
            if len(x_edges_for_plot) == 0 or len(x_counts) == 0:
                logging.debug("Skipping X marginal update: empty data")
                # Set dummy data to prevent display issues
                ndxplorer.g_xhist_m.set_data([0, 1], [0, 0])
            else:
                # For Steps curves, ensure data is valid for current scale
                try:
                    ndxplorer.g_xhist_m.set_data(x_edges_for_plot, x_counts)
                except Exception as e:
                    logging.warning(f"Error setting X marginal data: {e}, using dummy data")
                    ndxplorer.g_xhist_m.set_data([0, 1], [0, 0])
            
            _autoscale_horizontal_hist(ndxplorer.g_xplot, x_bin_edges, x_counts)

        # Update Y marginal
        y_hist = ndxplorer._histogram.get("y")
        if y_hist is not None:
            y_bin_edges = y_hist.edges
            y_counts = y_hist.counts
            
            # Ensure arrays have same length to prevent IndexError
            y_edges_for_plot = y_bin_edges[1:]
            if len(y_edges_for_plot) != len(y_counts):
                min_len = min(len(y_edges_for_plot), len(y_counts))
                y_edges_for_plot = y_edges_for_plot[:min_len]
                y_counts = y_counts[:min_len]
            
            # Additional validation for Steps curve style
            if len(y_edges_for_plot) == 0 or len(y_counts) == 0:
                logging.debug("Skipping Y marginal update: empty data")
                # Set dummy data to prevent display issues (swapped order for vertical)
                ndxplorer.g_yhist_m.set_data([0, 0], [0, 1])
            else:
                # For Steps curves, ensure data is valid for current scale
                try:
                    # For Y marginal: counts on X-axis (horizontal), edges on Y-axis (vertical)
                    ndxplorer.g_yhist_m.set_data(y_counts, y_edges_for_plot)
                except Exception as e:
                    logging.warning(f"Error setting Y marginal data: {e}, using dummy data")
                    ndxplorer.g_yhist_m.set_data([0, 0], [0, 1])
            
            _autoscale_vertical_hist(ndxplorer.g_yplot, y_bin_edges, y_counts)

        # Update Z marginal if enabled
        if (
            hasattr(ndxplorer, "groupBox_3")
            and ndxplorer.groupBox_3.isChecked()
            and "z" in ndxplorer._histogram
        ):
            z_hist = ndxplorer._histogram.get("z")
            if z_hist is not None:
                z_bin_edges = z_hist.edges
                z_counts = z_hist.counts
                
                # Ensure arrays have same length to prevent IndexError
                z_edges_for_plot = z_bin_edges[1:]
                if len(z_edges_for_plot) != len(z_counts):
                    min_len = min(len(z_edges_for_plot), len(z_counts))
                    z_edges_for_plot = z_edges_for_plot[:min_len]
                    z_counts = z_counts[:min_len]
                
                ndxplorer.g_zhist_m.set_data(z_edges_for_plot, z_counts)
                _autoscale_horizontal_hist(ndxplorer.g_zplot, z_bin_edges, z_counts)

        # Replot the marginal distributions
        ndxplorer.g_xplot.replot()
        ndxplorer.g_yplot.replot()
        if hasattr(ndxplorer, "g_zplot"):
            ndxplorer.g_zplot.replot()
        
        # Ensure marginal plots are visible
        try:
            if hasattr(ndxplorer, 'g_xplot') and ndxplorer.g_xplot:
                ndxplorer.g_xplot.setVisible(True)
            if hasattr(ndxplorer, 'g_yplot') and ndxplorer.g_yplot:
                ndxplorer.g_yplot.setVisible(True)
            if hasattr(ndxplorer, 'g_zplot') and ndxplorer.g_zplot:
                ndxplorer.g_zplot.setVisible(ndxplorer.groupBox_3.isChecked() if hasattr(ndxplorer, 'groupBox_3') else True)
            logging.debug("Ensured marginal plots are visible")
        except Exception as e:
            logging.warning(f"Failed to ensure marginal plots visibility: {e}")
            
    except Exception as e:
        logging.warning(f"Error updating marginal plots from cache: {e}")


def _compute_edge_range(edges: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(edges, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0, 1.0
    start = float(arr[0])
    stop = float(arr[-1])
    if stop == start:
        stop = start + 1.0
    return start, stop


def _compute_count_upper(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    max_val = float(np.max(arr))
    if max_val <= 0.0:
        return 1.0
    return max_val * 1.05


def _autoscale_horizontal_hist(plot: QwtPlot, bin_edges: np.ndarray, counts: np.ndarray) -> None:
    """Autoscale a horizontal histogram plot."""
    if not QWT_AVAILABLE or QwtPlot is None:
        logging.warning("QWT not available, skipping histogram autoscaling")
        return
        
    xmin, xmax = _compute_edge_range(bin_edges)
    ymax = _compute_count_upper(counts)
    
    # Validate axis ranges to prevent NaN crash in Qwt
    if not (np.isfinite(xmin) and np.isfinite(xmax)):
        logging.warning("Invalid X axis range (NaN/Inf detected), using default range")
        xmin, xmax = 0.0, 1.0
    if not np.isfinite(ymax):
        logging.warning("Invalid Y axis range (NaN/Inf detected), using default range")
        ymax = 1.0
    
    for axis in (QwtPlot.xBottom, QwtPlot.xTop):
        plot.setAxisScale(axis, xmin, xmax)
    for axis in (QwtPlot.yLeft, QwtPlot.yRight):
        plot.setAxisScale(axis, 0.0, ymax)


def _autoscale_vertical_hist(plot: QwtPlot, bin_edges: np.ndarray, counts: np.ndarray) -> None:
    """Autoscale a vertical histogram plot."""
    if not QWT_AVAILABLE or QwtPlot is None:
        logging.warning("QWT not available, skipping histogram autoscaling")
        return
        
    ymin, ymax = _compute_edge_range(bin_edges)
    xmax = _compute_count_upper(counts)
    
    # Validate axis ranges to prevent NaN crash in Qwt
    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        logging.warning("Invalid Y axis range (NaN/Inf detected), using default range")
        ymin, ymax = 0.0, 1.0
    if not np.isfinite(xmax):
        logging.warning("Invalid X axis range (NaN/Inf detected), using default range")
        xmax = 1.0
    
    for axis in (QwtPlot.yLeft, QwtPlot.yRight):
        plot.setAxisScale(axis, ymin, ymax)
    for axis in (QwtPlot.xBottom, QwtPlot.xTop):
        plot.setAxisScale(axis, 0.0, xmax)


def update_plots(ndxplorer, skip_clustering: bool = False, skip_cache_invalidation: bool = False) -> None:
    if not getattr(ndxplorer, "_deferred_init_done", False) or ndxplorer.g_2dplot is None:
        logging.info("update_plots: deferred_init not done or g_2dplot is None")
        return
    logging.info("update_plots(skip_clustering=%s, skip_cache_invalidation=%s)", skip_clustering, skip_cache_invalidation)
    if not skip_cache_invalidation:
        ndxplorer.invalidate_values_cache()

    # Use the data_source property which handles both _data_source and data_manager
    data_source = ndxplorer.data_source
    is_empty = data_source.empty
    has_shape = data_source.values.shape[0] > 0 if hasattr(data_source.values, 'shape') else False
    logging.info(f"update_plots: data_source.empty={is_empty}, has_data={has_shape}")
    
    if data_source.empty or data_source.values.shape[0] == 0:
        logging.info("update_plots: Data source is empty, showing empty plots")
        if hasattr(ndxplorer, "_set_data_loaded"):
            ndxplorer._set_data_loaded(False)
        _show_empty_plots(ndxplorer)
        return

    ndxplorer.update_parameter_names()
    ndxplorer.update_cmap()

    logging.info(f"update_plots: Checking clustering: _use_clustering={ndxplorer._use_clustering}, skip_clustering={skip_clustering}")
    if ndxplorer._use_clustering and ndxplorer._cluster_labels is None and not skip_clustering:
        global hdbscan  # noqa: PLW0603
        if hdbscan is None:
            try:  # pragma: no cover - optional
                import hdbscan as _hdbscan  # type: ignore

                hdbscan = _hdbscan
                logging.debug("Imported hdbscan library")
            except ImportError:
                hdbscan = None
        if hdbscan:
            ndxplorer.on_apply_clustering()
            return

    data_ready = ndxplorer.is_data_ready()
    logging.info(f"update_plots: data_ready={data_ready}")
    if data_ready:
        logging.info("Calling update_histograms from update_plots")
        update_histograms(ndxplorer)
    else:
        logging.info("update_plots: data/axes not ready, skipping histogram update")

    # Keep the NDxplorer background/logo visible whenever no usable data is present.
    if getattr(ndxplorer, "_data_source", None) is None or getattr(ndxplorer._data_source, "empty", True):
        if hasattr(ndxplorer, "_set_data_loaded"):
            ndxplorer._set_data_loaded(False)
        _show_background(ndxplorer)
        # When there's no data, still refresh the empty plot scaffolding.
        _show_empty_plots(ndxplorer)
        return
    if not data_ready:
        # Data source exists but axes selections aren't ready yet (e.g., combos blank) —
        # keep the background visible until histograms can be computed.
        # BUT: if we already have data loaded, don't toggle background to avoid flicker during selection updates
        has_data_loaded = getattr(ndxplorer, "_has_real_data", False)
        if not has_data_loaded:
            if hasattr(ndxplorer, "_set_data_loaded"):
                ndxplorer._set_data_loaded(False)
            _show_background(ndxplorer)
        return

    _hide_background(ndxplorer)
    if hasattr(ndxplorer, "_set_data_loaded"):
        ndxplorer._set_data_loaded(True)
    
    # Ensure plot container is visible
    try:
        plot_container = getattr(ndxplorer, "_plot_container", None)
        if plot_container is not None:
            plot_container.setVisible(True)
            plot_container.raise_()
            logging.info("Made plot container visible and raised to front")
    except Exception as e:
        logging.debug(f"Could not make plot container visible: {e}")

    # Force marginal plots to be visible after data is ready
    try:
        if hasattr(ndxplorer, 'g_xplot') and ndxplorer.g_xplot:
            ndxplorer.g_xplot.setVisible(True)
            ndxplorer.g_xplot.raise_()
            logging.info("Forced X marginal plot to be visible")
        if hasattr(ndxplorer, 'g_yplot') and ndxplorer.g_yplot:
            ndxplorer.g_yplot.setVisible(True)
            ndxplorer.g_yplot.raise_()
            logging.info("Forced Y marginal plot to be visible")
        if hasattr(ndxplorer, 'g_zplot') and ndxplorer.g_zplot:
            z_visible = ndxplorer.groupBox_3.isChecked() if hasattr(ndxplorer, 'groupBox_3') else True
            ndxplorer.g_zplot.setVisible(z_visible)
            if z_visible:
                ndxplorer.g_zplot.raise_()
            logging.info(f"Forced Z marginal plot visibility: {z_visible}")
    except Exception as e:
        logging.warning(f"Failed to force marginal plots visibility: {e}")

    # Check if histograms were successfully computed
    x_hist = ndxplorer._histogram.get("x")
    y_hist = ndxplorer._histogram.get("y")
    
    # Validate histogram format (edges, counts) or (H, x_edges, y_edges)
    def is_valid_histogram(hist):
        if hist is None:
            return False
        if isinstance(hist, tuple) and len(hist) == 2:
            edges, counts = hist
            return len(edges) >= 2 and len(counts) >= 1
        if isinstance(hist, tuple) and len(hist) == 3:
            H, x_edges, y_edges = hist
            return H.size > 0 and len(x_edges) >= 2 and len(y_edges) >= 2
        return False
    
    if not is_valid_histogram(x_hist):
        logging.error("X histogram computation failed or returned empty result")
        _show_empty_plots(ndxplorer)
        return
    
    if not is_valid_histogram(y_hist):
        logging.error("Y histogram computation failed or returned empty result")
        _show_empty_plots(ndxplorer)
        return

    # Extract 1D histogram data (edges, counts)
    x_bin_edges, x_counts = x_hist
    x_edges_for_plot = x_bin_edges[1:]
    
    # Update X marginal
    ndxplorer.g_xhist_m.set_data(x_edges_for_plot, x_counts)
    _autoscale_horizontal_hist(ndxplorer.g_xplot, x_bin_edges, x_counts)
    ndxplorer.g_xplot.replot()

    # Extract Y histogram data (edges, counts)
    y_bin_edges, y_counts = y_hist
    y_edges_for_plot = y_bin_edges[1:]
    
    # Update Y marginal: counts on X-axis (horizontal), edges on Y-axis (vertical)
    ndxplorer.g_yhist_m.set_data(y_counts, y_edges_for_plot)
    _autoscale_vertical_hist(ndxplorer.g_yplot, y_bin_edges, y_counts)
    ndxplorer.g_yplot.replot()

    # Handle Z histogram if enabled
    z_hist = ndxplorer._histogram.get("z")
    if (
        hasattr(ndxplorer, "groupBox_3")
        and ndxplorer.groupBox_3.isChecked()
        and z_hist is not None
        and is_valid_histogram(z_hist)
    ):
        z_bin_edges, z_counts = z_hist
        z_edges_for_plot = z_bin_edges[1:]
        ndxplorer.g_zhist_m.set_data(z_edges_for_plot, z_counts)
        _autoscale_horizontal_hist(ndxplorer.g_zplot, z_bin_edges, z_counts)
        ndxplorer.g_zplot.replot()

    ndxplorer.update_spinbox_limits()
    ndxplorer.update_2d_plot()
    ndxplorer.g_2dplot.replot()

    # Final check: ensure marginal plots are still visible and updated
    try:
        if hasattr(ndxplorer, 'g_xplot') and ndxplorer.g_xplot:
            ndxplorer.g_xplot.setVisible(True)
            ndxplorer.g_xplot.raise_()
            ndxplorer.g_xplot.replot()
        if hasattr(ndxplorer, 'g_yplot') and ndxplorer.g_yplot:
            ndxplorer.g_yplot.setVisible(True)
            ndxplorer.g_yplot.raise_()
            ndxplorer.g_yplot.replot()
        if hasattr(ndxplorer, 'g_zplot') and ndxplorer.g_zplot:
            z_visible = ndxplorer.groupBox_3.isChecked() if hasattr(ndxplorer, 'groupBox_3') else True
            ndxplorer.g_zplot.setVisible(z_visible)
            if z_visible:
                ndxplorer.g_zplot.raise_()
                ndxplorer.g_zplot.replot()
        logging.info("Final marginal plot visibility and replot completed")
    except Exception as e:
        logging.warning(f"Failed in final marginal plot update: {e}")


def _show_empty_plots(ndxplorer):
    """Show empty placeholder plots when no data is available."""
    if not QWT_AVAILABLE or QwtPlot is None:
        logging.warning("QWT not available, skipping empty plot setup")
        return
        
    ndxplorer.g_xhist_m.set_data([0, 1], [0, 0])
    ndxplorer.g_yhist_m.set_data([0, 0], [0, 1])
    ndxplorer.g_zhist_m.set_data([0, 1], [0, 0])
    ndxplorer.cax.set_data(np.zeros((1, 1)))
    _show_background(ndxplorer)
    
    # Handle both backends for axis scales
    if getattr(ndxplorer, '_use_simple_backend', True):
        ndxplorer.g_2dplot.set_axis_scale('xBottom', 0, 1)
        ndxplorer.g_2dplot.set_axis_scale('yLeft', 0, 1)
    else:
        ndxplorer.g_2dplot.setAxisScale(QwtPlot.xBottom, 0, 1)
        ndxplorer.g_2dplot.setAxisScale(QwtPlot.yLeft, 0, 1)
    ndxplorer.g_xplot.setAxisScale(QwtPlot.xBottom, 0, 1)
    ndxplorer.g_xplot.setAxisScale(QwtPlot.xTop, 0, 1)
    ndxplorer.g_xplot.setAxisScale(QwtPlot.yLeft, 0, 1)
    ndxplorer.g_yplot.setAxisScale(QwtPlot.yLeft, 0, 1)
    ndxplorer.g_yplot.setAxisScale(QwtPlot.yRight, 0, 1)
    ndxplorer.g_yplot.setAxisScale(QwtPlot.xBottom, 0, 1)
    ndxplorer.g_zplot.setAxisScale(QwtPlot.xBottom, 0, 1)
    ndxplorer.g_zplot.setAxisScale(QwtPlot.yLeft, 0, 1)
    ndxplorer.g_xplot.replot()
    ndxplorer.g_yplot.replot()
    ndxplorer.g_zplot.replot()
    ndxplorer.g_2dplot.replot()


def auto_contrast(ndxplorer: "NDXplorer", skip_if_preserve: bool = True):
    """Auto-adjust contrast based on 2D histogram percentiles."""
    # Skip contrast update if preserving contrast during selection operations
    if skip_if_preserve and getattr(ndxplorer, '_preserve_contrast', False):
        logging.debug("auto_contrast: preserving contrast, skipping auto contrast")
        return
        
    logging.debug("Auto contrast triggered")
    
    # Check if histogram exists before proceeding
    if not hasattr(ndxplorer, '_histogram') or not ndxplorer._histogram:
        logging.debug("No histogram dictionary available")
        return
    
    try:
        if "2d" not in ndxplorer._histogram:
            logging.debug("No 2D histogram data available")
            return

        # Extract data from 2D histogram (handle both old tuple and new clean formats)
        hist_2d = ndxplorer._histogram.get("2d")
        if hist_2d is None:
            logging.debug("No 2D histogram available yet")
            return
        
        if hasattr(hist_2d, 'H'):
            # New clean Histogram2D object
            hist = hist_2d.H
        elif isinstance(hist_2d, tuple) and len(hist_2d) == 3:
            # Old tuple format (H, x_edges, y_edges)
            hist, _, _ = hist_2d
        else:
            logging.debug("Histogram format not ready yet (expected during initialization)")
            return
            
        if (
            hist is None
            or hist.size == 0
            or np.all(hist == 0)
            or np.all(np.isnan(hist))
        ):
            logging.debug("Histogram empty/all zero/NaN – using defaults")
            ndxplorer.vmin = 0
            ndxplorer.vmax = 1
            ndxplorer.on_vmin_vmax_changed()
            return

        if ndxplorer.checkBoxLogCounts.isChecked():
            min_positive = np.min(hist[hist > 0]) if np.any(hist > 0) else 1e-10
            hist_processed = np.maximum(hist, min_positive / 10)
            hist_processed = np.log10(hist_processed)
            hist_processed = np.nan_to_num(hist_processed)
        else:
            hist_processed = hist

        non_zero_values = hist_processed[hist_processed > 0]
        if non_zero_values.size > 0:
            vmin, vmax = compute_percentile_range_optimized(non_zero_values, 1, 99)
            if vmin == vmax:
                vmin = 0.9 * vmin if vmin != 0 else 0
                vmax = 1.1 * vmax if vmax != 0 else 1

            logging.debug("Setting auto contrast: vmin=%s vmax=%s", vmin, vmax)
            ndxplorer.vmin = vmin
            ndxplorer.vmax = vmax
            ndxplorer.on_vmin_vmax_changed()
        else:
            logging.debug("No non-zero values in histogram – using defaults")
            ndxplorer.vmin = 0
            ndxplorer.vmax = 1
            ndxplorer.on_vmin_vmax_changed()
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        logging.warning("Error in auto contrast: %s", exc)
        ndxplorer.vmin = 0
        ndxplorer.vmax = 1
        ndxplorer.on_vmin_vmax_changed()


def update_spinbox_limits(ndxplorer, low_pct: float = 0.1, high_pct: float = 99) -> None:
    # Skip contrast update if preserving contrast during selection operations
    if getattr(ndxplorer, '_preserve_contrast', False):
        logging.debug("update_spinbox_limits: preserving contrast, skipping update")
        return
        
    logging.debug(
        "update_spinbox_limits(low_pct=%s, high_pct=%s)", low_pct, high_pct
    )
    try:
        # Extract data from 2D histogram (handle both old tuple and new clean formats)
        hist_2d = ndxplorer._histogram.get("2d")
        if hist_2d is None:
            logging.debug("No 2D histogram available yet for marginal plot update")
            return
        
        if hasattr(hist_2d, 'H'):
            # New clean Histogram2D object
            H = hist_2d.H
        elif isinstance(hist_2d, tuple) and len(hist_2d) == 3:
            # Old tuple format (H, x_edges, y_edges)
            H, _, _ = hist_2d
        else:
            logging.debug("update_spinbox_limits: invalid histogram format")
            return
    except Exception:
        logging.debug("update_spinbox_limits: histogram missing")
        return

    mask = np.isfinite(H)
    if not np.any(mask):
        logging.debug("update_spinbox_limits: no finite bins")
        return

    data = H[mask]
    if ndxplorer.checkBoxLogCounts.isChecked():
        data = data[data > 0]
        if data.size == 0:
            return
        data = np.log10(data)

    vmin, vmax = compute_percentile_range_optimized(data, low_pct, high_pct)
    if ndxplorer.checkBoxLogCounts.isChecked():
        vmin = 10 ** vmin
        vmax = 10 ** vmax
    ndxplorer.vmin = vmin
    ndxplorer.vmax = vmax
    ndxplorer.on_vmin_vmax_changed()


def _show_background(ndxplorer):
    """Display the NDxplorer background/logo and hide the histogram image layer."""
    if hasattr(ndxplorer, "bg_image_item") and ndxplorer.bg_image_item is not None:
        ndxplorer.bg_image_item.setVisible(True)
    if getattr(ndxplorer, "cax", None) is not None:
        ndxplorer.cax.setVisible(False)


def _hide_background(ndxplorer):
    """Hide the background/logo so the histogram image is fully visible."""
    if hasattr(ndxplorer, "bg_image_item") and ndxplorer.bg_image_item is not None:
        ndxplorer.bg_image_item.setVisible(False)
    if getattr(ndxplorer, "cax", None) is not None:
        ndxplorer.cax.setVisible(True)


__all__ = [
    'update_histograms',
    'update_plots',
    'auto_contrast',
    'update_spinbox_limits',
]
