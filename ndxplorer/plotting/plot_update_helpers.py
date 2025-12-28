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


def update_histograms(ndxplorer) -> None:
    logging.debug("update_histograms")
    if not ndxplorer.is_data_ready():
        logging.info("Skipping update_histograms: data/axes not ready")
        return

    # Use the new background computation system if available
    if (hasattr(ndxplorer.plot_control, 'compute_histograms_background') and 
        hasattr(ndxplorer.plot_control, '_background_computation_enabled') and 
        ndxplorer.plot_control._background_computation_enabled):
        
        try:
            # Import the helper functions
            from ..utils.histogram_computation import (
                extract_histogram_params_from_plot_control,
                should_recompute_histograms,
                resolve_weights
            )
            
            # Extract histogram parameters
            histogram_params = extract_histogram_params_from_plot_control(ndxplorer.plot_control)
            
            # Check if recomputation is needed
            if not should_recompute_histograms(ndxplorer.plot_control, histogram_params):
                logging.info("Using cached histograms")
                if hasattr(ndxplorer, '_histogram') and ndxplorer._histogram:
                    ndxplorer.lineEditCountCurrent.setText(str(len(ndxplorer.x_values)))
                return
            
            # Resolve weights
            weights = resolve_weights(ndxplorer.plot_control, ndxplorer.data_source)
            
            # Use background computation
            ndxplorer.plot_control.compute_histograms_background(histogram_params, weights)
            logging.debug("Scheduled background histogram computation")
            return
            
        except Exception as e:
            logging.warning(f"Background histogram system failed, falling back to immediate computation: {e}")
            # Fall back to the original immediate computation
    
    # Original immediate computation as fallback
    _update_histograms_immediate(ndxplorer)


def _update_histograms_immediate(ndxplorer) -> None:
    """Original immediate histogram computation - used as fallback."""
    recompute_needed = True
    p1_idx = ndxplorer.plot_control.p1[0]
    p2_idx = ndxplorer.plot_control.p2[0]
    p3_idx = ndxplorer.plot_control.p3[0]
    use_weights = (
        hasattr(ndxplorer, "checkBoxWeight") and ndxplorer.checkBoxWeight.isChecked()
    )
    weight_param = (
        ndxplorer.comboBoxWeight.currentText()
        if use_weights and hasattr(ndxplorer, "comboBoxWeight")
        else ""
    )
    z_enabled = (
        hasattr(ndxplorer, "checkBoxEnableZ") and ndxplorer.checkBoxEnableZ.isChecked()
    )

    if hasattr(ndxplorer, "_cached_hist_params") and ndxplorer._cached_hist_params is not None:
        cached = ndxplorer._cached_hist_params
        mask_id = getattr(ndxplorer, "_cached_values_mask_id", None)
        if (
            cached.get("p1_idx") == p1_idx
            and cached.get("p2_idx") == p2_idx
            and cached.get("p3_idx") == p3_idx
            and cached.get("use_weights") == use_weights
            and cached.get("weight_param") == weight_param
            and cached.get("z_enabled") == z_enabled
            and cached.get("mask_id") == mask_id
            and cached.get("normed_x") == ndxplorer.plot_control.normed_hist_x
            and cached.get("normed_y") == ndxplorer.plot_control.normed_hist_y
            and cached.get("normed_z") == ndxplorer.plot_control.normed_hist_z
            and cached.get("x_bins_1d") == str(ndxplorer.get_x_bins()[0])
            and cached.get("y_bins_1d") == str(ndxplorer.get_y_bins()[0])
            and (
                not z_enabled
                or cached.get("z_bins_1d") == str(ndxplorer.get_z_bins()[0])
            )
        ):
            recompute_needed = False
            logging.info("Using cached histograms")

    if not recompute_needed and getattr(ndxplorer, "_histogram", None):
        ndxplorer.lineEditCountCurrent.setText(str(len(ndxplorer.x_values)))
        return

    d1 = ndxplorer.x_values
    d2 = ndxplorer.y_values
    d3 = ndxplorer.z_values

    ndxplorer.lineEditCountCurrent.setText(str(len(d1)))
    x_bins_1d, x_bins_2d = ndxplorer.get_x_bins()
    y_bins_1d, y_bins_2d = ndxplorer.get_y_bins()
    z_bins_1d, _ = ndxplorer.get_z_bins()

    x_bins_1d = ndxplorer.sanitize_bins(
        x_bins_1d, d1, default_count=getattr(ndxplorer.plot_control, "n_xhist_1d", 50)
    )
    y_bins_1d = ndxplorer.sanitize_bins(
        y_bins_1d, d2, default_count=getattr(ndxplorer.plot_control, "n_yhist_1d", 50)
    )
    z_bins_1d = ndxplorer.sanitize_bins(
        z_bins_1d, d3, default_count=getattr(ndxplorer.plot_control, "n_zhist_1d", 50)
    )
    x_bins_2d = ndxplorer.sanitize_bins(
        x_bins_2d, d1, default_count=getattr(ndxplorer.plot_control, "n_xhist_2d", 50)
    )
    y_bins_2d = ndxplorer.sanitize_bins(
        y_bins_2d, d2, default_count=getattr(ndxplorer.plot_control, "n_yhist_2d", 50)
    )

    weights = None
    if use_weights:
        weight_idx = -1
        if ndxplorer._data_source is not None and hasattr(ndxplorer._data_source, "parameter_names"):
            param_names = ndxplorer._data_source.parameter_names
            if weight_param in param_names:
                weight_idx = param_names.index(weight_param)
        if weight_idx >= 0:
            weight_values = ndxplorer.values[weight_idx].astype("float64")
            if len(weight_values) == len(d1):
                weights = weight_values
            else:
                logging.warning(
                    "Weights array shape (%s) doesn't match data array shape (%s). Disabling weights.",
                    len(weight_values),
                    len(d1),
                )
        else:
            logging.warning(
                "Weight parameter '%s' not found in data source. Disabling weights.",
                weight_param,
            )

    ndxplorer._histogram["x"] = _safe_histogram(
        d1,
        x_bins_1d,
        weights,
        ndxplorer.plot_control.normed_hist_x,
    )
    ndxplorer._histogram["y"] = _safe_histogram(
        d2,
        y_bins_1d,
        weights,
        ndxplorer.plot_control.normed_hist_y,
    )

    if z_enabled:
        z_weights = None
        if weights is not None and weight_param != ndxplorer.plot_control.z_label:
            z_weights = weights
        ndxplorer._histogram["z"] = _safe_histogram(
            d3,
            z_bins_1d,
            z_weights,
            ndxplorer.plot_control.normed_hist_z,
        )

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            H, x_edges, y_edges = np.histogram2d(
                x=d1, y=d2, bins=[x_bins_2d, y_bins_2d], weights=weights, density=False
            )
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        ndxplorer._histogram["2d"] = H, x_edges, y_edges
    except ValueError as exc:
        logging.warning("Could not compute 2D histogram: %s", exc)

    # Enhanced cache with data hash for better invalidation
    data_hash = hash((len(d1), len(d2), p1_idx, p2_idx, p3_idx))
    ndxplorer._cached_hist_params = {
        "p1_idx": p1_idx,
        "p2_idx": p2_idx,
        "p3_idx": p3_idx,
        "use_weights": use_weights,
        "weight_param": weight_param,
        "z_enabled": z_enabled,
        "mask_id": getattr(ndxplorer, "_cached_values_mask_id", None),
        "data_hash": data_hash,
        "normed_x": ndxplorer.plot_control.normed_hist_x,
        "normed_y": ndxplorer.plot_control.normed_hist_y,
        "normed_z": ndxplorer.plot_control.normed_hist_z,
        "x_bins_1d": str(x_bins_1d),
        "y_bins_1d": str(y_bins_1d),
        "z_bins_1d": str(z_bins_1d),
    }
    
    # Cache histogram for current frame in time series mode (not stacked)
    if (
        hasattr(ndxplorer.plot_control, '_frame_param')
        and ndxplorer.plot_control._frame_param is not None
        and hasattr(ndxplorer.plot_control, 'checkBoxStackFrames')
        and not ndxplorer.plot_control.checkBoxStackFrames.isChecked()
    ):
        # Store count with histogram for display
        cache_data = dict(ndxplorer._histogram)
        cache_data['_count'] = len(d1)
        ndxplorer.plot_control.cache_current_frame_histogram(cache_data)


def _safe_histogram(data, bins, weights, normed):
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            counts, bin_edges = np.histogram(
                data, bins=bins, weights=weights, density=normed
            )
        return bin_edges, counts
    except ValueError as exc:
        logging.warning("Could not compute histogram with weights: %s", exc)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                counts, bin_edges = np.histogram(
                    data, bins=bins, density=normed
                )
            return bin_edges, counts
        except ValueError as exc2:
            logging.warning("Could not compute histogram without weights: %s", exc2)
            valid = data[np.isfinite(data)]
            if len(valid) > 1:
                auto_bins = np.linspace(np.min(valid), np.max(valid), len(bins))
                with np.errstate(divide="ignore", invalid="ignore"):
                    counts, bin_edges = np.histogram(
                        valid, bins=auto_bins, density=normed
                    )
                return bin_edges, counts
            return (np.array([0, 1]), np.array([0]))


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
    for axis in (QwtPlot.yLeft, QwtPlot.yRight):
        plot.setAxisScale(axis, ymin, ymax)
    for axis in (QwtPlot.xBottom, QwtPlot.xTop):
        plot.setAxisScale(axis, 0.0, xmax)


def update_plots(ndxplorer, skip_clustering: bool = False, skip_cache_invalidation: bool = False) -> None:
    if not getattr(ndxplorer, "_deferred_init_done", False) or ndxplorer.g_2dplot is None:
        return
    logging.debug("update_plots(skip_clustering=%s, skip_cache_invalidation=%s)", skip_clustering, skip_cache_invalidation)
    if not skip_cache_invalidation:
        ndxplorer.invalidate_values_cache()

    if ndxplorer._data_source.empty or ndxplorer._data_source.values.shape[0] == 0:
        if hasattr(ndxplorer, "_set_data_loaded"):
            ndxplorer._set_data_loaded(False)
        _show_empty_plots(ndxplorer)
        return

    ndxplorer.update_parameter_names()
    ndxplorer.update_cmap()

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
    if data_ready:
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

    x_bin_edges = ndxplorer._histogram["x"][0]
    x_counts = ndxplorer._histogram["x"][1]
    ndxplorer.g_xhist_m.set_data(x_bin_edges[1:], x_counts)
    _autoscale_horizontal_hist(ndxplorer.g_xplot, x_bin_edges, x_counts)

    y_bin_edges = ndxplorer._histogram["y"][0]
    y_counts = ndxplorer._histogram["y"][1]
    ndxplorer.g_yhist_m.set_data(y_counts, y_bin_edges[1:])
    _autoscale_vertical_hist(ndxplorer.g_yplot, y_bin_edges, y_counts)

    if (
        hasattr(ndxplorer, "checkBoxEnableZ")
        and ndxplorer.checkBoxEnableZ.isChecked()
        and "z" in ndxplorer._histogram
    ):
        z_bin_edges = ndxplorer._histogram["z"][0]
        z_counts = ndxplorer._histogram["z"][1]
        ndxplorer.g_zhist_m.set_data(z_bin_edges[1:], z_counts)
        _autoscale_horizontal_hist(ndxplorer.g_zplot, z_bin_edges, z_counts)

    ndxplorer.update_spinbox_limits()
    ndxplorer.update_2d_plot()
    ndxplorer.g_xplot.replot()
    ndxplorer.g_yplot.replot()
    ndxplorer.g_zplot.replot()
    ndxplorer.g_2dplot.replot()


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


def auto_contrast(ndxplorer) -> None:
    """Auto-adjust vmin/vmax for the 2D histogram."""
    # Skip auto contrast if preserving contrast during selection operations
    if getattr(ndxplorer, '_preserve_contrast', False):
        logging.debug("auto_contrast: preserving contrast, skipping auto contrast")
        return
        
    logging.debug("Auto contrast triggered")
    try:
        if "2d" not in ndxplorer._histogram:
            logging.debug("No 2D histogram data available")
            return

        hist, _, _ = ndxplorer._histogram["2d"]
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
            vmin = np.percentile(non_zero_values, 1)
            vmax = np.percentile(non_zero_values, 99)

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
        H, _, _ = ndxplorer._histogram["2d"]
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

    vmin = np.percentile(data, low_pct) if data.size >= 2 else float(np.min(data))
    vmax = np.percentile(data, high_pct) if data.size >= 2 else float(np.max(data))
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
