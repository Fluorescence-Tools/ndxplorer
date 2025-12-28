"""Histogram-related helper routines extracted from plot_main."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..logging_config import logging
from ..utils.performance_optimizations import (
    compute_histogram2d_adaptive,
    compute_histogram1d_adaptive,
    downsample_for_display,
    get_performance_monitor,
)

if False:  # pragma: no cover - type checking hints without runtime import
    from ..core.plot_main import NDXplorer


@dataclass
class HistogramParams:
    p1_idx: int
    p2_idx: int
    p3_idx: int
    use_weights: bool
    weight_param: str
    z_enabled: bool
    mask_id: Optional[int]
    normed_x: bool
    normed_y: bool
    normed_z: bool
    x_bins_1d: str
    y_bins_1d: str
    z_bins_1d: Optional[str]


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
) -> np.ndarray:
    """Ensure bins are a strictly increasing 1D array."""
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
            return np.linspace(vmin, vmax, n + 1)
        return np.array([0.0, 1.0])
    except Exception:
        return np.array([0.0, 1.0])


def is_data_ready(ndxplorer: "NDXplorer") -> bool:
    """Return True if data and axis selections are ready for histogram computation."""
    try:
        if ndxplorer._data_source is None or ndxplorer._data_source.empty:
            return False
        values = ndxplorer._data_source.values
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
    p1_idx = ndxplorer.plot_control.p1[0]
    p2_idx = ndxplorer.plot_control.p2[0]
    p3_idx = ndxplorer.plot_control.p3[0]
    use_weights = hasattr(ndxplorer, "checkBoxWeight") and ndxplorer.checkBoxWeight.isChecked()
    weight_param = (
        ndxplorer.comboBoxWeight.currentText()
        if use_weights and hasattr(ndxplorer, "comboBoxWeight")
        else ""
    )
    z_enabled = hasattr(ndxplorer, "checkBoxEnableZ") and ndxplorer.checkBoxEnableZ.isChecked()
    mask_id = getattr(ndxplorer, "_cached_values_mask_id", None)
    x_bins_1d = str(ndxplorer.get_x_bins()[0])
    y_bins_1d = str(ndxplorer.get_y_bins()[0])
    z_bins_1d = str(ndxplorer.get_z_bins()[0]) if z_enabled else None
    return HistogramParams(
        p1_idx=p1_idx,
        p2_idx=p2_idx,
        p3_idx=p3_idx,
        use_weights=use_weights,
        weight_param=weight_param,
        z_enabled=z_enabled,
        mask_id=mask_id,
        normed_x=ndxplorer.plot_control.normed_hist_x,
        normed_y=ndxplorer.plot_control.normed_hist_y,
        normed_z=ndxplorer.plot_control.normed_hist_z,
        x_bins_1d=x_bins_1d,
        y_bins_1d=y_bins_1d,
        z_bins_1d=z_bins_1d,
    )


def should_recompute(ndxplorer: "NDXplorer", params: HistogramParams) -> bool:
    """Return True when cached histograms no longer match current parameters."""
    cached = getattr(ndxplorer, "_cached_hist_params", None)
    if cached is None:
        return True
    if cached.get("p1_idx") != params.p1_idx:
        return True
    if cached.get("p2_idx") != params.p2_idx:
        return True
    if cached.get("p3_idx") != params.p3_idx:
        return True
    if cached.get("use_weights") != params.use_weights:
        return True
    if cached.get("weight_param") != params.weight_param:
        return True
    if cached.get("z_enabled") != params.z_enabled:
        return True
    if cached.get("mask_id") != params.mask_id:
        return True
    if cached.get("normed_x") != params.normed_x:
        return True
    if cached.get("normed_y") != params.normed_y:
        return True
    if cached.get("normed_z") != params.normed_z:
        return True
    if cached.get("x_bins_1d") != params.x_bins_1d:
        return True
    if cached.get("y_bins_1d") != params.y_bins_1d:
        return True
    if params.z_enabled and cached.get("z_bins_1d") != params.z_bins_1d:
        return True
    return False


def save_cache(ndxplorer: "NDXplorer", params: HistogramParams) -> None:
    """Store histogram parameters for cache comparisons."""
    ndxplorer._cached_hist_params = {
        "p1_idx": params.p1_idx,
        "p2_idx": params.p2_idx,
        "p3_idx": params.p3_idx,
        "use_weights": params.use_weights,
        "weight_param": params.weight_param,
        "z_enabled": params.z_enabled,
        "mask_id": params.mask_id,
        "normed_x": params.normed_x,
        "normed_y": params.normed_y,
        "normed_z": params.normed_z,
        "x_bins_1d": params.x_bins_1d,
        "y_bins_1d": params.y_bins_1d,
        "z_bins_1d": params.z_bins_1d,
    }


def resolve_weights(ndxplorer: "NDXplorer", use_weights: bool, d1) -> Optional[np.ndarray]:
    """Return weight array matching d1 length or None (float32 for memory efficiency)."""
    if not use_weights:
        return None
    weight_param = ndxplorer.comboBoxWeight.currentText()
    weight_idx = -1
    if ndxplorer._data_source is not None and hasattr(ndxplorer._data_source, "parameter_names"):
        param_names = ndxplorer._data_source.parameter_names
        if weight_param in param_names:
            weight_idx = param_names.index(weight_param)
    if weight_idx < 0:
        logging.warning("Weight parameter '%s' not found in data source. Disabling weights.", weight_param)
        return None
    # Use float32 for memory efficiency - sufficient precision for weights
    weight_values = ndxplorer.values[weight_idx].astype(np.float32)
    if len(weight_values) != len(d1):
        logging.warning(
            "Weights array shape (%d) doesn't match data array shape (%d). Disabling weights.",
            len(weight_values),
            len(d1),
        )
        return None
    return weight_values


def histogram_with_fallback(data, bins, normed, weights=None):
    """Compute histogram with automatic fallback bin generation.
    
    Uses Numba-accelerated computation for large datasets when available.
    Optimized to avoid unnecessary dtype conversions for float32 data.
    """
    try:
        # Avoid copy if data is already contiguous float32/float64
        data_arr = np.asarray(data)
        if data_arr.dtype not in (np.float32, np.float64):
            data_arr = data_arr.astype(np.float64, copy=False)
        
        bins_arr = np.asarray(bins)
        if bins_arr.dtype != np.float64:
            bins_arr = bins_arr.astype(np.float64, copy=False)
        
        # Use optimized adaptive histogram for large datasets
        result = compute_histogram1d_adaptive(
            data_arr if data_arr.dtype == np.float64 else data_arr.astype(np.float64),
            bins_arr,
            weights=weights,
            density=normed,
        )
        return (result.edges, result.counts)
    except (ValueError, TypeError) as exc:
        logging.warning("Could not compute histogram: %s", exc)
        try:
            if len(data) > 0 and np.isfinite(data).any():
                valid_data = data[np.isfinite(data)]
                if len(valid_data) > 1:
                    if bins is not None and len(bins) > 1:
                        n = len(bins) - 1
                    else:
                        n = 50
                    auto_bins = np.linspace(
                        np.min(valid_data),
                        np.max(valid_data),
                        int(max(2, n)) + 1,
                    )
                    result = compute_histogram1d_adaptive(
                        valid_data.astype(np.float64),
                        auto_bins,
                        density=normed,
                    )
                    return (result.edges, result.counts)
                else:
                    return (np.array([0, 1]), np.array([1]))
            return (np.array([0, 1]), np.array([0]))
        except Exception as nested:
            logging.error("Failed to compute histogram with fallback: %s", nested)
            return (np.array([0, 1]), np.array([0]))


def update_histograms(ndxplorer: "NDXplorer") -> None:
    """Main entry point to recompute histograms, updating UI as needed."""
    logging.debug("update_histograms")
    if not is_data_ready(ndxplorer):
        logging.info("Skipping update_histograms: data/axes not ready")
        return

    params = extract_histogram_params(ndxplorer)
    if not should_recompute(ndxplorer, params) and getattr(ndxplorer, "_histogram", None):
        ndxplorer.lineEditCountCurrent.setText(str(len(ndxplorer.x_values)))
        logging.info("Using cached histograms")
        return

    perf = get_performance_monitor()
    perf.start_timer("ndx_histogram_update")

    d1 = ndxplorer.x_values
    d2 = ndxplorer.y_values
    d3 = ndxplorer.z_values
    ndxplorer.lineEditCountCurrent.setText(str(len(d1)))

    x_bins_1d, x_bins_2d = ndxplorer.get_x_bins()
    y_bins_1d, y_bins_2d = ndxplorer.get_y_bins()
    z_bins_1d, _ = ndxplorer.get_z_bins()

    x_bins_1d = sanitize_bins(
        x_bins_1d,
        d1,
        default_count=getattr(ndxplorer.plot_control, "n_xhist_1d", 50),
    )
    y_bins_1d = sanitize_bins(
        y_bins_1d,
        d2,
        default_count=getattr(ndxplorer.plot_control, "n_yhist_1d", 50),
    )
    z_bins_1d = sanitize_bins(
        z_bins_1d,
        d3,
        default_count=getattr(ndxplorer.plot_control, "n_zhist_1d", 50),
    )
    x_bins_2d = sanitize_bins(
        x_bins_2d,
        d1,
        default_count=getattr(ndxplorer.plot_control, "n_xhist_2d", 50),
    )
    y_bins_2d = sanitize_bins(
        y_bins_2d,
        d2,
        default_count=getattr(ndxplorer.plot_control, "n_yhist_2d", 50),
    )

    weights = resolve_weights(ndxplorer, params.use_weights, d1)
    ndxplorer._histogram["x"] = histogram_with_fallback(
        d1, x_bins_1d, ndxplorer.plot_control.normed_hist_x, weights=weights
    )
    ndxplorer._histogram["y"] = histogram_with_fallback(
        d2, y_bins_1d, ndxplorer.plot_control.normed_hist_y, weights=weights
    )

    if params.z_enabled:
        z_weights = None
        if weights is not None:
            weight_param = ndxplorer.comboBoxWeight.currentText()
            z_param = ndxplorer.plot_control.z_label
            if weight_param != z_param:
                z_weights = weights
        ndxplorer._histogram["z"] = histogram_with_fallback(
            d3,
            z_bins_1d,
            ndxplorer.plot_control.normed_hist_z,
            weights=z_weights,
        )
    else:
        ndxplorer._histogram.pop("z", None)

    hist2d = compute_histogram2d_adaptive(
        d1,
        d2,
        x_bins_2d,
        y_bins_2d,
        weights=weights,
    )
    ndxplorer._histogram["2d"] = hist2d.data
    ndxplorer._histogram_metadata = {
        "chunked": hist2d.chunked,
        "total_points": hist2d.n_points,
        "threshold": len(d1),
    }
    if hist2d.chunked:
        logging.info(
            "Histogram update used chunked accumulation for %d points",
            hist2d.n_points,
        )

    save_cache(ndxplorer, params)
    perf.log_memory_usage("ndx_histogram_update")
    perf.end_timer("ndx_histogram_update")
