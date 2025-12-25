"""Axis-related helpers extracted from plot_main."""

from __future__ import annotations

import numpy as np

from ..logging_config import logging


def check_and_set_image_axes(ndxplorer: "NDXplorer") -> bool:
    logging.debug("check_and_set_image_axes()")
    data_source = getattr(ndxplorer, "_data_source", None)
    if data_source is None or data_source.empty:
        logging.debug("No data loaded, skipping image axes check")
        return False

    param_names = data_source.parameter_names
    logging.debug("Parameter names: %s", param_names)

    has_x_pixel = any("x pixel" in name.lower() for name in param_names)
    has_y_pixel = any("y pixel" in name.lower() for name in param_names)

    if not (has_x_pixel and has_y_pixel):
        return False

    logging.info("Image data detected (X pixel and Y pixel columns found)")
    x_pixel_param = next((name for name in param_names if "x pixel" in name.lower()), None)
    y_pixel_param = next((name for name in param_names if "y pixel" in name.lower()), None)

    x_success = ndxplorer.plot_control.set_axis_by_name("x", x_pixel_param, block_signals=True)
    y_success = ndxplorer.plot_control.set_axis_by_name("y", y_pixel_param, block_signals=True)
    if not x_success:
        logging.warning("Failed to set X axis to %s", x_pixel_param)
    if not y_success:
        logging.warning("Failed to set Y axis to %s", y_pixel_param)

    photon_param = next((name for name in param_names if "number of photons" in name.lower()), None)
    logging.debug("Weight parameter: %s", photon_param)
    weight_success = ndxplorer.plot_control.set_axis_by_name(
        "weight", photon_param, match_contains=True, block_signals=True
    )
    if weight_success:
        logging.debug("Set weighting to %s", photon_param)
    else:
        logging.debug("No matching weight parameter found, using default")

    x_values = data_source.values[param_names.index(x_pixel_param), :]
    y_values = data_source.values[param_names.index(y_pixel_param), :]
    logging.debug("x_pixel_param: %s, x_values: %s", x_pixel_param, x_values)
    logging.debug("y_pixel_param: %s, y_values: %s", y_pixel_param, y_values)

    x_pixels = int(np.max(x_values)) + 1
    y_pixels = int(np.max(y_values)) + 1
    logging.info("Image dimensions: %sx%s pixels", x_pixels, y_pixels)

    ndxplorer.plot_control.n_xhist_2d = x_pixels
    ndxplorer.plot_control.n_yhist_2d = y_pixels
    ndxplorer.plot_control.xmin = 0
    ndxplorer.plot_control.xmax = x_pixels - 1
    ndxplorer.plot_control.ymin = 0
    ndxplorer.plot_control.ymax = y_pixels - 1

    logging.debug("Set binning and ranges to match pixel dimensions")
    logging.debug("Applying auto contrast to image")
    ndxplorer.on_auto_contrast()
    try:
        ndxplorer.update_plots()
    except Exception:
        pass
    return True


def apply_default_axes_from_settings(ndxplorer: "NDXplorer") -> bool:
    logging.debug("apply_default_axes_from_settings()")
    try:
        defaults = ndxplorer.settings.get("default_axes", {}) if hasattr(ndxplorer, "settings") else {}
    except Exception:
        defaults = {}
    if not isinstance(defaults, dict) or not defaults:
        logging.debug("No default_axes configured in settings; skipping.")
        return False

    try:
        param_names = list(ndxplorer.data_source.parameter_names)
    except Exception:
        param_names = []

    changed = False

    def _set_axis(ax_key, axis_name):
        if not axis_name or not isinstance(axis_name, str):
            return False
        if axis_name in param_names:
            ok = ndxplorer.plot_control.set_axis_by_name(ax_key, axis_name, match_contains=False, block_signals=True)
            return bool(ok)
        ok = ndxplorer.plot_control.set_axis_by_name(ax_key, axis_name, match_contains=True, block_signals=True)
        return bool(ok)

    for ax_key in ("x", "y", "z"):
        name = defaults.get(ax_key)
        if _set_axis(ax_key, name):
            changed = True

    wname = defaults.get("weight")
    if wname and _set_axis("weight", wname):
        changed = True

    if changed:
        try:
            ndxplorer.plot_control.on_x_axis_changed()
        except Exception:
            pass
        try:
            ndxplorer.plot_control.on_y_axis_changed()
        except Exception:
            pass
        try:
            ndxplorer.plot_control.on_z_axis_changed()
        except Exception:
            pass
        logging.info("Applied default axes from settings.")
        return True

    logging.debug("No default axes were applied (names may not match current dataset).")
    return False


def _filtered_values(values, scale: str):
    arr = np.asarray(values)
    finite_mask = np.isfinite(arr)
    filtered = arr[finite_mask]
    if scale == "log":
        filtered = filtered[filtered > 0]
    return filtered


def compute_axis_min(values, scale: str = "lin") -> float:
    filtered = _filtered_values(values, scale)
    if filtered.size == 0:
        return 0.0
    return float(np.min(filtered))


def compute_axis_max(values, scale: str = "lin") -> float:
    filtered = _filtered_values(values, scale)
    if filtered.size == 0:
        return 0.0
    return float(np.max(filtered))
