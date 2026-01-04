"""Axis-related helpers extracted from plot_main."""

from __future__ import annotations

import numpy as np

from ..logging_config import logging


def check_and_set_image_axes(ndxplorer: "NDXplorer") -> bool:
    logging.debug("check_and_set_image_axes()")
    # Use the data_source property which handles both _data_source and data_manager
    try:
        data_source = ndxplorer.data_source
    except Exception as exc:
        logging.debug(f"Could not get data_source: {exc}")
        return False
    
    if data_source is None or data_source.empty:
        logging.info("Image detection skipped: data_source is None or empty")
        return False

    param_names = data_source.parameter_names
    logging.info(f"Image detection: checking parameter names: {list(param_names)}")

    has_x_pixel = any("x pixel" in name.lower() for name in param_names)
    has_y_pixel = any("y pixel" in name.lower() for name in param_names)
    
    logging.info(f"Image detection: has_x_pixel={has_x_pixel}, has_y_pixel={has_y_pixel}")

    if not (has_x_pixel and has_y_pixel):
        logging.info("Image detection failed: X pixel or Y pixel columns not found")
        return False

    logging.info("Image data detected (X pixel and Y pixel columns found)")
    x_pixel_param = next((name for name in param_names if "x pixel" in name.lower()), None)
    y_pixel_param = next((name for name in param_names if "y pixel" in name.lower()), None)
    
    t_pixel_param = next((name for name in param_names if "t pixel" in name.lower()), None)
    z_pixel_param = next((name for name in param_names if "z pixel" in name.lower()), None)

    # Set X and Y axes to the pixel parameters
    # Use match_contains=False for exact matching to avoid matching "T pixel" 
    x_success = ndxplorer.plot_control.set_axis_by_name("x", x_pixel_param, match_contains=False, block_signals=True)
    y_success = ndxplorer.plot_control.set_axis_by_name("y", y_pixel_param, match_contains=False, block_signals=True)
    
    if not x_success:
        logging.warning("Failed to set X axis to %s", x_pixel_param)
    if not y_success:
        logging.warning("Failed to set Y axis to %s", y_pixel_param)
    
    # Even if setting axes failed, continue with image setup (return True at end)
    # This prevents apply_default_axes_from_settings from overriding

    photon_param = next((name for name in param_names if "number of photons" in name.lower() or "Number of Photons" in name), None)
    logging.debug("Weight parameter: %s", photon_param)
    weight_success = ndxplorer.plot_control.set_axis_by_name(
        "weight", photon_param, match_contains=True, block_signals=True
    )
    if weight_success:
        logging.debug("Set weighting to %s", photon_param)
        try:
            ndxplorer.weight_param = photon_param
            ndxplorer.weight_enabled = True
            # Automatically check the weight checkbox when weight parameter is detected
            ndxplorer.checkBoxWeight.setChecked(True)
            logging.info("Automatically enabled weight checkbox for %s", photon_param)
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("Failed to enable weight parameter: %s", exc)
    else:
        logging.debug("No matching weight parameter found, using default")

    x_values = data_source.values[param_names.index(x_pixel_param), :]
    y_values = data_source.values[param_names.index(y_pixel_param), :]
    logging.debug("x_pixel_param: %s, x_values: %s", x_pixel_param, x_values)
    logging.debug("y_pixel_param: %s, y_values: %s", y_pixel_param, y_values)

    # Handle empty data arrays safely
    if len(x_values) > 0:
        x_pixels = int(np.max(x_values)) + 1
    else:
        x_pixels = 256  # Default size
    if len(y_values) > 0:
        y_pixels = int(np.max(y_values)) + 1
    else:
        y_pixels = 256  # Default size
    logging.info("Image dimensions: %sx%s pixels", x_pixels, y_pixels)

    logging.info(f"Setting histogram bins to match image dimensions: {x_pixels}x{y_pixels}")
    ndxplorer.plot_control.n_xhist_1d = x_pixels
    ndxplorer.plot_control.n_yhist_1d = y_pixels
    ndxplorer.plot_control.n_xhist_2d = x_pixels
    ndxplorer.plot_control.n_yhist_2d = y_pixels
    
    # Also update UI spinboxes so bins are read correctly
    if hasattr(ndxplorer.plot_control, 'spinBoxNXHist1D'):
        ndxplorer.plot_control.spinBoxNXHist1D.setValue(x_pixels)
    if hasattr(ndxplorer.plot_control, 'spinBoxNYHist1D'):
        ndxplorer.plot_control.spinBoxNYHist1D.setValue(y_pixels)
    if hasattr(ndxplorer.plot_control, 'spinBoxNXHist2D'):
        ndxplorer.plot_control.spinBoxNXHist2D.setValue(x_pixels)
    if hasattr(ndxplorer.plot_control, 'spinBoxNYHist2D'):
        ndxplorer.plot_control.spinBoxNYHist2D.setValue(y_pixels)
    
    logging.info(f"Bins set: n_xhist_2d={ndxplorer.plot_control.n_xhist_2d}, n_yhist_2d={ndxplorer.plot_control.n_yhist_2d}")
    ndxplorer.plot_control.xmin = 0
    ndxplorer.plot_control.ymin = 0
    
    # Set max values with error handling for empty arrays
    try:
        ndxplorer.plot_control.xmax = np.max(x_values) if len(x_values) > 0 else x_pixels
    except (ValueError, IndexError):
        ndxplorer.plot_control.xmax = x_pixels
    
    try:
        ndxplorer.plot_control.ymax = np.max(y_values) if len(y_values) > 0 else y_pixels
    except (ValueError, IndexError):
        ndxplorer.plot_control.ymax = y_pixels

    frame_param = t_pixel_param or z_pixel_param
    if frame_param:
        frame_values = data_source.values[param_names.index(frame_param), :]
        n_frames = int(np.max(frame_values)) + 1
        logging.info("Frame stack detected (%s): %d frames", frame_param, n_frames)
        
        ndxplorer.plot_control.setup_frame_selection(frame_param, n_frames)
    else:
        ndxplorer.plot_control.hide_frame_selection()

    logging.debug("Set binning and ranges to match pixel dimensions")
    # Don't call update_plots here - it will be called by _apply_axes_and_refresh
    # after this function returns True
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
