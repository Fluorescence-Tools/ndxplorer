"""Helper functions for loading and saving ndXplorer settings."""

from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Optional

import yaml
from qtpy import QtWidgets

from .logging_config import logging
from .settings import get_settings_path

if False:  # pragma: no cover - only for type checking without runtime import
    from .plot_main import NDXplorer


def save_axis_settings(
    ndxplorer: "NDXplorer", settings_json_fn: Optional[str] = None
) -> None:
    """Persist current axis configuration to a JSON file."""
    logging.debug("save_axis_settings")
    if isinstance(settings_json_fn, bool):
        settings_json_fn = None

    if not settings_json_fn:
        settings_path = get_settings_path()
        default_filename = str(settings_path / "mfd.axis.json")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            ndxplorer,
            "Axis settings file",
            default_filename,
            "Axis file (*.axis.json)",
        )
        if not filename:
            return
        settings_json_fn = filename

    try:
        if hasattr(ndxplorer, "plot_control") and ndxplorer.plot_control is not None:
            ndxplorer.plot_control.update_x_axis_settings()
            ndxplorer.plot_control.update_y_axis_settings()
            ndxplorer.plot_control.update_z_axis_settings()
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Could not refresh axis settings from UI before saving: %s", exc)

    baseline = {}
    target_path = Path(settings_json_fn)
    try:
        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as handle:
                baseline = json.load(handle) or {}
        else:
            packaged = Path(__file__).parent / "settings" / "mfd.axis.json"
            if packaged.exists():
                with open(packaged, "r", encoding="utf-8") as handle:
                    baseline = json.load(handle) or {}
    except Exception as exc:
        logging.warning(
            "Failed to load baseline axis settings, starting from empty. Reason: %s",
            exc,
        )
        baseline = {}

    in_use_names = []
    try:
        in_use_names = [
            getattr(ndxplorer.plot_control, "x_label", "") or "",
            getattr(ndxplorer.plot_control, "y_label", "") or "",
            getattr(ndxplorer.plot_control, "z_label", "") or "",
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Could not determine in-use axis names: %s", exc)

    current = getattr(ndxplorer.plot_control, "axis_settings", {}) or {}
    changed = {}
    for name in in_use_names:
        if not name:
            continue

        cur = current.get(name)
        if not isinstance(cur, dict):
            continue

        base_val = baseline.get(name)
        if base_val != cur:
            baseline[name] = cur
            changed[name] = {"from": base_val, "to": cur}

    print(f"Saving axis settings to {settings_json_fn}")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(baseline, handle, indent=4)
        if changed:
            logging.info("Updated axis settings for: %s", ", ".join(changed.keys()))
        else:
            logging.info("No changes detected for current axes; saved file unchanged")
    except Exception as exc:
        logging.error("Failed to save axis settings to %s: %s", settings_json_fn, exc)


def set_default_axis(ndxplorer: "NDXplorer") -> None:
    """Persist the currently selected axes/weight/colormap as defaults."""
    logging.debug("set_default_axis")
    settings_path = getattr(ndxplorer, "_settings_json_path", None)
    if not settings_path:
        try:
            settings_path = str(get_settings_path() / "mfd.settings.json")
        except Exception:  # pragma: no cover - defensive
            settings_path = None
    if not settings_path:
        QtWidgets.QMessageBox.warning(
            ndxplorer, "Set default axis", "Could not determine settings file path."
        )
        return

    try:
        with open(settings_path, "r", encoding="utf-8") as handle:
            settings_data = json.load(handle) or {}
    except Exception as exc:
        logging.debug(
            "Could not read settings file '%s', using in-memory settings. Reason: %s",
            settings_path,
            exc,
        )
        settings_data = (
            dict(ndxplorer.settings)
            if hasattr(ndxplorer, "settings") and isinstance(ndxplorer.settings, dict)
            else {}
        )

    try:
        x_name = ndxplorer.plot_control.p1[1]
        y_name = ndxplorer.plot_control.p2[1]
        z_name = ndxplorer.plot_control.p3[1]
    except Exception as exc:
        QtWidgets.QMessageBox.warning(
            ndxplorer, "Set default axis", f"Unable to read current axis selections: {exc}"
        )
        return

    weight_name = None
    try:
        if hasattr(ndxplorer, "comboBoxWeight") and ndxplorer.comboBoxWeight is not None:
            weight_name = str(ndxplorer.comboBoxWeight.currentText())
    except Exception:  # pragma: no cover - defensive
        weight_name = None

    settings_data.setdefault("default_axes", {})
    settings_data["default_axes"].update({"x": x_name, "y": y_name, "z": z_name})
    if weight_name:
        settings_data["default_axes"]["weight"] = weight_name
    try:
        settings_data["colormap"] = ndxplorer.current_cmap
    except Exception:  # pragma: no cover - optional
        pass

    try:
        with open(settings_path, "w", encoding="utf-8") as handle:
            json.dump(settings_data, handle, indent=4)
        try:
            ndxplorer.settings.update(settings_data)
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            ndxplorer, "Set default axis", "Default axis settings have been updated."
        )
        logging.info(
            "Updated default_axes in '%s' to %s",
            settings_path,
            settings_data.get("default_axes"),
        )
    except Exception as exc:
        logging.error("Failed to save default axes to '%s': %s", settings_path, exc)
        QtWidgets.QMessageBox.critical(
            ndxplorer,
            "Set default axis",
            f"Failed to save default axis settings:\n{exc}",
        )


def load_settings(
    ndxplorer: "NDXplorer", settings_json_fn: Optional[str] = None
) -> None:
    """Load settings JSON plus referenced axis/label/equation files."""
    logging.debug("onLoad_settings")
    if isinstance(settings_json_fn, bool):
        settings_json_fn = None
    if settings_json_fn is None:
        file_sel = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "ndXplorer settings file",
            ndxplorer.working_path,
            "ndXplorer settings (*.settings.json)",
        )
        if isinstance(file_sel, (tuple, list)):
            settings_json_fn = file_sel[0]
        else:
            settings_json_fn = file_sel
    if not settings_json_fn:
        return

    try:
        ndxplorer._settings_json_path = str(settings_json_fn)
    except Exception:
        ndxplorer._settings_json_path = None

    if not pathlib.Path(settings_json_fn).exists():
        logging.warning(f"Settings file not found: {settings_json_fn}. Falling back to defaults.")
        default_settings_dir = pathlib.Path(__file__).parent / "settings"
        settings_json_fn = default_settings_dir / "mfd.settings.json"
        if not settings_json_fn.exists():
            logging.error(f"Default settings file not found: {settings_json_fn}")
            return

    with open(settings_json_fn, "r", encoding="utf-8") as handle:
        data = json.load(handle)
        ndxplorer.settings.update(data)

    if "colormap" in ndxplorer.settings:
        ndxplorer.set_default_colormap(ndxplorer.settings["colormap"])

    settings_dir = pathlib.Path(settings_json_fn).parent
    default_settings_dir = pathlib.Path(__file__).parent / "settings"

    fn_axis = settings_dir / ndxplorer.settings["axis"]
    if not fn_axis.exists():
        fn_axis = default_settings_dir / ndxplorer.settings["axis"]
    
    if fn_axis.exists():
        with open(str(fn_axis), "r") as handle:
            axis_data = json.load(handle)
            ndxplorer.plot_control.axis_settings.update(axis_data)
    else:
        logging.warning(f"Axis settings file not found: {fn_axis}")

    if "axis_labels" in ndxplorer.settings:
        fn_axis_labels = settings_dir / ndxplorer.settings["axis_labels"]
        if not fn_axis_labels.exists():
            fn_axis_labels = default_settings_dir / ndxplorer.settings["axis_labels"]

        ndxplorer.axis_label_settings = {
            "enable_all_labels": True,
            "axis_labels": {
                "y_plot": {"top": True, "right": True},
                "x_plot": {"top": True},
                "z_plot": {"bottom": True, "left": True},
            },
            "fonts": {
                "tick_size_pt": 8,
                "title_size_pt": 10,
                "title_weight": 700,
                "color": "#000000",
            },
        }

        if fn_axis_labels.exists():
            try:
                with open(str(fn_axis_labels), "r") as handle:
                    labels_data = yaml.load(handle, Loader=yaml.FullLoader)
                    if labels_data is not None:
                        ndxplorer.axis_label_settings.update(labels_data)
            except Exception as exc:
                logging.warning("Error loading axis label settings: %s", exc)
        else:
            logging.warning("Axis label settings file not found: %s", fn_axis_labels)

        try:
            fonts = ndxplorer.axis_label_settings.get("fonts", {})
            if not hasattr(ndxplorer, "font_settings"):
                ndxplorer.font_settings = {}
            ndxplorer.font_settings.update(fonts)
            ndxplorer.apply_fonts()
        except Exception as exc:
            logging.debug("Could not apply font settings: %s", exc)

    fn_equations = settings_dir / ndxplorer.settings["equations"]
    if not fn_equations.exists():
        fn_equations = default_settings_dir / ndxplorer.settings["equations"]
    
    if fn_equations.exists():
        with open(str(fn_equations), "r") as handle:
            ndxplorer.equations = yaml.load(handle, Loader=yaml.FullLoader)
    else:
        logging.warning(f"Equations file not found: {fn_equations}")

    fn_constants = settings_dir / ndxplorer.settings["constants"]
    if not fn_constants.exists():
        fn_constants = default_settings_dir / ndxplorer.settings["constants"]
    
    if fn_constants.exists():
        with open(str(fn_constants), "r") as handle:
            ndxplorer.constants.update(json.load(handle))
    else:
        logging.warning(f"Constants file not found: {fn_constants}")

    ndxplorer.equation_editor.load_file(str(fn_equations))
