"""UI-related helper functions for NDXplorer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtGui import QFont

from ..logging_config import logging

if TYPE_CHECKING:  # pragma: no cover
    from ..core.plot_main import NDXplorer


def apply_fonts(ndxplorer: "NDXplorer") -> None:
    """Apply configured font settings to available plots."""
    if not getattr(ndxplorer, "_deferred_init_done", False) or ndxplorer.g_2dplot is None:
        return
    logging.debug("apply_fonts")
    try:
        fs = getattr(ndxplorer, "font_settings", {})
        tick_size = int(fs.get("tick_size_pt", 8))
        qf_tick = QFont("Segoe UI", tick_size)
    except Exception:
        qf_tick = QFont("Segoe UI", 8)

    plots = [ndxplorer.g_xplot, ndxplorer.g_yplot, ndxplorer.g_2dplot]
    if getattr(ndxplorer, "g_zplot", None) is not None:
        plots.append(ndxplorer.g_zplot)
    if getattr(ndxplorer, "overlay_plot", None) is not None:
        plots.append(ndxplorer.overlay_plot)

    for gp in plots:
        for axis in ("bottom", "top", "left", "right"):
            try:
                gp.set_axis_font(axis, qf_tick)
            except Exception:
                pass


def _set_axis_title(plot, axis: str, title: str) -> None:
    """Set an axis title on guiqwt or pyqtgraph-backed plots."""
    if plot is None:
        return
    if hasattr(plot, "set_axis_title"):
        plot.set_axis_title(axis, title)
        return
    if hasattr(plot, "setAxisTitle"):
        plot.setAxisTitle(axis, title)
        return
    if hasattr(plot, "setLabel"):
        plot.setLabel(axis, text=title or "")
        return
    plot_item = None
    if hasattr(plot, "getPlotItem"):
        try:
            plot_item = plot.getPlotItem()
        except Exception:
            plot_item = None
    if plot_item is None and hasattr(plot, "plot_widget") and hasattr(plot.plot_widget, "getPlotItem"):
        try:
            plot_item = plot.plot_widget.getPlotItem()
        except Exception:
            plot_item = None
    if plot_item is not None:
        try:
            plot_item.getAxis(axis).setLabel(text=title or "")
        except Exception:
            pass


def arrange_docks_preserving_geometry(ndxplorer: "NDXplorer") -> None:
    """Tabify the primary docks while ensuring the window size stays unchanged."""
    try:
        size_before = ndxplorer.size()
    except Exception:  # pragma: no cover - defensive guard
        logging.debug("arrange_docks_preserving_geometry: main window has no size yet")
        size_before = None

    docks = [
        getattr(ndxplorer, "dockWidget_PlotControl", None),
        getattr(ndxplorer, "dockWidget_Parameters", None),
        getattr(ndxplorer, "dockWidget_Overlays", None),
    ]
    docks = [dock for dock in docks if dock is not None]
    if len(docks) < 2:
        logging.debug("arrange_docks_preserving_geometry: not enough docks to tabify")
        return

    for i, dock in enumerate(docks[:-1]):
        ndxplorer.tabifyDockWidget(dock, docks[i + 1])
    docks[0].raise_()

    equations_dock = getattr(ndxplorer, "dockWidget_Equations", None)
    if equations_dock is not None:
        equations_dock.setVisible(False)

    if size_before is not None:
        size_after = ndxplorer.size()
        assert size_before == size_after, (
            "Arranging docks altered the NDxplorer window size before initialization."
        )

def update_parameter_names(ndxplorer: "NDXplorer") -> None:
    """Update axis labels/titles per current parameter selection and settings."""
    logging.debug("update_parameter_names")
    p1, p1_name = ndxplorer.plot_control.p1
    p2, p2_name = ndxplorer.plot_control.p2
    p3, p3_name = ndxplorer.plot_control.p3

    def fmt(title: str) -> str:
        if not title:
            return ""
        size_pt = None
        weight = 700
        color = None
        try:
            fs = getattr(ndxplorer, "font_settings", {})
            weight = int(fs.get("title_weight", 700))
            size_pt = fs.get("title_size_pt")
            color = fs.get("color")
        except Exception:
            pass
        color_css = f"; color:{color}" if color else ""
        if size_pt is not None:
            return f"<span style='font-weight:{weight}; font-size:{size_pt}pt{color_css}'>{title}</span>"
        return f"<span style='font-weight:{weight}; font-size:115%{color_css}'>{title}</span>"

    axis_settings = getattr(ndxplorer, "axis_label_settings", None)
    if axis_settings:
        enable_all = axis_settings.get("enable_all_labels", True)
        labels_cfg = axis_settings.get("axis_labels", {})
        y_cfg = labels_cfg.get("y_plot", {})
        x_cfg = labels_cfg.get("x_plot", {})
        z_cfg = labels_cfg.get("z_plot", {})

        _set_axis_title(ndxplorer.g_yplot, "top", fmt(p2_name) if (enable_all or y_cfg.get("top", True)) else "")
        _set_axis_title(ndxplorer.g_yplot, "right", fmt(p2_name) if (enable_all or y_cfg.get("right", True)) else "")
        _set_axis_title(ndxplorer.g_xplot, "top", fmt(p1_name) if (enable_all or x_cfg.get("top", True)) else "")

        if getattr(ndxplorer, "g_zplot", None) is not None:
            _set_axis_title(ndxplorer.g_zplot, "bottom", fmt(p3_name) if (enable_all or z_cfg.get("bottom", True)) else "")
            _set_axis_title(ndxplorer.g_zplot, "left", fmt(p3_name) if (enable_all or z_cfg.get("left", True)) else "")
    else:
        _set_axis_title(ndxplorer.g_yplot, "top", fmt(p2_name))
        _set_axis_title(ndxplorer.g_yplot, "right", fmt(p2_name))
        _set_axis_title(ndxplorer.g_xplot, "top", fmt(p1_name))
        if getattr(ndxplorer, "g_zplot", None) is not None:
            _set_axis_title(ndxplorer.g_zplot, "bottom", fmt(p3_name))
            _set_axis_title(ndxplorer.g_zplot, "left", fmt(p3_name))
