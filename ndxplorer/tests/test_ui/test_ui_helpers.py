"""Tests for UI helper compatibility."""

from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtWidgets

from ndxplorer.utils.ui_helpers import update_parameter_names


def test_update_parameter_names_supports_pyqtgraph_plotwidgets() -> None:
    """Axis titles should update on plain pyqtgraph plot widgets."""
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Pyqtgraph PlotWidget is enough to reproduce the arm64 failure path.
    import pyqtgraph as pg

    x_plot = pg.PlotWidget()
    y_plot = pg.PlotWidget()
    z_plot = pg.PlotWidget()

    ndxplorer = SimpleNamespace(
        plot_control=SimpleNamespace(
            p1=(0, "X axis"),
            p2=(1, "Y axis"),
            p3=(2, "Z axis"),
        ),
        g_xplot=x_plot,
        g_yplot=y_plot,
        g_zplot=z_plot,
        axis_label_settings=None,
        font_settings={},
    )

    update_parameter_names(ndxplorer)

    assert "X axis" in x_plot.getPlotItem().getAxis("top").labelText
    assert "Y axis" in y_plot.getPlotItem().getAxis("top").labelText
    assert "Z axis" in z_plot.getPlotItem().getAxis("bottom").labelText
