"""Tests for pyqtgraph plotting compatibility widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from qtpy import QtWidgets

from ndxplorer.plotting.pg_image_widget import PGHistogramPlot, PGImageWidget
from ndxplorer.plotting.plot_update_helpers import (
    _autoscale_horizontal_hist,
    _autoscale_vertical_hist,
)


@pytest.fixture(scope="module")
def qapp() -> QtWidgets.QApplication:
    """Create or reuse a QApplication."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_pg_image_widget_supports_existing_image_api(qapp: QtWidgets.QApplication) -> None:
    """The pyqtgraph image widget should match the current image backend API."""
    pytest.importorskip("pyqtgraph")
    widget = PGImageWidget()
    data = np.arange(9, dtype=float).reshape(3, 3)

    widget.set_data(data)
    widget.set_colormap("viridis", 0.0, 8.0)
    widget.set_lut_range([1.0, 7.0])
    widget.enable_axis("xBottom", True)
    widget.set_axis_title("xBottom", "X axis")
    widget.set_axis_scale("xBottom", 0, 2)
    widget.set_axis_scale("yLeft", 0, 2)

    assert widget.data is not None
    assert widget.data.shape == (3, 3)
    assert widget.axis_enabled("xBottom") is True


def test_pg_histogram_plot_supports_existing_histogram_api(qapp: QtWidgets.QApplication) -> None:
    """The pyqtgraph marginal plot should expose guiqwt-style calls."""
    pytest.importorskip("pyqtgraph")
    plot = PGHistogramPlot()
    item = plot.add_histogram(color="#0066cc")

    item.set_data([0, 1, 2], [3, 4, 5])
    plot.enableAxis("bottom", False)
    plot.enableAxis("top", True)
    plot.set_axis_title("top", "Parameter")
    plot.setAxisScale("xBottom", 0, 2)
    plot.set_axis_scale("bottom", "linear")
    selection = plot.add_range_selection(0.25, 0.5)
    selection.set_range(1.0, 2.0)

    assert plot.axisEnabled("bottom") is False
    assert plot.axisEnabled("top") is True
    assert selection.get_range() == (1.0, 2.0)


def test_pg_histogram_item_draws_steps_from_bin_edges(qapp: QtWidgets.QApplication) -> None:
    """Histogram edges and counts should render as steps, not diagonal lines."""
    pytest.importorskip("pyqtgraph")
    horizontal_plot = PGHistogramPlot()
    vertical_plot = PGHistogramPlot()
    horizontal = horizontal_plot.add_histogram(color="#0066cc")
    vertical = vertical_plot.add_histogram(color="#0066cc", orientation="vertical")

    horizontal.set_data([0, 1, 2, 3], [4, 5, 6])
    vertical.set_data([4, 5, 6], [0, 1, 2, 3])

    x_step, y_step = horizontal._item.getData()
    x_vertical, y_vertical = vertical._item.getData()
    np.testing.assert_allclose(x_step, [0, 1, 1, 2, 2, 3])
    np.testing.assert_allclose(y_step, [4, 4, 5, 5, 6, 6])
    np.testing.assert_allclose(x_vertical, [4, 4, 5, 5, 6, 6])
    np.testing.assert_allclose(y_vertical, [0, 1, 1, 2, 2, 3])
    assert vertical._baseline_item is not None
    baseline_x, baseline_y = vertical._baseline_item.getData()
    np.testing.assert_allclose(baseline_x, [0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(baseline_y, [0, 1, 1, 2, 2, 3])


def test_pg_histogram_autoscale_uses_compatibility_api(qapp: QtWidgets.QApplication) -> None:
    """Autoscale helpers should work without a Qwt plot instance."""
    pytest.importorskip("pyqtgraph")
    x_plot = PGHistogramPlot()
    y_plot = PGHistogramPlot()

    _autoscale_horizontal_hist(x_plot, np.array([1.0, 10.0, 100.0]), np.array([2.0, 5.0]))
    _autoscale_vertical_hist(y_plot, np.array([1.0, 10.0, 100.0]), np.array([2.0, 5.0]))

    x_range, x_count_range = x_plot.viewRange()
    y_count_range, y_range = y_plot.viewRange()
    assert x_range == [1.0, 100.0]
    assert x_count_range == [0.0, 5.25]
    assert y_range == [1.0, 100.0]
    assert y_count_range == [0.0, 5.25]


def test_pg_histogram_log_axis_scale_uses_raw_data_range(qapp: QtWidgets.QApplication) -> None:
    """Qwt-style raw log ranges should become pyqtgraph log-view ranges."""
    pytest.importorskip("pyqtgraph")
    plot = PGHistogramPlot()

    plot.set_axis_scale("bottom", "log")
    plot.setAxisScale("bottom", 1.0, 100.0)

    x_range, _ = plot.viewRange()
    assert x_range == [0.0, 2.0]
