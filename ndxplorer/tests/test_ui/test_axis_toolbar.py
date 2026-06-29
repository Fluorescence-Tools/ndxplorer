"""Tests for the ndXplorer axis controls placement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy import QtWidgets

from ndxplorer.core.plot_main import NDXplorer
from ndxplorer.plotting.plot_control import SurfacePlotWidget


@pytest.fixture(scope="session")
def qapp() -> QtWidgets.QApplication:
    """Create or reuse a QApplication instance."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_plot_control_shows_per_axis_buttons(qapp: QtWidgets.QApplication) -> None:
    """The Set/Auto buttons live next to each axis, not in a consolidated row."""
    widget = SurfacePlotWidget()

    for name in (
        "toolButtonSetXAxis",
        "toolButtonAutoX",
        "toolButtonSetYAxis",
        "toolButtonAutoY",
        "toolButtonSetZAxis",
        "toolButtonAutoZ",
    ):
        button = getattr(widget, name)
        assert button is not None
        # Not explicitly hidden (visibility otherwise depends on the parent being
        # shown); the key regression is that the code no longer force-hides them.
        assert not button.isHidden()

    widget.deleteLater()


def test_main_toolbar_does_not_duplicate_axis_actions(qapp: QtWidgets.QApplication) -> None:
    """The consolidated axis toolbar row is removed; the per-axis buttons own it."""
    toolbar = QtWidgets.QToolBar()
    actions = {
        "actionUpdate_x_axis_settings": QtWidgets.QAction("Set X"),
        "actionAuto_range_x": QtWidgets.QAction("Auto X"),
        "actionUpdate_y_axis_settings": QtWidgets.QAction("Set Y"),
        "actionAuto_range_y": QtWidgets.QAction("Auto Y"),
        "actionUpdate_z_axis_settings": QtWidgets.QAction("Set Z"),
        "actionAuto_range_z": QtWidgets.QAction("Auto Z"),
    }
    fake = SimpleNamespace(
        toolBar=toolbar,
        plot_control=SimpleNamespace(**actions),
    )

    NDXplorer._setup_axis_toolbar_actions(fake)

    # The toolbar is cleared and hidden; the axis actions are not duplicated there.
    assert toolbar.actions() == []
    assert not toolbar.isVisible()
