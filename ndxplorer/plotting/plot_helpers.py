"""Helper utilities extracted from plot_main to keep the module manageable."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QFont

from .pg_image_widget import PGHistogramPlot, PGImageWidget
from ..logging_config import logging
from ..utils.mouse_event_filter import MouseEventFilter
from ..widgets import ScientificSpinBox

if TYPE_CHECKING:  # pragma: no cover
    from ..core.plot_main import NDXplorer


def setup_histogram_spinboxes(ndxplorer: "NDXplorer") -> None:
    """Create the scientific spin boxes used for vmin/vmax selection."""
    ndxplorer.doubleSpinBox_vmin = ScientificSpinBox(ndxplorer, format_str="%.2e")
    ndxplorer.doubleSpinBox_vmax = ScientificSpinBox(ndxplorer, format_str="%.2e")

    for spinbox in (ndxplorer.doubleSpinBox_vmin, ndxplorer.doubleSpinBox_vmax):
        spinbox.setRange(-1e10, 1e10)
        spinbox.setSingleStep(0.1)

    ndxplorer.doubleSpinBox_vmin.setValue(0.0)
    ndxplorer.doubleSpinBox_vmax.setValue(1.0)

    layout = ndxplorer.horizontalLayout_3
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(ndxplorer.doubleSpinBox_vmin)
    layout.addWidget(ndxplorer.doubleSpinBox_vmax)

    ndxplorer.doubleSpinBox_vmin.valueChanged.connect(ndxplorer.on_vmin_vmax_changed)
    ndxplorer.doubleSpinBox_vmax.valueChanged.connect(ndxplorer.on_vmin_vmax_changed)


class _BackgroundLabel(QtWidgets.QLabel):
    """QLabel that displays an image centered with aspect ratio preserved."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap(image_path)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(100, 100)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap.isNull():
            return
        target_size = self.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self._pixmap.scaled(
            target_size,
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation,
        )
        if (
            scaled.width() > target_size.width()
            or scaled.height() > target_size.height()
        ):
            x_offset = max((scaled.width() - target_size.width()) // 2, 0)
            y_offset = max((scaled.height() - target_size.height()) // 2, 0)
            scaled = scaled.copy(
                x_offset,
                y_offset,
                target_size.width(),
                target_size.height(),
            )
        self.setPixmap(scaled)



def _get_layout(ndxplorer: "NDXplorer", primary: str, fallbacks: tuple[str, ...] = ()) -> QtWidgets.QLayout:
    layout = getattr(ndxplorer, primary, None)
    if layout is not None:
        return layout
    for name in fallbacks:
        layout = getattr(ndxplorer, name, None)
        if layout is not None:
            logging.warning("Using fallback layout '%s' for '%s'", name, primary)
            return layout
    raise AttributeError(f"Could not find layout '{primary}' (fallbacks tried: {fallbacks})")


def _top_row_target_height(ndxplorer: "NDXplorer") -> int:
    controls_widget = getattr(ndxplorer, "widget_6", None)
    for method in ("sizeHint", "minimumSizeHint", "size"):
        if controls_widget is None:
            break
        size = getattr(controls_widget, method)()
        if size.height() > 0:
            return size.height()
    return 140


def setup_plot_placeholders(ndxplorer: "NDXplorer") -> None:
    """Create placeholder widgets that maintain correct layout until real plots are created.
    
    These placeholders have the same size constraints as the real plots so the layout
    appears correct immediately when the window is shown.
    """
    # Z-axis placeholder (marginal histogram at top of plot_control)
    ndxplorer._placeholder_z = QtWidgets.QFrame()
    ndxplorer._placeholder_z.setFrameStyle(QtWidgets.QFrame.StyledPanel)
    ndxplorer._placeholder_z.setStyleSheet("background-color: white;")
    ndxplorer._placeholder_z.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
    )
    ndxplorer.plot_control.verticalLayout_4.addWidget(ndxplorer._placeholder_z)

    # X-axis placeholder (horizontal histogram above 2D plot)
    ndxplorer._placeholder_x = QtWidgets.QFrame()
    ndxplorer._placeholder_x.setFrameStyle(QtWidgets.QFrame.StyledPanel)
    ndxplorer._placeholder_x.setStyleSheet("background-color: white;")
    ndxplorer._placeholder_x.setMaximumHeight(100)
    ndxplorer._placeholder_x.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
    )
    target_height = _top_row_target_height(ndxplorer)
    ndxplorer._placeholder_x.setMaximumHeight(target_height)
    x_layout = _get_layout(ndxplorer, "verticalLayout_xhist", ("verticalLayout_5",))
    ndxplorer._x_hist_layout = x_layout
    x_layout.addWidget(ndxplorer._placeholder_x)

    # Y-axis placeholder (vertical histogram to right of 2D plot)
    ndxplorer._placeholder_y = QtWidgets.QFrame()
    ndxplorer._placeholder_y.setFrameStyle(QtWidgets.QFrame.StyledPanel)
    ndxplorer._placeholder_y.setStyleSheet("background-color: white;")
    ndxplorer._placeholder_y.setMaximumWidth(150)
    ndxplorer._placeholder_y.setMaximumWidth(150)
    ndxplorer._placeholder_y.setSizePolicy(
        QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
    )
    ndxplorer._placeholder_y.setMaximumWidth(120)
    y_layout = _get_layout(ndxplorer, "verticalLayout_yhist", ("verticalLayout_7",))
    ndxplorer._y_hist_layout = y_layout
    y_layout.addWidget(ndxplorer._placeholder_y)

    # 2D plot placeholder (main histogram area) - use background.png
    bg_path = _find_background_image()
    if bg_path:
        ndxplorer._placeholder_2d = _BackgroundLabel(bg_path)
    else:
        ndxplorer._placeholder_2d = QtWidgets.QFrame()
        ndxplorer._placeholder_2d.setStyleSheet("background-color: white;")
    ndxplorer._placeholder_2d.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
    )
    ndxplorer.verticalLayout_11.addWidget(ndxplorer._placeholder_2d)


def _replace_placeholder(layout, placeholder, new_widget) -> None:
    """Replace a placeholder widget in a layout with the real widget."""
    if placeholder is None:
        layout.addWidget(new_widget)
        return
    idx = layout.indexOf(placeholder)
    if idx >= 0:
        layout.removeWidget(placeholder)
        placeholder.deleteLater()
        layout.insertWidget(idx, new_widget)
    else:
        layout.addWidget(new_widget)


def configure_dynamic_selection_controls(ndxplorer: "NDXplorer") -> None:
    """Wire dynamic selection, Z enable and weighting controls."""
    plot_control = ndxplorer.plot_control

    ndxplorer.checkBoxDynamicSelection = plot_control.checkBoxDynamicSelection
    ndxplorer.checkBoxDynamicSelection.setToolTip(
        "When checked, 2D and 1D histograms (except Z) will only display "
        "data selected by region selector"
    )
    ndxplorer.checkBoxDynamicSelection.setChecked(ndxplorer._dynamic_selection)
    ndxplorer.checkBoxDynamicSelection.stateChanged.connect(
        ndxplorer.on_dynamic_selection_changed
    )

    ndxplorer._last_z_range = None

    ndxplorer.groupBox_3 = plot_control.groupBox_3
    ndxplorer.groupBox_3.setToolTip("When checked, the Z-axis plot is displayed")
    ndxplorer.groupBox_3.toggled.connect(ndxplorer.on_enable_z_changed)

    ndxplorer.checkBoxWeight = plot_control.checkBoxWeight
    ndxplorer.checkBoxWeight.setToolTip(
        "If checked, histograms are weighted by selected parameter"
    )
    ndxplorer.checkBoxWeight.stateChanged.connect(ndxplorer.on_weight_changed)

    ndxplorer.comboBoxWeight = plot_control.comboBoxWeight
    ndxplorer.comboBoxWeight.setToolTip("Select parameter to use as weights")
    ndxplorer.comboBoxWeight.setEnabled(ndxplorer.checkBoxWeight.isChecked())
    ndxplorer.comboBoxWeight.currentIndexChanged.connect(
        ndxplorer.on_weight_param_changed
    )

    ndxplorer.z_range_check_timer = QtCore.QTimer(ndxplorer)
    ndxplorer.z_range_check_timer.setInterval(500)
    ndxplorer._z_timer_connected = False
    ndxplorer.checkBoxDynamicSelection.toggled.connect(
        ndxplorer.on_dynamic_selection_toggled
    )
    ndxplorer.on_dynamic_selection_toggled(
        ndxplorer.checkBoxDynamicSelection.isChecked()
    )


def setup_histogram_plots(ndxplorer: "NDXplorer") -> None:
    """Create the marginal histogram plots for X, Y and Z, replacing placeholders."""
    logging.info("Setting up histogram plots")

    logging.info("Creating pyqtgraph marginal plots")
    ndxplorer.g_zplot = PGHistogramPlot(parent=ndxplorer)
    ndxplorer.g_zhist_m = ndxplorer.g_zplot.add_histogram(color="#ff00ff", fill=0.5)
    ndxplorer.selection_z = ndxplorer.g_zplot.add_range_selection(0.25, 0.5)
    _replace_placeholder(
        ndxplorer.plot_control.verticalLayout_4,
        getattr(ndxplorer, "_placeholder_z", None),
        ndxplorer.g_zplot,
    )
    ndxplorer.g_zplot.setVisible(ndxplorer.groupBox_3.isChecked())

    ndxplorer.g_xplot = PGHistogramPlot(parent=ndxplorer)
    ndxplorer.g_xplot.enableAxis("bottom", False)
    ndxplorer.g_xplot.enableAxis("top", True)
    ndxplorer.g_xplot.enableAxis("left", False)
    ndxplorer.g_xplot.enableAxis("right", False)
    ndxplorer.g_xhist_m = ndxplorer.g_xplot.add_histogram(color="#0066cc", fill=0.5)
    x_layout = getattr(ndxplorer, "_x_hist_layout", None)
    if x_layout is None:
        x_layout = _get_layout(ndxplorer, "verticalLayout_xhist", ("verticalLayout_5",))
        ndxplorer._x_hist_layout = x_layout
    _replace_placeholder(
        x_layout,
        getattr(ndxplorer, "_placeholder_x", None),
        ndxplorer.g_xplot,
    )

    ndxplorer.g_yplot = PGHistogramPlot(parent=ndxplorer)
    ndxplorer.g_yplot.enableAxis("bottom", False)
    ndxplorer.g_yplot.enableAxis("top", False)
    ndxplorer.g_yplot.enableAxis("left", False)
    ndxplorer.g_yplot.enableAxis("right", True)
    ndxplorer.g_yhist_m = ndxplorer.g_yplot.add_histogram(
        color="#00aa00",
        fill=0.15,
        orientation="vertical",
    )
    y_layout = getattr(ndxplorer, "_y_hist_layout", None)
    if y_layout is None:
        y_layout = _get_layout(ndxplorer, "verticalLayout_yhist", ("verticalLayout_7",))
        ndxplorer._y_hist_layout = y_layout
    _replace_placeholder(
        y_layout,
        getattr(ndxplorer, "_placeholder_y", None),
        ndxplorer.g_yplot,
    )

    target_height = _top_row_target_height(ndxplorer)
    ndxplorer.g_xplot.setMinimumHeight(0)
    ndxplorer.g_xplot.setMaximumHeight(target_height)

    controls_widget = getattr(ndxplorer, "widget_6", None)
    target_width = controls_widget.sizeHint().width() if controls_widget is not None else 150
    if target_width <= 0:
        target_width = 150
    ndxplorer.g_yplot.setMinimumWidth(0)
    ndxplorer.g_yplot.setMaximumWidth(target_width)
    ndxplorer.g_zplot.setMinimumHeight(0)
    ndxplorer.g_zplot.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)


def _configure_image_widget(ndxplorer: "NDXplorer", cmap: str, widget: QtWidgets.QWidget) -> None:
    """Configure a pyqtgraph image widget."""
    ndxplorer.g_2dplot = widget
    ndxplorer.cax = widget
    ndxplorer._use_simple_backend = True

    data = np.zeros((10, 10))
    ndxplorer.g_2dplot.set_data(data)

    bg_image_path = _find_background_image()
    if bg_image_path:
        ndxplorer.g_2dplot.set_background_image(bg_image_path)
    else:
        logging.warning("Background image not found: %s", bg_image_path)

    ndxplorer.set_default_colormap(cmap)
    ndxplorer.g_2dplot.set_axis_font("left", QFont("Courier"))
    ndxplorer.font_settings = {
        "tick_size_pt": 8,
        "title_size_pt": 10,
        "title_weight": 700,
        "color": "#000000",
    }

    ndxplorer.g_2dplot.enable_axis("xBottom", False)
    ndxplorer.g_2dplot.enable_axis("xTop", False)
    ndxplorer.g_2dplot.enable_axis("yLeft", False)
    ndxplorer.g_2dplot.enable_axis("yRight", False)
    ndxplorer.g_2dplot.setMouseTracking(True)

    try:
        ndxplorer.g_2dplot.installEventFilter(ndxplorer)
    except Exception:
        pass


def setup_2d_histogram_plot(ndxplorer: "NDXplorer", cmap: str) -> None:
    """Create the 2D histogram plot widget using pyqtgraph."""
    logging.info("Creating pyqtgraph 2D histogram plot")
    _configure_image_widget(ndxplorer, cmap, PGImageWidget(parent=ndxplorer))
    ndxplorer._use_pyqtgraph_backend = True


def setup_overlay_plot(ndxplorer: "NDXplorer") -> None:
    """Create a transparent overlay plot stacked on top of the 2D histogram."""
    from ..widgets.drawing_overlay_widget import DrawingOverlayWidget
    ndxplorer.overlay_plot = DrawingOverlayWidget(parent=ndxplorer)
    ndxplorer.mouse_event_filter = MouseEventFilter(ndxplorer)
    ndxplorer.overlay_plot.installEventFilter(ndxplorer.mouse_event_filter)

    plot_container = QtWidgets.QWidget()
    plot_container.setAutoFillBackground(False)
    plot_container.setStyleSheet("background-color: transparent;")
    plot_container.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
    )
    plot_layout = QtWidgets.QGridLayout(plot_container)
    plot_layout.setContentsMargins(0, 0, 0, 0)
    plot_layout.setSpacing(0)
    plot_layout.addWidget(ndxplorer.g_2dplot, 0, 0)
    plot_layout.addWidget(ndxplorer.overlay_plot, 0, 0)

    stack_widget = QtWidgets.QStackedWidget()
    stack_widget.setContentsMargins(0, 0, 0, 0)
    stack_widget.setAutoFillBackground(False)
    ndxplorer._plot_container = plot_container

    bg_label_path = _find_background_image()
    if bg_label_path:
        ndxplorer._background_label = _BackgroundLabel(bg_label_path)
    else:
        bg_label = QtWidgets.QLabel()
        bg_label.setStyleSheet("background-color: #111111;")
        ndxplorer._background_label = bg_label
    stack_widget.addWidget(ndxplorer._background_label)
    stack_widget.addWidget(plot_container)
    ndxplorer._plot_stack_widget = stack_widget

    _replace_placeholder(
        ndxplorer.verticalLayout_11,
        getattr(ndxplorer, "_placeholder_2d", None),
        stack_widget,
    )
    try:
        ndxplorer._set_data_loaded(getattr(ndxplorer, "_has_real_data", False))
    except AttributeError:
        pass


def _find_background_image() -> Optional[str]:
    """Locate the NDxplorer background image if shipped."""
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "ui", "background.png"),
        os.path.join(os.path.dirname(base_dir), "ui", "background.png"),
    ]
    for path in candidates:
        norm_path = os.path.abspath(os.path.normpath(path))
        if os.path.exists(norm_path):
            return norm_path
    logging.info(
        "NDXplorer background image not found. Searched: %s",
        ", ".join(candidates),
    )
    return None


__all__ = [
    'setup_histogram_spinboxes',
    'setup_plot_placeholders',
    'configure_dynamic_selection_controls',
    'setup_histogram_plots',
    'setup_2d_histogram_plot',
    'setup_overlay_plot',
]
