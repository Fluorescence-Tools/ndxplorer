"""Helper utilities extracted from plot_main to keep the module manageable."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QFont, QImage

try:
    import guiqwt.curve
    import guiqwt.plot
    import guiqwt.styles
    from guiqwt.builder import make
    from guiqwt.plot import CurveDialog
    from qwt.plot import QwtPlot
    from qwt.plot_canvas import QwtPlotCanvas
    GUIQWT_AVAILABLE = True
except ImportError:
    GUIQWT_AVAILABLE = False
    # Create dummy objects for type checking
    guiqwt = None
    make = None
    CurveDialog = None
    QwtPlot = None
    QwtPlotCanvas = None

from .image_items import FixedImageItem
from .simple_image_widget import SimpleImageWidget
from ..logging_config import logging
from ..utils.mouse_event_filter import MouseEventFilter
from ..widgets import ScientificSpinBox

# Check environment variable for backend selection
# NDXPLORER_2D_BACKEND can be "guiqwt" or "simple" (default: "simple")
USE_SIMPLE_BACKEND = os.environ.get('NDXPLORER_2D_BACKEND', 'simple').lower() != 'guiqwt'

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
    
    if not GUIQWT_AVAILABLE:
        logging.warning("guiqwt not available, skipping histogram plot setup")
        return
    
    logging.info("Creating Z marginal plot")
    win_z = CurveDialog()
    ndxplorer.g_zplot = win_z.get_plot()
    curveparam_z = guiqwt.styles.CurveParam("Curve", icon="curve.png")
    curveparam_z.curvestyle = "Steps"
    curveparam_z.line.color = "#ff00ff"
    curveparam_z.shade = 0.5
    curveparam_z.line.width = 2.0
    ndxplorer.g_zhist_m = guiqwt.curve.CurveItem(curveparam=curveparam_z)
    ndxplorer.g_zplot.add_item(ndxplorer.g_zhist_m)
    ndxplorer.selection_z = make.range(0.25, 0.5)
    ndxplorer.g_zplot.add_item(ndxplorer.selection_z)
    _replace_placeholder(
        ndxplorer.plot_control.verticalLayout_4,
        getattr(ndxplorer, "_placeholder_z", None),
        ndxplorer.g_zplot,
    )
    ndxplorer.g_zplot.setVisible(ndxplorer.groupBox_3.isChecked())

    logging.info("Creating X marginal plot")
    win_x = CurveDialog()
    ndxplorer.g_xplot = win_x.get_plot()
    ndxplorer.g_xplot.enableAxis(QwtPlot.xBottom, False)
    ndxplorer.g_xplot.enableAxis(QwtPlot.xTop, True)
    ndxplorer.g_xplot.enableAxis(QwtPlot.yLeft, False)
    ndxplorer.g_xplot.enableAxis(QwtPlot.yRight, False)

    curveparam_x = guiqwt.styles.CurveParam("Curve", icon="curve.png")
    curveparam_x.curvestyle = "Steps"
    curveparam_x.line.color = "#0066cc"
    curveparam_x.shade = 0.5
    curveparam_x.line.width = 2.0
    ndxplorer.g_xhist_m = guiqwt.curve.CurveItem(curveparam=curveparam_x)
    ndxplorer.g_xplot.add_item(ndxplorer.g_xhist_m)
    x_layout = getattr(ndxplorer, "_x_hist_layout", None)
    if x_layout is None:
        x_layout = _get_layout(ndxplorer, "verticalLayout_xhist", ("verticalLayout_5",))
        ndxplorer._x_hist_layout = x_layout
    _replace_placeholder(
        x_layout,
        getattr(ndxplorer, "_placeholder_x", None),
        ndxplorer.g_xplot,
    )

    logging.info("Creating Y marginal plot")
    win_y = CurveDialog()
    ndxplorer.g_yplot = win_y.get_plot()
    ndxplorer.g_yplot.enableAxis(QwtPlot.xBottom, False)
    ndxplorer.g_yplot.enableAxis(QwtPlot.xTop, False)
    ndxplorer.g_yplot.enableAxis(QwtPlot.yLeft, False)
    ndxplorer.g_yplot.enableAxis(QwtPlot.yRight, True)

    curveparam_y = guiqwt.styles.CurveParam()
    curveparam_y.curvestyle = "Steps"
    curveparam_y.line.color = "#00ff00"
    curveparam_y.shade = 0.1
    curveparam_y.line.width = 2.0
    ndxplorer.g_yhist_m = guiqwt.curve.CurveItem(curveparam=curveparam_y)
    ndxplorer.g_yplot.add_item(ndxplorer.g_yhist_m)
    y_layout = getattr(ndxplorer, "_y_hist_layout", None)
    if y_layout is None:
        y_layout = _get_layout(ndxplorer, "verticalLayout_yhist", ("verticalLayout_7",))
        ndxplorer._y_hist_layout = y_layout
    _replace_placeholder(
        y_layout,
        getattr(ndxplorer, "_placeholder_y", None),
        ndxplorer.g_yplot,
    )

    logging.info("Configuring marginal plot canvases")
    for canvas in (
        ndxplorer.g_xplot.canvas(),
        ndxplorer.g_yplot.canvas(),
        ndxplorer.g_zplot.canvas(),
    ):
        canvas.setStyleSheet("background-color: white;")
        try:
            canvas.setPaintAttribute(QwtPlotCanvas.BackingStore, False)
            canvas.setPaintAttribute(QwtPlotCanvas.Opaque, True)
            canvas.setPaintAttribute(QwtPlotCanvas.HackStyledBackground, False)
            canvas.setPaintAttribute(QwtPlotCanvas.ImmediatePaint, True)
        except Exception:
            pass

    # Constrain x-axis marginal height to match control widget height
    target_height = _top_row_target_height(ndxplorer)
    ndxplorer.g_xplot.setMinimumHeight(0)
    ndxplorer.g_xplot.setMaximumHeight(target_height)
    
    # Constrain y-axis marginal width to match control widget width
    controls_widget = getattr(ndxplorer, "widget_6", None)
    if controls_widget is not None:
        target_width = controls_widget.sizeHint().width()
        if target_width <= 0:
            target_width = 150
    else:
        target_width = 150
    ndxplorer.g_yplot.setMinimumWidth(0)
    ndxplorer.g_yplot.setMaximumWidth(target_width)
    
    # Z-plot can expand freely
    ndxplorer.g_zplot.setMinimumHeight(0)
    ndxplorer.g_zplot.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)


def setup_2d_histogram_plot(ndxplorer: "NDXplorer", cmap: str) -> None:
    """Create the 2D histogram plot widget (guiqwt or simple backend based on env var)."""
    
    if USE_SIMPLE_BACKEND:
        # Use simple Qt-based image widget
        logging.info("Using simple Qt backend for 2D histogram plot")
        ndxplorer.g_2dplot = SimpleImageWidget(parent=ndxplorer)
        ndxplorer.cax = ndxplorer.g_2dplot
        ndxplorer._use_simple_backend = True
        
        # Set initial empty data
        data = np.zeros((10, 10))
        ndxplorer.g_2dplot.set_data(data)
        
        # Load background image for when there's no data
        bg_image_path = _find_background_image()
        if bg_image_path:
            ndxplorer.g_2dplot.set_background_image(bg_image_path)
        else:
            logging.warning("Background image not found: %s", bg_image_path)
        
        # Set default colormap
        ndxplorer.set_default_colormap(cmap)
        
        # Set axis font
        ndxplorer.g_2dplot.set_axis_font("left", QFont("Courier"))
        
        # Store font settings
        ndxplorer.font_settings = {
            "tick_size_pt": 8,
            "title_size_pt": 10,
            "title_weight": 700,
            "color": "#000000",
        }
        
        # Disable all axes by default
        ndxplorer.g_2dplot.enable_axis("xBottom", False)
        ndxplorer.g_2dplot.enable_axis("xTop", False)
        ndxplorer.g_2dplot.enable_axis("yLeft", False)
        ndxplorer.g_2dplot.enable_axis("yRight", False)
        
        # Enable mouse tracking
        ndxplorer.g_2dplot.setMouseTracking(True)
        
        # Install event filter
        try:
            ndxplorer.g_2dplot.installEventFilter(ndxplorer)
        except Exception:
            pass
    else:
        # Use guiqwt backend
        if not GUIQWT_AVAILABLE:
            logging.error("guiqwt backend requested but not available, falling back to simple backend")
            # Fall back to simple backend by creating SimpleImageWidget directly
            logging.info("Using simple Qt backend for 2D histogram plot (fallback)")
            ndxplorer.g_2dplot = SimpleImageWidget(parent=ndxplorer)
            ndxplorer.cax = ndxplorer.g_2dplot
            ndxplorer._use_simple_backend = True
            
            data = np.zeros((10, 10))
            ndxplorer.g_2dplot.set_data(data)
            
            bg_image_path = _find_background_image()
            if bg_image_path:
                ndxplorer.g_2dplot.set_background_image(bg_image_path)
            
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
            return
        
        logging.info("Using guiqwt backend for 2D histogram plot")
        ndxplorer._use_simple_backend = False
        
        win_2d = guiqwt.plot.ImageDialog(edit=False, toolbar=False)
        ndxplorer.g_2dplot = win_2d.get_plot()

        # Create empty initial data for the main image
        data = np.zeros((10, 10))
        ndxplorer.cax = FixedImageItem(data=data)
        ndxplorer.g_2dplot.add_item(ndxplorer.cax)

        # Load background image for when there's no data
        bg_image_path = _find_background_image()
        ndxplorer.bg_image_item = None
        if bg_image_path:
            bg_qimage = QImage(bg_image_path)
            if not bg_qimage.isNull():
                bg_qimage = bg_qimage.convertToFormat(QImage.Format_RGBA8888)
                width = bg_qimage.width()
                height = bg_qimage.height()
                ptr = bg_qimage.bits()
                ptr.setsize(height * width * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                ndxplorer.bg_image_item = FixedImageItem(data=arr[:, :, 0].copy())
                ndxplorer.g_2dplot.add_item(ndxplorer.bg_image_item)
                try:
                    ndxplorer.bg_image_item.setZ(ndxplorer.cax.z() + 1.0)
                except Exception:
                    ndxplorer.bg_image_item.setZ(1.0)
                ndxplorer.bg_image_item.setVisible(True)
        else:
            logging.warning("Background image not found: %s", bg_image_path)

        try:
            ndxplorer.cax.setZ(0.0)
        except Exception:
            pass

        ndxplorer.set_default_colormap(cmap)
        ndxplorer.g_2dplot.set_axis_font("left", QFont("Courier"))
        ndxplorer.g_2dplot.set_axis_font("bottom", QFont("Courier"))
        ndxplorer.font_settings = {
            "tick_size_pt": 8,
            "title_size_pt": 10,
            "title_weight": 700,
            "color": "#000000",
        }
        try:
            ndxplorer.apply_fonts()
        except Exception:
            pass

        ndxplorer.g_2dplot.enableAxis(QwtPlot.xBottom, False)
        ndxplorer.g_2dplot.enableAxis(QwtPlot.xTop, False)
        ndxplorer.g_2dplot.enableAxis(QwtPlot.yLeft, False)
        ndxplorer.g_2dplot.enableAxis(QwtPlot.yRight, False)
        ndxplorer.g_2dplot.canvas().setStyleSheet("background-color: white;")
        ndxplorer.g_2dplot.canvas().setMouseTracking(True)
        try:
            ndxplorer.g_2dplot.canvas().installEventFilter(ndxplorer)
        except Exception:
            pass


def setup_overlay_plot(ndxplorer: "NDXplorer") -> None:
    """Create a transparent overlay plot stacked on top of the 2D histogram."""
    # Use PyQt overlay widget for simple backend, guiqwt for guiqwt backend
    use_simple = getattr(ndxplorer, '_use_simple_backend', True)
    
    if use_simple:
        # Use PyQt-based overlay widget
        from ..widgets.drawing_overlay_widget import DrawingOverlayWidget
        ndxplorer.overlay_plot = DrawingOverlayWidget(parent=ndxplorer)
        ndxplorer.mouse_event_filter = MouseEventFilter(ndxplorer)
        ndxplorer.overlay_plot.installEventFilter(ndxplorer.mouse_event_filter)
    else:
        # Use guiqwt overlay (original implementation)
        if not GUIQWT_AVAILABLE:
            logging.warning("guiqwt not available, skipping overlay plot setup")
            return
        
        ndxplorer.overlay_plot = guiqwt.curve.CurvePlot(parent=ndxplorer)
        ndxplorer.mouse_event_filter = MouseEventFilter(ndxplorer)
        ndxplorer.overlay_plot.canvas().installEventFilter(ndxplorer.mouse_event_filter)
        ndxplorer.overlay_plot.grid.setVisible(False)
        ndxplorer.overlay_plot.setAutoFillBackground(False)
        ndxplorer.overlay_plot.setStyleSheet("background-color: transparent;")
        ndxplorer.overlay_plot.setFrameStyle(QtWidgets.QFrame.NoFrame)
        ndxplorer.overlay_plot.canvas().setAutoFillBackground(False)
        ndxplorer.overlay_plot.canvas().setStyleSheet("background-color: transparent;")
        ndxplorer.overlay_plot.canvas().setFrameStyle(QtWidgets.QFrame.NoFrame)
        ndxplorer.overlay_plot.canvas().setPaintAttribute(
            QwtPlotCanvas.BackingStore, False
        )
        ndxplorer.overlay_plot.canvas().setPaintAttribute(QwtPlotCanvas.Opaque, False)
        ndxplorer.overlay_plot.canvas().setPaintAttribute(
            QwtPlotCanvas.HackStyledBackground, False
        )
        ndxplorer.overlay_plot.canvas().setPaintAttribute(
            QwtPlotCanvas.ImmediatePaint, True
        )

        ndxplorer.overlay_plot.enableAxis(QwtPlot.xBottom, False)
        ndxplorer.overlay_plot.enableAxis(QwtPlot.xTop, False)
        ndxplorer.overlay_plot.enableAxis(QwtPlot.yLeft, False)
        ndxplorer.overlay_plot.enableAxis(QwtPlot.yRight, False)

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
