from typing import Dict, List, Optional, Tuple, Set

import sys
import os
import json
import yaml
import typing

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    import umap
except ImportError:
    umap = None

from . plot_control import SurfacePlotWidget
from qtpy.QtCore import QThread, Signal
from . parameter_editor import ParameterEditor
from . curve_overlay import CurveOverlayWidget, CurveEvaluator
try:
    from chisurf.gui.tools.code_editor import CodeEditor
except:
    from . qsci_editor import CodeEditor
try:
    from chisurf import logging
except:
    import logging
    logging.basicConfig()

from . data_source import DataSource
from guidata.widgets.dataframeeditor import DataFrameEditor

import pathlib
import pandas as pd

try:
    from chisurf.gui import QtGui, QtCore, uic, QtWidgets
    from chisurf.gui.QtGui import QFont, QImage
except ImportError:
    from qtpy import QtCore, uic
    from qtpy import QtGui, QtWidgets
    from qtpy.QtGui import QFont, QImage

from qwt.plot import QwtPlot
from qwt.plot_canvas import QwtPlotCanvas

import guiqwt.signals
import guiqwt.plot
import guiqwt.image
import guiqwt.curve
import guiqwt.styles
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

# Custom ImageItem class that fixes the float to int conversion issue
class FixedImageItem(guiqwt.image.ImageItem):
    def draw(self, painter, xMap, yMap, canvasRect):
        x1, y1, x2, y2 = canvasRect.getCoords()
        i1, i2 = xMap.invTransform(x1), xMap.invTransform(x2)
        j1, j2 = yMap.invTransform(y1), yMap.invTransform(y2)

        xl, yt, xr, yb = self.boundingRect().getCoords()
        dest = (
            xMap.transform(xl),
            yMap.transform(yt),
            xMap.transform(xr) + 1,
            yMap.transform(yb) + 1,
        )

        # Convert float to int for W and H
        W = int(canvasRect.right())
        H = int(canvasRect.bottom())
        if self._offscreen.shape != (H, W):
            self._offscreen = np.empty((H, W), np.uint32)
            self._image = QImage(self._offscreen, W, H, QImage.Format_ARGB32)
            self._image.ndarray = self._offscreen
            self.notify_new_offscreen()
        self.draw_image(painter, canvasRect, (i1, j1, i2, j2), dest, xMap, yMap)
        self.draw_border(painter, xMap, yMap, canvasRect)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from guiqwt.colormap import get_colormap_list

from . import reader
from . import writer
from .clustering_dialog import ClusteringDialog
from .column_selection_dialog import ColumnSelectionDialog


class NDXplorer(QtWidgets.QMainWindow):

    def invalidate_values_cache(self) -> None:
        """
        Manually clear the cached 'values'. Call this whenever something
        changes that would invalidate the mask or the data.
        """
        logging.log(0, "Invalidating values cache")
        self._cached_values = None
        self._cached_values_selections = None
        self._cached_values_p13 = None
        self._cached_values_mask_inf = None
        self._cached_values_mask_nan = None

    @property
    def data_source(self) -> DataSource:
        logging.log(0, "Getting data_source")
        if self._data_source.empty:
            values = self._default_data_source
            logging.log(0, "Using default data source")
        else:
            values = self._data_source
            logging.log(0, f"Using actual data source with {self._data_source.values.shape[1] if not self._data_source.empty else 0} data points")
        return values

    @data_source.setter
    def data_source(self, v: DataSource) -> None:
        logging.log(0, f"Setting data_source with {v.values.shape[1] if not v.empty else 0} data points")
        self._data_source = v
        # Whenever the underlying DataSource changes, invalidate the cached 'values'
        self.invalidate_values_cache()
        logging.log(0, "Computing columns with equations and constants")
        self._data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )

    @property
    def x_values(self) -> np.ndarray:
        logging.log(0, f"Getting x_values for parameter: {self.plot_control.p1[1]}")
        return self.values[self.plot_control.p1[0]].astype('float64')

    @property
    def y_values(self) -> np.ndarray:
        logging.log(0, f"Getting y_values for parameter: {self.plot_control.p2[1]}")
        return self.values[self.plot_control.p2[0]].astype('float64')

    @property
    def z_values(self)-> np.ndarray:
        logging.log(0, f"Getting z_values for parameter: {self.plot_control.p3[1]}")
        return self.values[self.plot_control.p3[0]].astype('float64')

    @property
    def value_mask(self):
        selections = self.plot_control.get_selections()
        mask_inf = self._mask_inf
        mask_nan = self._mask_nan
        p13 = (self.plot_control.p1[0], self.plot_control.p2[0], self.plot_control.p3[0])
        logging.log(0, f"Value mask parameters: p13={p13}, mask_inf={mask_inf}, mask_nan={mask_nan}, selections={len(selections)}")

        # Check if dynamic selection is enabled
        dynamic_selection = self._dynamic_selection and hasattr(self, 'selection_z')

        # Check if clustering is enabled and a specific cluster is selected
        selected_cluster = self.plot_control.selected_cluster
        use_clustering = self._use_clustering and selected_cluster >= 0

        # Step 2: Check if our existing cache is still valid
        cache_is_valid = (
            self._cached_values is not None
            and self._cached_values_selections == selections
            and self._cached_values_p13 == p13
            and self._cached_values_mask_inf == mask_inf
            and self._cached_values_mask_nan == mask_nan
            and getattr(self, '_cached_values_dynamic_selection', None) == dynamic_selection
            and getattr(self, '_cached_values_z_range', None) == getattr(self, '_last_z_range', None)
            and getattr(self, '_cached_values_use_clustering', None) == use_clustering
            and getattr(self, '_cached_values_selected_cluster', None) == selected_cluster
        )
        if cache_is_valid:
            logging.log(0, "Using cached values")
            return self._cached_values

        # Step 3: If cache is invalid or empty, compute fresh data
        logging.log(0, "Cache invalid, computing fresh data")
        mask = self.data_source.get_mask(
            selections=selections,
            idxs=[p13[0], p13[1], p13[2]],
            mask_inf=mask_inf,
            mask_nan=mask_nan
        )

        # Apply additional filters

        # If dynamic selection is enabled, filter the data based on the Z selection range
        if dynamic_selection:
            # Get the current Z selection range
            z_range = self.selection_z.get_range()
            z_min = min(z_range)
            z_max = max(z_range)

            # Store the current Z selection range for change detection
            self._last_z_range = z_range

            # Get z values
            d3 = self.data_source.values[p13[2]]

            # Create a mask for values within the Z selection range
            z_mask = (d3 >= z_min) & (d3 <= z_max)

            # Update the combined mask
            mask = mask & z_mask

            # Log the number of points in the selection
            logging.log(0, f"Dynamic selection: {np.sum(z_mask)} points selected out of {len(d3)}")

        # If clustering is enabled and a specific cluster is selected, filter by cluster
        if use_clustering:
            # Get cluster labels from the dataframe instead of using self._cluster_labels
            try:
                if 'Cluster Label' in self.data_source.data.columns:
                    # Get cluster labels from the dataframe
                    cluster_labels = self.data_source.data['Cluster Label'].values

                    # Create a mask for the selected cluster
                    cluster_mask = (cluster_labels == selected_cluster)

                    # Create a 2D mask from the 1D cluster mask
                    # The mask should be True for points that are NOT in the selected cluster
                    n_parameter, n_data_points = mask.shape
                    new_mask = np.zeros_like(mask)

                    # For each data point not in the selected cluster, mask it across all parameters
                    new_mask[:, ~cluster_mask] = True

                    # Combine with the existing mask (keep points masked in either mask)
                    mask = mask | new_mask

                    # Calculate how many points are in the selected cluster and not masked
                    # A point is not masked if all parameters for that point are not masked
                    # So we need to check if any column in the mask for that point is False
                    points_in_cluster = np.sum(cluster_mask)
                    points_in_cluster_after_masking = np.sum(~np.any(mask[:, cluster_mask], axis=0))

                    # Log the number of points in the selected cluster after masking
                    logging.log(0, f"Cluster selection: {points_in_cluster_after_masking} points in cluster {selected_cluster} (out of {points_in_cluster} total in this cluster)")
                else:
                    logging.warning("'Cluster Label' column not found in dataframe. Skipping cluster filtering.")
                    logging.warning("This can happen if clustering has not been performed yet.")
            except Exception as e:
                logging.warning(f"Error applying cluster filter: {str(e)}")
                logging.warning("Skipping cluster filtering.")

        # Step 4: Store in the cache for next time
        self._cached_values_selections = selections
        self._cached_values_p13 = p13
        self._cached_values_mask_inf = mask_inf
        self._cached_values_mask_nan = mask_nan
        self._cached_values_dynamic_selection = dynamic_selection
        self._cached_values_z_range = getattr(self, '_last_z_range', None)
        self._cached_values_use_clustering = use_clustering
        self._cached_values_selected_cluster = selected_cluster
        logging.log(0, "Values cached for future use")

        return mask

    @property
    def values(self) -> np.ndarray:
        """
        Return a 2D array of data (selected columns only), applying the
        user-defined mask for Inf/NaN. The result is cached to avoid repeated
        computation when .values is accessed multiple times.
        """
        logging.log(0, "Getting values with masking")

        mask = self.value_mask
        all_values = self.data_source.values

        x = np.ma.array(all_values, mask=mask)
        oCol, oRow = x.shape
        logging.log(0, f"Original data shape: {oCol}x{oRow}")
        re = np.ma.compressed(x)
        nD = re.shape[0]
        re = re.reshape((oCol, int(nD / oCol)))
        logging.log(0, f"Reshaped data shape: {re.shape}")

        return re

    @property
    def ymax(self) -> float:
        logging.log(0, "Getting ymax")
        result = max(self.y_values)
        logging.log(0, f"ymax = {result}")
        return result

    @property
    def zmin(self):
        logging.log(0, "Getting zmin")
        v = self.z_values[self.z_values > -np.inf]
        logging.log(0, f"Filtered out {len(self.z_values) - len(v)} infinite values")
        if self.plot_control.scale_z == "log":
            v_before = len(v)
            v = v[np.where(v > 0)[0]]
            logging.log(0, f"Log scale: filtered out {v_before - len(v)} non-positive values")
        try:
            result = min(v)
            logging.log(0, f"zmin = {result}")
            return result
        except ValueError:
            logging.log(0, "No valid values for zmin, returning 0")
            return 0

    @property
    def zmax(self) -> float:
        logging.log(0, "Getting zmax")
        result = max(self.z_values)
        logging.log(0, f"zmax = {result}")
        return result

    @property
    def working_path(self):
        logging.log(0, "Getting working_path")
        path = self.lineEditWorkingPath.text()
        logging.log(0, f"working_path = {path}")
        return path

    @working_path.setter
    def working_path(self, v):
        logging.log(0, f"Setting working_path to {v}")
        if pathlib.Path(v).is_dir():
            logging.log(0, f"Path {v} is a valid directory, updating working path")
            self.lineEditWorkingPath.setText(v)
        else:
            logging.log(0, f"Path {v} is not a valid directory, working path not updated")

    @property
    def xmin(self) -> float:
        logging.log(0, "Getting xmin")
        v = self.x_values[self.x_values > -np.inf]
        logging.log(0, f"Filtered out {len(self.x_values) - len(v)} infinite values")
        if self.plot_control.scale_x == "log":
            v_before = len(v)
            v = v[np.where(v > 0)[0]]
            logging.log(0, f"Log scale: filtered out {v_before - len(v)} non-positive values")
        try:
            result = min(v)
            logging.log(0, f"xmin = {result}")
            return result
        except ValueError:
            logging.log(0, "No valid values for xmin, returning 0")
            return 0

    @property
    def xmax(self) -> float:
        logging.log(0, "Getting xmax")
        result = max(self.x_values)
        logging.log(0, f"xmax = {result}")
        return result

    @property
    def ymin(self) -> float:
        logging.log(0, "Getting ymin")
        v = self.y_values[self.y_values > -np.inf]
        logging.log(0, f"Filtered out {len(self.y_values) - len(v)} infinite values")
        if self.plot_control.scale_y == "log":
            v_before = len(v)
            v = v[np.where(v > 0)[0]]
            logging.log(0, f"Log scale: filtered out {v_before - len(v)} non-positive values")
        try:
            result = min(v)
            logging.log(0, f"ymin = {result}")
            return result
        except ValueError:
            logging.log(0, "No valid values for ymin, returning 0")
            return 0

    @property
    def vmin(self):
        logging.log(0, "Getting vmin")
        result = self.doubleSpinBox_vmin.value()
        logging.log(0, f"vmin = {result}")
        return result

    @vmin.setter
    def vmin(self, v):
        logging.log(0, f"Setting vmin to {v}")
        return self.doubleSpinBox_vmin.setValue(v)

    @property
    def vmax(self):
        logging.log(0, "Getting vmax")
        result = self.doubleSpinBox_vmax.value()
        logging.log(0, f"vmax = {result}")
        return result

    @vmax.setter
    def vmax(self, v):
        logging.log(0, f"Setting vmax to {v}")
        return self.doubleSpinBox_vmax.setValue(v)

    @property
    def current_cmap(self) -> str:
        logging.log(0, "Getting current_cmap")
        result = self.comboBoxCmap.currentText()
        logging.log(0, f"current_cmap = {result}")
        return result

    def update_cmap(self, cmap_name = None):
        """
        Update the colormap of the imshow plot based on the selected cmap.
        """
        logging.log(0, f"Updating colormap with cmap_name={cmap_name}")
        if cmap_name is None:
            cmap_name = self.current_cmap
            logging.log(0, f"Using current colormap: {cmap_name}")

        # Apply the colormap to the image
        self.cax.set_color_map(cmap_name)
        self.g_2dplot.replot()  # Redraw the plot
        logging.log(0, f"Colormap updated to {cmap_name}")

    def populate_colormap_combobox(self):
        """Populate the QComboBox with guiqwt colormap names."""
        logging.log(0, "Populating colormap combobox")
        colormap_names = sorted(get_colormap_list())  # Get all guiqwt colormap names
        logging.log(0, f"Found {len(colormap_names)} colormaps")
        self.comboBoxCmap.addItems(colormap_names)  # Add them to the QComboBox

        # Set default selection
        if self.current_cmap in colormap_names:
            default_index = colormap_names.index(self.current_cmap)
            self.comboBoxCmap.setCurrentIndex(default_index)
            logging.log(0, f"Set default colormap to {self.current_cmap} at index {default_index}")
        else:
            logging.log(0, f"Default colormap {self.current_cmap} not found in available colormaps")

    def on_vmin_vmax_changed(self):
        logging.log(0, "vmin/vmax values changed")
        # Get current values from the spin boxes using the properties
        current_vmin = self.vmin  # this should read from doubleSpinBox_vmin.value()
        current_vmax = self.vmax  # similarly for doubleSpinBox_vmax.value()
        logging.log(0, f"Setting colormap limits to vmin={current_vmin}, vmax={current_vmax}")

        # Update the colormap limits for the 2D histogram image
        self.cax.set_lut_range([current_vmin, current_vmax])
        self.g_2dplot.replot()  # Redraw the plot to reflect the change
        logging.log(0, "Colormap limits updated")

    def set_default_colormap(self, default_cmap):
        """Set the default colormap in the QComboBox."""
        logging.log(0, f"Setting default colormap to {default_cmap}")
        index = self.comboBoxCmap.findText(default_cmap)  # Find the index of the colormap
        if index != -1:  # Ensure it exists in the list
            logging.log(0, f"Found colormap {default_cmap} at index {index}")
            self.comboBoxCmap.setCurrentIndex(index)  # Set the QComboBox to the colormap

            # Apply the colormap to the image
            self.cax.set_color_map(default_cmap)
            self.g_2dplot.replot()

            logging.log(0, f"Default colormap set to {default_cmap}")
        else:
            logging.log(0, f"Colormap {default_cmap} not found in available colormaps")

    def __init__(
            self,
            data_source=None,  # type: DataSource
            settings_json_fn=None,  # type: str
            parent=None,
            cmap: str = 'jet',
            theme_file = "theme.qss"
    ) -> None:
        if isinstance(data_source, DataSource):
            self._data_source = data_source
        super(NDXplorer, self).__init__(parent=parent)

        self.settings = dict()  # type: Dict
        self.equations = list()  # type: List[Dict[str, str]]
        self.constants = dict()  # type: Dict[str, float]
        self._histogram = {
            "x": (),
            "y": (),
            "z": (),
            "2d": ()
        }
        self._mask_inf = True  # type: bool
        self._mask_nan = True  # type: bool
        self._dynamic_selection = False  # type: bool
        self._data_source = DataSource()  # type: DataSource
        self._default_data_source = DataSource(
            ["Tau (green)", "Proximity ratio", "r Experimental (green)"],
            np.vstack(
                [
                    np.random.multivariate_normal(
                        [4.1, 0.0, 0.05], [[0.1, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], size=500
                    ),
                    np.random.multivariate_normal(
                        [2.0, 0.5, 0.15], [[0.1, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], size=500
                    )
                ]
            )
        )

        # Clustering settings
        self._use_clustering = False  # type: bool, always enabled now

        # Common clustering variables
        self._cluster_labels = None  # type: Optional[np.ndarray]
        self._cluster_probabilities = None  # type: Optional[np.ndarray]
        self.clustering_worker = None  # Initialize the worker instance

        # Create clustering dialog early to use its parameters
        self.clustering_dialog = ClusteringDialog(parent=self)

        # Initialize the cache variables to None
        self._cached_values = None
        self._cached_values_selections = None
        self._cached_values_p13 = None
        self._cached_values_mask_inf = None
        self._cached_values_mask_nan = None

        self.plot_control = SurfacePlotWidget(self)
        self.equation_editor = CodeEditor(parent=self)
        self.curve_overlay_widget = CurveOverlayWidget(self)
        self.curve_evaluator = CurveEvaluator()
        self.curve_items = []  # List to store curve items

        uic.loadUi(os.path.dirname(__file__) + '/plot_main.ui', self)
        self.verticalLayout_3.addWidget(self.plot_control)
        self.verticalLayout_15.addWidget(self.equation_editor)
        self.verticalLayout_10.addWidget(self.curve_overlay_widget)

        # Connect curve overlay signals
        self.curve_overlay_widget.curvesChanged.connect(self.update_curve_overlays)

        # -------------------------------------------------------------
        # Load a stylesheet from an external file and apply it here
        # -------------------------------------------------------------
        theme_file = os.path.join(os.path.dirname(__file__), theme_file)
        # or prompt the user to pick a file with QFileDialog
        # theme_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Theme", "", "Style Sheets (*.qss)")
        if os.path.exists(theme_file):
            with open(theme_file, "r") as f:
                style_sheet = f.read()
            self.setStyleSheet(style_sheet)
        else:
            logging.warning(f"Theme file not found: {theme_file}")

        def save_cb():
            logging.log(0, "Save CB")
            json_str = self.equation_editor.text()
            self.equations = yaml.load(json_str)
        self.equation_editor.save_callback = save_cb

        self.populate_colormap_combobox()

        # Add clustering button to show/hide dialog
        self.setup_clustering_button()

        # Add dynamic selection checkbox
        self.checkBoxDynamicSelection = self.plot_control.checkBoxDynamicSelection
        self.checkBoxDynamicSelection.setToolTip("When checked, 2D and 1D histograms (except Z) "
                                                 "will only display data selected by region selector")
        self.checkBoxDynamicSelection.setChecked(self._dynamic_selection)
        self.checkBoxDynamicSelection.stateChanged.connect(self.on_dynamic_selection_changed)

        # Store the last Z selection range to detect changes
        self._last_z_range = None

        # Create a timer to check for Z selection range changes
        self.z_range_check_timer = QtCore.QTimer(self)
        self.z_range_check_timer.timeout.connect(self.check_z_range_changes)
        # Check every 500 ms
        self.z_range_check_timer.start(500)

        # Plots
        #############
        # z-axis
        win_z = CurveDialog()
        self.g_zplot = win_z.get_plot()
        curveparam = guiqwt.styles.CurveParam("Curve", icon='curve.png')
        curveparam.curvestyle = "Steps"
        curveparam.line.color = '#ff00ff'
        curveparam.shade = 0.5
        curveparam.line.width = 2.0
        self.g_zhist_m = guiqwt.curve.CurveItem(curveparam=curveparam)
        self.g_zplot.add_item(self.g_zhist_m)
        self.selection_z = make.range(.25, .5)
        self.g_zplot.add_item(self.selection_z)

        self.plot_control.verticalLayout_4.addWidget(self.g_zplot)

        # x-axis
        win_x = CurveDialog()
        self.g_xplot = win_x.get_plot()
        self.g_xplot.enableAxis(QwtPlot.xBottom, True)
        self.g_xplot.enableAxis(QwtPlot.xTop, False)
        self.g_xplot.enableAxis(QwtPlot.yLeft, False)
        self.g_xplot.enableAxis(QwtPlot.yRight, False)

        curveparam = guiqwt.styles.CurveParam("Curve", icon='curve.png')
        curveparam.curvestyle = "Steps"
        curveparam.line.color = '#0066cc'
        curveparam.shade = 0.5
        curveparam.line.width = 2.0
        self.g_xhist_m = guiqwt.curve.CurveItem(curveparam=curveparam)
        self.g_xplot.add_item(self.g_xhist_m)
        self.verticalLayout_5.addWidget(self.g_xplot)

        # y-axis
        win_y = CurveDialog()
        self.g_yplot = win_y.get_plot()
        self.g_yplot.enableAxis(QwtPlot.xBottom, False)
        self.g_yplot.enableAxis(QwtPlot.xTop, False)
        self.g_yplot.enableAxis(QwtPlot.yLeft, True)
        self.g_yplot.enableAxis(QwtPlot.yRight, False)

        curveparam = guiqwt.styles.CurveParam()
        curveparam.curvestyle = "Steps"
        curveparam.line.color = '#00ff00'
        curveparam.shade = 0.1
        curveparam.line.width = 2.0
        self.g_yhist_m = guiqwt.curve.CurveItem(curveparam=curveparam)
        self.g_yplot.add_item(self.g_yhist_m)
        self.verticalLayout_7.addWidget(self.g_yplot)

        # -------------------------------------------------------------
        # Force background colors to white
        # -------------------------------------------------------------
        # For guiqwt / QwtPlot-based plots:
        self.g_xplot.canvas().setStyleSheet("background-color: white;")
        self.g_yplot.canvas().setStyleSheet("background-color: white;")
        self.g_zplot.canvas().setStyleSheet("background-color: white;")

        # 2D-Histogram using guiqwt
        # Create a guiqwt plot widget
        win_2d = guiqwt.plot.ImageDialog(edit=False, toolbar=False)
        self.g_2dplot = win_2d.get_plot()

        # Create initial data
        d = np.ones((200, 200))
        d[120, 120] = 250
        d[150, 120] = 250

        # Create an image item with our fixed subclass
        self.cax = FixedImageItem(data=d)
        self.g_2dplot.add_item(self.cax)

        # Set default colormap
        self.set_default_colormap(cmap)

        # Configure the plot
        self.g_2dplot.set_axis_font("left", QFont("Courier"))
        self.g_2dplot.set_axis_font("bottom", QFont("Courier"))

        # Enable axes that we want to link with marginal plots
        self.g_2dplot.enableAxis(QwtPlot.xBottom, False)
        self.g_2dplot.enableAxis(QwtPlot.xTop, False)
        self.g_2dplot.enableAxis(QwtPlot.yLeft, False)
        self.g_2dplot.enableAxis(QwtPlot.yRight, False)

        # Set background color to white
        self.g_2dplot.canvas().setStyleSheet("background-color: white;")

        # Disable zoom and panning on the 2D plot
        self.g_2dplot.canvas().setMouseTracking(False)

        # Create an event filter to ignore mouse events for zoom and panning
        class MouseEventFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                # Allow right-click events for context menu
                if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
                    return False  # Process right-click events normally

                # Ignore other mouse events that would trigger zoom and panning
                if event.type() in [QtCore.QEvent.MouseButtonPress, 
                                   QtCore.QEvent.MouseButtonRelease,
                                   QtCore.QEvent.MouseButtonDblClick,
                                   QtCore.QEvent.MouseMove,
                                   QtCore.QEvent.Wheel]:
                    return True  # Ignore mouse events

                return False  # Process other events normally

        # Install the event filter on the canvas
        self.mouse_event_filter = MouseEventFilter(self)
        self.g_2dplot.canvas().installEventFilter(self.mouse_event_filter)

        # Create a separate plot for curve overlays
        self.overlay_plot = guiqwt.curve.CurvePlot(parent=self)

        # Disable grid lines in the overlay plot
        self.overlay_plot.grid.setVisible(False)

        # Make the overlay plot transparent
        self.overlay_plot.setAutoFillBackground(False)
        self.overlay_plot.setStyleSheet("background-color: transparent;")
        self.overlay_plot.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self.overlay_plot.canvas().setAutoFillBackground(False)
        self.overlay_plot.canvas().setStyleSheet("background-color: transparent;")
        self.overlay_plot.canvas().setFrameStyle(QtWidgets.QFrame.NoFrame)

        # Disable paint attributes that might interfere with transparency
        self.overlay_plot.canvas().setPaintAttribute(QwtPlotCanvas.BackingStore, False)
        self.overlay_plot.canvas().setPaintAttribute(QwtPlotCanvas.Opaque, False)
        self.overlay_plot.canvas().setPaintAttribute(QwtPlotCanvas.HackStyledBackground, False)
        self.overlay_plot.canvas().setPaintAttribute(QwtPlotCanvas.ImmediatePaint, True)

        # Configure the overlay plot to match the main plot
        self.overlay_plot.enableAxis(QwtPlot.xBottom, False)
        self.overlay_plot.enableAxis(QwtPlot.xTop, False)
        self.overlay_plot.enableAxis(QwtPlot.yLeft, False)
        self.overlay_plot.enableAxis(QwtPlot.yRight, False)

        # Create a widget to hold both plots
        plot_container = QtWidgets.QWidget()
        plot_container.setAutoFillBackground(False)
        plot_container.setStyleSheet("background-color: transparent;")
        plot_layout = QtWidgets.QGridLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.g_2dplot, 0, 0)
        plot_layout.addWidget(self.overlay_plot, 0, 0)  # Overlay at the same position

        # Add the container widget to your PyQt layout
        self.verticalLayout_11.addWidget(plot_container)

        self.g_xplot.setMaximumHeight(150)
        self.g_yplot.setMaximumWidth(150)
        self.g_zplot.setMaximumHeight(150)

        # Load settings
        ###############
        if settings_json_fn is None:
            settings_json_fn = pathlib.Path(__file__).parent / "settings" / "mfd.settings.json"
        self.onLoad_settings(settings_json_fn=str(settings_json_fn))

        # Parameter control
        ######################
        def parameter_update():
            self.constants = self.parameter_control.dict
            self.data_source.compute_columns(
                constants=self.constants,
                equations=self.equations
            )
            self.update_plots()
        self.parameter_control = ParameterEditor(
            parent=self,
            json_file=str(pathlib.Path(__file__).parent / "settings/mfd.constants.json"),
            callback=parameter_update
        )
        self.verticalLayout_4.addWidget(self.parameter_control)

        # Actions
        #############
        # Working path
        self.actionSelect_working_path.triggered.connect(self.onSelectWorkingPath)

        # Load / Save
        self.actionOpenChiSurfSampling.triggered.connect(self.onOpenChiSurfSampling)
        self.actionOpenParisDataset.triggered.connect(self.onOpenSmFRET)
        self.actionOpenCsv.triggered.connect(self.onOpenCsv)
        self.actionBurst_IDs.triggered.connect(self.onSaveBurstIDs)

        # Settings
        self.actionLoad_settings.triggered.connect(self.onLoad_settings)
        self.actionSave_axis_settings.triggered.connect(self.onSaveAxisSettings)
        # GUI updates
        self.actionUpdate_plot.triggered.connect(lambda: self.update_plots())
        self.actionClear_plot.triggered.connect(self.clear_plots)
        self.actionMask_toggle_changed.triggered.connect(self.onMaskChanged)
        # UMAP
        self.actionUMAP.triggered.connect(self.onShowUMAP)

        # Connect toolButton_3 to show data in DataFrameEditor
        self.toolButton_3.clicked.connect(self.show_dataframe_editor)
        self.toolButton_3.setEnabled(True)  # Enable the button
        # Axis range

        self.overlay_plot.canvas().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.overlay_plot.canvas().customContextMenuRequested.connect(self.on_canvas_context_menu)

        # Assuming these combo boxes control the parameter selections for the 2D plot:
        self.plot_control.comboBoxSelX.currentIndexChanged.connect(self.update_spinbox_limits)
        self.plot_control.comboBoxSelY.currentIndexChanged.connect(self.update_spinbox_limits)
        self.plot_control.comboBoxSelZ.currentIndexChanged.connect(self.update_spinbox_limits)

        # In your __init__ or setup method, after creating the spin boxes:
        self.doubleSpinBox_vmin.valueChanged.connect(self.on_vmin_vmax_changed)
        self.doubleSpinBox_vmax.valueChanged.connect(self.on_vmin_vmax_changed)

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        docks = [self.dockWidget_PlotControl, self.dockWidget_Parameters, self.dockWidget_Overlays]
        for i, d in enumerate(docks[:-1]):
            self.tabifyDockWidget(d, docks[i+1])
        self.dockWidget_PlotControl.raise_()
        self.dockWidget_Equations.setVisible(False)
        self.update()

    def show_dataframe_editor(self):
        """
        Show the data in the data source using DataFrameEditor.
        """
        if self._data_source.empty:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No data loaded—nothing to show."
            )
            return

        dlg = DataFrameEditor(self)
        # Set up the editor on the current DataFrame
        if not dlg.setup_and_check(self._data_source.data, title="Data Source"):
            return

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # User hit OK: grab the possibly-modified DataFrame back
            self._data_source.data = dlg.get_value()
            # Refresh the plots
            self.update_plots()

    def clear_plots(self):
        logging.log(0, "clearing plots")
        # 1. Clear the user data => empty => fallback to _default_data_source
        self._data_source.clear()

        # 2. Clear the selection table so no old mask references remain
        #    (But remember, this does NOT fix comboBoxSelX/Y/Z)
        self.plot_control.onClearSelection()

        # 3. Update once so 'plot_control.update()' sees empty _data_source =>
        #    repopulates combo boxes with the default dataset columns
        self.update()

        # 4. Force combo box indices to match the default columns.
        #    Example: we want [ "Tau (green)", "Proximity ratio", "r Experimental (green)" ]
        default_names = self._default_data_source.parameter_names
        ix_tau = default_names.index("Tau (green)")
        ix_prox = default_names.index("Proximity ratio")
        ix_r = default_names.index("r Experimental (green)")

        self.plot_control.comboBoxSelX.setCurrentIndex(ix_tau)
        self.plot_control.comboBoxSelY.setCurrentIndex(ix_prox)
        self.plot_control.comboBoxSelZ.setCurrentIndex(ix_r)

        # 5. Trigger a final update for correct plots
        self.update()

    def copy_1d_hists_to_clipboard_csv(self):
        try:
            x_hist = self._histogram["x"]  # tuple: (bin_edges, counts)
            y_hist = self._histogram["y"]
            z_hist = self._histogram["z"]
        except Exception as e:
            logging.error("1D histogram data not available: {}".format(e))
            return

        import io
        output = io.StringIO()

        # Extract histogram components for X, Y, and Z
        x_edges, x_counts = x_hist
        y_edges, y_counts = y_hist
        z_edges, z_counts = z_hist

        n_x = len(x_counts)
        n_y = len(y_counts)
        n_z = len(z_counts)
        n_rows = max(n_x, n_y, n_z)

        # Write header with columns for each histogram
        header = "\t".join([
            "X Bin Start", "X Bin End", "X Count",
            "Y Bin Start", "Y Bin End", "Y Count",
            "Z Bin Start", "Z Bin End", "Z Count"
        ])
        output.write(header + "\n")

        # Write each row
        for i in range(n_rows):
            if i < n_x:
                x_bin_start = f"{x_edges[i]:12.4e}"
                x_bin_end = f"{x_edges[i + 1]:12.4e}"
                x_count = f"{x_counts[i]:12.4e}"
            else:
                x_bin_start = x_bin_end = x_count = ""
            if i < n_y:
                y_bin_start = f"{y_edges[i]:12.4e}"
                y_bin_end = f"{y_edges[i + 1]:12.4e}"
                y_count = f"{y_counts[i]:12.4e}"
            else:
                y_bin_start = y_bin_end = y_count = ""
            if i < n_z:
                z_bin_start = f"{z_edges[i]:12.4e}"
                z_bin_end = f"{z_edges[i + 1]:12.4e}"
                z_count = f"{z_counts[i]:12.4e}"
            else:
                z_bin_start = z_bin_end = z_count = ""

            row = "\t".join([
                x_bin_start, x_bin_end, x_count,
                y_bin_start, y_bin_end, y_count,
                z_bin_start, z_bin_end, z_count
            ])
            output.write(row + "\n")

        csv_text = output.getvalue()
        output.close()

        # Copy the resulting CSV text to the clipboard
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(csv_text)
        logging.log(0, "1D histograms data copied to clipboard as CSV with side-by-side columns.")

    def on_canvas_context_menu(self, pos):
        menu = QtWidgets.QMenu(self.g_2dplot.canvas())
        action_csv = menu.addAction("Copy 2D Histogram (CSV)")
        #action_json = menu.addAction("Copy 2D Histogram (JSON)")
        action_csv1d = menu.addAction("Copy 1D Histograms (CSV)")
        action = menu.exec_(self.g_2dplot.canvas().mapToGlobal(pos))
        if action == action_csv:
            self.copy_2d_hist_to_clipboard_csv()
        #elif action == action_json:
        #    self.copy_2d_hist_to_clipboard_json()
        elif action == action_csv1d:
            self.copy_1d_hists_to_clipboard_csv()

    def onMaskChanged(self) -> None:
        """
        Whenever the user toggles the Inf/NaN masks,
        invalidate the cache and re-plot.
        """
        self._mask_inf = self.checkBoxMaskInf.isChecked()
        self._mask_nan = self.checkBoxMaskNaN.isChecked()
        self.invalidate_values_cache()
        self.update_plots()

    def onShowUMAP(self) -> None:
        """
        Show the UMAP plot.
        This method is triggered when the user clicks the UMAP action in the View menu.
        """
        try:
            # Create the clustering dialog if it doesn't exist
            if self.clustering_dialog is None:
                self.create_clustering_dialog()

            # Show the dialog if it's not visible
            if not self.clustering_dialog.isVisible():
                # Update the dialog with current settings before showing it
                self.update_clustering_dialog()
                self.clustering_dialog.show()
        except RuntimeError:
            # If we get a RuntimeError, it means the UI elements have been deleted
            # In this case, we need to recreate the dialog
            logging.warning("Clustering dialog UI elements have been deleted. Recreating dialog.")
            self.clustering_dialog = None
            self.create_clustering_dialog()
            self.clustering_dialog.show()

        # Get UMAP parameters directly from clustering dialog
        # This avoids accessing potentially deleted UI elements
        params = {
            "n_neighbors": self.clustering_dialog._umap_n_neighbors,
            "min_dist": self.clustering_dialog._umap_min_dist,
            "n_components": self.clustering_dialog._umap_n_components
        }

        # Create the UMAP plot
        self.create_umap_plot(
            self.clustering_dialog._cluster_columns,
            params
        )

    def onSelectWorkingPath(self):
        working_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select current path', self.working_path)
        self.lineEditWorkingPath.blockSignals(True)
        self.lineEditWorkingPath.setText(working_path)
        self.lineEditWorkingPath.blockSignals(False)

    def onSaveBurstIDs(self, evt=None, folder=None):
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Folder for Burst IDs', self.working_path
            )
        logging.info(f"Saving burst IDs to {folder}...")
        writer.save_burst_ids(
            folder_name=folder,
            selections=self.plot_control.get_selections(),
            data_source=self.data_source
        )

    def onSaveClusteringData(self, evt=None, folder=None):
        """
        Save clustering data to a folder.

        Args:
            evt: Event that triggered this method (not used)
            folder: Folder where clustering data will be saved. If None, a folder selection dialog will be shown.
        """
        # Check if we have cluster labels
        if self._cluster_labels is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Clustering Data",
                "No clustering data available. Please apply clustering before saving."
            )
            return

        # Get folder to save data
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Folder for Clustering Data', self.working_path
            )

        if not folder:  # User cancelled the dialog
            return

        # Prepare parameters dictionary based on the clustering method
        if self.clustering_dialog._cluster_method == "hdbscan":
            parameters = {
                "min_samples": self.clustering_dialog._cluster_min_samples,
                "min_cluster_size": self.clustering_dialog._cluster_min_cluster_size
            }
        elif self.clustering_dialog._cluster_method == "kmeans":
            parameters = {
                "n_clusters": self.clustering_dialog._cluster_n_clusters
            }
        else:
            parameters = {}

        # Save clustering data
        logging.info(f"Saving clustering data to {folder}...")
        writer.save_clustering_data(
            folder_name=folder,
            data_source=self.data_source,
            cluster_method=self.clustering_dialog._cluster_method,
            cluster_labels=self._cluster_labels,
            cluster_probabilities=self._cluster_probabilities,
            cluster_columns=self.clustering_dialog._cluster_columns,
            parameters=parameters
        )

    def onSaveAxisSettings(
            self,
            settings_json_fn=None  # type: str
    ):
        if settings_json_fn is None:
            settings_json_fn = QtWidgets.QFileDialog.getSaveFileName(
                self ,
                'Axis settings file',
                self.working_path,
                'Axis file (*.axis.json)'
            )
        with open(settings_json_fn, "w") as fp:
            json.dump(
                self.plot_control.axis_settings,
                fp,
                indent=4
            )

    def onLoad_settings(
            self,
            settings_json_fn=None  # type: str
    ):
        if settings_json_fn is None:
            settings_json_fn = QtWidgets.QFileDialog.getOpenFileName(
                None, 'ndXplorer settings file', self.working_path, 'ndXplorer settings (*.settings.json)'
            )
        with open(settings_json_fn, "r") as fp:
            d = json.load(fp)
            self.settings.update(d)
        fn_axis = pathlib.Path(settings_json_fn).parent / self.settings["axis"]
        with open(str(fn_axis), "r") as fp:
            d = json.load(fp)
            self.plot_control.axis_settings.update(d)
        fn_equations = pathlib.Path(settings_json_fn).parent / self.settings["equations"]
        with open(str(fn_equations), "r") as fp:
            d = yaml.load(fp, Loader=yaml.FullLoader)
            self.equations = d
        fn_constants = pathlib.Path(settings_json_fn).parent / self.settings["constants"]
        with open(str(fn_constants), "r") as fp:
            d = json.load(fp)
            self.constants.update(d)
        self.equation_editor.load_file(str(fn_equations))

    def open_files(
            self,
            file_handles: typing.List[str] = None,
            file_type: str = None
    ):
        wp = str(self.working_path)
        if file_type in ["cs_sampling", "er4"]:
            if not hasattr(file_handles, '__iter__'):
                file_handles, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'ChiSurf sampling files', wp, 'Sampling files (*.*)')
            logging.log(0, "Opening files: {}".format(file_handles))
            data_reader = reader.read_csv_sampling
        elif file_type in ["burst_dir"]:
            file_handles = QtWidgets.QFileDialog.getExistingDirectory(None, 'Open burst analysis folder', self.working_path)
            data_reader = reader.read_burst_analysis
        else: #if file_type in [None, "csv"]:
            if file_handles is None:
                file_handles = QtWidgets.QFileDialog.getOpenFileNames(None, 'Comma separated value files', self.working_path, 'Text files (*.*)')
            data_reader = reader.read_csv
        if file_handles:
            # self.working_path = str(pathlib.Path(file_handles[0]).parent)
            self._data_source = data_reader(file_handles)
            self.update()

    def onOpenCsv(
            self,
            filenames: List[str] = None
    ):
        self.open_files(file_type="csv", file_handles=filenames)

    def onOpenChiSurfSampling(
            self,
            filenames=None  # type: List[str]
    ):
        self.open_files(file_type="cs_sampling", file_handles=filenames)

    def onOpenSmFRET(self):
        self.open_files(file_type="burst_dir")

    def update(self, *args, **kwargs):
        super(NDXplorer, self).update()
        self.data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )
        self.lineEditCountTotal.setText(str(self.data_source.size))
        self.plot_control.update()  # plot_control.update() - also updates plots

    def update_parameter_names(self):
        p1, p1_name = self.plot_control.p1
        p2, p2_name = self.plot_control.p2
        self.g_yplot.set_axis_title("top", p2_name)
        self.g_xplot.set_axis_title("top", p1_name)

    def get_bins(self, arange, scale, n_1d, n_2d):
        xmin, xmax = arange
        # set log scales
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

    def get_x_bins(self):
        return self.get_bins(
            self.plot_control.x_range,
            self.plot_control.scale_x,
            self.plot_control.n_xhist_1d,
            self.plot_control.n_xhist_2d
        )

    def get_y_bins(self):
        return self.get_bins(
            self.plot_control.y_range,
            self.plot_control.scale_y,
            self.plot_control.n_yhist_1d,
            self.plot_control.n_yhist_2d
        )

    def get_z_bins(self):
        bins = self.get_bins(
            self.plot_control.z_range,
            self.plot_control.scale_z,
            self.plot_control.n_zhist_1d,
            10
        )
        return bins

    def update_histograms(self):
        # Get the values that are already filtered by value_mask
        # These properties use self.values which applies the value_mask
        d1 = self.x_values
        d2 = self.y_values
        d3 = self.z_values

        # Update GUI - Number of displayed data points
        self.lineEditCountCurrent.setText(str(len(d1)))
        x_bins_1d, x_bins_2d = self.get_x_bins()
        y_bins_1d, y_bins_2d = self.get_y_bins()
        z_bins_1d, _ = self.get_z_bins()

        # X, Y, Z Histogram
        ###################
        # Use the filtered data for all histograms
        self._histogram["x"] = np.histogram(d1, bins=x_bins_1d, density=self.plot_control.normed_hist_x)[::-1]
        self._histogram["y"] = np.histogram(d2, bins=y_bins_1d, density=self.plot_control.normed_hist_y)[::-1]
        self._histogram["z"] = np.histogram(d3, bins=z_bins_1d, density=self.plot_control.normed_hist_z)[::-1]

        # 2D Histogram
        ####################
        try:
            # Use the filtered data for the 2D histogram if dynamic selection is enabled
            H, x_edges, y_edges = np.histogram2d(x=d1, y=d2, bins=[x_bins_2d, y_bins_2d], density=True)
            self._histogram["2d"] = H, x_edges, y_edges
        except ValueError:
            logging.log(1, "Did not compute 2D histogram")

    def copy_2d_hist_to_clipboard_json(self):
        try:
            H, x_edges, y_edges = self._histogram["2d"]
        except Exception as e:
            logging.error("No 2D histogram data available: {}".format(e))
            return

        # Convert NumPy arrays to lists for serialization
        data_dict = {
            "H": H.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist()
        }
        # Serialize the data as a nicely formatted JSON string
        text_data = json.dumps(data_dict, indent=2)

        # Use Qt's clipboard to store the data
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text_data)
        logging.log(0, "2D histogram data copied to clipboard.")

    def copy_2d_hist_to_clipboard_csv(self):
        try:
            H, x_edges, y_edges = self._histogram["2d"]
        except Exception as e:
            logging.error("No 2D histogram data available: {}".format(e))
            return

        # Compute bin centers from bin edges
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        import io
        output = io.StringIO()

        # Build header row with tabs and uniform formatting
        header_cells = ["y/x"] + [f"{x:10.4f}" for x in x_centers]
        output.write("\t".join(header_cells) + "\n")

        # Build rows: each row starts with the y center, then the corresponding histogram counts.
        # Note: H is assumed to be shaped (len(x_centers), len(y_centers)).
        for j, y in enumerate(y_centers):
            row_cells = [f"{y:10.4e}"]  # y center formatted uniformly
            for i in range(len(x_centers)):
                row_cells.append(f"{H[i, j]:10.4e}")
            output.write("\t".join(row_cells) + "\n")

        csv_text = output.getvalue()
        output.close()

        # Copy the nicely formatted CSV text to the clipboard
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(csv_text)
        logging.log(0, "2D histogram data copied to clipboard as CSV (formatted with tabs).")

    def update_plots(self, skip_clustering=False):
        """
        Update all plots with the current data.

        Args:
            skip_clustering: If True, skip the clustering step even if clustering is enabled.
                            This is useful when update_plots is called after clustering is done
                            or when loading data.
        """
        # If there's no data (or fewer than 3 columns), display the background image
        if self._data_source.empty or self._data_source.values.shape[0] == 0:
            # Clear any old histogram displays
            self.g_xhist_m.set_data([], [])
            self.g_yhist_m.set_data([], [])
            self.g_zhist_m.set_data([], [])

            # Construct the path to the background image
            bg_path = os.path.join(os.path.dirname(__file__), 'ui', 'background.png')
            try:
                # Load the background image using matplotlib's imread
                bg_img = plt.imread(bg_path)
            except Exception as e:
                logging.error("Could not load background image: %s", e)
                bg_img = np.zeros((1, 1))  # fallback to an empty array if needed

            # Set the background image in the 2D histogram axis
            self.cax.set_data(bg_img)

            # Set default axis scales for empty data
            self.g_2dplot.setAxisScale(QwtPlot.xBottom, 0, 1)
            self.g_2dplot.setAxisScale(QwtPlot.yLeft, 0, 1)

            # Replot the 2D plot
            self.g_2dplot.replot()
            return

        # Update parameter names, colormap, and recalc histograms
        self.update_parameter_names()
        self.update_cmap()

        # Apply HDBSCAN clustering if enabled and not skipped
        # Skip clustering when loading data (skip_clustering=True)
        if self._use_clustering and self._cluster_labels is None and hdbscan and not skip_clustering:
            # Start the clustering in a separate thread
            self.on_apply_clustering()
            # Return early to avoid updating the plots until clustering is done
            return

        # Update histograms
        self.update_histograms()

        # ----------------------------------------------------
        # 1. X histogram
        # _histogram["x"] = (bin_edges, counts), reversed via [::-1] in your code
        x_bin_edges = self._histogram["x"][0]
        x_counts = self._histogram["x"][1]
        # Plot with X=bin_edges[1:], Y=counts
        self.g_xhist_m.set_data(x_bin_edges[1:], x_counts)

        # 2. Y histogram (rotated)
        # _histogram["y"] = (bin_edges, counts)
        y_bin_edges = self._histogram["y"][0]
        y_counts = self._histogram["y"][1]
        # Plot with X=counts, Y=bin_edges[1:]
        self.g_yhist_m.set_data(y_counts, y_bin_edges[1:])

        # 3. Z histogram
        # _histogram["z"] = (bin_edges, counts)
        z_bin_edges = self._histogram["z"][0]
        z_counts = self._histogram["z"][1]
        # Plot with X=bin_edges[1:], Y=counts
        self.g_zhist_m.set_data(z_bin_edges[1:], z_counts)

        # ----------------------------------------------------
        # Manually set axis scales to start at 0 for the count axis

        # X histogram => x-axis: bin edges, y-axis: counts
        y_max_x = np.max(x_counts) if len(x_counts) else 1
        self.g_xplot.setAxisScale(QwtPlot.yLeft, 0, y_max_x * 1.05)
        self.g_xplot.setAxisScale(QwtPlot.xBottom, x_bin_edges[0], x_bin_edges[-1])

        # Y histogram => x-axis: counts, y-axis: bin edges
        x_max_y = np.max(y_counts) if len(y_counts) else 1
        self.g_yplot.setAxisScale(QwtPlot.xBottom, 0, x_max_y * 1.05)
        self.g_yplot.setAxisScale(QwtPlot.yLeft, y_bin_edges[0], y_bin_edges[-1])

        # Z histogram => x-axis: bin edges, y-axis: counts
        y_max_z = np.max(z_counts) if len(z_counts) else 1
        self.g_zplot.setAxisScale(QwtPlot.yLeft, 0, y_max_z * 1.05)
        self.g_zplot.setAxisScale(QwtPlot.xBottom, z_bin_edges[0], z_bin_edges[-1])

        # ----------------------------------------------------
        # 2D histogram updates as before
        self.update_2d_plot()

        # ----------------------------------------------------
        # Set the 2D plot axis scales from 0 to the number of bins
        # This ensures the 2D plot always displays the full range of bins
        try:
            H, _, _ = self._histogram["2d"]
            n_bins_x, n_bins_y = H.shape
            self.g_2dplot.setAxisScale(QwtPlot.xBottom, 0, n_bins_x -1)
            self.g_2dplot.setAxisScale(QwtPlot.yLeft, 0, n_bins_y -1)
        except (ValueError, KeyError):
            # If there's no 2D histogram data, use default scales
            self.g_2dplot.setAxisScale(QwtPlot.xBottom, 0, 1)
            self.g_2dplot.setAxisScale(QwtPlot.yLeft, 0, 1)

        # ----------------------------------------------------
        # Finally replot everything
        self.g_xplot.replot()
        self.g_yplot.replot()
        self.g_zplot.replot()
        self.g_2dplot.replot()  # Replot the 2D plot to apply the axis scale changes

    def update_spinbox_limits(self, low_pct=0.1, high_pct=99):
        """
        Recompute the 2D histogram limits based on the newly selected parameters
        and update the vmin/vmax spin boxes.
        """
        # --- 0) Guard: is there any data at all? ---
        if self._data_source.empty:
            return

        # --- 2) Guard: are our selected column indices valid? ---
        p1_idx, p2_idx, p3_idx = self.plot_control.p1[0], self.plot_control.p2[0], self.plot_control.p3[0]
        try:
            # make sure values array has at least p1, p2, p3 rows
            n_rows, _ = self.values.shape
        except Exception:
            # values property may raise if data not ready
            return

        if not (0 <= p1_idx < n_rows and 0 <= p2_idx < n_rows and 0 <= p3_idx < n_rows):
            return

        # 1) Recompute the 2D histogram
        self.update_histograms()
        try:
            H, *_ = self._histogram["2d"]
        except Exception:
            return

        # 2) Decide which data to percentile over
        if self.checkBoxLogCounts.isChecked():
            # only positive bins, then log
            mask = H > 0
            data = np.log10(H[mask]) if np.any(mask) else np.array([])
        else:
            # only positive bins
            data = H[H > 0]

        # 3) Guard against empty data
        if data.size < 2:
            # too few nonzero bins → just use the full range
            raw = H if not self.checkBoxLogCounts.isChecked() else np.log10(np.nan_to_num(H))
            vmin, vmax = float(np.nanmin(raw)), float(np.nanmax(raw))
        else:
            # 4) Compute robust cutoffs
            vmin, vmax = np.percentile(data, [low_pct, high_pct])
            # if log scale, convert back to linear for the clim
            if self.checkBoxLogCounts.isChecked():
                vmin, vmax = 10 ** vmin, 10 ** vmax

        # 5) Push to spin‐boxes and the image
        self.vmin, self.vmax = vmin, vmax
        self.cax.set_lut_range([vmin, vmax])
        self.g_2dplot.replot()

    def setup_clustering_button(self):
        """
        Set up a button to show/hide the clustering dialog.
        """
        self.pushButtonShowClusteringDialog.clicked.connect(self.toggle_clustering_dialog)

    def toggle_clustering_dialog(self):
        """
        Show or hide the clustering dialog.
        """
        if self.clustering_dialog is None:
            self.create_clustering_dialog()

        if self.clustering_dialog.isVisible():
            self.clustering_dialog.hide()
        else:
            # Update dialog with current settings
            self.update_clustering_dialog()
            self.clustering_dialog.show()

    def create_clustering_dialog(self):
        """
        Create the clustering dialog if it doesn't exist.
        """
        if self.clustering_dialog is None:
            self.clustering_dialog = ClusteringDialog(parent=self)

            # Set initial values
            self.update_clustering_dialog()

            # Connect signals
            self.clustering_dialog.clustering_done.connect(self.on_clustering_done)
            self.clustering_dialog.clustering_error.connect(self.on_clustering_error)
            self.clustering_dialog.progress_updated.connect(self.on_clustering_progress)

    def update_clustering_dialog(self):
        """
        Update the clustering dialog UI elements.
        """
        if self.clustering_dialog is None:
            return

        # Only update UI elements if the dialog is visible
        # This prevents errors when trying to access UI elements that might have been deleted
        if self.clustering_dialog.isVisible():
            try:
                # Update button text to show number of selected columns
                num_selected = len(self.clustering_dialog._cluster_columns)
                if num_selected > 0:
                    self.clustering_dialog.pushButtonSelectColumns.setText(f"Select Columns (Recommended) ({num_selected})")
                else:
                    self.clustering_dialog.pushButtonSelectColumns.setText("Select Columns (Recommended)")

                # Update save button state
                self.clustering_dialog.pushButtonSaveClustering.setEnabled(
                    self._cluster_labels is not None
                )
            except RuntimeError:
                # If we get a RuntimeError, it means the UI elements have been deleted
                # In this case, we need to recreate the dialog
                logging.warning("Clustering dialog UI elements have been deleted. Recreating dialog.")
                self.clustering_dialog = None
                self.clustering_dialog = ClusteringDialog(parent=self)
                self.clustering_dialog.clustering_done.connect(self.on_clustering_done)
                self.clustering_dialog.clustering_error.connect(self.on_clustering_error)
                self.clustering_dialog.progress_updated.connect(self.on_clustering_progress)


    def start_clustering_from_dialog(self, method, columns, params):
        """
        Start clustering with parameters from the dialog.

        Args:
            method: The clustering method to use (e.g., 'kmeans', 'hdbscan')
            columns: Set of column names to use for clustering
            params: Dictionary of parameters for the clustering method
        """
        # Update the dialog parameters with the ones passed to this method
        self.clustering_dialog._cluster_method = method
        self.clustering_dialog._cluster_columns = columns

        # Start clustering
        self.on_apply_clustering()

    def cancel_clustering(self):
        """
        Cancel the current clustering operation.
        """
        self.on_cancel_clustering()

    def on_select_columns(self):
        """
        Open a dialog to select columns for clustering.
        """
        # If clustering dialog exists and is visible, use its method
        if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
            self.clustering_dialog.on_select_columns()
            return

        # Otherwise, implement the functionality directly
        # Get current parameter names from data source
        parameter_names = self.data_source.parameter_names

        # Create and show the dialog
        dialog = ColumnSelectionDialog(
            parent=self,
            column_names=parameter_names,
            selected_columns=self.clustering_dialog._cluster_columns
        )

        # If dialog is accepted, update selected columns
        if dialog.exec_():
            self.clustering_dialog._cluster_columns = dialog.get_selected_columns()

            # Update button text to show number of selected columns
            num_selected = len(self.clustering_dialog._cluster_columns)
            if num_selected > 0:
                self.clustering_dialog.pushButtonSelectColumns.setText(f"Select Columns (Recommended) ({num_selected})")
            else:
                self.clustering_dialog.pushButtonSelectColumns.setText("Select Columns (Recommended)")



    # Worker class for performing clustering in a separate thread
    class ClusteringWorker(QThread):
        # Signal emitted when clustering is done
        clustering_done = Signal(tuple)
        # Signal emitted when an error occurs
        clustering_error = Signal(str)
        # Signal emitted to report progress
        progress_updated = Signal(int)

        def __init__(self, parent, method, params):
            super().__init__(parent)
            self.parent = parent
            self.method = method
            self.params = params
            self._stop_requested = False

        def stop(self):
            """Request the worker to stop processing"""
            self._stop_requested = True
            logging.info("Clustering stop requested")

        def run(self):
            try:
                # Perform clustering in the worker thread
                result = self.parent.perform_clustering(
                    method=self.method,
                    **self.params,
                    worker=self  # Pass the worker instance to allow progress updates and cancellation
                )
                # Emit signal with the result only if not stopped
                if not self._stop_requested:
                    self.clustering_done.emit(result)
            except Exception as e:
                # Log the error
                logging.error(f"Error in clustering worker thread: {str(e)}")
                # Emit error signal only if not stopped
                if not self._stop_requested:
                    self.clustering_error.emit(str(e))
                    # Emit done signal with None values to ensure the UI is updated
                    self.clustering_done.emit((None, None))

    def on_apply_clustering(self):
        """
        Apply clustering with current parameters and update plots.
        """
        logging.log(0, "Applying clustering with current parameters")

        # Set the _use_clustering flag to True
        self._use_clustering = True
        logging.log(0, "Set _use_clustering flag to True")

        # Get parameters from the clustering dialog
        cluster_method = self.clustering_dialog._cluster_method
        cluster_columns = self.clustering_dialog._cluster_columns

        # Check if the required library is available
        if cluster_method == "hdbscan" and not hdbscan:
            QtWidgets.QMessageBox.warning(
                self,
                "HDBSCAN Not Available",
                "HDBSCAN is not installed. Please install it using pip or conda."
            )
            if self.clustering_dialog is not None:
                self.clustering_dialog.checkBoxClustering.setChecked(False)
            return
        elif cluster_method == "kmeans" and not KMeans:
            QtWidgets.QMessageBox.warning(
                self,
                "scikit-learn Not Available",
                "scikit-learn is not installed. Please install it using pip or conda."
            )
            if self.clustering_dialog is not None:
                self.clustering_dialog.checkBoxClustering.setChecked(False)
            return

        # Check if any columns are selected for clustering
        if not cluster_columns:
            # No columns selected, use default (x, y, z) values
            logging.info("No columns selected for clustering. Using x, y, z values.")

        # Prepare parameters based on the selected method
        params = {}
        if cluster_method == "hdbscan":
            params = {
                "min_samples": self.clustering_dialog._cluster_min_samples,
                "min_cluster_size": self.clustering_dialog._cluster_min_cluster_size
            }
        elif cluster_method == "kmeans":
            params = {
                "n_clusters": self.clustering_dialog._cluster_n_clusters
            }

        # Clean up any existing worker thread
        if hasattr(self, 'clustering_worker') and self.clustering_worker is not None:
            # Disconnect any existing connections
            try:
                self.clustering_worker.clustering_done.disconnect(self.on_clustering_done)
                self.clustering_worker.clustering_error.disconnect(self.on_clustering_error)
                self.clustering_worker.progress_updated.disconnect(self.on_clustering_progress)
            except (TypeError, RuntimeError):
                # No connections exist, or the signal was not connected to this slot
                pass

            # Wait for the thread to finish if it's still running
            if self.clustering_worker.isRunning():
                self.clustering_worker.wait()

            # Delete the old worker
            self.clustering_worker.deleteLater()

        # Create a new worker thread
        self.clustering_worker = self.ClusteringWorker(
            self, 
            cluster_method,
            params
        )

        # Connect the signals to slots
        self.clustering_worker.clustering_done.connect(self.on_clustering_done)
        self.clustering_worker.clustering_error.connect(self.on_clustering_error)
        self.clustering_worker.progress_updated.connect(self.on_clustering_progress)

        # Start the worker thread
        self.clustering_worker.start()

    def on_cancel_clustering(self):
        """
        Cancel the current clustering operation.
        """
        logging.log(0, "Cancelling clustering operation")

        if hasattr(self, 'clustering_worker') and self.clustering_worker is not None and self.clustering_worker.isRunning():
            # Request the worker to stop
            self.clustering_worker.stop()
            logging.log(0, "Requested clustering worker to stop")

            # Update dialog UI if it exists
            if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
                self.clustering_dialog.pushButtonCancelClustering.setText("Cancelling...")
                self.clustering_dialog.pushButtonCancelClustering.setEnabled(False)
                logging.log(0, "Updated clustering dialog UI for cancellation")

            # The worker will emit clustering_done with None values when it's done
            logging.log(0, "Waiting for worker to complete cancellation")

    def on_clustering_progress(self, progress):
        """
        Update the progress bar with the current clustering progress.

        Args:
            progress: Integer value between 0 and 100 representing the progress percentage
        """
        logging.log(0, f"Clustering progress: {progress}%")

        # Update progress in dialog if it exists
        if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
            self.clustering_dialog.update_progress(progress)
            logging.log(0, f"Updated clustering dialog progress bar to {progress}%")

    def on_clustering_error(self, error_message):
        """
        Handle errors that occur during clustering.

        Args:
            error_message: String containing the error message
        """
        logging.log(0, f"Clustering error: {error_message}")

        # Clear any partial clustering results
        self._cluster_labels = None
        self._cluster_probabilities = None

        # Set the _use_clustering flag to False
        self._use_clustering = False
        logging.log(0, "Set _use_clustering flag to False due to error")

        # Update dialog if it exists
        if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
            self.clustering_dialog.clustering_completed(success=False)

        # Display an error message to the user
        QtWidgets.QMessageBox.critical(
            self,
            "Clustering Error",
            f"An error occurred during clustering:\n\n{error_message}\n\nPlease try again with different parameters."
        )

    def on_clustering_done(self, result):
        """
        Handle the completion of clustering.

        Args:
            result: Tuple containing cluster labels and probabilities
        """
        logging.log(0, "Clustering completed")

        # Update the cluster labels and probabilities
        self._cluster_labels, self._cluster_probabilities = result

        # Initialize the cluster data shape if it doesn't exist
        if not hasattr(self, '_cluster_data_shape'):
            self._cluster_data_shape = len(self._cluster_labels) if self._cluster_labels is not None else 0
            logging.log(0, f"Initialized cluster data shape: {self._cluster_data_shape}")

        # If result is None, it means clustering was cancelled or failed
        if result[0] is None:
            logging.info("Clustering was cancelled or failed")

            # Set the _use_clustering flag to False
            self._use_clustering = False
            logging.log(0, "Set _use_clustering flag to False due to cancellation or failure")

            # Update dialog if it exists
            if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
                self.clustering_dialog.clustering_completed(success=False)

            return

        # Adjust the spinBoxCluster range based on the number of clusters
        if self._cluster_labels is not None:
            # Get the unique cluster labels
            unique_clusters = np.unique(self._cluster_labels)
            # Count the number of clusters (excluding noise points with label -1)
            n_clusters = len([c for c in unique_clusters if c >= 0])
            # Set the maximum value of spinBoxCluster to (n_clusters - 1)
            self.plot_control.spinBoxCluster.setMaximum(n_clusters - 1)
            logging.log(0, f"Adjusted spinBoxCluster range to (-1, {n_clusters - 1})")

        # Update dialog if it exists
        if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
            self.clustering_dialog.clustering_completed(success=True)

        # Display a message to the user that clustering is complete
        QtWidgets.QMessageBox.information(
            self,
            "Clustering Complete",
            f"Clustering using {self.clustering_dialog._cluster_method.upper()} has been completed successfully."
        )


    def perform_clustering(self, method=None, worker=None, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Perform clustering on the current data using the specified method.

        Args:
            method: The clustering method to use. If None, uses the method from clustering dialog.
            worker: The ClusteringWorker instance that called this method, used for progress updates and cancellation.
            **kwargs: Additional parameters for the clustering method.
                For HDBSCAN:
                    min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
                    min_cluster_size: Minimum number of points for a cluster.
                For K-means:
                    n_clusters: Number of clusters to form.
                For UMAP enhancement:
                    use_umap_enhancement: Whether to use UMAP for dimensionality reduction before clustering.
                    umap_n_neighbors: Number of neighbors to consider for each point in UMAP.
                    umap_min_dist: Minimum distance between points in the UMAP embedding.
                    umap_n_components: Number of components (dimensions) for the UMAP embedding.

        Returns:
            Tuple containing:
            - cluster_labels: Array of cluster labels for each data point
            - cluster_probabilities: Array of cluster membership probabilities (or None for K-means)
        """
        if method is None:
            method = self.clustering_dialog._cluster_method

        if method == "hdbscan" and not hdbscan:
            logging.error("HDBSCAN is not installed. Cannot perform clustering.")
            return None, None
        elif method == "kmeans" and not KMeans:
            logging.error("scikit-learn is not installed. Cannot perform clustering.")
            return None, None

        if self._data_source.empty:
            return None, None

        # Extract parameters for the selected method
        if method == "hdbscan":
            min_samples = kwargs.get("min_samples", self.clustering_dialog._cluster_min_samples)
            min_cluster_size = kwargs.get("min_cluster_size", self.clustering_dialog._cluster_min_cluster_size)
        elif method == "kmeans":
            n_clusters = kwargs.get("n_clusters", self.clustering_dialog._cluster_n_clusters)

        # Report progress: 10% - Starting data preparation
        if worker:
            worker.progress_updated.emit(10)
            # Check if stop was requested
            if worker._stop_requested:
                logging.info("Clustering cancelled during data preparation")
                return None, None

        # Get the data for clustering based on selected columns
        if self.clustering_dialog._cluster_columns:
            # Use selected columns
            df = self._data_source.data
            selected_data = []

            for column in self.clustering_dialog._cluster_columns:
                if column in df.columns:
                    # Convert to numeric and handle errors
                    values = pd.to_numeric(df[column], errors='coerce').values
                    selected_data.append(values)

            if not selected_data:  # If no valid columns were found
                logging.warning("No valid columns selected for clustering. Using x, y, z values.")
                data = np.column_stack((self.x_values, self.y_values, self.z_values))
            else:
                data = np.column_stack(selected_data)
        else:
            # If no columns are selected, use x, y, z values
            logging.info("No columns selected for clustering. Using x, y, z values.")
            data = np.column_stack((self.x_values, self.y_values, self.z_values))

        # Report progress: 20% - Data collected
        if worker:
            worker.progress_updated.emit(20)
            # Check if stop was requested
            if worker._stop_requested:
                logging.info("Clustering cancelled after data collection")
                return None, None

        # Remove any rows with NaN or Inf values
        mask = ~np.any(np.isnan(data) | np.isinf(data), axis=1)
        clean_data = data[mask]

        # Report progress: 30% - Data cleaned
        if worker:
            worker.progress_updated.emit(30)
            # Check if stop was requested
            if worker._stop_requested:
                logging.info("Clustering cancelled after data cleaning")
                return None, None

        # Check if UMAP enhancement is enabled
        use_umap_enhancement = kwargs.get("use_umap_enhancement", False)

        # Apply UMAP dimensionality reduction if enhancement is enabled
        if use_umap_enhancement and umap is not None:
            logging.info("Applying UMAP dimensionality reduction before clustering")

            # Get UMAP parameters
            umap_n_neighbors = kwargs.get("umap_n_neighbors", 15)
            umap_min_dist = kwargs.get("umap_min_dist", 0.1)
            umap_n_components = kwargs.get("umap_n_components", 2)

            # Check if we have enough data points for UMAP
            if len(clean_data) < umap_n_neighbors:
                logging.warning(f"Not enough data points for UMAP. Need at least {umap_n_neighbors} (n_neighbors parameter).")
                return None, None

            # Report progress: 35% - Starting UMAP
            if worker:
                worker.progress_updated.emit(35)
                # Check if stop was requested
                if worker._stop_requested:
                    logging.info("Clustering cancelled before UMAP")
                    return None, None

            try:
                # Create and fit the UMAP reducer
                reducer = umap.UMAP(
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_components=umap_n_components,
                    random_state=42  # For reproducibility
                )

                # Fit and transform the data
                clean_data = reducer.fit_transform(clean_data)
                logging.info(f"Data dimensionality reduced to {umap_n_components} using UMAP")

            except Exception as e:
                logging.error(f"Error during UMAP dimensionality reduction: {str(e)}")
                # Continue with original data if UMAP fails
                logging.info("Continuing with original data")

        # Check if we have enough data points
        if method == "hdbscan" and len(clean_data) < min_cluster_size:
            logging.warning(f"Not enough data points for HDBSCAN clustering. Need at least {min_cluster_size}.")
            return None, None
        elif method == "kmeans" and len(clean_data) < n_clusters:
            logging.warning(f"Not enough data points for K-means clustering. Need at least {n_clusters} (one per cluster).")
            return None, None

        try:
            # Report progress: 40% - Starting clustering algorithm
            if worker:
                worker.progress_updated.emit(40)
                # Check if stop was requested
                if worker._stop_requested:
                    logging.info("Clustering cancelled before algorithm start")
                    return None, None

            if method == "hdbscan":
                # Create and fit the HDBSCAN clusterer
                clusterer = hdbscan.HDBSCAN(
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                    prediction_data=True
                )

                # This is the most CPU-intensive part
                clusterer.fit(clean_data)

                # Report progress: 70% - HDBSCAN clustering completed
                if worker:
                    worker.progress_updated.emit(70)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled after HDBSCAN fit")
                        return None, None

                # Get cluster labels and probabilities
                labels = clusterer.labels_
                probabilities = clusterer.probabilities_

                # Create full-sized arrays with NaN for filtered points
                full_labels = np.full(len(data), -1, dtype=np.int32)
                full_probabilities = np.zeros(len(data))

                # Fill in the values for non-filtered points
                full_labels[mask] = labels
                full_probabilities[mask] = probabilities

            elif method == "kmeans":
                # Create and fit the K-means clusterer
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=42  # For reproducibility
                )

                # This is the most CPU-intensive part
                clusterer.fit(clean_data)

                # Report progress: 70% - K-means clustering completed
                if worker:
                    worker.progress_updated.emit(70)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled after K-means fit")
                        return None, None

                # Get cluster labels
                labels = clusterer.labels_

                # Create full-sized arrays with NaN for filtered points
                full_labels = np.full(len(data), -1, dtype=np.int32)

                # Fill in the values for non-filtered points
                full_labels[mask] = labels

                # For K-means, we don't have probabilities, so we use the distance to the cluster center
                # as a proxy for probability (inverse of distance)
                distances = np.zeros(len(clean_data))

                # Report progress: 80% - Starting distance calculations
                if worker:
                    worker.progress_updated.emit(80)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled before distance calculations")
                        return None, None

                # Calculate distances in batches to reduce CPU load and allow cancellation
                batch_size = 1000
                for batch_start in range(0, len(clean_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(clean_data))

                    # Check if stop was requested before processing each batch
                    if worker and worker._stop_requested:
                        logging.info(f"Clustering cancelled during distance calculations at batch {batch_start}-{batch_end}")
                        return None, None

                    for i in range(batch_start, batch_end):
                        cluster_idx = labels[i]
                        if cluster_idx >= 0:  # Skip noise points
                            center = clusterer.cluster_centers_[cluster_idx]
                            distances[i] = np.linalg.norm(clean_data[i] - center)

                    # Update progress during batch processing
                    if worker:
                        progress = 80 + int((batch_end / len(clean_data)) * 10)
                        worker.progress_updated.emit(progress)

                # Normalize distances to [0, 1] range and invert (closer = higher probability)
                if len(distances) > 0:
                    max_dist = np.max(distances) if np.max(distances) > 0 else 1
                    probabilities = 1 - (distances / max_dist)
                else:
                    probabilities = np.array([])

                # Create full-sized array for probabilities
                full_probabilities = np.zeros(len(data))
                full_probabilities[mask] = probabilities

            else:
                logging.error(f"Unsupported clustering method: {method}")
                return None, None

            # Report progress: 90% - Updating data frame
            if worker:
                worker.progress_updated.emit(90)
                # Check if stop was requested
                if worker._stop_requested:
                    logging.info("Clustering cancelled before data frame update")
                    return None, None

            # Add cluster labels and probabilities to the data frame
            df = self._data_source.data
            df['Cluster Label'] = full_labels
            df['Cluster Probability'] = full_probabilities
            self._data_source.data = df  # Update the data frame to trigger parameter_names update

            # Update the plot control to include the new columns
            # Note: This should be done in the main thread, not here
            # We'll handle this in the on_clustering_done method

            # Report progress: 100% - Clustering completed
            if worker:
                worker.progress_updated.emit(100)

            # Store the data shape used for clustering
            self._cluster_data_shape = len(data)
            logging.log(0, f"Stored cluster data shape: {self._cluster_data_shape}")

            return full_labels, full_probabilities

        except Exception as e:
            logging.error(f"Error during {method} clustering: {str(e)}")
            return None, None

    def keyPressEvent(self, event):
        """
        Handle key press events.

        Args:
            event: The key event
        """
        # Check if the 'c' key was pressed
        if event.key() == QtCore.Qt.Key_C:
            # Open the clustering dialog
            self.toggle_clustering_dialog()
        else:
            # For other keys, call the parent class implementation
            super(NDXplorer, self).keyPressEvent(event)

    def closeEvent(self, event):
        """
        Handle the window close event.
        Clean up resources before closing.
        """
        # Stop the Z range check timer
        if hasattr(self, 'z_range_check_timer'):
            self.z_range_check_timer.stop()

        # Clean up any existing worker thread
        if hasattr(self, 'clustering_worker') and self.clustering_worker is not None:
            # Disconnect any existing connections
            try:
                self.clustering_worker.clustering_done.disconnect(self.on_clustering_done)
                self.clustering_worker.clustering_error.disconnect(self.on_clustering_error)
                self.clustering_worker.progress_updated.disconnect(self.on_clustering_progress)
            except (TypeError, RuntimeError):
                # No connections exist, or the signal was not connected to this slot
                pass

            # Stop the worker if it's running
            if self.clustering_worker.isRunning():
                self.clustering_worker.stop()
                self.clustering_worker.wait()

            # Delete the worker
            self.clustering_worker.deleteLater()
            self.clustering_worker = None

        # Call the base class implementation
        super(NDXplorer, self).closeEvent(event)

    def check_z_range_changes(self):
        """
        Check if the Z selection range has changed and update the histograms if necessary.
        This method is called periodically by a timer.
        """
        if not self._dynamic_selection or not hasattr(self, 'selection_z'):
            return

        # Get the current Z selection range
        current_range = self.selection_z.get_range()

        # If the range has changed, update the histograms
        if self._last_z_range != current_range:
            logging.log(0, f"Z selection range changed from {self._last_z_range} to {current_range}")
            self._last_z_range = current_range
            # Update histograms and plots
            self.update_histograms()
            self.update_plots(skip_clustering=True)

    def on_dynamic_selection_changed(self, state):
        """
        Handle changes to the dynamic selection checkbox.

        Args:
            state: The new state of the checkbox (Qt.Checked or Qt.Unchecked)
        """
        self._dynamic_selection = bool(state)
        # Update histograms to reflect the new selection state
        self.update_histograms()
        # Update plots to display the new histograms
        self.update_plots(skip_clustering=True)

    def create_umap_plot(self, columns, params):
        """
        Create and display a UMAP plot in a separate window using PyQtGraph.

        Args:
            columns: Set of column names to use for UMAP
            params: Dictionary of parameters for UMAP
                n_neighbors: Number of neighbors to consider for each point
                min_dist: Minimum distance between points in the embedding
                n_components: Number of components (dimensions) for the embedding
        """
        logging.log(0, f"Creating UMAP plot with params: {params}")

        # Check if UMAP is available
        if not umap:
            QtWidgets.QMessageBox.warning(
                self,
                "UMAP Not Available",
                "UMAP is not installed. Please install it using pip or conda."
            )
            return

        # Store the UMAP windows as instance variables to prevent garbage collection
        if not hasattr(self, 'umap_windows'):
            self.umap_windows = []

        # Close any existing UMAP windows
        for window in self.umap_windows:
            window.close()
        self.umap_windows = []

        # Get the data for UMAP based on selected columns
        if columns:
            # Use selected columns
            df = self._data_source.data
            selected_data = []

            for column in columns:
                if column in df.columns:
                    # Convert to numeric and handle errors
                    values = pd.to_numeric(df[column], errors='coerce').values
                    selected_data.append(values)

            if not selected_data:  # If no valid columns were found
                logging.warning("No valid columns selected for UMAP. Using x, y, z values.")
                data = np.column_stack((self.x_values, self.y_values, self.z_values))
            else:
                data = np.column_stack(selected_data)
        else:
            # If no columns are selected, use x, y, z values
            logging.info("No columns selected for UMAP. Using x, y, z values.")
            data = np.column_stack((self.x_values, self.y_values, self.z_values))

        # Remove any rows with NaN or Inf values
        mask = ~np.any(np.isnan(data) | np.isinf(data), axis=1)
        clean_data = data[mask]

        # Check if we have enough data points
        if len(clean_data) < params['n_neighbors']:
            QtWidgets.QMessageBox.warning(
                self,
                "Not Enough Data",
                f"Not enough data points for UMAP. Need at least {params['n_neighbors']} (n_neighbors parameter)."
            )
            return

        try:
            # Create and fit the UMAP reducer
            reducer = umap.UMAP(
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                n_components=params['n_components'],
                random_state=42  # For reproducibility
            )

            # Fit and transform the data
            embedding = reducer.fit_transform(clean_data)

            # Import pyqtgraph
            import pyqtgraph as pg

            # Create the plot based on the number of components
            if params['n_components'] == 2:
                # Create a new window for the UMAP plot
                umap_window = QtWidgets.QMainWindow()
                umap_window.setWindowTitle('UMAP Projection')
                umap_window.resize(800, 600)

                # Add the window to the list of UMAP windows
                self.umap_windows.append(umap_window)

                # Create central widget and layout
                central_widget = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(central_widget)

                # Create plot widget
                plot_widget = pg.PlotWidget(title='UMAP Projection')
                plot_widget.setLabel('bottom', 'UMAP 1')
                plot_widget.setLabel('left', 'UMAP 2')

                # If cluster labels are available, color points by cluster
                if hasattr(self, '_cluster_labels') and self._cluster_labels is not None:
                    # Get cluster labels for non-filtered points
                    cluster_labels = self._cluster_labels[mask]

                    # Get unique cluster labels
                    unique_labels = np.unique(cluster_labels)

                    # Create a colormap
                    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

                    # Create a legend
                    legend = pg.LegendItem(offset=(70, 30))
                    legend.setParentItem(plot_widget.graphicsItem())

                    # Create a scatter plot item for each cluster
                    for i, label in enumerate(unique_labels):
                        mask_label = cluster_labels == label

                        # Convert color to RGBA format for PyQtGraph
                        color = colors[i]
                        rgba = (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*100))

                        scatter_item = pg.ScatterPlotItem(
                            x=embedding[mask_label, 0],
                            y=embedding[mask_label, 1],
                            size=5,
                            pen=None,
                            brush=pg.mkBrush(*rgba),
                            name=f"Cluster {label}"
                        )
                        plot_widget.addItem(scatter_item)

                        # Add item to legend
                        legend.addItem(scatter_item, f"Cluster {label}")
                else:
                    # Create scatter plot item with default color
                    scatter = pg.ScatterPlotItem(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        size=5,
                        pen=None,
                        brush=pg.mkBrush(255, 255, 255, 100)
                    )
                    plot_widget.addItem(scatter)

                # Add plot widget to layout
                layout.addWidget(plot_widget)

                # Set central widget
                umap_window.setCentralWidget(central_widget)

                # Show the main plot window
                umap_window.show()
                # Bring the window to the front
                umap_window.activateWindow()
                umap_window.raise_()

            elif params['n_components'] == 3:
                # Import pyqtgraph.opengl for 3D plotting
                import pyqtgraph.opengl as gl

                # Create a new window for the UMAP plot
                umap_window = QtWidgets.QMainWindow()
                umap_window.setWindowTitle('UMAP Projection (3D)')
                umap_window.resize(800, 600)

                # Add the window to the list of UMAP windows
                self.umap_windows.append(umap_window)

                # Create central widget and layout
                central_widget = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(central_widget)

                # Create 3D view widget
                view_widget = gl.GLViewWidget()

                # If cluster labels are available, color points by cluster
                if hasattr(self, '_cluster_labels') and self._cluster_labels is not None:
                    # Get cluster labels for non-filtered points
                    cluster_labels = self._cluster_labels[mask]

                    # Get unique cluster labels
                    unique_labels = np.unique(cluster_labels)

                    # Create a colormap
                    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

                    # Create a scatter plot for each cluster
                    for i, label in enumerate(unique_labels):
                        mask_label = cluster_labels == label

                        # Convert color to RGBA format for PyQtGraph
                        color = colors[i]

                        scatter_item = gl.GLScatterPlotItem(
                            pos=embedding[mask_label],
                            size=5,
                            color=(color[0], color[1], color[2], 0.5),
                            pxMode=True
                        )
                        view_widget.addItem(scatter_item)

                    # Create a separate 2D plot widget for the legend
                    legend_widget = pg.PlotWidget(title='Legend')
                    legend_widget.setFixedHeight(len(unique_labels) * 30 + 50)  # Adjust height based on number of clusters
                    legend_widget.getPlotItem().hideAxis('left')
                    legend_widget.getPlotItem().hideAxis('bottom')

                    # Create a legend
                    legend = pg.LegendItem(offset=(10, 10))
                    legend.setParentItem(legend_widget.getPlotItem())

                    # Add items to the legend
                    for i, label in enumerate(unique_labels):
                        color = colors[i]
                        rgba = (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*100))

                        # Create a dummy scatter item for the legend
                        dummy_scatter = pg.ScatterPlotItem(
                            x=[0], y=[0],
                            size=5,
                            pen=None,
                            brush=pg.mkBrush(*rgba)
                        )

                        # Add to legend
                        legend.addItem(dummy_scatter, f"Cluster {label}")

                    # Add legend widget to layout
                    layout.addWidget(legend_widget)
                else:
                    # Create 3D scatter plot with default color
                    scatter_plot = gl.GLScatterPlotItem(
                        pos=embedding,
                        size=5,
                        color=(1, 1, 1, 0.5),
                        pxMode=True
                    )
                    view_widget.addItem(scatter_plot)

                # Add axes
                axes = gl.GLAxisItem()
                axes.setSize(x=1, y=1, z=1)
                view_widget.addItem(axes)

                # Add view widget to layout
                layout.addWidget(view_widget)

                # Set central widget
                umap_window.setCentralWidget(central_widget)

                # Show the main plot window
                umap_window.show()
                # Bring the window to the front
                umap_window.activateWindow()
                umap_window.raise_()

        except Exception as e:
            logging.error(f"Error during UMAP: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "UMAP Error",
                f"An error occurred during UMAP: {str(e)}"
            )

    def update_2d_plot(self):
        try:
            new_data, x_edges, y_edges = self._histogram["2d"]
        except ValueError:
            return None

        log_counts = self.checkBoxLogCounts.isChecked()
        if log_counts:
            new_data = np.log10(new_data)
            new_data = np.nan_to_num(new_data)

        # Update the data of the displayed image
        # Rotate the data to match the original orientation
        self.cax.set_data(np.flip(np.rot90(new_data, k=3), axis=1))

        # Set the intensity range using the vmin and vmax properties from the UI
        self.cax.set_lut_range([self.vmin, self.vmax])

        # Redraw the main plot to update the display
        self.g_2dplot.replot()

        # Update curve overlays
        self.update_curve_overlays()

    def bin_to_x_value(self, bin_idx, x_edges):
        """Convert a bin index to an x value (center of the bin)."""
        if bin_idx < 0 or bin_idx >= len(x_edges) - 1:
            return None
        return (x_edges[bin_idx] + x_edges[bin_idx + 1]) / 2

    def bin_to_y_value(self, bin_idx, y_edges):
        """Convert a bin index to a y value (center of the bin)."""
        if bin_idx < 0 or bin_idx >= len(y_edges) - 1:
            return None
        return (y_edges[bin_idx] + y_edges[bin_idx + 1]) / 2

    def x_value_to_bin(self, x_value, x_edges):
        """Convert an x value to a bin index with linear interpolation."""
        for i in range(len(x_edges) - 1):
            if x_edges[i] <= x_value <= x_edges[i + 1]:
                # Calculate the relative position within the bin (0.0 to 1.0)
                bin_width = x_edges[i + 1] - x_edges[i]
                if bin_width == 0:  # Avoid division by zero
                    return float(i)
                relative_pos = (x_value - x_edges[i]) / bin_width
                # Return the bin index plus the relative position
                return float(i) + relative_pos
        return None

    def y_value_to_bin(self, y_value, y_edges):
        """Convert a y value to a bin index with linear interpolation."""
        for i in range(len(y_edges) - 1):
            if y_edges[i] <= y_value <= y_edges[i + 1]:
                # Calculate the relative position within the bin (0.0 to 1.0)
                bin_width = y_edges[i + 1] - y_edges[i]
                if bin_width == 0:  # Avoid division by zero
                    return float(i)
                relative_pos = (y_value - y_edges[i]) / bin_width
                # Return the bin index plus the relative position
                return float(i) + relative_pos
        return None

    def update_curve_overlays(self):
        """Update the curve overlays on the 2D histogram."""
        # Remove existing curve items
        for curve_item in self.curve_items:
            self.overlay_plot.del_item(curve_item)
        self.curve_items = []

        try:
            # Get the 2D histogram data and edges
            _, x_edges, y_edges = self._histogram["2d"]
        except (ValueError, KeyError):
            return

        # Get visible curves from the overlay widget
        visible_curves = self.curve_overlay_widget.get_visible_curves()

        # Get the number of points to use for curve computation
        num_points = self.curve_overlay_widget.get_num_points()

        # Synchronize the overlay plot's axes with the main plot
        self.overlay_plot.setAxisScale(QwtPlot.xBottom, 0, len(x_edges) - 1)
        self.overlay_plot.setAxisScale(QwtPlot.yLeft, 0, len(y_edges) - 1)

        for equation, parameters, color in visible_curves:
            # Create x values array with the specified number of points
            # Use the same scaling function (linear or logarithmic) that was used to create the bins
            x_min = x_edges[0]
            x_max = x_edges[-1]

            # Check if x-axis is using logarithmic scale
            if self.plot_control.scale_x == "log":
                if x_min <= 0:
                    x_min = 1e-6
                if x_max <= 0:
                    x_max = 1e-6
                x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_points)
            else:
                x_values = np.linspace(x_min, x_max, num_points)

            # Evaluate the equation
            y_values = self.curve_evaluator.evaluate(equation, x_values, parameters)

            if y_values is None:
                continue  # Skip if evaluation failed

            # Convert x and y values to bin coordinates for plotting
            # Note: The 2D histogram is rotated 90 degrees in the plot
            x_coords = []
            y_coords = []

            # Check if y-axis is using logarithmic scale and adjust y values accordingly
            if self.plot_control.scale_y == "log":
                # For logarithmic y-axis, we need to ensure y values are positive
                y_values = np.maximum(y_values, 1e-6)

            for i, (x, y) in enumerate(zip(x_values, y_values)):
                # Check if y is within the y range
                if y < y_edges[0] or y > y_edges[-1]:
                    continue

                # Convert to bin coordinates
                # Note: The 2D histogram is rotated 90 degrees in the plot
                # so we need to swap x and y coordinates
                y_bin = self.y_value_to_bin(y, y_edges)
                if y_bin is None:
                    continue

                # Convert x value to bin index
                x_bin = self.x_value_to_bin(x, x_edges)
                if x_bin is None:
                    continue

                # Add points to the curve
                # The y-coordinate is the bin index (not the value)
                # The x-coordinate is the bin index (not the value)
                x_coords.append(x_bin)  # x bin index
                y_coords.append(y_bin)  # y bin index

            if not x_coords:
                continue  # Skip if no valid points

            # Create a curve item
            curveparam = guiqwt.styles.CurveParam()
            curveparam.line.color = color  # Use the selected color
            curveparam.line.width = 2.0
            curve_item = guiqwt.curve.CurveItem(curveparam=curveparam)
            curve_item.set_data(x_coords, y_coords)

            # Add the curve to the overlay plot
            self.overlay_plot.add_item(curve_item)
            self.curve_items.append(curve_item)

        # Redraw the overlay plot to update the display
        self.overlay_plot.replot()
