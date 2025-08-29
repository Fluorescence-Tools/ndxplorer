from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .logging_config import logging

import os
import json
import yaml
import typing
import pathlib
import importlib.util

# Import settings functions
from .settings import get_settings_path, ensure_default_settings

import numpy as np

# Delay imports of heavy libraries
hdbscan = None  # For clustering
KMeans = None   # For clustering
GaussianMixture = None  # For Gaussian Mixture Modeling
umap = None     # For dimensionality reduction
napari = None   # For image visualization in external viewer

from . plot_control import SurfacePlotWidget
from . parameter_editor import ParameterEditor
from . curve_overlay import CurveOverlayWidget, CurveEvaluator
from . import reader
from . import writer
from .clustering_dialog import ClusteringDialog
from .column_selection_dialog import ColumnSelectionDialog
from . import plot_umap
from .clustering import ClusteringManager, ClusteringWorker
from .widgets import ScientificSpinBox
from .mouse_event_filter import MouseEventFilter
from .axis_control_dialog import AxisControlDialog
from guiqwt.colormap import get_colormap_list

try:
    from chisurf.gui.tools.code_editor import CodeEditor
except ImportError:
    from ndxplorer.widgets.code_editor import CodeEditor

from . data_source import DataSource

try:
    from chisurf.gui import QtGui, QtCore, uic, QtWidgets
    from chisurf.gui.QtGui import QFont, QImage
except ImportError:
    from qtpy import QtCore, uic
    from qtpy import QtGui, QtWidgets
    from qtpy.QtGui import QFont, QImage

import guiqwt.signals
import guiqwt.plot
import guiqwt.image
import guiqwt.curve
import guiqwt.styles

from guiqwt.plot import CurveDialog
from guiqwt.builder import make
from guidata.widgets.dataframeeditor import DataFrameEditor
from qwt.plot import QwtPlot
from qwt.plot_canvas import QwtPlotCanvas
from .image_items import FixedImageItem



class NDXplorer(QtWidgets.QMainWindow):

    def invalidate_values_cache(self) -> None:
        """
        Manually clear the cached 'values'. Call this whenever something
        changes that would invalidate the mask or the data.
        """
        logging.debug("Invalidating values cache")
        self._cached_values = None
        self._cached_values_selections = None
        self._cached_values_p13 = None
        self._cached_values_mask_inf = None
        self._cached_values_mask_nan = None
        # Clear the new cache variables
        self._cached_filtered_values = None
        self._cached_values_mask_id = None
        # Clear axis cache variables
        self._cached_x_values = None
        self._cached_x_param_idx = None
        self._cached_y_values = None
        self._cached_y_param_idx = None
        self._cached_z_values = None
        self._cached_z_param_idx = None
        # Clear histogram cache
        self._cached_hist_params = None

    @property
    def data_source(self) -> DataSource:
        logging.debug("Getting data_source")
        if self._data_source.empty:
            values = self._default_data_source
            logging.debug("Using default data source")
        else:
            values = self._data_source
            logging.debug(f"Using actual data source with {self._data_source.values.shape[1] if not self._data_source.empty else 0} data points")
        return values

    @data_source.setter
    def data_source(self, v: DataSource) -> None:
        logging.info(f"Setting data_source with {v.values.shape[1] if not v.empty else 0} data points")
        self._data_source = v
        # Whenever the underlying DataSource changes, invalidate the cached 'values'
        self.invalidate_values_cache()
        logging.debug("Computing columns with equations and constants")
        self._data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )

    @property
    def x_values(self) -> np.ndarray:
        logging.debug(f"Getting x_values for parameter: {self.plot_control.x_label}")
        # Check if we have a cached result that's still valid
        if hasattr(self, '_cached_x_values') and self._cached_x_values is not None:
            # Check if the parameter index and values cache are still valid
            if (getattr(self, '_cached_x_param_idx', None) == self.plot_control.p1[0] and
                getattr(self, '_cached_values_mask_id', None) == id(self.value_mask)):
                logging.debug("Using cached x_values")
                return self._cached_x_values

        # Get the values and extract the x column
        values = self.values
        x_values = values[self.plot_control.p1[0]].astype('float64')

        # Cache the result and parameter index
        self._cached_x_values = x_values
        self._cached_x_param_idx = self.plot_control.p1[0]

        return x_values

    @property
    def y_values(self) -> np.ndarray:
        logging.debug(f"Getting y_values for parameter: {self.plot_control.y_label}")
        # Check if we have a cached result that's still valid
        if hasattr(self, '_cached_y_values') and self._cached_y_values is not None:
            # Check if the parameter index and values cache are still valid
            if (getattr(self, '_cached_y_param_idx', None) == self.plot_control.p2[0] and
                getattr(self, '_cached_values_mask_id', None) == id(self.value_mask)):
                logging.debug("Using cached y_values")
                return self._cached_y_values

        # Get the values and extract the y column
        values = self.values
        y_values = values[self.plot_control.p2[0]].astype('float64')

        # Cache the result and parameter index
        self._cached_y_values = y_values
        self._cached_y_param_idx = self.plot_control.p2[0]

        return y_values

    @property
    def z_values(self)-> np.ndarray:
        logging.debug(f"Getting z_values for parameter: {self.plot_control.z_label}")
        # Check if we have a cached result that's still valid
        if hasattr(self, '_cached_z_values') and self._cached_z_values is not None:
            # Check if the parameter index and values cache are still valid
            if (getattr(self, '_cached_z_param_idx', None) == self.plot_control.p3[0] and
                getattr(self, '_cached_values_mask_id', None) == id(self.value_mask)):
                logging.debug("Using cached z_values")
                return self._cached_z_values

        # Get the values and extract the z column
        values = self.values
        z_values = values[self.plot_control.p3[0]].astype('float64')

        # Cache the result and parameter index
        self._cached_z_values = z_values
        self._cached_z_param_idx = self.plot_control.p3[0]

        return z_values

    @property
    def weight_enabled(self) -> bool:
        """
        Get the current weight enabled status.
        
        Returns:
            bool: True if weighting is enabled, False otherwise
        """
        return self.current_weight_enabled
        
    @weight_enabled.setter
    def weight_enabled(self, value: bool) -> None:
        """
        Set the weight enabled status.
        
        Args:
            value: Boolean indicating whether weighting should be enabled
        """
        self.current_weight_enabled = bool(value)
        # Update the UI to match
        self.checkBoxWeight.setChecked(self.current_weight_enabled)
        
    @property
    def weight_param(self) -> str:
        """
        Get the current weight parameter.
        
        Returns:
            str: The name of the current weight parameter, or "None" if weighting is disabled
        """
        return self.current_weight_param
        
    @weight_param.setter
    def weight_param(self, value: str) -> None:
        """
        Set the weight parameter.
        
        Args:
            value: The name of the parameter to use for weighting
        """
        if not isinstance(value, str):
            return
            
        self.current_weight_param = value
        
        # Update the UI to match if the parameter exists
        if self.current_weight_param != "None":
            index = self.comboBoxWeight.findText(self.current_weight_param)
            if index >= 0:
                self.comboBoxWeight.setCurrentIndex(index)
    
    @property
    def value_mask(self):
        selections = self.plot_control.get_selections()
        mask_inf = self._mask_inf
        mask_nan = self._mask_nan
        p13 = (self.plot_control.p1[0], self.plot_control.p2[0], self.plot_control.p3[0])
        logging.debug(f"Value mask parameters: p13={p13}, mask_inf={mask_inf}, mask_nan={mask_nan}, selections={len(selections)}")

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
            logging.debug("Using cached values")
            return self._cached_values

        # Step 3: If cache is invalid or empty, compute fresh data
        logging.debug("Cache invalid, computing fresh data")
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
            logging.debug(f"Dynamic selection: {np.sum(z_mask)} points selected out of {len(d3)}")

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
                    logging.debug(f"Cluster selection: {points_in_cluster_after_masking} points in cluster {selected_cluster} (out of {points_in_cluster} total in this cluster)")
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
        logging.debug("Values cached for future use")

        return mask

    @property
    def values(self) -> np.ndarray:
        """
        Return a 2D array of data (selected columns only), applying the
        user-defined mask for Inf/NaN. The result is cached to avoid repeated
        computation when .values is accessed multiple times.
        """
        logging.debug("Getting values with masking")

        # Check if we have a cached result that's still valid
        if hasattr(self, '_cached_filtered_values') and self._cached_filtered_values is not None:
            # The value_mask property already checks if the mask is still valid
            # If it returns the cached mask, we can use our cached filtered values
            if getattr(self, '_cached_values_mask_id', None) == id(self.value_mask):
                logging.debug("Using cached filtered values")
                return self._cached_filtered_values

        mask = self.value_mask
        all_values = self.data_source.values

        x = np.ma.array(all_values, mask=mask)
        oCol, oRow = x.shape
        logging.debug(f"Original data shape: {oCol}x{oRow}")
        re = np.ma.compressed(x)
        nD = re.shape[0]
        re = re.reshape((oCol, int(nD / oCol)))
        logging.debug(f"Reshaped data shape: {re.shape}")

        # Cache the result and the mask ID for future use
        self._cached_filtered_values = re
        self._cached_values_mask_id = id(mask)

        return re

    @property
    def ymax(self) -> float:
        logging.debug("Getting ymax")
        result = max(self.y_values)
        logging.debug(f"ymax = {result}")
        return result

    @property
    def zmin(self):
        logging.debug("Getting zmin")
        arr = np.asarray(self.z_values)
        finite_mask = np.isfinite(arr)
        v = arr[finite_mask]
        logging.debug(f"Filtered out {len(arr) - len(v)} non-finite values")
        if getattr(self.plot_control, 'scale_z', 'lin') == "log":
            v_before = len(v)
            v = v[v > 0]
            logging.debug(f"Log scale: filtered out {v_before - len(v)} non-positive values")
        if v.size == 0:
            logging.debug("No valid values for zmin, returning 0")
            return 0
        result = float(np.min(v))
        logging.debug(f"zmin = {result}")
        return result

    @property
    def zmax(self) -> float:
        logging.debug("Getting zmax")
        result = max(self.z_values)
        logging.debug(f"zmax = {result}")
        return result

    @property
    def working_path(self):
        logging.debug("Getting working_path")
        path = self.lineEditWorkingPath.text()
        logging.debug(f"working_path = {path}")
        return path

    @working_path.setter
    def working_path(self, v):
        logging.info(f"Setting working_path to {v}")
        if pathlib.Path(v).is_dir():
            logging.debug(f"Path {v} is a valid directory, updating working path")
            self.lineEditWorkingPath.setText(v)
        else:
            logging.warning(f"Path {v} is not a valid directory, working path not updated")

    @property
    def xmin(self) -> float:
        logging.debug("Getting xmin")
        # Cache x_values once to avoid race between repeated property accesses
        arr = np.asarray(self.x_values)
        # Keep only finite values
        finite_mask = np.isfinite(arr)
        v = arr[finite_mask]
        logging.debug(f"Filtered out {len(arr) - len(v)} non-finite values")
        # For log scale, remove non-positive values
        if getattr(self.plot_control, 'scale_x', 'lin') == "log":
            v_before = len(v)
            v = v[v > 0]
            logging.debug(f"Log scale: filtered out {v_before - len(v)} non-positive values")
        if v.size == 0:
            logging.debug("No valid values for xmin, returning 0")
            return 0
        result = float(np.min(v))
        logging.debug(f"xmin = {result}")
        return result

    @property
    def xmax(self) -> float:
        logging.debug("Getting xmax")
        result = max(self.x_values)
        logging.debug(f"xmax = {result}")
        return result

    @property
    def ymin(self) -> float:
        logging.debug("Getting ymin")
        arr = np.asarray(self.y_values)
        finite_mask = np.isfinite(arr)
        v = arr[finite_mask]
        logging.debug(f"Filtered out {len(arr) - len(v)} non-finite values")
        if getattr(self.plot_control, 'scale_y', 'lin') == "log":
            v_before = len(v)
            v = v[v > 0]
            logging.debug(f"Log scale: filtered out {v_before - len(v)} non-positive values")
        if v.size == 0:
            logging.debug("No valid values for ymin, returning 0")
            return 0
        result = float(np.min(v))
        logging.debug(f"ymin = {result}")
        return result

    @property
    def vmin(self):
        logging.debug("Getting vmin")
        result = self.doubleSpinBox_vmin.value()
        logging.debug(f"vmin = {result}")
        return result

    @vmin.setter
    def vmin(self, v):
        logging.debug(f"Setting vmin to {v}")
        return self.doubleSpinBox_vmin.setValue(v)

    @property
    def vmax(self):
        logging.debug("Getting vmax")
        result = self.doubleSpinBox_vmax.value()
        logging.debug(f"vmax = {result}")
        return result

    @vmax.setter
    def vmax(self, v):
        logging.debug(f"Setting vmax to {v}")
        return self.doubleSpinBox_vmax.setValue(v)

    @property
    def current_cmap(self) -> str:
        logging.debug("Getting current_cmap")
        result = self.comboBoxCmap.currentText()
        logging.debug(f"current_cmap = {result}")
        return result

    def update_cmap(self, cmap_name = None):
        """
        Update the colormap of the imshow plot based on the selected cmap.
        """
        logging.debug(f"Updating colormap with cmap_name={cmap_name}")
        if cmap_name is None:
            cmap_name = self.current_cmap
            logging.debug(f"Using current colormap: {cmap_name}")

        # Apply the colormap to the image
        self.cax.set_color_map(cmap_name)
        self.g_2dplot.replot()  # Redraw the plot
        logging.debug(f"Colormap updated to {cmap_name}")

    def populate_colormap_combobox(self):
        """Populate the QComboBox with guiqwt colormap names."""
        logging.debug("Populating colormap combobox")
        colormap_names = sorted(get_colormap_list())  # Get all guiqwt colormap names
        logging.debug(f"Found {len(colormap_names)} colormaps")
        self.comboBoxCmap.addItems(colormap_names)  # Add them to the QComboBox

        # Set default selection
        if self.current_cmap in colormap_names:
            default_index = colormap_names.index(self.current_cmap)
            self.comboBoxCmap.setCurrentIndex(default_index)
            logging.info( f"Set default colormap to {self.current_cmap} at index {default_index}")
        else:
            logging.info( f"Default colormap {self.current_cmap} not found in available colormaps")

    def on_vmin_vmax_changed(self):
        logging.info( "vmin/vmax values changed")
        # Get current values from the spin boxes using the properties
        current_vmin = self.vmin  # this should read from doubleSpinBox_vmin.value()
        current_vmax = self.vmax  # similarly for doubleSpinBox_vmax.value()
        logging.info( f"Setting colormap limits to vmin={current_vmin}, vmax={current_vmax}")

        # Update the colormap limits for the 2D histogram image
        self.cax.set_lut_range([current_vmin, current_vmax])
        self.g_2dplot.replot()  # Redraw the plot to reflect the change
        logging.info( "Colormap limits updated")

    def set_default_colormap(self, default_cmap):
        """Set the default colormap in the QComboBox."""
        logging.info( f"Setting default colormap to {default_cmap}")
        index = self.comboBoxCmap.findText(default_cmap)  # Find the index of the colormap
        if index != -1:  # Ensure it exists in the list
            logging.info( f"Found colormap {default_cmap} at index {index}")
            self.comboBoxCmap.setCurrentIndex(index)  # Set the QComboBox to the colormap

            # Apply the colormap to the image
            self.cax.set_color_map(default_cmap)
            self.g_2dplot.replot()

            logging.info( f"Default colormap set to {default_cmap}")
        else:
            logging.info( f"Colormap {default_cmap} not found in available colormaps")

    def __init__(
            self,
            data_source=None,  # type: DataSource
            settings_json_fn=None,  # type: str
            parent=None,
            cmap: str = 'gist_earth',
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
        self._use_clustering = False

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
        
        # Initialize weight tracking attributes
        self.current_weight_enabled = False
        self.current_weight_param = "None"

        self.plot_control = SurfacePlotWidget(self)
        self.equation_editor = CodeEditor(parent=self)
        self.curve_overlay_widget = CurveOverlayWidget(self)
        self.curve_evaluator = CurveEvaluator()
        self.curve_items = []  # List to store curve items

        uic.loadUi(os.path.dirname(__file__) + '/plot_main.ui', self)
        self.verticalLayout_3.addWidget(self.plot_control)
        self.verticalLayout_15.addWidget(self.equation_editor)
        self.verticalLayout_10.addWidget(self.curve_overlay_widget)

        # Connect screenshot tool button if present
        try:
            if hasattr(self, 'toolButton_screenshot') and self.toolButton_screenshot is not None:
                self.toolButton_screenshot.clicked.connect(self.on_take_screenshot)
        except Exception as e:
            logging.debug(f"Could not connect screenshot button: {e}")

        # Make Fit action checkable and wire it to the Fit dock visibility
        self.actionFit_Gaussians.toggled.connect(self.dockWidget_Fit.setVisible)
        self.dockWidget_Fit.visibilityChanged.connect(self._on_fit_dock_visibility_changed)
        self.dockWidget_Fit.setVisible(False)

        # Report tool
        self.actionMake_Report.triggered.connect(self.onShowReportWizard)

        # Enable drag & drop on working path line edit
        try:
            self._install_working_path_drop()
        except Exception as e:
            logging.debug(f"Failed to enable working path drop: {e}")

        # Create scientific notation spin boxes for vmin and vmax
        self.doubleSpinBox_vmin = ScientificSpinBox(self, format_str="%.2e")
        self.doubleSpinBox_vmax = ScientificSpinBox(self, format_str="%.2e")

        # Set initial values and ranges
        self.doubleSpinBox_vmin.setRange(-1e10, 1e10)
        self.doubleSpinBox_vmax.setRange(-1e10, 1e10)
        self.doubleSpinBox_vmin.setValue(0.0)
        self.doubleSpinBox_vmax.setValue(1.0)
        self.doubleSpinBox_vmin.setSingleStep(0.1)
        self.doubleSpinBox_vmax.setSingleStep(0.1)

        # Create a container widget for the spin boxes
        spin_box_layout = self.horizontalLayout_3
        spin_box_layout.setContentsMargins(0, 0, 0, 0)
        spin_box_layout.setSpacing(0)  # Minimal spacing between widgets

        # Add all widgets to a single row for maximum compactness
        spin_box_layout.addWidget(self.doubleSpinBox_vmin)
        spin_box_layout.addWidget(self.doubleSpinBox_vmax)

        # Connect value changed signals to update colormap
        self.doubleSpinBox_vmin.valueChanged.connect(self.on_vmin_vmax_changed)
        self.doubleSpinBox_vmax.valueChanged.connect(self.on_vmin_vmax_changed)

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
            logging.info( "Save CB")
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

        # Initialize and connect the checkBoxEnableZ
        self.checkBoxEnableZ = self.plot_control.checkBoxEnableZ
        self.checkBoxEnableZ.setToolTip("When checked, the Z-axis plot is displayed")
        self.checkBoxEnableZ.stateChanged.connect(self.on_enable_z_changed)

        # Initialize and connect the checkBoxWeight
        self.checkBoxWeight = self.plot_control.checkBoxWeight
        self.checkBoxWeight.setToolTip("If checked, histograms are weighted by selected parameter")
        self.checkBoxWeight.stateChanged.connect(self.on_weight_changed)

        # Initialize comboBoxWeight
        self.comboBoxWeight = self.plot_control.comboBoxWeight
        self.comboBoxWeight.setToolTip("Select parameter to use as weights")
        self.comboBoxWeight.setEnabled(self.checkBoxWeight.isChecked())
        # Connect signal to update histograms when selection changes
        self.comboBoxWeight.currentIndexChanged.connect(self.on_weight_param_changed)
        # Initial population will be done in on_weight_changed

        # Set initial visibility of z-axis plot based on checkbox state
        # This will be properly set after the z-axis plot is created

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

        # Set initial visibility of z-axis plot based on checkbox state
        self.g_zplot.setVisible(self.checkBoxEnableZ.isChecked())

        # x-axis
        win_x = CurveDialog()
        self.g_xplot = win_x.get_plot()
        self.g_xplot.enableAxis(QwtPlot.xBottom, False)
        self.g_xplot.enableAxis(QwtPlot.xTop, True)
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
        self.g_yplot.enableAxis(QwtPlot.yLeft, False)
        self.g_yplot.enableAxis(QwtPlot.yRight, True)

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
        
        # Load background image
        bg_image_path = os.path.join(os.path.dirname(__file__), 'ui', 'background.png')
        if os.path.exists(bg_image_path):
            # Load the image using QImage
            bg_qimage = QImage(bg_image_path)
            if not bg_qimage.isNull():
                # Convert QImage to numpy array
                bg_qimage = bg_qimage.convertToFormat(QImage.Format_RGBA8888)
                width = bg_qimage.width()
                height = bg_qimage.height()
                ptr = bg_qimage.bits()
                ptr.setsize(height * width * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                
                # Create a 2D array from the image (using the first channel)
                bg_data = arr[:, :, 0].copy()
                
                # Create a FixedImageItem for the background
                self.bg_image_item = FixedImageItem(data=bg_data)
                self.g_2dplot.add_item(self.bg_image_item)
                
                # Initially show the background image
                self.bg_image_item.setVisible(True)
            else:
                logging.warning(f"Failed to load background image: {bg_image_path}")
                self.bg_image_item = None
        else:
            logging.warning(f"Background image not found: {bg_image_path}")
            self.bg_image_item = None

        # Set default colormap
        self.set_default_colormap(cmap)

        # Configure the plot
        self.g_2dplot.set_axis_font("left", QFont("Courier"))
        self.g_2dplot.set_axis_font("bottom", QFont("Courier"))

        # Initialize default font settings and apply
        self.font_settings = {
            "tick_size_pt": 8,
            "title_size_pt": 10,
            "title_weight": 700,
            "color": "#000000"
        }
        try:
            self.apply_fonts()
        except Exception:
            pass

        # Enable axes that we want to link with marginal plots
        self.g_2dplot.enableAxis(QwtPlot.xBottom, False)
        self.g_2dplot.enableAxis(QwtPlot.xTop, False)
        self.g_2dplot.enableAxis(QwtPlot.yLeft, False)
        self.g_2dplot.enableAxis(QwtPlot.yRight, False)

        # Set background color to white
        self.g_2dplot.canvas().setStyleSheet("background-color: white;")

        # Enable mouse tracking for region selection
        self.g_2dplot.canvas().setMouseTracking(True)
        # Listen to canvas resize events to keep orientation consistent on internal resizes
        try:
            self.g_2dplot.canvas().installEventFilter(self)
        except Exception:
            pass

        # Create a separate plot for curve overlays
        self.overlay_plot = guiqwt.curve.CurvePlot(parent=self)

        # Install the event filter on the overlay plot canvas
        self.mouse_event_filter = MouseEventFilter(self)
        self.overlay_plot.canvas().installEventFilter(self.mouse_event_filter)

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

        # -----------------------------------------------------------------
        # Gaussian Fit controls: attach from a separate module for cleanliness
        # -----------------------------------------------------------------
        from .gaussian_fit import GaussianFit
        self.gaussian_fit = GaussianFit(self)

        self.g_xplot.setMaximumHeight(150)
        self.g_yplot.setMaximumWidth(150)
        self.g_zplot.setMaximumHeight(150)

        # Load settings
        ###############
        if settings_json_fn is None:
            # Ensure default settings exist in the user's settings folder
            ensure_default_settings()
            # Get the path to the settings folder
            settings_path = get_settings_path()
            settings_json_fn = settings_path / "mfd.settings.json"
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
        # Get the settings path
        settings_path = get_settings_path()
        self.parameter_control = ParameterEditor(
            parent=self,
            json_file=str(settings_path / "mfd.constants.json"),
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
        self.actionOpenMfdHdf5.triggered.connect(self.onOpenMfdHdf5)
        self.actionBurst_IDs.triggered.connect(self.onSaveBurstIDs)

        # Settings
        self.actionLoad_settings.triggered.connect(self.onLoad_settings)
        self.actionSave_axis_settings.triggered.connect(self.onSaveAxisSettings)
        self.actionSet_default_axis.triggered.connect(self.onSetDefaultAxis)
        # GUI updates
        self.actionUpdate_plot.triggered.connect(lambda: self.update_plots())
        self.actionClear_plot.triggered.connect(self.clear_plots)
        self.actionMask_toggle_changed.triggered.connect(self.onMaskChanged)
        # UMAP
        self.actionUMAP.triggered.connect(self.onShowUMAP)
        self.actionAxisControl.triggered.connect(self.onShowAxisControl)

        # Connect toolButton_3 to show data in DataFrameEditor
        self.toolButton_3.clicked.connect(self.show_dataframe_editor)

        # Connect toolButton_AutoContrast to auto contrast function
        self.toolButton_AutoContrast.clicked.connect(self.on_auto_contrast)
        self.toolButton_3.setEnabled(True)  # Enable the button
        
        # Connect to save parameters
        self.toolButton_parameter_save.clicked.connect(self.save_parameters)
        self.actionConstants.triggered.connect(self.save_parameters)
        # Axis range

        self.overlay_plot.canvas().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.overlay_plot.canvas().customContextMenuRequested.connect(self.on_canvas_context_menu)

        # Assuming these combo boxes control the parameter selections for the 2D plot:
        self.plot_control.comboBoxSelX.currentIndexChanged.connect(self.update_spinbox_limits)
        self.plot_control.comboBoxSelY.currentIndexChanged.connect(self.update_spinbox_limits)
        self.plot_control.comboBoxSelZ.currentIndexChanged.connect(self.update_spinbox_limits)

        # Connections for spin boxes are already set up above

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
        logging.info( "clearing plots")
        # 0. Also clear the working path line edit (global clear should reset path)
        try:
            self.lineEditWorkingPath.blockSignals(True)
            try:
                self.lineEditWorkingPath.clear()
            except Exception:
                # Fallback in case clear() is not available
                self.lineEditWorkingPath.setText("")
        except Exception:
            pass
        finally:
            try:
                self.lineEditWorkingPath.blockSignals(False)
            except Exception:
                pass

        # 1. Clear the user data => empty => fallback to _default_data_source
        self._data_source.clear()

        # 2. Clear the selection table so no old mask references remain
        #    (But remember, this does NOT fix comboBoxSelX/Y/Z)
        self.plot_control.onClearSelection()

        # 2.5. Clear all overlays (curve overlays and Gaussian overlays)
        try:
            # Remove any existing curve overlay items from the overlay plot
            if hasattr(self, 'curve_items') and hasattr(self, 'overlay_plot') and self.overlay_plot is not None:
                for item in list(self.curve_items):
                    try:
                        self.overlay_plot.del_item(item)
                    except Exception:
                        pass
                try:
                    self.curve_items.clear()
                except Exception:
                    self.curve_items = []
            # Clear curve overlay widgets (also resets internal state and emits signal)
            if hasattr(self, 'curve_overlay_widget') and self.curve_overlay_widget is not None:
                self.curve_overlay_widget.clear_curves()
            # Clear Gaussian overlays (ellipses, marginals, and table)
            try:
                self.on_clear_gaussians()
            except Exception:
                # If GaussianFit not initialized, ignore
                pass
            # Ensure overlay canvas is refreshed
            if hasattr(self, 'overlay_plot') and self.overlay_plot is not None:
                try:
                    self.overlay_plot.replot()
                except Exception:
                    pass
        except Exception:
            pass

        # 2.6. Clear clustering data
        if hasattr(self, '_cluster_labels'):
            self._cluster_labels = None

        # 2.7. Clear cluster probabilities if they exist
        if hasattr(self, '_cluster_probabilities'):
            self._cluster_probabilities = None

        # 2.8. Hide clustering dialog if it's visible
        if hasattr(self, 'clustering_dialog') and self.clustering_dialog is not None and self.clustering_dialog.isVisible():
            self.clustering_dialog.hide()

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

        # 5. Directly call update_plots to ensure all graphs are cleared
        self.update_plots()

    def copy_1d_hists_to_clipboard_csv(self):
        try:
            x_hist = self._histogram["x"]  # tuple: (bin_edges, counts)
            y_hist = self._histogram["y"]
            z_hist = self._histogram.get("z", ())  # Use get with default empty tuple
        except Exception as e:
            logging.error("1D histogram data not available: {}".format(e))
            return

        import io
        output = io.StringIO()

        # Extract histogram components for X, Y, and Z
        x_edges, x_counts = x_hist
        y_edges, y_counts = y_hist
        
        # Check if z_hist is not empty before unpacking
        if z_hist:
            z_edges, z_counts = z_hist
        else:
            z_edges, z_counts = [], []

        n_x = len(x_counts)
        n_y = len(y_counts)
        n_z = len(z_counts)
        n_rows = max(n_x, n_y, n_z)

        # Write header with columns for each histogram
        header_columns = [
            "X Bin Start", "X Bin End", "X Count",
            "Y Bin Start", "Y Bin End", "Y Count"
        ]
        
        # Only include Z columns if z_hist exists
        if z_hist:
            header_columns.extend(["Z Bin Start", "Z Bin End", "Z Count"])
            
        header = "\t".join(header_columns)
        output.write(header + "\n")

        # Write each row
        for i in range(n_rows):
            row_values = []
            
            # X histogram values
            if i < n_x:
                x_bin_start = f"{x_edges[i]:12.4e}"
                x_bin_end = f"{x_edges[i + 1]:12.4e}"
                x_count = f"{x_counts[i]:12.4e}"
            else:
                x_bin_start = x_bin_end = x_count = ""
            row_values.extend([x_bin_start, x_bin_end, x_count])
            
            # Y histogram values
            if i < n_y:
                y_bin_start = f"{y_edges[i]:12.4e}"
                y_bin_end = f"{y_edges[i + 1]:12.4e}"
                y_count = f"{y_counts[i]:12.4e}"
            else:
                y_bin_start = y_bin_end = y_count = ""
            row_values.extend([y_bin_start, y_bin_end, y_count])
            
            # Z histogram values (only if z_hist exists)
            if z_hist and i < n_z:
                z_bin_start = f"{z_edges[i]:12.4e}"
                z_bin_end = f"{z_edges[i + 1]:12.4e}"
                z_count = f"{z_counts[i]:12.4e}"
                row_values.extend([z_bin_start, z_bin_end, z_count])
            
            row = "\t".join(row_values)
            output.write(row + "\n")

        csv_text = output.getvalue()
        output.close()

        # Copy the resulting CSV text to the clipboard
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(csv_text)
        logging.info( "1D histograms data copied to clipboard as CSV with side-by-side columns.")

    def is_napari_available(self):
        """
        Check if napari is installed and available for use.
        
        This method uses lazy importing to avoid loading napari until it's needed.
        It first checks if napari has already been imported, and if not, attempts
        to import it. If the import fails, it logs a debug message and returns False.
        
        Returns:
            bool: True if napari is available, False otherwise
        """
        global napari
        if napari is None:
            try:
                import napari
                logging.debug("Imported napari library")
                return True
            except ImportError:
                napari = None
                logging.debug("napari library not available")
                return False
        return True
        
    def send_to_napari(self):
        """
        Send the current 2D histogram image to napari as a new layer.
        
        This method transfers the current 2D histogram data to napari for
        visualization. It performs the following steps:
        1. Checks if napari is installed and available
        2. Gets the current 2D histogram data
        3. Creates a napari viewer or uses an existing one
        4. Adds the histogram as a new image layer with appropriate labels
        
        If napari is not installed, it shows a warning message to the user.
        If no 2D histogram data is available, it logs a warning and returns.
        
        The image is transposed to ensure correct orientation in napari.
        """
        # Check if napari is available
        if not self.is_napari_available():
            QtWidgets.QMessageBox.warning(
                self,
                "Napari Not Available",
                "Napari is not installed. Please install it using pip or conda."
            )
            return
            
        # Get the 2D histogram data
        if not hasattr(self, '_histogram') or '2d' not in self._histogram:
            logging.warning("No 2D histogram data available to send to napari")
            return
            
        # Get the 2D histogram data and transpose it for correct orientation in napari
        hist_data = self._histogram['2d'][0].T
        
        # Get axis labels
        x_label = self.plot_control.x_label if hasattr(self.plot_control, 'x_label') else "X"
        y_label = self.plot_control.y_label if hasattr(self.plot_control, 'y_label') else "Y"
        weight_label = self.plot_control.weight_parameter
        
        # Create a napari viewer if one doesn't exist
        viewer = napari.current_viewer()
        if viewer is None:
            viewer = napari.Viewer()
            
        # Add the image as a new layer
        viewer.add_image(
            hist_data,
            name=f"NDXplorer: {x_label} vs {y_label} {weight_label}",
            colormap='viridis',
            scale=[1, 1]
        )
        
        logging.info(f"Sent 2D histogram to napari: {x_label} vs {y_label}")
    
    def on_canvas_context_menu(self, pos):
        """
        Display a context menu when right-clicking on the 2D plot canvas.
        
        This method creates a context menu with options to:
        - Copy the 2D histogram data as CSV
        - Copy the 1D histograms data as CSV
        - Send the current 2D histogram to Napari (if installed)
        
        Args:
            pos: The position where the context menu should be displayed
        """
        menu = QtWidgets.QMenu(self.g_2dplot.canvas())
        action_csv = menu.addAction("Copy 2D Histogram (CSV)")
        #action_json = menu.addAction("Copy 2D Histogram (JSON)")
        action_csv1d = menu.addAction("Copy 1D Histograms (CSV)")
        
        # Add napari option - allows sending the current 2D histogram to Napari
        action_napari = menu.addAction("Send to Napari")
        
        action = menu.exec_(self.g_2dplot.canvas().mapToGlobal(pos))
        if action == action_csv:
            self.copy_2d_hist_to_clipboard_csv()
        #elif action == action_json:
        #    self.copy_2d_hist_to_clipboard_json()
        elif action == action_csv1d:
            self.copy_1d_hists_to_clipboard_csv()
        elif action == action_napari:
            self.send_to_napari()

    def onMaskChanged(self) -> None:
        """
        Whenever the user toggles the Inf/NaN masks,
        invalidate the cache and re-plot.
        """
        self._mask_inf = self.checkBoxMaskInf.isChecked()
        self._mask_nan = self.checkBoxMaskNaN.isChecked()
        self.invalidate_values_cache()
        self.update_plots()

    def onShowAxisControl(self) -> None:
        """
        Show the Axis Control dialog.
        This method is triggered when the user clicks the Axis Control action in the View menu.
        """
        # Create and show the axis control dialog
        dialog = AxisControlDialog(parent=self)
        dialog.exec_()
        
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

    def onShowReportWizard(self):
        """Open the Report Tool dialog."""
        try:
            from .report_tool import ReportWizard
            dlg = ReportWizard(parent=self)
            dlg.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Report Tool Error", str(e))

    def on_take_screenshot(self):
        """Capture a screenshot of the NDXplorer window and ask the user where to save it.
        Also copies the screenshot to the system clipboard.
        """
        try:
            # Grab the entire window as a pixmap
            pixmap = self.grab()

            # Copy to clipboard (in addition to asking to save)
            try:
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    clipboard = app.clipboard()
                    if clipboard is not None:
                        clipboard.setPixmap(pixmap)
            except Exception as e:
                logging.debug(f"Failed to copy screenshot to clipboard: {e}")

            # Build a default filename in the working path
            try:
                base_dir = self.working_path if getattr(self, 'working_path', None) else os.getcwd()
            except Exception:
                base_dir = os.getcwd()

            # Create default name with timestamp
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            default_name = os.path.join(base_dir, f"ndxplorer_screenshot_{ts}.png")

            # Ask the user where to save
            filename, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Save Screenshot',
                default_name,
                'PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)'
            )
            if not filename:
                return

            # Determine image format from extension
            ext = os.path.splitext(filename)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                img_format = 'JPEG'
            elif ext == '.bmp':
                img_format = 'BMP'
            else:
                img_format = 'PNG'
                if not ext:
                    # Append default extension if none provided
                    filename = filename + '.png'

            ok = pixmap.save(filename, img_format)
            if not ok:
                QtWidgets.QMessageBox.warning(self, 'Save Screenshot', f'Failed to save screenshot to:\n{filename}')
            else:
                logging.info(f"Saved screenshot to {filename}")
        except Exception as e:
            logging.error(f"Error while taking screenshot: {e}")
            QtWidgets.QMessageBox.critical(self, 'Save Screenshot', f'An error occurred while saving the screenshot:\n{e}')

    def onSelectWorkingPath(self):
        working_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select current path', self.working_path)
        # If user cancels the dialog, do not change the working path
        if not working_path:
            return
        self.lineEditWorkingPath.blockSignals(True)
        self.lineEditWorkingPath.setText(working_path)
        self.lineEditWorkingPath.blockSignals(False)

    def _install_working_path_drop(self):
        """
        Enable dropping of folders and files on the working path line edit.
        - If a folder is dropped: open as burst analysis folder and set working path.
        - If an HDF5 (.h5/.hdf5) file is dropped: open via onOpenMfdHdf5.
        - If a CSV (.csv) file is dropped: open via onOpenCsv.
        """
        le = self.lineEditWorkingPath
        try:
            le.setAcceptDrops(True)
        except Exception:
            pass

        def dragEnterEvent(event):
            try:
                md = event.mimeData()
                if md and md.hasUrls():
                    urls = md.urls()
                    if urls:
                        # Check first URL for type acceptance
                        p = Path(urls[0].toLocalFile())
                        if p.exists() and (p.is_dir() or p.suffix.lower() in ('.h5', '.hdf5', '.csv')):
                            event.acceptProposedAction()
                            return
                event.ignore()
            except Exception as e:
                logging.debug(f"dragEnterEvent error: {e}")
                event.ignore()

        def dropEvent(event):
            try:
                md = event.mimeData()
                if not (md and md.hasUrls()):
                    event.ignore()
                    return
                paths = [Path(u.toLocalFile()) for u in md.urls()]
                # Prefer directories first
                dirs = [p for p in paths if p.exists() and p.is_dir()]
                if dirs:
                    p = dirs[0]
                    event.acceptProposedAction()
                    try:
                        self.lineEditWorkingPath.setText(str(p))
                    except Exception:
                        pass
                    try:
                        # Open dropped analysis folder
                        self.open_files(file_type="burst_dir", file_handles=str(p), append=False)
                    except Exception as e:
                        logging.error(f"Failed to open burst analysis folder from drop: {e}")
                    return

                files = [p for p in paths if p.exists() and p.is_file()]
                if files:
                    csvs = [str(p) for p in files if p.suffix.lower() == '.csv']
                    h5s = [str(p) for p in files if p.suffix.lower() in ('.h5', '.hdf5')]

                    if csvs:
                        event.acceptProposedAction()
                        try:
                            self.onOpenCsv(None, filenames=csvs, append=False, merge_mode='columns')
                        except Exception as e:
                            logging.error(f"Failed to open CSV from drop: {e}")
                        return

                    if h5s:
                        event.acceptProposedAction()
                        try:
                            self.onOpenMfdHdf5(None, filenames=h5s, append=False, merge_mode='columns')
                        except Exception as e:
                            logging.error(f"Failed to open HDF5 from drop: {e}")
                        return

                event.ignore()
            except Exception as e:
                logging.debug(f"dropEvent error: {e}")
                event.ignore()

        # Monkey patch events onto the line edit (minimal invasive change)
        le.dragEnterEvent = dragEnterEvent
        le.dropEvent = dropEvent

    def onSaveBurstIDs(self, evt=None, folder=None):
        """
        Save burst IDs to a folder and show dialog for microtime histogram.
        
        Args:
            evt: Event that triggered this method (not used)
            folder: Folder where burst IDs will be saved. If None, a folder selection dialog will be shown.
        """
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Folder for Burst IDs', self.working_path
            )
        
        # If user cancelled the dialog, return
        if not folder:
            return
            
        logging.info(f"Saving burst IDs to {folder}...")
        writer.save_burst_ids(
            folder_name=folder,
            selections=self.plot_control.get_selections(),
            data_source=self.data_source
        )
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Process BIDs")
        
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        
        # Add message
        label = QtWidgets.QLabel("Process Burst IDs.")
        layout.addWidget(label)
        
        # Add checkbox (unchecked by default)
        checkbox = QtWidgets.QCheckBox("Compute microtime histogram")
        checkbox.setChecked(True)
        layout.addWidget(checkbox)
        
        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Set layout and show dialog
        dialog.setLayout(layout)
        
        # If dialog is accepted (OK clicked), proceed with export
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Get checkbox state for auto_transfer parameter
            export_chisurf = checkbox.isChecked()
            if not export_chisurf:
                logging.info("Skipping auto transfer to ChiSurf.")
                return
            else:
                logging.info("Exporting burst IDs to ChiSurf...")
            try:
                # Try to find setup name from photon_selection_parameters.json
                setup_name = None
                bid_folder = Path(folder)

                # Look for photon_selection_parameters.json in the Info folder
                # First check if there's an Info folder in the parent directory
                logging.info(f"Looking for setup name in {bid_folder}...")
                info_folder = bid_folder.parent / "Info"
                if not info_folder.exists():
                    # Try looking for Info folder in the grandparent directory
                    info_folder = bid_folder.parent.parent / "Info"
                    logging.info(f"Looking for setup name in {info_folder}...")
                
                if info_folder.exists():
                    params_file = info_folder / "photon_selection_parameters.json"
                    if params_file.exists():
                        try:
                            with open(params_file, 'r') as f:
                                params = json.load(f)
                                setup_name = params.get("selected_setup")
                                if setup_name:
                                    logging.info(f"Found setup name '{setup_name}' in photon_selection_parameters.json")
                        except Exception as e:
                            logging.error(f"Error reading photon_selection_parameters.json: {e}")
                
                # Check if we can import MicrotimeHistogram
                from chisurf.plugins.microtime_histogram.wizard import MicrotimeHistogram
                
                # Get or create the MicrotimeHistogram instance
                histogram = MicrotimeHistogram.get_instance()
                histogram.show()
                histogram.raise_()  # Bring window to front
                
                # Load the BID folder in the existing instance
                histogram.load_bid_folder(folder, setup_name=setup_name)
                                    
            except Exception as e:
                logging.error(f"Failed to launch microtime histogram plugin: {str(e)}")
                # Show error message to user
                QtWidgets.QMessageBox.warning(
                    self,
                    "Export Error",
                    f"Failed to export to ChiSurf: {str(e)}"
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

    def save_parameters(self):
        """
        Save the current parameters to a JSON file in the user's settings folder.
        If chisurf module exists, parameters are saved in the user folder.
        """
        # Get the settings path (this will use chisurf user folder if available)
        settings_path = get_settings_path()
        
        # Create the filename for the parameters
        param_filename = str(settings_path / "mfd.constants.json")
        
        # Save the parameters
        with open(param_filename, "w") as fp:
            json.dump(
                self.parameter_control.dict,
                fp,
                indent=4
            )
        
        logging.info( f"Parameters saved to {param_filename}")
    
    def onSaveAxisSettings(
            self,
            event,
            settings_json_fn=None  # type: str
    ):
        # Qt signals like triggered(bool) may pass a boolean; treat it as no filename provided
        if isinstance(settings_json_fn, bool):
            settings_json_fn = None
        
        # Choose target file (default to user's settings mfd.axis.json)
        if not settings_json_fn:
            settings_path = get_settings_path()
            default_filename = str(settings_path / "mfd.axis.json")
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Axis settings file',
                default_filename,
                'Axis file (*.axis.json)'
            )
            if not filename:
                return
            settings_json_fn = filename
        
        # Ensure we have the most recent UI values stored for the current axes
        try:
            if hasattr(self, 'plot_control') and self.plot_control is not None:
                self.plot_control.update_x_axis_settings()
                self.plot_control.update_y_axis_settings()
                self.plot_control.update_z_axis_settings()
        except Exception as e:
            logging.debug(f"Could not refresh axis settings from UI before saving: {e}")
        
        # Load baseline axis settings from target file if exists; otherwise from packaged defaults
        baseline = {}
        target_path = Path(settings_json_fn)
        try:
            if target_path.exists():
                with open(target_path, 'r', encoding='utf-8') as fp:
                    baseline = json.load(fp) or {}
            else:
                # fall back to default packaged settings
                packaged = Path(__file__).parent / 'settings' / 'mfd.axis.json'
                if packaged.exists():
                    with open(packaged, 'r', encoding='utf-8') as fp:
                        baseline = json.load(fp) or {}
        except Exception as e:
            logging.warning(f"Failed to load baseline axis settings, starting from empty. Reason: {e}")
            baseline = {}
        
        # Collect current axis names in use (x, y, z)
        in_use_names = []
        try:
            in_use_names = [
                getattr(self.plot_control, 'x_label', '') or '',
                getattr(self.plot_control, 'y_label', '') or '',
                getattr(self.plot_control, 'z_label', '') or ''
            ]
        except Exception as e:
            logging.debug(f"Could not determine in-use axis names: {e}")
        
        # Merge: for each in-use axis name, overwrite baseline entry if current differs or not present
        current = getattr(self.plot_control, 'axis_settings', {}) or {}
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
                changed[name] = {'from': base_val, 'to': cur}
        
        # Save merged baseline back to the chosen file
        print(f"Saving axis settings to {settings_json_fn}")
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as fp:
                json.dump(baseline, fp, indent=4)
            if changed:
                logging.info(f"Updated axis settings for: {', '.join(changed.keys())}")
            else:
                logging.info("No changes detected for current axes; saved existing settings file unchanged")
        except Exception as e:
            logging.error(f"Failed to save axis settings to {settings_json_fn}: {e}")

    def onSetDefaultAxis(self):
        """
        Set the current axis selections (and weight, if available) as the new defaults
        in the active ndxplorer settings JSON under the 'default_axes' key.
        """
        # Determine target settings file
        settings_path = getattr(self, "_settings_json_path", None)
        if not settings_path:
            try:
                settings_path = str(get_settings_path() / "mfd.settings.json")
            except Exception:
                settings_path = None
        if not settings_path:
            QtWidgets.QMessageBox.warning(self, "Set default axis", "Could not determine settings file path.")
            return

        # Load settings from file (fallback to in-memory settings on failure)
        try:
            with open(settings_path, "r", encoding="utf-8") as fp:
                settings_data = json.load(fp) or {}
        except Exception as e:
            logging.debug(f"Could not read settings file '{settings_path}', using in-memory settings. Reason: {e}")
            settings_data = dict(self.settings) if hasattr(self, 'settings') and isinstance(self.settings, dict) else {}

        # Get the current axis names
        try:
            x_name = self.plot_control.p1[1]
            y_name = self.plot_control.p2[1]
            z_name = self.plot_control.p3[1]
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Set default axis", f"Unable to read current axis selections: {e}")
            return

        weight_name = None
        try:
            if hasattr(self, 'comboBoxWeight') and self.comboBoxWeight is not None:
                weight_name = str(self.comboBoxWeight.currentText())
        except Exception:
            weight_name = None

        # Update default_axes and colormap
        settings_data.setdefault("default_axes", {})
        settings_data["default_axes"].update({
            "x": x_name,
            "y": y_name,
            "z": z_name
        })
        if weight_name:
            settings_data["default_axes"]["weight"] = weight_name
        # Also persist the current colormap selection so it becomes the user default
        try:
            settings_data["colormap"] = self.current_cmap
        except Exception:
            pass

        # Save back to file
        try:
            with open(settings_path, "w", encoding="utf-8") as fp:
                json.dump(settings_data, fp, indent=4)
            # Update in-memory settings
            try:
                self.settings.update(settings_data)
            except Exception:
                pass
            QtWidgets.QMessageBox.information(self, "Set default axis", "Default axis settings have been updated.")
            logging.info(f"Updated default_axes in '{settings_path}' to {settings_data.get('default_axes')}")
        except Exception as e:
            logging.error(f"Failed to save default axes to '{settings_path}': {e}")
            QtWidgets.QMessageBox.critical(self, "Set default axis", f"Failed to save default axis settings:\n{e}")

    def onLoad_settings(
            self,
            settings_json_fn=None  # type: str
    ):
        # Handle Qt signals possibly passing a boolean
        if isinstance(settings_json_fn, bool):
            settings_json_fn = None
        if settings_json_fn is None:
            file_sel = QtWidgets.QFileDialog.getOpenFileName(
                None, 'ndXplorer settings file', self.working_path, 'ndXplorer settings (*.settings.json)'
            )
            if isinstance(file_sel, (tuple, list)):
                settings_json_fn = file_sel[0]
            else:
                settings_json_fn = file_sel
        if not settings_json_fn:
            return
        # Remember which settings file is active for later updates
        try:
            self._settings_json_path = str(settings_json_fn)
        except Exception:
            self._settings_json_path = None
        with open(settings_json_fn, "r", encoding="utf-8") as fp:
            d = json.load(fp)
            self.settings.update(d)

        # Check if colormap is specified in settings and apply it
        if "colormap" in self.settings:
            self.set_default_colormap(self.settings["colormap"])

        # Get the settings directory and default settings directory
        settings_dir = pathlib.Path(settings_json_fn).parent
        default_settings_dir = pathlib.Path(__file__).parent / "settings"

        # Load axis settings
        fn_axis = settings_dir / self.settings["axis"]
        if not fn_axis.exists():
            # Fall back to default settings directory
            fn_axis = default_settings_dir / self.settings["axis"]
        with open(str(fn_axis), "r") as fp:
            d = json.load(fp)
            self.plot_control.axis_settings.update(d)

        # Load axis label settings
        # This section loads the configuration for enabling/disabling axis labels
        if "axis_labels" in self.settings:
            fn_axis_labels = settings_dir / self.settings["axis_labels"]
            if not fn_axis_labels.exists():
                # Fall back to default settings directory
                fn_axis_labels = default_settings_dir / self.settings["axis_labels"]
            
            # Initialize default axis label settings
            # These defaults will be used if the file doesn't exist or has missing settings
            self.axis_label_settings = {
                # Global setting to enable/disable all axis labels
                "enable_all_labels": True,
                # Individual settings for each plot type and axis
                "axis_labels": {
                    # Y-plot axis labels (top and right axes)
                    "y_plot": {"top": True, "right": True},
                    # X-plot axis labels (top axis)
                    "x_plot": {"top": True},
                    # Z-plot axis labels (bottom and left axes)
                    "z_plot": {"bottom": True, "left": True}
                },
                # Font settings (optional)
                "fonts": {
                    "tick_size_pt": 8,
                    "title_size_pt": 10,
                    "title_weight": 700,
                    "color": "#000000"
                }
            }
            
            # Load settings from file if it exists
            if fn_axis_labels.exists():
                try:
                    with open(str(fn_axis_labels), "r") as fp:
                        d = yaml.load(fp, Loader=yaml.FullLoader)
                        if d is not None:
                            # Update the default settings with values from the file
                            # This preserves default values for any missing settings
                            self.axis_label_settings.update(d)
                except Exception as e:
                    logging.warning(f"Error loading axis label settings: {e}")
            else:
                logging.warning(f"Axis label settings file not found: {fn_axis_labels}")

            # Update font_settings from axis_label_settings.fonts
            try:
                fonts = self.axis_label_settings.get("fonts", {})
                if not hasattr(self, 'font_settings'):
                    self.font_settings = {}
                self.font_settings.update(fonts)
                # Apply fonts immediately
                self.apply_fonts()
            except Exception as e:
                logging.debug(f"Could not apply font settings: {e}")

        # Load equations
        fn_equations = settings_dir / self.settings["equations"]
        if not fn_equations.exists():
            # Fall back to default settings directory
            fn_equations = default_settings_dir / self.settings["equations"]
        with open(str(fn_equations), "r") as fp:
            d = yaml.load(fp, Loader=yaml.FullLoader)
            self.equations = d

        # Load constants
        fn_constants = settings_dir / self.settings["constants"]
        if not fn_constants.exists():
            # Fall back to default settings directory
            fn_constants = default_settings_dir / self.settings["constants"]
        with open(str(fn_constants), "r") as fp:
            d = json.load(fp)
            self.constants.update(d)

        self.equation_editor.load_file(str(fn_equations))

    def open_files(
            self,
            file_handles: typing.List[str] = None,
            file_type: str = None,
            append: bool = False,
            merge_mode: str = 'columns'
    ):
        logging.info("NDXplorer: Opening files..")
        logging.debug(f"File handles: {file_handles}")
        logging.debug(f"File type: {file_type}")
        logging.debug(f"Append mode: {append}")
        logging.debug(f"Merge mode: {merge_mode}")
        wp = str(self.working_path)

        if file_type in ["cs_sampling", "er4"]:
            if not hasattr(file_handles, '__iter__'):
                file_handles, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    self, 'ChiSurf sampling files', wp, 'Sampling files (*.*)')
            # Update working path based on selection
            try:
                if file_handles:
                    first_path = str(file_handles[0])
                    dir_path = str(Path(first_path).parent)
                    self.working_path = dir_path
            except Exception as e:
                logging.debug(f"Could not update working path for cs_sampling: {e}")
            logging.info(f"Opening files ({file_type}): {file_handles}")
            data_reader = reader.read_csv_sampling

        elif file_type in ["burst_dir"]:
            if file_handles is None:
                file_handles = QtWidgets.QFileDialog.getExistingDirectory(
                    self, 'Open burst analysis folder', self.working_path)
            # Update working path to selected directory
            try:
                if file_handles:
                    self.working_path = str(file_handles)
            except Exception as e:
                logging.debug(f"Could not update working path for burst_dir: {e}")
            data_reader = reader.read_burst_analysis

        elif file_type in ["mfd_hdf5"]:
            if file_handles is None:
                file_handles, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    self, 'MFD HDF5 files', wp,
                    'HDF5 files (*.h5 *.hdf5);;ZIP files (*.zip);;All Files (*.*)')
            # Update working path based on selection
            try:
                if file_handles:
                    first_path = str(file_handles[0])
                    dir_path = str(Path(first_path).parent)
                    self.working_path = dir_path
            except Exception as e:
                logging.debug(f"Could not update working path for mfd_hdf5: {e}")
            logging.info(f"Opening MFD HDF5/Zip files: {file_handles}")

            if not file_handles:
                return  # user cancelled

            # Build a combined DataSource; for ZIPs, decide HDF5 vs burst by contents
            combined_data_source = None
            for file_path in file_handles:
                p = str(file_path)
                is_zip = p.lower().endswith('.zip')

                if is_zip:
                    # Inspect once
                    try:
                        import zipfile as _zip
                        with _zip.ZipFile(p, 'r') as zf:
                            names = zf.namelist()
                        has_h5 = any(n.lower().endswith(('.h5', '.hdf5')) for n in names)
                        logging.debug(f"ZIP '{p}' contains HDF5: {has_h5}")
                    except Exception as e:
                        logging.debug(f"Could not inspect zip '{p}': {e}")
                        has_h5 = False

                    temp_ds = reader.read_mfd_hdf5([p]) if has_h5 else reader.read_burst_analysis(p)
                else:
                    temp_ds = reader.read_mfd_hdf5([p])

                if combined_data_source is None:
                    combined_data_source = temp_ds
                else:
                    combined_data_source.merge(temp_ds, mode=merge_mode)

            # Append or replace
            if append and hasattr(self,
                                  '_data_source') and self._data_source is not None and not self._data_source.empty:
                if self._data_source.merge(combined_data_source, mode=merge_mode):
                    self.update()
            else:
                self._data_source = combined_data_source
                self.update()

            # Detect image axes if any; if not applied, try default axes from settings, then EXIT
            _img_applied = self.check_and_set_image_axes()
            if not _img_applied:
                try:
                    self.apply_default_axes_from_settings()
                except Exception as _e:
                    logging.debug(f"Could not apply default axes after HDF5 load: {_e}")
            logging.debug("Handled mfd_hdf5; returning before generic loader.")
            return

        else:
            if file_handles is None:
                file_handles, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    self, 'Comma separated value files', self.working_path, 'Text files (*.*)')
            # Update working path based on selection
            try:
                if file_handles:
                    first_path = str(file_handles[0])
                    dir_path = str(Path(first_path).parent)
                    self.working_path = dir_path
            except Exception as e:
                logging.debug(f"Could not update working path for csv: {e}")
            data_reader = reader.read_csv

        # Generic loader path (CSV / cs_sampling / burst_dir)
        if file_handles:
            logging.info(f"Opening files ({file_type or 'csv'}): {file_handles}")

            if append and hasattr(self,
                                  '_data_source') and self._data_source is not None and not self._data_source.empty:
                new_data_source = data_reader(file_handles)
                if self._data_source.merge(new_data_source, mode=merge_mode):
                    self.update()
            else:
                self._data_source = data_reader(file_handles)
                self.update()

        _img_applied = self.check_and_set_image_axes()
        if not _img_applied:
            try:
                self.apply_default_axes_from_settings()
            except Exception as _e:
                logging.debug(f"Could not apply default axes after generic load: {_e}")
        # Ensure plots are updated after applying axes
        self.on_auto_contrast()

    def show_merge_dialog(self, title):
        """
        Show a dialog to ask the user how to merge new data with existing data.

        Args:
            title (str): The title of the dialog

        Returns:
            tuple: (append, merge_mode) where:
                - append (bool): Whether to append the new data
                - merge_mode (str): The merge mode ('columns' or 'rows')
                - None if the dialog was cancelled
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout()

        # Question label
        label = QtWidgets.QLabel('How do you want to merge the new data?')
        layout.addWidget(label)

        # Radio buttons for merge options
        replace_rb = QtWidgets.QRadioButton('Replace existing data')
        append_columns_rb = QtWidgets.QRadioButton('Append as columns (add new columns, rows must match)')
        append_rows_rb = QtWidgets.QRadioButton('Append as rows (add new rows of existing columns)')

        # Set default selection
        replace_rb.setChecked(True)

        layout.addWidget(replace_rb)
        layout.addWidget(append_columns_rb)
        layout.addWidget(append_rows_rb)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        # Show dialog
        result = dialog.exec_()

        if result != QtWidgets.QDialog.Accepted:
            return None

        # Determine append and merge_mode based on selection
        append = append_columns_rb.isChecked() or append_rows_rb.isChecked()
        merge_mode = 'columns'
        if append_columns_rb.isChecked():
            merge_mode = 'columns'
        elif append_rows_rb.isChecked():
            merge_mode = 'rows'

        return append, merge_mode

    def onOpenCsv(
            self,
            event,
            filenames: List[str] = None,
            append: bool = False,
            merge_mode: str = 'columns'
    ):
        # If no filenames provided, check if we should append
        if filenames is None and hasattr(self, '_data_source') and self._data_source is not None and not self._data_source.empty:
            result = self.show_merge_dialog('Open CSV Files')
            if result is None:
                return
            append, merge_mode = result

        self.open_files(file_type="csv", file_handles=filenames, append=append, merge_mode=merge_mode)

    def onOpenChiSurfSampling(
            self,
            filenames=None,  # type: List[str]
            append: bool = False,
            merge_mode: str = 'columns'
    ):
        # If no filenames provided, check if we should append
        if filenames is None and hasattr(self, '_data_source') and self._data_source is not None and not self._data_source.empty:
            result = self.show_merge_dialog('Open ChiSurf Sampling Files')
            if result is None:
                return
            append, merge_mode = result

        self.open_files(file_type="cs_sampling", file_handles=filenames, append=append, merge_mode=merge_mode)

    def onOpenMfdHdf5(
            self,
            event,
            filenames=None,  # type: List[str]
            append: bool = False,
            merge_mode: str = 'columns'
    ):
        """
        Open MFD HDF5 files.

        Args:
            filenames: List of HDF5 file paths
            append: Whether to append to existing data
            merge_mode: How to merge the data - 'columns' or 'rows'
        """
        # If no filenames provided, check if we should append
        if filenames is None and hasattr(self, '_data_source') and self._data_source is not None and not self._data_source.empty:
            result = self.show_merge_dialog('Open MFD HDF5 Files')
            if result is None:
                return
            append, merge_mode = result

        self.open_files(file_type="mfd_hdf5", file_handles=filenames, append=append, merge_mode=merge_mode)

    def onOpenSmFRET(self, merge_mode: str = 'columns'):
        # Check if we should append
        append = False
        if hasattr(self, '_data_source') and self._data_source is not None and not self._data_source.empty:
            result = self.show_merge_dialog('Open SmFRET Files')
            if result is None:
                return
            append, merge_mode = result

        self.open_files(file_type="burst_dir", append=append, merge_mode=merge_mode)

    def update(self, *args, **kwargs):
        super(NDXplorer, self).update()
        self.data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )
        self.lineEditCountTotal.setText(str(self.data_source.size))
        self.plot_control.update()  # plot_control.update() - also updates plots

    def apply_fonts(self):
        """
        Apply font settings (tick/title sizes) to all plots. Family is fixed.
        """
        try:
            fs = getattr(self, 'font_settings', {})
            tick_size = int(fs.get('tick_size_pt', 8))
            qf_tick = QFont("Segoe UI", int(tick_size))
        except Exception:
            qf_tick = QFont("Segoe UI", 8)
        # Apply to x,y,z,2d,overlay if present
        plots = [self.g_xplot, self.g_yplot, self.g_2dplot]
        try:
            if hasattr(self, 'g_zplot') and self.g_zplot is not None:
                plots.append(self.g_zplot)
        except Exception:
            pass
        try:
            if hasattr(self, 'overlay_plot') and self.overlay_plot is not None:
                plots.append(self.overlay_plot)
        except Exception:
            pass
        for gp in plots:
            try:
                gp.set_axis_font("bottom", qf_tick)
            except Exception:
                pass
            try:
                gp.set_axis_font("top", qf_tick)
            except Exception:
                pass
            try:
                gp.set_axis_font("left", qf_tick)
            except Exception:
                pass
            try:
                gp.set_axis_font("right", qf_tick)
            except Exception:
                pass
        # Titles are handled in update_parameter_names via fmt()

    def update_parameter_names(self):
        """
        Update the axis titles with the current parameter names.
        
        This method sets the axis titles for the plots based on the selected parameters.
        It also respects the axis label configuration settings, allowing labels to be
        enabled or disabled according to the user's preferences.
        In addition, it formats axis titles to be bold and slightly larger for better readability.
        """
        # Get the current parameter names from the plot control
        p1, p1_name = self.plot_control.p1
        p2, p2_name = self.plot_control.p2
        p3, p3_name = self.plot_control.p3

        # Helper: format titles as bold and slightly larger using Qt rich text
        def fmt(title: str) -> str:
            if not title:
                return ""
            # Use configured title font size/weight/color if available
            size_pt = None
            weight = 700
            color = None
            try:
                if hasattr(self, 'font_settings'):
                    weight = int(self.font_settings.get('title_weight', 700))
                    size_pt = float(self.font_settings.get('title_size_pt'))
                    color = self.font_settings.get('color')
            except Exception:
                pass
            color_css = f"; color:{color}" if color else ""
            if size_pt is not None:
                return f"<span style='font-weight:{weight}; font-size:{size_pt}pt{color_css}'>{title}</span>"
            else:
                return f"<span style='font-weight:{weight}; font-size:115%{color_css}'>{title}</span>"
        
        # Check if axis label settings are available
        # These settings are loaded from the axis_labels.yaml file
        if hasattr(self, 'axis_label_settings'):
            # Get the global enable/disable setting
            # If true, all labels are enabled unless individually disabled
            # If false, all labels are disabled unless individually enabled
            enable_all_labels = self.axis_label_settings.get('enable_all_labels', True)
            
            # Get individual axis label settings for each plot type
            axis_labels = self.axis_label_settings.get('axis_labels', {})
            y_plot_settings = axis_labels.get('y_plot', {})
            x_plot_settings = axis_labels.get('x_plot', {})
            z_plot_settings = axis_labels.get('z_plot', {})
            
            # Set y-plot top axis title if enabled
            # The label is shown if either:
            # 1. enable_all_labels is true and the individual setting is not explicitly false, or
            # 2. enable_all_labels is false but the individual setting is explicitly true
            if enable_all_labels or y_plot_settings.get('top', True):
                self.g_yplot.set_axis_title("top", fmt(p2_name))
            else:
                # Set empty title to hide the label
                self.g_yplot.set_axis_title("top", "")
            
            # Set y-plot right axis title if enabled
            if enable_all_labels or y_plot_settings.get('right', True):
                self.g_yplot.set_axis_title("right", fmt(p2_name))
            else:
                # Set empty title to hide the label
                self.g_yplot.set_axis_title("right", "")
            
            # Set x-plot top axis title if enabled
            if enable_all_labels or x_plot_settings.get('top', True):
                self.g_xplot.set_axis_title("top", fmt(p1_name))
            else:
                # Set empty title to hide the label
                self.g_xplot.set_axis_title("top", "")

            # Set z-plot bottom axis title if enabled
            if enable_all_labels or z_plot_settings.get('bottom', True):
                try:
                    self.g_zplot.set_axis_title("bottom", fmt(p3_name))
                except Exception:
                    pass
            else:
                try:
                    self.g_zplot.set_axis_title("bottom", "")
                except Exception:
                    pass

            # Set z-plot left axis title if enabled
            if enable_all_labels or z_plot_settings.get('left', True):
                try:
                    self.g_zplot.set_axis_title("left", fmt(p3_name))
                except Exception:
                    pass
            else:
                try:
                    self.g_zplot.set_axis_title("left", "")
                except Exception:
                    pass
        else:
            # No axis label settings available, use default behavior
            # All labels are shown by default
            self.g_yplot.set_axis_title("top", fmt(p2_name))
            self.g_yplot.set_axis_title("right", fmt(p2_name))
            self.g_xplot.set_axis_title("top", fmt(p1_name))
            try:
                self.g_zplot.set_axis_title("bottom", fmt(p3_name))
                self.g_zplot.set_axis_title("left", fmt(p3_name))
            except Exception:
                pass

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
        # Check if we need to recompute histograms
        # We can skip recomputation if the data, bins, and weights haven't changed
        recompute_needed = True

        # Get current parameters for comparison
        p1_idx = self.plot_control.p1[0]
        p2_idx = self.plot_control.p2[0]
        p3_idx = self.plot_control.p3[0]
        use_weights = hasattr(self, 'checkBoxWeight') and self.checkBoxWeight.isChecked()
        weight_param = self.comboBoxWeight.currentText() if use_weights and hasattr(self, 'comboBoxWeight') else ""
        z_enabled = hasattr(self, 'checkBoxEnableZ') and self.checkBoxEnableZ.isChecked()

        # Check if we have cached parameters that match current settings
        if hasattr(self, '_cached_hist_params') and self._cached_hist_params is not None:
            cached_params = self._cached_hist_params
            mask_id = getattr(self, '_cached_values_mask_id', None)

            # Compare current parameters with cached ones
            if (cached_params.get('p1_idx') == p1_idx and
                cached_params.get('p2_idx') == p2_idx and
                cached_params.get('p3_idx') == p3_idx and
                cached_params.get('use_weights') == use_weights and
                cached_params.get('weight_param') == weight_param and
                cached_params.get('z_enabled') == z_enabled and
                cached_params.get('mask_id') == mask_id and
                cached_params.get('normed_x') == self.plot_control.normed_hist_x and
                cached_params.get('normed_y') == self.plot_control.normed_hist_y and
                cached_params.get('normed_z') == self.plot_control.normed_hist_z and
                cached_params.get('x_bins_1d') == str(self.get_x_bins()[0]) and
                cached_params.get('y_bins_1d') == str(self.get_y_bins()[0]) and
                (not z_enabled or cached_params.get('z_bins_1d') == str(self.get_z_bins()[0]))):

                # All parameters match, no need to recompute
                recompute_needed = False
                logging.info( "Using cached histograms")

        if not recompute_needed and '_histogram' in self.__dict__ and self._histogram:
            # Update GUI with cached data
            self.lineEditCountCurrent.setText(str(len(self.x_values)))
            return

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

        # Check if we should weight histograms by selected parameter
        weights = None
        if use_weights:
            # Get the selected parameter from comboBoxWeight
            weight_param = self.comboBoxWeight.currentText()

            # Find the parameter index in the data source
            weight_idx = -1
            if self._data_source is not None and hasattr(self._data_source, 'parameter_names'):
                param_names = self._data_source.parameter_names
                if weight_param in param_names:
                    weight_idx = param_names.index(weight_param)

            # Get the weight values
            if weight_idx >= 0:
                # Get the values that are already filtered by value_mask
                weight_values = self.values[weight_idx].astype('float64')

                # Make sure weights have the same shape as the data arrays
                if len(weight_values) == len(d1):
                    weights = weight_values
                else:
                    logging.warning(f"Weights array shape ({len(weight_values)}) doesn't match data array shape ({len(d1)}). Disabling weights.")
            else:
                logging.warning(f"Weight parameter '{weight_param}' not found in data source. Disabling weights.")

        # X, Y, Z Histogram
        ###################
        # Use the filtered data for all histograms
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                self._histogram["x"] = np.histogram(d1, bins=x_bins_1d, weights=weights, density=self.plot_control.normed_hist_x)[::-1]
        except ValueError as e:
            logging.warning(f"Could not compute X histogram with weights: {str(e)}")
            # Fallback to histogram without weights
            with np.errstate(divide='ignore', invalid='ignore'):
                self._histogram["x"] = np.histogram(d1, bins=x_bins_1d, density=self.plot_control.normed_hist_x)[::-1]

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                self._histogram["y"] = np.histogram(d2, bins=y_bins_1d, weights=weights, density=self.plot_control.normed_hist_y)[::-1]
        except ValueError as e:
            logging.warning(f"Could not compute Y histogram with weights: {str(e)}")
            # Fallback to histogram without weights
            with np.errstate(divide='ignore', invalid='ignore'):
                self._histogram["y"] = np.histogram(d2, bins=y_bins_1d, density=self.plot_control.normed_hist_y)[::-1]

        # Only compute z histogram if z-axis is enabled
        if z_enabled:
            # Don't use weights for z histogram if the weight parameter is the same as the z parameter (would be self-weighting)
            z_weights = None
            if weights is not None and use_weights:
                # Get the selected weight parameter and z parameter
                weight_param = self.comboBoxWeight.currentText()
                z_param = self.plot_control.z_label

                # Only use weights if they're different parameters
                if weight_param != z_param:
                    z_weights = weights

            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._histogram["z"] = np.histogram(d3, bins=z_bins_1d, weights=z_weights, density=self.plot_control.normed_hist_z)[::-1]
            except ValueError as e:
                logging.warning(f"Could not compute Z histogram: {str(e)}")
                # Create a simple histogram without weights as fallback
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._histogram["z"] = np.histogram(d3, bins=z_bins_1d, density=self.plot_control.normed_hist_z)[::-1]

        # 2D Histogram
        ####################
        try:
            # Use the filtered data for the 2D histogram
            # weights should already be checked for shape compatibility above
            with np.errstate(divide='ignore', invalid='ignore'):
                H, x_edges, y_edges = np.histogram2d(x=d1, y=d2, bins=[x_bins_2d, y_bins_2d], weights=weights, density=True)
            # Sanitize H to remove NaNs/Infs resulting from empty bins or zero area
            H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
            self._histogram["2d"] = H, x_edges, y_edges
        except ValueError as e:
            logging.warning(f"Could not compute 2D histogram: {str(e)}")

        # Cache the parameters used for this computation
        self._cached_hist_params = {
            'p1_idx': p1_idx,
            'p2_idx': p2_idx,
            'p3_idx': p3_idx,
            'use_weights': use_weights,
            'weight_param': weight_param,
            'z_enabled': z_enabled,
            'mask_id': getattr(self, '_cached_values_mask_id', None),
            'normed_x': self.plot_control.normed_hist_x,
            'normed_y': self.plot_control.normed_hist_y,
            'normed_z': self.plot_control.normed_hist_z,
            'x_bins_1d': str(x_bins_1d),
            'y_bins_1d': str(y_bins_1d),
            'z_bins_1d': str(z_bins_1d)
        }

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
        logging.info( "2D histogram data copied to clipboard.")

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
        logging.info( "2D histogram data copied to clipboard as CSV (formatted with tabs).")

    def update_plots(self, skip_clustering=False):
        """
        Update all plots with the current data.

        Args:
            skip_clustering: If True, skip the clustering step even if clustering is enabled.
                            This is useful when update_plots is called after clustering is done
                            or when loading data.
        """
        # Invalidate the values cache to ensure we're using the latest data
        # This is especially important when selections have changed
        self.invalidate_values_cache()

        # If there's no data (or fewer than 3 columns), display an empty plot
        if self._data_source.empty or self._data_source.values.shape[0] == 0:
            # Set valid initial data for the curve items (single point at 0,0)
            # This prevents errors when autoscaling with empty data
            self.g_xhist_m.set_data([0, 1], [0, 0])
            self.g_yhist_m.set_data([0, 0], [0, 1])
            self.g_zhist_m.set_data([0, 1], [0, 0])

            # Create an empty array for the 2D plot
            empty_img = np.zeros((1, 1))

            # Set the empty image in the 2D histogram axis
            self.cax.set_data(empty_img)
            
            # Show the background image when there's no data
            if hasattr(self, 'bg_image_item') and self.bg_image_item is not None:
                self.bg_image_item.setVisible(True)

            # Set default axis scales for empty data
            self.g_2dplot.setAxisScale(QwtPlot.xBottom, 0, 1)
            self.g_2dplot.setAxisScale(QwtPlot.yLeft, 0, 1)

            # Set default axis scales for x and y histograms
            self.g_xplot.setAxisScale(QwtPlot.xBottom, 0, 1)
            self.g_xplot.setAxisScale(QwtPlot.xTop, 0, 1)  # Link top axis to bottom axis
            self.g_xplot.setAxisScale(QwtPlot.yLeft, 0, 1)

            self.g_yplot.setAxisScale(QwtPlot.yLeft, 0, 1)
            self.g_yplot.setAxisScale(QwtPlot.yRight, 0, 1)  # Link right axis to left axis
            self.g_yplot.setAxisScale(QwtPlot.xBottom, 0, 1)

            # Set default axis scales for z histogram
            self.g_zplot.setAxisScale(QwtPlot.xBottom, 0, 1)
            self.g_zplot.setAxisScale(QwtPlot.yLeft, 0, 1)

            # Replot all plots with the valid initial data
            try:
                self.g_xplot.replot()
            except ValueError as e:
                logging.warning(f"Error reploting x-plot: {e}")

            try:
                self.g_yplot.replot()
            except ValueError as e:
                logging.warning(f"Error reploting y-plot: {e}")

            try:
                self.g_zplot.replot()
            except ValueError as e:
                logging.warning(f"Error reploting z-plot: {e}")

            self.g_2dplot.replot()
            return

        # Update parameter names, colormap, and recalc histograms
        self.update_parameter_names()
        self.update_cmap()

        # Apply HDBSCAN clustering if enabled and not skipped
        # Skip clustering when loading data (skip_clustering=True)
        if self._use_clustering and self._cluster_labels is None and not skip_clustering:
            # Lazy import of hdbscan
            global hdbscan
            if hdbscan is None:
                try:
                    import hdbscan
                    logging.debug("Imported hdbscan library")
                except ImportError:
                    hdbscan = None

            if hdbscan:
                # Start the clustering in a separate thread
                self.on_apply_clustering()
                # Return early to avoid updating the plots until clustering is done
                return

        # Update histograms
        self.update_histograms()
        
        # Hide the background image when data is loaded
        if hasattr(self, 'bg_image_item') and self.bg_image_item is not None:
            self.bg_image_item.setVisible(False)

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

        # 3. Z histogram - only if enabled
        if hasattr(self, 'checkBoxEnableZ') and self.checkBoxEnableZ.isChecked() and "z" in self._histogram:
            # _histogram["z"] = (bin_edges, counts)
            z_bin_edges = self._histogram["z"][0]
            z_counts = self._histogram["z"][1]
            # Plot with X=bin_edges[1:], Y=counts
            self.g_zhist_m.set_data(z_bin_edges[1:], z_counts)

            # Z histogram => x-axis: bin edges, y-axis: counts
            y_max_z = np.max(z_counts) if len(z_counts) else 1
            self.g_zplot.setAxisScale(QwtPlot.yLeft, 0, y_max_z * 1.05)
            self.g_zplot.setAxisScale(QwtPlot.xBottom, z_bin_edges[0], z_bin_edges[-1])
        else:
            # Set valid initial data for the z histogram when disabled
            # This prevents errors when autoscaling with empty data
            self.g_zhist_m.set_data([0, 1], [0, 0])

            # Set default axis scales for z histogram
            self.g_zplot.setAxisScale(QwtPlot.xBottom, 0, 1)
            self.g_zplot.setAxisScale(QwtPlot.yLeft, 0, 1)

        # ----------------------------------------------------
        # Manually set axis scales to start at 0 for the count axis

        # X histogram => x-axis: bin edges, y-axis: counts
        y_max_x = np.max(x_counts) if len(x_counts) else 1
        self.g_xplot.setAxisScale(QwtPlot.yLeft, 0, y_max_x * 1.05)
        self.g_xplot.setAxisScale(QwtPlot.xBottom, x_bin_edges[0], x_bin_edges[-1])
        # Link top axis to bottom axis
        self.g_xplot.setAxisScale(QwtPlot.xTop, x_bin_edges[0], x_bin_edges[-1])

        # Y histogram => x-axis: counts, y-axis: bin edges
        x_max_y = np.max(y_counts) if len(y_counts) else 1
        self.g_yplot.setAxisScale(QwtPlot.xBottom, 0, x_max_y * 1.05)
        self.g_yplot.setAxisScale(QwtPlot.yLeft, y_bin_edges[0], y_bin_edges[-1])
        # Link right axis to left axis
        self.g_yplot.setAxisScale(QwtPlot.yRight, y_bin_edges[0], y_bin_edges[-1])

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

        # Only replot x-plot if it has valid data
        if "x" in self._histogram:
            x_counts = self._histogram["x"][1]
            if len(x_counts) > 0 and not np.all(np.isnan(x_counts)):
                try:
                    self.g_xplot.replot()
                except ValueError as e:
                    logging.warning(f"Error reploting x-plot: {e}")

        # Only replot y-plot if it has valid data
        if "y" in self._histogram:
            y_counts = self._histogram["y"][1]
            if len(y_counts) > 0 and not np.all(np.isnan(y_counts)):
                try:
                    self.g_yplot.replot()
                except ValueError as e:
                    logging.warning(f"Error reploting y-plot: {e}")

        # Only replot z-plot if it has valid data
        if hasattr(self, 'checkBoxEnableZ') and self.checkBoxEnableZ.isChecked() and "z" in self._histogram:
            z_counts = self._histogram["z"][1]
            if len(z_counts) > 0 and not np.all(np.isnan(z_counts)):
                try:
                    self.g_zplot.replot()
                except ValueError as e:
                    logging.warning(f"Error reploting z-plot: {e}")

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



    # Clustering functionality has been moved to clustering.py

    def on_apply_clustering(self):
        """
        Apply clustering with current parameters and update plots.
        """
        logging.info("Applying clustering with current parameters")

        # Set the _use_clustering flag to True
        self._use_clustering = True
        logging.debug("Set _use_clustering flag to True")

        # Get parameters from the clustering dialog
        cluster_method = self.clustering_dialog._cluster_method
        cluster_columns = self.clustering_dialog._cluster_columns

        # Check if the required library is available
        if cluster_method == "hdbscan":
            # Lazy import of hdbscan
            global hdbscan
            if hdbscan is None:
                try:
                    import hdbscan
                    logging.debug("Imported hdbscan library")
                except ImportError:
                    hdbscan = None

            if not hdbscan:
                QtWidgets.QMessageBox.warning(
                    self,
                    "HDBSCAN Not Available",
                    "HDBSCAN is not installed. Please install it using pip or conda."
                )
                if self.clustering_dialog is not None:
                    self.clustering_dialog.checkBoxClustering.setChecked(False)
                return
        elif cluster_method == "kmeans":
            # Lazy import of KMeans
            global KMeans
            if KMeans is None:
                try:
                    from sklearn.cluster import KMeans
                    logging.debug("Imported KMeans library")
                except ImportError:
                    KMeans = None

            if not KMeans:
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
            logging.debug("No columns selected for clustering. Using x, y, z values.")

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

        # Create a new worker thread using the imported ClusteringWorker class
        self.clustering_worker = ClusteringWorker(
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
        logging.info("Cancelling clustering operation")

        if hasattr(self, 'clustering_worker') and self.clustering_worker is not None and self.clustering_worker.isRunning():
            # Request the worker to stop
            self.clustering_worker.stop()
            logging.debug("Requested clustering worker to stop")

            # Update dialog UI if it exists
            if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
                self.clustering_dialog.pushButtonCancelClustering.setText("Cancelling...")
                self.clustering_dialog.pushButtonCancelClustering.setEnabled(False)
                logging.debug("Updated clustering dialog UI for cancellation")

            # The worker will emit clustering_done with None values when it's done
            logging.debug("Waiting for worker to complete cancellation")

    def on_clustering_progress(self, progress):
        """
        Update the progress bar with the current clustering progress.

        Args:
            progress: Integer value between 0 and 100 representing the progress percentage
        """
        logging.debug(f"Clustering progress: {progress}%")

        # Update progress in dialog if it exists
        if self.clustering_dialog is not None and self.clustering_dialog.isVisible():
            self.clustering_dialog.update_progress(progress)
            logging.debug(f"Updated clustering dialog progress bar to {progress}%")

    def on_clustering_error(self, error_message):
        """
        Handle errors that occur during clustering.

        Args:
            error_message: String containing the error message
        """
        logging.warning(f"Clustering error: {error_message}")

        # Clear any partial clustering results
        self._cluster_labels = None
        self._cluster_probabilities = None

        # Set the _use_clustering flag to False
        self._use_clustering = False
        logging.debug("Set _use_clustering flag to False due to error")

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
        logging.info("Clustering completed")

        # Update the cluster labels and probabilities
        self._cluster_labels, self._cluster_probabilities = result

        # Initialize the cluster data shape if it doesn't exist
        if not hasattr(self, '_cluster_data_shape'):
            self._cluster_data_shape = len(self._cluster_labels) if self._cluster_labels is not None else 0
            logging.debug(f"Initialized cluster data shape: {self._cluster_data_shape}")

        # If result is None, it means clustering was cancelled or failed
        if result[0] is None:
            logging.info("Clustering was cancelled or failed")

            # Set the _use_clustering flag to False
            self._use_clustering = False
            logging.debug("Set _use_clustering flag to False due to cancellation or failure")

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
            logging.debug(f"Adjusted spinBoxCluster range to (-1, {n_clusters - 1})")

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
        This is now a wrapper around the ClusteringManager's perform_clustering method.

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
        if not hasattr(self, '_clustering_manager'):
            self._clustering_manager = ClusteringManager(self._data_source)
        else:
            # Update data source in case it has changed
            self._clustering_manager._data_source = self._data_source

        if method is None:
            method = self.clustering_dialog._cluster_method

        # Extract parameters for the selected method
        if method == "hdbscan":
            min_samples = kwargs.get("min_samples", self.clustering_dialog._cluster_min_samples)
            min_cluster_size = kwargs.get("min_cluster_size", self.clustering_dialog._cluster_min_cluster_size)
            kwargs["min_samples"] = min_samples
            kwargs["min_cluster_size"] = min_cluster_size
        elif method == "kmeans":
            n_clusters = kwargs.get("n_clusters", self.clustering_dialog._cluster_n_clusters)
            kwargs["n_clusters"] = n_clusters

        # Get the data for clustering based on selected columns
        if self.clustering_dialog._cluster_columns:
            kwargs["columns"] = self.clustering_dialog._cluster_columns
        else:
            # If no columns are selected, use x, y, z values
            kwargs["x_values"] = self.x_values
            kwargs["y_values"] = self.y_values
            kwargs["z_values"] = self.z_values

        # Perform clustering using the ClusteringManager
        result = self._clustering_manager.perform_clustering(method=method, worker=worker, **kwargs)

        # If clustering was successful, update the data frame
        if result[0] is not None and result[1] is not None:
            full_labels, full_probabilities = result

            # Add cluster labels and probabilities to the data frame
            df = self._data_source.data
            df['Cluster Label'] = full_labels
            df['Cluster Probability'] = full_probabilities
            self._data_source.data = df  # Update the data frame to trigger parameter_names update

            # Store the data shape used for clustering
            self._cluster_data_shape = self._clustering_manager._cluster_data_shape
            logging.info( f"Stored cluster data shape: {self._cluster_data_shape}")

        return result

    def keyPressEvent(self, event):
        """
        Handle key press events.

        Args:
            event: The key event
        """
        # Check if Ctrl+L was pressed to toggle clustering dialog
        if (event.modifiers() & QtCore.Qt.ControlModifier) and event.key() == QtCore.Qt.Key_L:
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
            logging.info( f"Z selection range changed from {self._last_z_range} to {current_range}")
            self._last_z_range = current_range
            # Update histograms and plots
            self.update_histograms()
            self.update_plots(skip_clustering=True)

    def on_enable_z_changed(self, state):
        """
        Handle changes to the enable Z checkbox.

        Args:
            state: The new state of the checkbox (Qt.Checked or Qt.Unchecked)
        """
        # Show or hide the z-axis plot based on the checkbox state
        self.g_zplot.setVisible(bool(state))

        # Update histograms and plots to reflect the new state
        # This will skip z-axis histogram computation if disabled
        self.update_histograms()
        self.update_plots(skip_clustering=True)

    def on_weight_param_changed(self, index):
        """
        Handle changes to the selected weight parameter.

        Args:
            index: The index of the newly selected item in the combobox
        """
        # Only update if weight is enabled
        if self.checkBoxWeight.isChecked():
            # Update histograms and plots to reflect the new weight parameter
            self.update_histograms()
            self.update_plots(skip_clustering=True)

    def on_weight_changed(self, state):
        """
        Handle changes to the weight checkbox.

        Args:
            state: The new state of the checkbox (Qt.Checked or Qt.Unchecked)
        """
        # Enable/disable comboBoxWeight based on checkbox state
        is_checked = bool(state)
        self.comboBoxWeight.setEnabled(is_checked)

        # If weight is checked, populate the comboBoxWeight with available parameters
        if is_checked:
            # Save current selection if any
            current_text = self.comboBoxWeight.currentText()

            # Clear and populate the combobox
            self.comboBoxWeight.clear()

            # Get parameter names from data source
            if self._data_source is not None and hasattr(self._data_source, 'parameter_names'):
                param_names = self._data_source.parameter_names
                for name in param_names:
                    self.comboBoxWeight.addItem(name)

                # Restore previous selection if it exists in the new list
                if current_text and current_text in param_names:
                    index = self.comboBoxWeight.findText(current_text)
                    if index >= 0:
                        self.comboBoxWeight.setCurrentIndex(index)
                # Otherwise, default to z-axis parameter for backward compatibility
                elif self.plot_control.z_label in param_names:
                    index = self.comboBoxWeight.findText(self.plot_control.z_label)
                    if index >= 0:
                        self.comboBoxWeight.setCurrentIndex(index)

        # Update histograms and plots to reflect the new state
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
        # Get cluster labels if available
        cluster_labels = None
        if hasattr(self, '_cluster_labels') and self._cluster_labels is not None:
            cluster_labels = self._cluster_labels

        # Call the UMAP plot function from the plot_umap module
        plot_umap.create_umap_plot(
            parent=self,
            columns=columns,
            params=params,
            data_source=self._data_source,
            x_values=self.x_values,
            y_values=self.y_values,
            z_values=self.z_values,
            cluster_labels=cluster_labels
        )

    def update_2d_plot(self):
        try:
            new_data, x_edges, y_edges = self._histogram["2d"]
        except (ValueError, KeyError):
            return None
            
        # Check if the data is empty or has zero size
        if new_data is None or new_data.size == 0 or np.all(np.isnan(new_data)):
            # Set a small valid array instead of empty data
            new_data = np.zeros((1, 1))
            logging.info( "Empty or invalid 2D histogram data detected, using placeholder")

        log_counts = self.checkBoxLogCounts.isChecked()
        if log_counts and new_data.size > 1:  # Only apply log transform if we have real data
            # Handle zeros and negative values before taking log10
            # Add a small positive value to avoid log(0) which would give -inf
            # This ensures we can see more details in the 2D histogram
            min_positive = np.min(new_data[new_data > 0]) if np.any(new_data > 0) else 1e-10
            new_data = np.maximum(new_data, min_positive / 10)  # Replace zeros/negatives with a small value
            new_data = np.log10(new_data)
            new_data = np.nan_to_num(new_data)

        try:
            # Update the data of the displayed image
            # numpy.histogram2d outputs shape (x_bins, y_bins);
            # ImageItem expects (rows=y, cols=x), so use transpose only.
            self.cax.set_data(new_data.T)
        except ValueError as e:
            logging.warning(f"Error setting 2D plot data: {str(e)}")
            # If setting data fails, try with a simple valid array
            self.cax.set_data(np.zeros((1, 1)))

        # Set the intensity range using the vmin and vmax properties from the UI
        self.cax.set_lut_range([self.vmin, self.vmax])

        # Redraw the main plot to update the display
        self.g_2dplot.replot()

        # Update curve overlays
        self.update_curve_overlays()

    def bin_to_value(self, bin_idx, edges):
        """Convert a bin index to a value (center of the bin).

        Args:
            bin_idx: The bin index
            edges: The bin edges array

        Returns:
            The center value of the bin, or None if the bin index is invalid
        """
        if bin_idx < 0 or bin_idx >= len(edges) - 1:
            return None
        return (edges[bin_idx] + edges[bin_idx + 1]) / 2

    def value_to_bin(self, value, edges):
        """Convert a value to a bin index with linear interpolation.

        Args:
            value: The value to convert
            edges: The bin edges array

        Returns:
            The bin index (as a float for interpolation), or None if the value is outside the range
        """
        for i in range(len(edges) - 1):
            if edges[i] <= value <= edges[i + 1]:
                # Calculate the relative position within the bin (0.0 to 1.0)
                bin_width = edges[i + 1] - edges[i]
                if bin_width == 0:  # Avoid division by zero
                    return float(i)
                relative_pos = (value - edges[i]) / bin_width
                # Return the bin index plus the relative position
                return float(i) + relative_pos
        return None

    def bin_to_x_value(self, bin_idx, x_edges):
        """Convert a bin index to an x value (center of the bin)."""
        return self.bin_to_value(bin_idx, x_edges)

    def bin_to_y_value(self, bin_idx, y_edges):
        """Convert a bin index to a y value (center of the bin)."""
        return self.bin_to_value(bin_idx, y_edges)

    def x_value_to_bin(self, x_value, x_edges):
        """Convert an x value to a bin index with linear interpolation."""
        return self.value_to_bin(x_value, x_edges)

    def y_value_to_bin(self, y_value, y_edges):
        """Convert a y value to a bin index with linear interpolation."""
        return self.value_to_bin(y_value, y_edges)

    def on_auto_contrast(self):
        """
        Automatically adjust vmin and vmax for the 2D histogram based on the data.
        This function calculates appropriate min and max values for better visualization.
        """
        logging.debug("Auto contrast triggered")
        try:
            # Get the 2D histogram data
            if "2d" not in self._histogram:
                logging.debug("No 2D histogram data available")
                return
                
            H, _, _ = self._histogram["2d"]

            # Skip if histogram is empty, contains only zeros, or is all NaN
            if H is None or H.size == 0 or np.all(H == 0) or np.all(np.isnan(H)):
                logging.debug("Histogram is empty, contains only zeros, or all NaN values")
                # Set default contrast values
                self.vmin = 0
                self.vmax = 1
                self.on_vmin_vmax_changed()
                return

            # Apply log transform if log counts is checked
            if self.checkBoxLogCounts.isChecked():
                # Handle zeros and negative values before taking log10
                min_positive = np.min(H[H > 0]) if np.any(H > 0) else 1e-10
                H_processed = np.maximum(H, min_positive / 10)  # Replace zeros/negatives with a small value
                H_processed = np.log10(H_processed)
                H_processed = np.nan_to_num(H_processed)
            else:
                H_processed = H

            # Calculate percentiles for robust min/max values
            # Ignore zeros which might be a large part of the histogram
            non_zero_values = H_processed[H_processed > 0]
            if non_zero_values.size > 0:
                vmin = np.percentile(non_zero_values, 1)  # 1st percentile for minimum
                vmax = np.percentile(non_zero_values, 99)  # 99th percentile for maximum

                # Ensure vmin and vmax are different to avoid display issues
                if vmin == vmax:
                    vmin = 0.9 * vmin if vmin != 0 else 0
                    vmax = 1.1 * vmax if vmax != 0 else 1

                logging.debug(f"Setting auto contrast: vmin={vmin}, vmax={vmax}")

                # Update the UI controls
                self.vmin = vmin
                self.vmax = vmax

                # Update the plot
                self.on_vmin_vmax_changed()
            else:
                logging.debug("No non-zero values in histogram")
                # Set default contrast values
                self.vmin = 0
                self.vmax = 1
                self.on_vmin_vmax_changed()
        except (ValueError, KeyError, TypeError, IndexError) as e:
            logging.warning(f"Error in auto contrast: {str(e)}")
            # Set default contrast values on error
            self.vmin = 0
            self.vmax = 1
            self.on_vmin_vmax_changed()

    def update_curve_overlays(self):
        """Update the curve overlays on the 2D histogram."""
        try:
            # Get the 2D histogram data and edges
            histogram_data = self._histogram["2d"]
            
            # Check if histogram data is valid
            if histogram_data is None or len(histogram_data) < 3 or histogram_data[0].size == 0:
                logging.debug("Empty histogram data, skipping curve overlay update")
                return
                
            # Call the update_curve_overlays method in the CurveOverlayWidget class
            self.curve_overlay_widget.update_curve_overlays(
                overlay_plot=self.overlay_plot,
                histogram_data=histogram_data,
                plot_control=self.plot_control,
                curve_evaluator=self.curve_evaluator,
                value_to_bin_func=self.value_to_bin
            )

            # Update the curve_items reference to maintain backward compatibility
            self.curve_items = self.curve_overlay_widget.curve_items
            
        except (ValueError, KeyError, IndexError, AttributeError) as e:
            logging.warning(f"Error updating curve overlays: {str(e)}")
            return

    # ========================= GAUSSIAN FITTING ==============================
    def on_fit_2d_gaussian(self):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_fit_2d_gaussian()

    def on_select_point_toggled(self, checked: bool):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_select_point_toggled(checked)

    def _on_point_selected(self, pos):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_point_selected(pos)

    def on_clear_gaussians(self):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_clear_gaussians()

    def _compute_moments(self, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._compute_moments(H, x_edges, y_edges)
        return None, None

    def _compute_local_moments(self, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, ix: int, iy: int, window: int = 5):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._compute_local_moments(H, x_edges, y_edges, ix, iy, window)
        return None, None

    def _add_gaussian_overlay(self, mu: Tuple[float, float], cov: np.ndarray, label: str = "", color: Optional[str] = None):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._add_gaussian_overlay(mu, cov, label, color)
        return None

    def _append_gaussian_row(self, mu: Tuple[float, float], cov: np.ndarray, w: float = 1.0):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._append_gaussian_row(mu, cov, w)
        return -1

    def _update_gaussian_row(self, row: int, mu: np.ndarray, cov: np.ndarray, w: float = None):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._update_gaussian_row(row, mu, cov, w)
        return None

    def _read_gaussian_table(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._read_gaussian_table()
        return []

    def _redraw_gaussian_overlays_from_table(self):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._redraw_gaussian_overlays_from_table()

    def _clear_gaussian_marginal_items(self):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._clear_gaussian_marginal_items()

    def _draw_gaussian_marginals_from_table(self, rows, colors=None):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._draw_gaussian_marginals_from_table(rows, colors)

    def on_toggle_gaussian_marginals(self, checked: bool):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_toggle_gaussian_marginals(checked)

    # ======================= END GAUSSIAN FITTING ===========================

    def check_and_set_image_axes(self):
        """
        Check if the loaded data contains image information (X pixel and Y pixel columns)
        and set the appropriate axes and weighting.
        
        Returns:
            bool: True if image axes were detected and applied; False otherwise.
        """
        logging.debug("Checking image axes")
        if self._data_source is None or self._data_source.empty:
            logging.debug("No data loaded, skipping image axes check")
            return False
            
        # Get parameter names from data source
        param_names = self._data_source.parameter_names
        logging.debug(f"Parameter names: {param_names}")
        
        # Check if both 'X pixel' and 'Y pixel' exist in the data (case-insensitive)
        has_x_pixel = any('x pixel' in name.lower() for name in param_names)
        has_y_pixel = any('y pixel' in name.lower() for name in param_names)
        
        if has_x_pixel and has_y_pixel:
            logging.info("Image data detected (X pixel and Y pixel columns found)")
            
            # Find the actual parameter names with correct case
            x_pixel_param = next((name for name in param_names if 'x pixel' in name.lower()), None)
            x_success = self.plot_control.set_axis_by_name('x', x_pixel_param, block_signals=True)
            if x_success:
                logging.debug(f"Set X axis to {x_pixel_param}")
            else:
                logging.warning(f"Failed to set X axis to {x_pixel_param}")

            y_pixel_param = next((name for name in param_names if 'y pixel' in name.lower()), None)
            y_success = self.plot_control.set_axis_by_name('y', y_pixel_param, block_signals=True)
            if y_success:
                logging.debug(f"Set Y axis to {y_pixel_param}")
            else:
                logging.warning(f"Failed to set Y axis to {y_pixel_param}")
            
            # Find and set weight parameter to "Number of Photons" (case-insensitive)
            self.weight_enabled = True
            photon_param = next((name for name in param_names if 'number of photons' in name.lower()), None)
            logging.debug(f"Weight parameter: {photon_param}")
            weight_success = self.plot_control.set_axis_by_name('weight', photon_param, match_contains=True, block_signals=True)
            if weight_success:
                logging.debug(f"Set weighting to {photon_param}")
            else:
                logging.debug("No matching weight parameter found, using default")

            # Get the number of pixels in X and Y dimensions
            x_values = self._data_source.values[param_names.index(x_pixel_param), :]
            y_values = self._data_source.values[param_names.index(y_pixel_param), :]
            x_pixels = len(set(x_values))
            y_pixels = len(set(y_values))
            logging.debug(f"x_pixel_param: {x_pixel_param}, x_values: {x_values}")
            logging.debug(f"y_pixel_param: {y_pixel_param}, y_values: {y_values}")
            logging.debug(f"x_pixels: {x_pixels}, y_pixels: {y_pixels}")

            x_pixels = int(np.max(x_values)) + 1  # +1 because pixels are 0-indexed
            y_pixels = int(np.max(y_values)) + 1

            logging.info(f"Image dimensions: {x_pixels}x{y_pixels} pixels")

            # Set binning to match pixel count
            self.plot_control.n_xhist_2d = x_pixels
            self.plot_control.n_yhist_2d = y_pixels

            # Set X range from 0 to max pixel
            self.plot_control.xmin = 0
            self.plot_control.xmax = x_pixels - 1

            # Set Y range from 0 to max pixel
            self.plot_control.ymin = 0
            self.plot_control.ymax = y_pixels - 1

            logging.debug(f"Set binning and ranges to match pixel dimensions")
            
            # Apply auto contrast as final action
            logging.debug("Applying auto contrast to image")
            self.on_auto_contrast()
            # Ensure plots reflect new image axes
            try:
                self.update_plots()
            except Exception:
                pass
            return True
        return False

    def apply_default_axes_from_settings(self):
        """
        Apply default axis selections from settings (if provided).
        This is used after a dataset is loaded to preselect X/Y/Z/weight axes.
        It will not override image axes (the caller should check first).
        """
        try:
            defaults = self.settings.get("default_axes", {}) if hasattr(self, 'settings') else {}
        except Exception:
            defaults = {}
        if not isinstance(defaults, dict) or not defaults:
            logging.debug("No default_axes configured in settings; skipping.")
            return False

        # Parameter names available
        try:
            param_names = list(self.data_source.parameter_names)
        except Exception:
            param_names = []

        changed = False
        # Helper to try set axis by exact match or contains
        def _set_axis(ax_key, axis_name):
            if not axis_name or not isinstance(axis_name, str):
                return False
            # Try exact match first
            if axis_name in param_names:
                ok = self.plot_control.set_axis_by_name(ax_key, axis_name, match_contains=False, block_signals=True)
                return bool(ok)
            # Fallback to contains
            ok = self.plot_control.set_axis_by_name(ax_key, axis_name, match_contains=True, block_signals=True)
            return bool(ok)

        # X, Y, Z axes
        for ax_key in ('x', 'y', 'z'):
            name = defaults.get(ax_key)
            if _set_axis(ax_key, name):
                changed = True

        # Weight parameter (optional)
        wname = defaults.get('weight')
        if wname:
            if _set_axis('weight', wname):
                try:
                    self.weight_enabled = True
                except Exception:
                    pass
                changed = True

        # Trigger axis changed handlers to apply ranges/bins/scales
        if changed:
            try:
                self.plot_control.on_x_axis_changed()
            except Exception:
                pass
            try:
                self.plot_control.on_y_axis_changed()
            except Exception:
                pass
            try:
                self.plot_control.on_z_axis_changed()
            except Exception:
                pass
            logging.info("Applied default axes from settings.")
            return True
        logging.debug("No default axes were applied (names may not match current dataset).")
        return False

    def on_gaussian_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_gaussian_table_item_changed(item)

    def eventFilter(self, obj, event):
        """
        Handle canvas resize events to refresh the 2D plot orientation and
        delegate to GaussianFit for other table-related events.
        """
        handled_by_gaussian = False
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            try:
                handled_by_gaussian = bool(self.gaussian_fit.eventFilter(obj, event))
            except Exception:
                handled_by_gaussian = False

        # If the 2D canvas is resized, schedule an update of the 2D plot
        try:
            if event.type() == QtCore.QEvent.Resize and hasattr(self, 'g_2dplot') and obj is self.g_2dplot.canvas():
                if not getattr(self, '_resize_update_pending', False):
                    self._resize_update_pending = True
                    def _do_update():
                        try:
                            self.update_2d_plot()
                        finally:
                            self._resize_update_pending = False
                    QtCore.QTimer.singleShot(0, _do_update)
        except Exception:
            pass

        if handled_by_gaussian:
            return True
        return super(NDXplorer, self).eventFilter(obj, event)

    def _delete_selected_gaussian_rows(self, rows: List[int]):
        """Delegate to GaussianFit."""
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._delete_selected_gaussian_rows(rows)

    def _on_fit_dock_visibility_changed(self, visible: bool):
        """Delegate to GaussianFit."""
        return self.gaussian_fit.on_fit_dock_visibility_changed(visible)

    def resizeEvent(self, event):
        """
        Trigger a 2D plot update on window resize to prevent orientation issues.
        Use a zero-timeout singleShot to run after layout has applied new sizes.
        Debounce scheduling to avoid flooding during continuous resizing.
        """
        # First perform the default resize handling
        super(NDXplorer, self).resizeEvent(event)
        # Then schedule an update of the 2D plot
        def _do_update():
            self.update_plots()
        QtCore.QTimer.singleShot(1, _do_update)
