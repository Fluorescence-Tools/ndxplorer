from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..logging_config import logging

import os
import json
import yaml
import typing
import pathlib
import importlib.util

import numpy as np
import pandas as pd

# Delay imports of heavy libraries
hdbscan = None  # For clustering
KMeans = None   # For clustering
GaussianMixture = None  # For Gaussian Mixture Modeling
umap = None     # For dimensionality reduction
napari = None   # For image visualization in external viewer

from ..plotting.plot_control import SurfacePlotWidget
from ..ui.parameter_editor import ParameterEditor
from ..plotting.curve_overlay import CurveOverlayWidget, CurveEvaluator
from ..io import reader
from ..io import writer
from ..utils import axis_helpers
from ..utils import colormap_helpers
from ..utils import napari_helpers
from ..utils import screenshot_helpers
from ..utils import ui_helpers
from ..utils import working_path_helpers
from ..io import file_operations
from ..utils import histogram_helpers
from ..analysis import clustering_helpers
from ..analysis import umap_helpers
from ..utils import settings_helpers
from ..settings import get_settings_path
from ..ui.clustering_dialog import ClusteringDialog
from ..ui.column_selection_dialog import ColumnSelectionDialog
from ..plotting import plot_umap
from ..analysis.clustering import ClusteringManager, ClusteringWorker
from ..widgets import ScientificSpinBox
from ..utils.mouse_event_filter import MouseEventFilter
from ..ui.axis_control_dialog import AxisControlDialog

# Import new modular plotting components
from ..plotting import api as plotting_api
from ..plotting import histograms as plot_histograms
from ..plotting import scatter as plot_scatter
from ..plotting import colormaps as plot_colormaps
from ..analysis.umap_progress import UMAPProgressDialog

# Defer guiqwt imports to reduce startup time
_guiqwt_signals = None
_guiqwt_plot = None
_guiqwt_image = None
_guiqwt_curve = None
_guiqwt_styles = None
_guiqwt_colormap_list = None
_CurveDialog = None
_make = None
_DataFrameEditor = None
_QwtPlot = None
_QwtPlotCanvas = None

def _ensure_guiqwt():
    """Lazy-load guiqwt modules on first use."""
    global _guiqwt_signals, _guiqwt_plot, _guiqwt_image, _guiqwt_curve, _guiqwt_styles
    global _guiqwt_colormap_list, _CurveDialog, _make, _DataFrameEditor, _QwtPlot, _QwtPlotCanvas
    if _guiqwt_signals is None:
        import guiqwt.signals
        import guiqwt.plot
        import guiqwt.image
        import guiqwt.curve
        import guiqwt.styles
        from guiqwt.colormap import get_colormap_list
        from guiqwt.plot import CurveDialog
        from guiqwt.builder import make
        from guidata.widgets.dataframeeditor import DataFrameEditor
        from qwt.plot import QwtPlot
        from qwt.plot_canvas import QwtPlotCanvas
        _guiqwt_signals = guiqwt.signals
        _guiqwt_plot = guiqwt.plot
        _guiqwt_image = guiqwt.image
        _guiqwt_curve = guiqwt.curve
        _guiqwt_styles = guiqwt.styles
        _guiqwt_colormap_list = get_colormap_list
        _CurveDialog = CurveDialog
        _make = make
        _DataFrameEditor = DataFrameEditor
        _QwtPlot = QwtPlot
        _QwtPlotCanvas = QwtPlotCanvas

try:
    from chisurf.plugins.misc.code_editor import CodeEditor
except ImportError:
    from ndxplorer.widgets.code_editor import CodeEditor

from .data_source import DataSource, RectangularDataSelection, MaskDataSelection
from .data import DataManager

try:
    from chisurf.gui import QtGui, QtCore, uic, QtWidgets
    from chisurf.gui.QtGui import QFont, QImage
except ImportError:
    from qtpy import QtCore, uic
    from qtpy import QtGui, QtWidgets
    from qtpy.QtGui import QFont, QImage

from ..plotting.image_items import FixedImageItem
from ..plotting.plot_helpers import (
    configure_dynamic_selection_controls,
    setup_histogram_plots,
    setup_histogram_spinboxes,
    setup_overlay_plot,
    setup_2d_histogram_plot,
    setup_plot_placeholders,
)
from ..utils.value_cache import get_value_mask, get_filtered_values
from ..utils.histogram_export import (
    copy_1d_histograms,
    copy_2d_hist_csv,
    copy_2d_hist_json,
)
from ..io.export_helpers import save_burst_ids, save_clustering_data
from ..io.file_open_helpers import (
    open_chisurf_sampling,
    open_csv,
    open_files,
    open_mfd_hdf5,
    open_smfret,
    show_merge_dialog,
)
from ..plotting import plot_update_helpers
from ..utils.mask_drawing_integration import MaskDrawingIntegration


class NDXplorer(QtWidgets.QMainWindow):

    def invalidate_values_cache(self) -> None:
        """
        Manually clear the cached 'values'. Call this whenever something
        changes that would invalidate the mask or the data.
        """
        logging.debug("Invalidating values cache")
        # New architecture: use data_manager
        if hasattr(self, 'data_manager'):
            self.data_manager.cache.invalidate_all()
        
        # Backward compatibility: keep old cache variables
        self._cached_values = None
        self._cached_values_selections = None
        self._cached_values_p13 = None
        self._cached_values_mask_inf = None
        self._cached_values_mask_nan = None
        self._cached_filtered_values = None
        self._cached_values_mask_id = None
        self._cached_x_values = None
        self._cached_x_param_idx = None
        self._cached_y_values = None
        self._cached_y_param_idx = None
        self._cached_z_values = None
        self._cached_z_param_idx = None
        self._cached_hist_params = None

    @property
    def data_source(self) -> DataSource:
        logging.debug("Getting data_source")
        # New architecture: use data_manager
        if hasattr(self, 'data_manager'):
            return self.data_manager.data_source
        # Fallback for initialization
        if self._data_source.empty and self._default_data_source is not None:
            return self._default_data_source
        return self._data_source

    @data_source.setter
    def data_source(self, v: DataSource) -> None:
        logging.info(f"Setting data_source with {v.values.shape[1] if not v.empty else 0} data points")
        
        # New architecture: use data_manager
        if hasattr(self, 'data_manager'):
            self.data_manager.constants = self.constants
            self.data_manager.equations = self.equations
            self.data_manager.data_source = v
        else:
            # Fallback during initialization
            self._data_source = v
            self.invalidate_values_cache()
            self._data_source.compute_columns(
                constants=self.constants,
                equations=self.equations
            )
        
        self._set_data_loaded(not v.empty)

    def _set_data_loaded(self, has_data: bool) -> None:
        has_data = bool(has_data)
        self._has_real_data = has_data
        stack = getattr(self, "_plot_stack_widget", None)
        bg_label = getattr(self, "_background_label", None)
        plot_container = getattr(self, "_plot_container", None)
        if stack is not None and bg_label is not None and plot_container is not None:
            target = plot_container if has_data else bg_label
            other = bg_label if has_data else plot_container
            try:
                stack.setCurrentWidget(target)
                target.setVisible(True)
                other.setVisible(False)
            except Exception:
                pass
        try:
            if has_data:
                plot_update_helpers._hide_background(self)
            else:
                plot_update_helpers._show_background(self)
        except Exception:
            pass

    @property
    def x_values(self) -> np.ndarray:
        """Get x-axis values, using cache when valid."""
        p1_idx = self.plot_control.p1[0]
        
        # New architecture: use data_manager
        if hasattr(self, 'data_manager'):
            return self.data_manager.get_axis_values('x', p1_idx, use_filtered=True)
        
        # Fallback: old implementation
        mask_id = getattr(self, '_cached_values_mask_id', None)
        if (
            getattr(self, '_cached_x_values', None) is not None
            and getattr(self, '_cached_x_param_idx', None) == p1_idx
            and mask_id is not None
        ):
            return self._cached_x_values

        values = self.values
        x_values = values[p1_idx]
        self._cached_x_values = x_values
        self._cached_x_param_idx = p1_idx
        return x_values

    @property
    def y_values(self) -> np.ndarray:
        """Get y-axis values, using cache when valid."""
        p2_idx = self.plot_control.p2[0]
        
        # New architecture: use data_manager
        if hasattr(self, 'data_manager'):
            return self.data_manager.get_axis_values('y', p2_idx, use_filtered=True)
        
        # Fallback: old implementation
        mask_id = getattr(self, '_cached_values_mask_id', None)
        if (
            getattr(self, '_cached_y_values', None) is not None
            and getattr(self, '_cached_y_param_idx', None) == p2_idx
            and mask_id is not None
        ):
            return self._cached_y_values

        values = self.values
        y_values = values[p2_idx]
        self._cached_y_values = y_values
        self._cached_y_param_idx = p2_idx
        return y_values

    @property
    def z_values(self) -> np.ndarray:
        """Get z-axis values, using cache when valid."""
        p3_idx = self.plot_control.p3[0]
        
        # New architecture: use data_manager
        if hasattr(self, 'data_manager'):
            return self.data_manager.get_axis_values('z', p3_idx, use_filtered=True)
        
        # Fallback: old implementation
        mask_id = getattr(self, '_cached_values_mask_id', None)
        if (
            getattr(self, '_cached_z_values', None) is not None
            and getattr(self, '_cached_z_param_idx', None) == p3_idx
            and mask_id is not None
        ):
            return self._cached_z_values

        values = self.values
        z_values = values[p3_idx]
        self._cached_z_values = z_values
        self._cached_z_param_idx = p3_idx
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
        return get_value_mask(self)

    @property
    def values(self) -> np.ndarray:
        """
        Return a 2D array of data (selected columns only), applying the
        user-defined mask for Inf/NaN. The result is cached to avoid repeated
        computation when .values is accessed multiple times.
        """
        return get_filtered_values(self)

    @property
    def ymax(self) -> float:
        logging.debug("Getting ymax")
        scale = getattr(self.plot_control, 'scale_y', 'lin')
        result = axis_helpers.compute_axis_max(self.y_values, scale)
        logging.debug(f"ymax = {result}")
        return result

    @property
    def zmin(self):
        logging.debug("Getting zmin")
        scale = getattr(self.plot_control, 'scale_z', 'lin')
        result = axis_helpers.compute_axis_min(self.z_values, scale)
        logging.debug(f"zmin = {result}")
        return result

    @property
    def zmax(self) -> float:
        logging.debug("Getting zmax")
        scale = getattr(self.plot_control, 'scale_z', 'lin')
        result = axis_helpers.compute_axis_max(self.z_values, scale)
        logging.debug(f"zmax = {result}")
        return result

    @property
    def xmin(self) -> float:
        logging.debug("Getting xmin")
        scale = getattr(self.plot_control, 'scale_x', 'lin')
        result = axis_helpers.compute_axis_min(self.x_values, scale)
        logging.debug(f"xmin = {result}")
        return result

    @property
    def xmax(self) -> float:
        logging.debug("Getting xmax")
        scale = getattr(self.plot_control, 'scale_x', 'lin')
        result = axis_helpers.compute_axis_max(self.x_values, scale)
        logging.debug(f"xmax = {result}")
        return result

    @property
    def ymin(self) -> float:
        logging.debug("Getting ymin")
        scale = getattr(self.plot_control, 'scale_y', 'lin')
        result = axis_helpers.compute_axis_min(self.y_values, scale)
        logging.debug(f"ymin = {result}")
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
    def vmin(self):
        logging.debug("Getting vmin")
        result = self.doubleSpinBox_vmin.value()
        logging.debug(f"vmin = {result}")
        return result

    @vmin.setter
    def vmin(self, v):
        logging.debug(f"Setting vmin to {v}")
        self.doubleSpinBox_vmin.blockSignals(True)
        self.doubleSpinBox_vmin.setValue(v)
        self.doubleSpinBox_vmin.blockSignals(False)

    @property
    def vmax(self):
        logging.debug("Getting vmax")
        result = self.doubleSpinBox_vmax.value()
        logging.debug(f"vmax = {result}")
        return result

    @vmax.setter
    def vmax(self, v):
        logging.debug(f"Setting vmax to {v}")
        self.doubleSpinBox_vmax.blockSignals(True)
        self.doubleSpinBox_vmax.setValue(v)
        self.doubleSpinBox_vmax.blockSignals(False)

    @property
    def current_cmap(self) -> str:
        """Get current colormap name."""
        return plot_colormaps.current_cmap(self)

    def update_cmap(self, cmap_name=None):
        """Update colormap using new colormaps module."""
        plot_colormaps.update_guiqwt_colormap(self, cmap_name)

    def populate_colormap_combobox(self):
        """Populate colormap combobox using new colormaps module."""
        plot_colormaps.populate_colormap_combobox(self)

    def on_vmin_vmax_changed(self):
        if not getattr(self, '_deferred_init_done', False) or self.g_2dplot is None:
            return
        
        # Debounce: schedule update instead of executing immediately
        if not hasattr(self, '_vmin_vmax_timer'):
            from qtpy import QtCore
            self._vmin_vmax_timer = QtCore.QTimer()
            self._vmin_vmax_timer.setSingleShot(True)
            self._vmin_vmax_timer.timeout.connect(self._apply_vmin_vmax_change)
        
        # Cancel any pending update and schedule a new one
        self._vmin_vmax_timer.stop()
        self._vmin_vmax_timer.start(50)  # 50ms debounce
    
    def _apply_vmin_vmax_change(self):
        """Actually apply the vmin/vmax change (called after debounce)."""
        if not getattr(self, '_deferred_init_done', False) or self.g_2dplot is None:
            return
        
        logging.info("vmin/vmax values changed")
        current_vmin = self.vmin
        current_vmax = self.vmax
        logging.info(f"Setting colormap limits to vmin={current_vmin}, vmax={current_vmax}")

        # Update the colormap limits for the 2D histogram image
        self.cax.set_lut_range([current_vmin, current_vmax])
        self.g_2dplot.replot()
        logging.info("Colormap limits updated")

    def set_default_colormap(self, default_cmap):
        """Set default colormap using new colormaps module."""
        plot_colormaps.set_default_colormap(self, default_cmap)

    # New unified plotting API methods
    def create_scatter_plot(self, **kwargs):
        """Create scatter plot using unified API."""
        return plotting_api.plot_scatter(self, **kwargs)

    def apply_colormap_to_data(self, data=None, **kwargs):
        """Apply colormap to data using unified API."""
        return plotting_api.apply_colormap(self, data, **kwargs)

    def get_plot_statistics(self, plot_type="histogram", **kwargs):
        """Get plot statistics using unified API."""
        return plotting_api.get_plot_statistics(self, plot_type, **kwargs)

    def export_plot_data(self, plot_type="histogram", **kwargs):
        """Export plot data using unified API."""
        return plotting_api.export_plot_data(self, plot_type, **kwargs)

    def refresh_all_plots(self):
        """Refresh all plots using unified API."""
        plotting_api.refresh_all_plots(self)

    def __init__(
            self,
            data_source=None,  # type: DataSource
            settings_json_fn=None,  # type: str
            parent=None,
            cmap: str = 'gist_earth'
    ) -> None:
        super(NDXplorer, self).__init__(parent=parent)
        

        # Store init params for deferred initialization
        self._init_cmap = cmap
        self._init_settings_json_fn = settings_json_fn
        self._deferred_init_done = False

        # Initialize data manager (new architecture)
        self.data_manager = DataManager()

        self.settings = dict()  # type: Dict
        self.equations = list()  # type: List[Dict[str, str]]
        self.constants = dict()  # type: Dict[str, float]
        self._histogram = {
            "x": (),
            "y": (),
            "z": (),
            "2d": ()
        }
        
        # Backward compatibility: delegate to data_manager
        self._mask_inf = True  # type: bool
        self._mask_nan = True  # type: bool
        self._dynamic_selection = False  # type: bool
        
        self._preserve_contrast = False  # type: bool
        self._has_real_data = False
        self._plot_stack_widget = None
        self._background_label = None
        self._plot_container = None
        
        # Backward compatibility: old data source properties
        self._data_source = DataSource()
        self._default_data_source = None  # Now handled by data_manager
        
        if isinstance(data_source, DataSource):
            self.data_source = data_source

        # Clustering settings
        self._use_clustering = False

        # Common clustering variables
        self._cluster_labels = None  # type: Optional[np.ndarray]
        self._cluster_probabilities = None  # type: Optional[np.ndarray]
        self.clustering_worker = None  # Initialize the worker instance

        # Create clustering dialog early to use its parameters
        self.clustering_dialog = ClusteringDialog(parent=self)

        # Initialize the cache variables to None (backward compatibility)
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

        ui_candidates = [
            Path(__file__).resolve().with_name("plot_main.ui"),
            Path(__file__).resolve().parents[1] / "plotting" / "plot_main.ui",
        ]
        ui_path = next((path for path in ui_candidates if path.exists()), None)
        if ui_path is None:
            raise FileNotFoundError(
                f"NDXplorer UI definition not found. Tried: {', '.join(str(p) for p in ui_candidates)}"
            )
        uic.loadUi(str(ui_path), self)
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

        setup_histogram_spinboxes(self)

        # Connect curve overlay signals
        self.curve_overlay_widget.curvesChanged.connect(self.update_curve_overlays)

        def save_cb():
            logging.info( "Save CB")
            json_str = self.equation_editor.text()
            self.equations = yaml.load(json_str)
        self.equation_editor.save_callback = save_cb

        self.populate_colormap_combobox()

        # Add clustering button to show/hide dialog
        self.setup_clustering_button()

        configure_dynamic_selection_controls(self)

        # Placeholder attributes for plots (created in _deferred_init)
        self.g_zplot = None
        self.g_xplot = None
        self.g_yplot = None
        self.g_2dplot = None
        self.overlay_plot = None
        self.cax = None
        self.bg_image_item = None
        self.gaussian_fit = None
        self.parameter_control = None

        # Create placeholder widgets to maintain correct layout while plots load
        setup_plot_placeholders(self)

        # Plot update batching timer (debounces repeated UI-triggered updates)
        self._plot_update_timer: Optional[QtCore.QTimer] = None
        self._plot_update_pending = False
        self._plot_update_requires_clustering = False
        self._initialize_plot_update_timer()

        # Arrange docks immediately so the UI looks right even before deferred init finishes
        ui_helpers.arrange_docks_preserving_geometry(self)

        # Schedule deferred initialization after window is shown
        # This makes the window appear faster
        QtCore.QTimer.singleShot(0, self._deferred_init)

    def _deferred_init(self):
        """Deferred initialization of heavy plot widgets for faster window appearance."""
        if self._deferred_init_done:
            return
        self._deferred_init_done = True

        cmap = self._init_cmap
        settings_json_fn = self._init_settings_json_fn

        setup_histogram_plots(self)
        setup_2d_histogram_plot(self, cmap)
        setup_overlay_plot(self)
        # Keep the NDxplorer splash/background visible until real data arrives
        plot_update_helpers._show_background(self)

        # -----------------------------------------------------------------
        # Gaussian Fit controls: attach from a separate module for cleanliness
        # -----------------------------------------------------------------
        from ..analysis.gaussian_fit import GaussianFit
        self.gaussian_fit = GaussianFit(self)

        self.g_xplot.setMinimumHeight(40)
        self.g_yplot.setMinimumWidth(40)
        self.g_zplot.setMinimumHeight(100)

        # Load settings
        ###############
        if settings_json_fn is None:
            # Ensure default settings exist in the user's settings folder
            from ..settings import ensure_default_settings
            from .. import settings_helpers
            ensure_default_settings()
            # Get the path to the settings folder
            settings_path = settings_helpers.get_settings_path()
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
        self.actionPerformanceSettings.triggered.connect(self.onPerformanceSettings)
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
        
        # Initialize mask drawing integration
        self._setup_mask_drawing()

        # Initialize UI enabled state based on current dataset
        try:
            self.update_ui_enabled_state()
        except Exception:
            pass
        
        # After deferred init completes, render any pending histograms
        # This handles the case where file loading computed histograms before plot objects existed
        try:
            if self._histogram and any(self._histogram.values()):
                from ..plotting.plot_update_helpers import _update_marginal_plots_from_cache
                _update_marginal_plots_from_cache(self)
                logging.info("Rendered pending histograms after deferred init completion")
        except Exception as e:
            logging.debug(f"Could not render pending histograms: {e}")
        
    def _setup_mask_drawing(self):
        """Setup mask drawing integration with the 2D plot."""
        logging.debug("Starting mask drawing setup...")
        try:
            # Initialize mask drawing integration
            logging.debug("Creating MaskDrawingIntegration instance...")
            self.mask_drawing = MaskDrawingIntegration(self)
            logging.debug("MaskDrawingIntegration instance created")
            
            # Connect mask widget signals
            logging.debug("Getting mask widget...")
            mask_widget = self.plot_control.mask_widget
            logging.debug(f"Mask widget: {mask_widget}")
            
            logging.debug("Connecting mask_changed signal...")
            mask_widget.mask_changed.connect(self._on_mask_changed)
            logging.debug("Signal connected")
            
            # Setup mask overlay on 2D plot
            logging.debug("Setting up mask overlay...")
            self.mask_drawing.setup_mask_overlay()
            logging.debug("Mask overlay setup complete")
            
            logging.info("Mask drawing integration initialized successfully")
        except Exception as e:
            logging.error(f"Could not setup mask drawing: {e}", exc_info=True)
            # Still set mask_drawing to None so we know it failed
            self.mask_drawing = None
    
    def _on_mask_changed(self, mask):
        """Handle mask changes from the mask widget."""
        try:
            # Update the mask overlay visualization
            if hasattr(self, 'mask_drawing'):
                self.mask_drawing.update_mask_overlay(mask)
            
            # Invalidate histogram cache when mask changes
            # This ensures background worker recomputes histograms with new mask
            if (hasattr(self.plot_control, '_histogram_cache') and 
                self.plot_control._histogram_cache is not None):
                self.plot_control.clear_histogram_cache()
                logging.debug("Cleared histogram cache due to mask change")
            
            # Clear frame histogram cache when mask changes
            # This is critical for movies/stacks - otherwise cached frame histograms persist
            if hasattr(self.plot_control, 'clear_frame_histogram_cache'):
                self.plot_control.clear_frame_histogram_cache()
                logging.debug("Cleared frame histogram cache due to mask change")
            
            # Request plot update to recompute histograms with new mask
            # Use skip_clustering=True for faster response (mask changes don't affect clustering)
            self.request_plot_update(skip_clustering=True)
            
        except Exception as e:
            logging.warning(f"Error handling mask change: {e}")
    
    def apply_mask_to_selection(self, category: Optional[int] = None):
        """
        Apply the current mask to create a selection.
        
        Parameters
        ----------
        category : Optional[int]
            If specified, only select pixels with this category.
            If None, select all non-zero pixels.
        """
        try:
            from ..utils import mask_helpers
            
            mask = self.plot_control.mask_widget.get_mask()
            if mask is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Mask",
                    "No mask to apply. Draw or load a mask first."
                )
                return
            
            # Get current histogram bounds
            if not hasattr(self, '_histogram') or self._histogram is None:
                return
            
            hist_data = self._histogram.get('2d')
            if hist_data is None:
                return
            
            H, xedges, yedges = hist_data
            
            # Get parameter names
            x_param = self.plot_control.comboBoxSelX.currentText()
            y_param = self.plot_control.comboBoxSelY.currentText()

            # Get parameter indices
            x_idx_param = -1
            y_idx_param = -1
            
            for i, name in enumerate(self._data_source.parameter_names):
                if name == x_param:
                    x_idx_param = i
                if name == y_param:
                    y_idx_param = i
            
            if x_idx_param == -1 or y_idx_param == -1:
                logging.error(f"Could not find parameter indices for {x_param} or {y_param}")
                return

            # Extract the specific category if requested
            if category is not None:
                binary_mask = (mask == category)
            else:
                binary_mask = (mask > 0)

            if not np.any(binary_mask):
                QtWidgets.QMessageBox.information(
                    self,
                    "Empty Selection",
                    "The mask does not contain any pixels for the specified category."
                )
                return
            
            # Log mask and histogram details
            logging.info(f"Applying mask to selection:")
            logging.info(f"  Mask shape: {mask.shape}, binary_mask shape: {binary_mask.shape}")
            logging.info(f"  Histogram H shape: {H.shape}")
            logging.info(f"  X edges: len={len(xedges)}, range=[{xedges[0]:.2f}, {xedges[-1]:.2f}]")
            logging.info(f"  Y edges: len={len(yedges)}, range=[{yedges[0]:.2f}, {yedges[-1]:.2f}]")
            logging.info(f"  X param: {x_param} (idx={x_idx_param})")
            logging.info(f"  Y param: {y_param} (idx={y_idx_param})")
            logging.info(f"  Mask pixels set: {np.count_nonzero(binary_mask)}/{binary_mask.size}")

            # Remove any existing mask selections for the same parameters to avoid conflicts
            # This prevents old masks from accumulating and interfering with new ones
            existing_mask_selections = []
            for i, sel in enumerate(self.plot_control._selections):
                if isinstance(sel, MaskDataSelection) and sel.idx1 == x_idx_param and sel.idx2 == y_idx_param:
                    existing_mask_selections.append(i)
            
            # Remove from internal list (in reverse to maintain indices)
            for idx in reversed(existing_mask_selections):
                removed = self.plot_control._selections.pop(idx)
                logging.info(f"  Removed existing mask selection: {removed.name} (id={removed.selection_id})")
            
            # Remove from UI table
            table = self.plot_control.tableWidget
            rows_to_remove = []
            for row in range(table.rowCount()):
                item0 = table.item(row, 0)
                if item0:
                    # Check if this is a mask selection by metadata or text
                    meta = None
                    try:
                        for role in [QtCore.Qt.UserRole, 32, QtCore.Qt.UserRole + 10]:
                            meta_raw = item0.data(role)
                            if meta_raw:
                                try:
                                    if isinstance(meta_raw, dict):
                                        meta = meta_raw
                                    else:
                                        meta = json.loads(str(meta_raw))
                                    if meta and isinstance(meta, dict) and "type" in meta:
                                        break
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    
                    is_mask = False
                    if meta and meta.get("type") == "Mask":
                        is_mask = True
                    elif "Bitmap" in table.item(row, 1).text() if table.item(row, 1) else False:
                        is_mask = True
                    
                    if is_mask:
                        rows_to_remove.append(row)
            
            # Remove rows in reverse order
            for row in reversed(rows_to_remove):
                table.removeRow(row)
                logging.info(f"  Removed mask selection UI row {row}")

            # 1. Create the actual selection object first
            selection = MaskDataSelection(
                idx1=x_idx_param,
                idx2=y_idx_param,
                mask=binary_mask,
                edges1=xedges,
                edges2=yedges,
                name=f"Mask ({x_param}, {y_param})"
            )
            
            logging.info(f"  Created MaskDataSelection with id={selection.selection_id}")
            
            # 2. Add it to the internal selections list
            self.plot_control._selections.append(selection)
            
            # 3. Add the UI representation (which triggers the update)
            self.plot_control.addMaskSelection(
                name=selection.name,
                mask=binary_mask,
                edges1=xedges,
                edges2=yedges,
                idx1=x_idx_param,
                idx2=y_idx_param,
                selection_id=selection.selection_id
            )
            
            # Request update
            self.request_plot_update()
            
        except Exception as e:
            logging.error(f"Error applying mask to selection: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to apply mask: {str(e)}"
            )

    def show_dataframe_editor(self):
        """
        Show the data in the data source using DataFrameEditor.
        """
        logging.debug("show_dataframe_editor")
        if self._data_source.empty:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No data loaded—nothing to show."
            )
            return

        _ensure_guiqwt()
        dlg = _DataFrameEditor(self)
        # Set up the editor on the current DataFrame
        if not dlg.setup_and_check(self._data_source.data, title="Data Source"):
            return

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # User hit OK: grab the possibly-modified DataFrame back
            self._data_source.data = dlg.get_value()
            # Refresh the plots
            self.update_plots()

    def clear_plots(self):
        logging.debug(f"clear_plots")
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
        if hasattr(self, 'data_manager') and self.data_manager is not None:
            default_data = self.data_manager.data_source
            if default_data is not None and not default_data.empty:
                default_names = default_data.parameter_names
                try:
                    ix_tau = default_names.index("Tau (green)")
                    ix_prox = default_names.index("Proximity ratio")
                    ix_r = default_names.index("r Experimental (green)")

                    self.plot_control.comboBoxSelX.setCurrentIndex(ix_tau)
                    self.plot_control.comboBoxSelY.setCurrentIndex(ix_prox)
                    self.plot_control.comboBoxSelZ.setCurrentIndex(ix_r)
                except ValueError as e:
                    logging.warning(f"Could not set default combo box indices: {e}")
            else:
                logging.warning("Default data source is empty or None")
        else:
            logging.warning("Data manager not available, cannot set default combo box indices")

        # 5. Directly call update_plots to ensure all graphs are cleared
        self.update_plots()

    def is_napari_available(self):
        return napari_helpers.is_napari_available(self)

    def prompt_install_napari(self) -> bool:
        return napari_helpers.prompt_install_napari(self)

    def install_napari_via_conda(self) -> Tuple[bool, Optional[str]]:
        return napari_helpers.install_napari_via_conda(self)
        
    def send_to_napari(self):
        napari_helpers.send_to_napari(self)
    
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
        logging.debug(f"on_canvas_context_menu")
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
        logging.debug(f"onMaskChanged")
        self._mask_inf = self.checkBoxMaskInf.isChecked()
        self._mask_nan = self.checkBoxMaskNaN.isChecked()
        self.invalidate_values_cache()
        self.update_plots()

    def onShowAxisControl(self) -> None:
        """
        Show the Axis Control dialog.
        This method is triggered when the user clicks the Axis Control action in the View menu.
        """
        logging.debug(f"onShowAxisControl")
        # Create and show the axis control dialog
        dialog = AxisControlDialog(parent=self)
        dialog.exec_()
        
    def onShowUMAP(self) -> None:
        umap_helpers.on_show_umap(self)

    def onShowReportWizard(self):
        """Open the Report Tool dialog."""
        logging.debug(f"onShowReportWizard")
        try:
            from .report_tool import ReportWizard
            dlg = ReportWizard(parent=self)
            dlg.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Report Tool Error", str(e))

    def on_take_screenshot(self):
        screenshot_helpers.take_screenshot(self)

    def onSelectWorkingPath(self):
        logging.debug(f"onSelectWorkingPath")
        working_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select current path', self.working_path)
        # If user cancels the dialog, do not change the working path
        if not working_path:
            return
        self.lineEditWorkingPath.blockSignals(True)
        self.lineEditWorkingPath.setText(working_path)
        self.lineEditWorkingPath.blockSignals(False)

    def _install_working_path_drop(self):
        working_path_helpers.install_working_path_drop(self)

    def on_save_burst_ids(self, evt=None, folder=None):
        save_burst_ids(self, folder)

    def onSaveBurstIDs(self, evt=None, folder=None):
        """Backward-compatible alias used by older macros."""
        self.on_save_burst_ids(evt=evt, folder=folder)

    def onSaveClusteringData(self, evt=None, folder=None):
        save_clustering_data(self, folder)

    def save_parameters(self):
        """
        Save the current parameters to a JSON file in the user's settings folder.
        If chisurf module exists, parameters are saved in the user folder.
        """
        logging.debug(f"save_parameters")
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
        settings_helpers.save_axis_settings(self, settings_json_fn=settings_json_fn)

    def onSetDefaultAxis(self):
        """
        Set the current axis selections (and weight, if available) as the new defaults
        in the active ndxplorer settings JSON under the 'default_axes' key.
        """
        settings_helpers.set_default_axis(self)

    def onLoad_settings(
            self,
            settings_json_fn=None  # type: str
    ) -> None:
        settings_helpers.load_settings(self, settings_json_fn=settings_json_fn)

    def onPerformanceSettings(self) -> None:
        """Show the performance settings dialog."""
        try:
            from ..ui.performance_settings_dialog import PerformanceSettingsDialog
            dlg = PerformanceSettingsDialog(parent=self)
            dlg.exec_()
        except Exception as e:
            logging.error(f"Failed to open performance settings dialog: {e}")
            QtWidgets.QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to open performance settings dialog: {e}"
            )

    def open_files(
        self,
        file_handles: typing.List[str] = None,
        file_type: str = None,
        append: bool = False,
        merge_mode: str = "columns",
    ):
        open_files(
            self,
            file_handles=file_handles,
            file_type=file_type,
            append=append,
            merge_mode=merge_mode,
        )

    def show_merge_dialog(self, title):
        return show_merge_dialog(self, title)

    def onOpenCsv(
        self,
        event,
        filenames: List[str] = None,
        append: bool = False,
        merge_mode: str = "columns",
    ):
        open_csv(self, filenames, append, merge_mode)

    def onOpenChiSurfSampling(
        self,
        filenames=None,
        append: bool = False,
        merge_mode: str = "columns",
    ):
        open_chisurf_sampling(self, filenames, append, merge_mode)

    def onOpenMfdHdf5(
        self,
        event,
        filenames=None,
        append: bool = False,
        merge_mode: str = "columns",
    ):
        open_mfd_hdf5(self, filenames, append, merge_mode)

    def onOpenSmFRET(self, merge_mode: str = "columns"):
        open_smfret(self, merge_mode=merge_mode)

    def update(self, *args, **kwargs):
        logging.debug(f"update")
        super(NDXplorer, self).update()
        self.data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )
        self.lineEditCountTotal.setText(str(self.data_source.size))
        self.plot_control.update()  # plot_control.update() - also updates plots
        # Keep UI enabled/disabled state in sync if someone calls update() directly
        try:
            self.update_ui_enabled_state()
        except Exception:
            pass

    def update_ui_enabled_state(self) -> None:
        """
        Enable/disable most of the UI when there is no dataset loaded.
        Only data-loading actions remain enabled in the disabled state.
        Rules:
        - When no dataset: disable docks, plots, analysis actions and tool buttons.
          Keep only file-open actions (CSV, ChiSurf sampling, MFD HDF5, Paris dataset)
          and working path selection enabled so the user can load data.
        - When dataset present: enable everything.
        """
        try:
            has_data = bool(getattr(self, "_data_source", None) is not None and not self._data_source.empty)
        except Exception:
            has_data = False

        # Whitelist of actions that are allowed even when no data is loaded
        allowed_when_empty = {
            "actionSelect_working_path",
            "actionOpenCsv",
            "actionOpenChiSurfSampling",
            "actionOpenMfdHdf5",
            "actionOpenParisDataset",
        }

        # Toggle all QAction members
        for name in dir(self):
            if not name.startswith("action"):
                continue
            try:
                act = getattr(self, name)
                # QAction has setEnabled; use duck typing
                if hasattr(act, "setEnabled"):
                    enable = has_data or (name in allowed_when_empty)
                    act.setEnabled(bool(enable))
            except Exception:
                pass

        # Toggle key dock widgets (disable content interactions when no data)
        for dock_name in [
            "dockWidget_PlotControl",
            "dockWidget_Parameters",
            "dockWidget_Overlays",
            "dockWidget_Fit",
            "dockWidget_Equations",
        ]:
            try:
                dock = getattr(self, dock_name, None)
                if dock is not None and hasattr(dock, "setEnabled"):
                    dock.setEnabled(bool(has_data))
            except Exception:
                pass

        # Toggle commonly used tool buttons and widgets
        for w_name in [
            "toolButton_screenshot",
            "toolButton_AutoContrast",
            "toolButton_3",  # DataFrame editor
            "toolButton_parameter_save",
            "comboBoxWeight",
            "checkBoxWeight",
            "groupBox_3",
        ]:
            try:
                w = getattr(self, w_name, None)
                if w is not None and hasattr(w, "setEnabled"):
                    w.setEnabled(bool(has_data))
            except Exception:
                pass

        # Keep the file/working-path widgets usable without data
        try:
            if hasattr(self, "lineEditWorkingPath") and self.lineEditWorkingPath is not None:
                self.lineEditWorkingPath.setEnabled(True)
        except Exception:
            pass

    def apply_fonts(self):
        ui_helpers.apply_fonts(self)

    def update_parameter_names(self):
        ui_helpers.update_parameter_names(self)

    def get_bins(self, arange, scale, n_1d, n_2d):
        return histogram_helpers.get_bins(self.plot_control, arange, scale, n_1d, n_2d)

    def get_x_bins(self):
        logging.debug(f"get_x_bins")
        return self.get_bins(
            self.plot_control.x_range,
            self.plot_control.scale_x,
            self.plot_control.n_xhist_1d,
            self.plot_control.n_xhist_2d
        )

    def get_y_bins(self):
        logging.debug(f"get_y_bins")
        return self.get_bins(
            self.plot_control.y_range,
            self.plot_control.scale_y,
            self.plot_control.n_yhist_1d,
            self.plot_control.n_yhist_2d
        )

    def get_z_bins(self):
        logging.debug(f"get_z_bins")
        bins = self.get_bins(
            self.plot_control.z_range,
            self.plot_control.scale_z,
            self.plot_control.n_zhist_1d,
            10
        )
        return bins

    def is_data_ready(self) -> bool:
        """Return True if data and axes are ready for histogram computation."""
        return histogram_helpers.is_data_ready(self)

    def are_bins_valid(self, bins) -> bool:
        return histogram_helpers.are_bins_valid(bins)

    def sanitize_bins(self, bins, data: np.ndarray, default_count: int = 50, scale: str = "linear") -> np.ndarray:
        return histogram_helpers.sanitize_bins(bins, data, default_count, scale)

    def update_histograms(self):
        """Update histograms using new histograms module."""
        plot_histograms.update_histogram_display(self)

    def copy_1d_hists_to_clipboard_csv(self):
        copy_1d_histograms(self)

    def copy_2d_hist_to_clipboard_json(self):
        copy_2d_hist_json(self)

    def copy_2d_hist_to_clipboard_csv(self):
        copy_2d_hist_csv(self)

    def update_plots(self, skip_clustering=False, skip_cache_invalidation=False):
        self._cancel_scheduled_plot_update()
        plot_update_helpers.update_plots(self, skip_clustering=skip_clustering, skip_cache_invalidation=skip_cache_invalidation)

    def request_plot_update(self, skip_clustering=False):
        """
        Schedule a plot update so rapid UI changes can be batched together.
        Falls back to immediate update when deferred init is not complete.
        """
        if not getattr(self, "_deferred_init_done", False):
            # Before full init we can't rely on timers—update immediately.
            return self.update_plots(skip_clustering=skip_clustering)

        timer = self._plot_update_timer
        if timer is None:
            self._initialize_plot_update_timer()
            timer = self._plot_update_timer
        if timer is None:
            # If timer creation still fails, run update immediately.
            return self.update_plots(skip_clustering=skip_clustering)

        self._plot_update_requires_clustering |= not skip_clustering
        self._plot_update_pending = True
        if not timer.isActive():
            timer.start()

    def _cancel_scheduled_plot_update(self):
        timer = self._plot_update_timer
        if timer is not None and timer.isActive():
            timer.stop()
        self._plot_update_pending = False
        self._plot_update_requires_clustering = False

    def _initialize_plot_update_timer(self) -> bool:
        """Create (or confirm) the batching timer used for plot updates."""
        if self._plot_update_timer is not None:
            return True
        try:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(40)  # ms, batches rapid UI signals
            timer.timeout.connect(self._execute_scheduled_plot_update)
            self._plot_update_timer = timer
        except Exception as exc:
            logging.warning("Failed to create plot update timer: %s", exc)
            self._plot_update_timer = None
            return False
        self._plot_update_pending = False
        self._plot_update_requires_clustering = False
        return True

    def _execute_scheduled_plot_update(self):
        if not self._plot_update_pending:
            return
        skip_clustering = not self._plot_update_requires_clustering
        self._cancel_scheduled_plot_update()
        self.update_plots(skip_clustering=skip_clustering)

    def update_spinbox_limits(self, low_pct=0.1, high_pct=99):
        plot_update_helpers.update_spinbox_limits(self, low_pct=low_pct, high_pct=high_pct)

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
            hist_2d = self._histogram["2d"]
            # Extract data from 2D histogram (handle both old tuple and new clean formats)
            if hasattr(hist_2d, 'H'):
                # New clean Histogram2D object
                H = hist_2d.H
            elif isinstance(hist_2d, tuple) and len(hist_2d) == 3:
                # Old tuple format (H, x_edges, y_edges)
                H, *_ = hist_2d
            else:
                return
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
        logging.debug("setup_clustering_button()")
        clustering_helpers.setup_button(self)

    def toggle_clustering_dialog(self):
        """
        Show or hide the clustering dialog.
        """
        logging.debug("toggle_clustering_dialog()")
        clustering_helpers.toggle_dialog(self)

    def create_clustering_dialog(self):
        """
        Create the clustering dialog if it doesn't exist.
        """
        logging.debug("create_clustering_dialog()")
        clustering_helpers.ensure_dialog(self)

    def update_clustering_dialog(self):
        """
        Update the clustering dialog UI elements.
        """
        logging.debug("update_clustering_dialog()")
        clustering_helpers.update_dialog(self)


    def start_clustering_from_dialog(self, method, columns, params):
        """
        Start clustering with parameters from the dialog.

        Args:
            method: The clustering method to use (e.g., 'kmeans', 'hdbscan')
            columns: Set of column names to use for clustering
            params: Dictionary of parameters for the clustering method
        """
        logging.debug(f"start_clustering_from_dialog(method={method}, columns={columns}, params={params})")
        clustering_helpers.start_clustering_from_dialog(self, method, columns, params)

    def cancel_clustering(self):
        """
        Cancel the current clustering operation.
        """
        logging.debug("cancel_clustering()")
        clustering_helpers.cancel_clustering(self)

    def on_select_columns(self):
        """
        Open a dialog to select columns for clustering.
        """
        logging.debug("on_select_columns()")
        clustering_helpers.select_columns(self)


    def on_apply_clustering(self):
        """
        Apply clustering with current parameters and update plots.
        """
        logging.info("Applying clustering with current parameters")
        self._use_clustering = True
        logging.debug("Set _use_clustering flag to True")
        clustering_helpers.apply_clustering(self)

    def on_cancel_clustering(self):
        """
        Cancel the current clustering operation.
        """
        logging.info("Cancelling clustering operation")
        clustering_helpers.cancel_clustering(self)

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
        return clustering_helpers.perform_clustering(self, method=method, worker=worker, **kwargs)

    def keyPressEvent(self, event):
        """
        Handle key press events.

        Args:
            event: The key event
        """
        logging.debug("keyPressEvent: event.key() = {}".format(event.key()))
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
        logging.debug("closeEvent()")
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
        logging.debug("check_z_range_changes()")
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
        Handle changes to the Z-axis groupbox toggle.

        Args:
            state: The new state of the groupbox (True for checked, False for unchecked)
        """
        logging.debug(f"on_enable_z_changed(state={state})")
        # Show or hide the z-axis plot based on the checkbox state
        self.g_zplot.setVisible(bool(state))
        
        # Enable/disable the Z axis update button based on the checkbox state
        if hasattr(self, 'plot_control') and hasattr(self.plot_control, 'toolButtonSetZAxis'):
            self.plot_control.toolButtonSetZAxis.setEnabled(bool(state))
            logging.debug(f"Z axis update button enabled: {bool(state)}")

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
        logging.debug(f"on_weight_param_changed(index={index})")
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
        logging.debug(f"on_weight_changed(state={state})")
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
        logging.debug(f"on_dynamic_selection_changed(state={state})")
        self._dynamic_selection = bool(state)
        # Update histograms to reflect the new selection state
        self.update_histograms()
        # Update plots to display the new histograms
        self.update_plots(skip_clustering=True)

    def on_dynamic_selection_toggled(self, checked: bool):
        """
        Enable/disable periodic Z-range change checks based on the dynamic selection toggle.
        Only connects the timer to check_z_range_changes when enabled.

        Args:
            checked: True if dynamic selection is enabled, False otherwise.
        """
        logging.debug(f"on_dynamic_selection_toggled(checked={checked})")
        try:
            if checked:
                # Ensure the timer is connected once
                if not getattr(self, '_z_timer_connected', False):
                    try:
                        # In case there is a stale connection
                        self.z_range_check_timer.timeout.disconnect(self.check_z_range_changes)
                    except (TypeError, RuntimeError):
                        pass
                    self.z_range_check_timer.timeout.connect(self.check_z_range_changes)
                    self._z_timer_connected = True
                if not self.z_range_check_timer.isActive():
                    self.z_range_check_timer.start()
            else:
                # Stop timer and disconnect the slot
                if self.z_range_check_timer.isActive():
                    self.z_range_check_timer.stop()
                if getattr(self, '_z_timer_connected', False):
                    try:
                        self.z_range_check_timer.timeout.disconnect(self.check_z_range_changes)
                    except (TypeError, RuntimeError):
                        pass
                    self._z_timer_connected = False
        except AttributeError:
            # In case called early during construction
            pass

    def add_umap_columns_to_dataframe(self, columns, params):
        """
        Add UMAP projection columns to the dataframe via helper utilities.
        """
        logging.info("Adding UMAP columns to dataframe using helper")
        return umap_helpers.add_umap_columns(
            ndxplorer=self,
            columns=columns,
            params=params,
            progress_dialog_factory=self._create_umap_progress_dialog,
        )

    def _create_umap_progress_dialog(self, title: str) -> UMAPProgressDialog:
        """Factory used by umap_helpers to create/show the worker dialog."""
        dialog = UMAPProgressDialog(self, title)
        dialog.show()
        return dialog

    def refresh_axis_comboboxes_preserving_selection(self) -> None:
        """Reload axis combobox contents while keeping current selections."""
        if (
            not hasattr(self, "plot_control")
            or self._data_source is None
            or getattr(self._data_source, "parameter_names", None) is None
        ):
            return

        pn = self._data_source.parameter_names
        combos = [
            getattr(self.plot_control, "comboBoxSelX", None),
            getattr(self.plot_control, "comboBoxSelY", None),
            getattr(self.plot_control, "comboBoxSelZ", None),
        ]
        current_texts = [combo.currentText() if combo is not None else "" for combo in combos]

        for combo in combos:
            if combo is not None:
                combo.blockSignals(True)

        try:
            for idx, combo in enumerate(combos):
                if combo is None:
                    continue
                combo.clear()
                combo.addItems(pn)
                previous = current_texts[idx]
                if previous in pn:
                    combo.setCurrentText(previous)
        finally:
            for combo in combos:
                if combo is not None:
                    combo.blockSignals(False)

    def create_umap_plot(self, columns, params):
        """
        Create a UMAP plot in a separate window using existing data.
        Note: This function does not add UMAP columns to the dataframe.
        Use add_umap_columns_to_dataframe() first if needed.

        Args:
            columns: Set of column names to use for UMAP
            params: Dictionary of parameters for UMAP
                n_neighbors: Number of neighbors to consider for each point
                min_dist: Minimum distance between points in the embedding
                n_components: Number of components (dimensions) for the embedding
                n_jobs: Number of parallel jobs for UMAP computation
        """
        logging.debug(f"create_umap_plot(columns={columns}, params={params})")

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
        """
        Update the 2D histogram plot using clean histogram objects.
        """
        logging.info("[DISPLAY] update_2d_plot() called")
        try:
            hist_2d = self._histogram.get("2d")
            if hist_2d is None:
                logging.debug("[DISPLAY] No 2D histogram available yet")
                return
            
            logging.info(f"[DISPLAY] Retrieved hist_2d type={type(hist_2d)}, has H attr={hasattr(hist_2d, 'H')}")
            
            # Extract data from 2D histogram (handle both old tuple and new clean formats)
            if hasattr(hist_2d, 'H'):
                # New clean Histogram2D object
                H = hist_2d.H
                x_edges = hist_2d.x_edges
                y_edges = hist_2d.y_edges
                logging.info(f"[DISPLAY] Extracted from Histogram2D: H shape={H.shape}, dtype={H.dtype}")
                logging.info(f"[DISPLAY] H contiguous={H.flags['C_CONTIGUOUS']}, min={np.min(H)}, max={np.max(H)}, sum={np.sum(H)}")
            elif isinstance(hist_2d, tuple) and len(hist_2d) == 3:
                # Old tuple format (H, x_edges, y_edges)
                H, x_edges, y_edges = hist_2d
                logging.info(f"[DISPLAY] Extracted from tuple: H shape={H.shape}, dtype={H.dtype}")
                logging.info(f"[DISPLAY] H contiguous={H.flags['C_CONTIGUOUS']}, min={np.min(H)}, max={np.max(H)}, sum={np.sum(H)}")
                logging.info(f"[DISPLAY] x_edges length={len(x_edges)}, y_edges length={len(y_edges)}")
            else:
                logging.error("[DISPLAY] Invalid 2D histogram format")
                return
                
        except (ValueError, TypeError) as e:
            logging.warning(f"No 2D histogram data available: {e}")
            return

        # Guard for empty/invalid
        if H is None or H.size == 0 or np.all(np.isnan(H)):
            H = np.zeros((1, 1))
            x_edges = np.array([0.0, 1.0])
            y_edges = np.array([0.0, 1.0])

        # Initialize mask shape when histogram is updated
        if hasattr(self, 'plot_control') and hasattr(self.plot_control, 'mask_widget'):
            # Calculate bin counts from edges
            nx_bins = len(x_edges) - 1
            ny_bins = len(y_edges) - 1
            # Set mask shape to match TRANSPOSED/DISPLAYED image: (ny_bins, nx_bins)
            mask_shape = (ny_bins, nx_bins)
            self.plot_control.mask_widget.set_mask_shape(mask_shape)
            logging.info(f"Set mask shape to {mask_shape} (ny={ny_bins}, nx={nx_bins}) for TRANSPOSED display, histogram shape {H.shape}, edges: x={len(x_edges)}, y={len(y_edges)}")
        
        # Optional log counts (safe for zeros)
        data = H.copy()
        if self.checkBoxLogCounts.isChecked():
            if np.any(data > 0):
                mpos = float(np.min(data[data > 0]))
            else:
                mpos = 1e-10
            data = np.maximum(data, mpos / 10.0)
            data = np.log10(data)
            data = np.nan_to_num(data)

        # H is already in (ny, nx) shape, no transpose needed for display
        img = np.ascontiguousarray(data)
        
        logging.info(f"[DISPLAY] Final image data before set_data():")
        logging.info(f"[DISPLAY]   Original H shape: {H.shape}, data shape: {data.shape}, img shape: {img.shape}")
        logging.info(f"[DISPLAY]   Expected img shape: ({len(y_edges)-1}, {len(x_edges)-1})")
        logging.info(f"[DISPLAY]   Shape match: {img.shape == (len(y_edges)-1, len(x_edges)-1)}")
        logging.info(f"[DISPLAY]   img dtype: {img.dtype}, contiguous: {img.flags['C_CONTIGUOUS']}")
        logging.debug(f"update_2d_plot: H min={np.min(H)}, max={np.max(H)}, data min={np.min(data)}, max={np.max(data)}")
        try:
            self.cax.set_data(img)
            logging.info(f"[DISPLAY] Successfully set image data to cax widget")
        except Exception as e:
            # Fallback to a trivial image if anything goes wrong
            logging.error(f"[DISPLAY] Failed to set image data: {e}")
            self.cax.set_data(np.zeros((1, 1)))

        # Apply colormap/contrast - handle both backends
        if getattr(self, '_use_simple_backend', True):
            # SimpleImageWidget uses set_colormap
            try:
                self.cax.set_colormap(self.cax._colormap_name, self.vmin, self.vmax)
            except Exception:
                pass
        else:
            # guiqwt uses set_lut_range
            try:
                self.cax.set_lut_range([self.vmin, self.vmax])
            except Exception:
                pass

        # Set axis scales - handle both backends
        if getattr(self, '_use_simple_backend', True):
            self.g_2dplot.set_axis_scale('xBottom', 0, len(x_edges)-1)
            self.g_2dplot.set_axis_scale('yLeft', 0, len(y_edges)-1)
            
            # Synchronize overlay plot axis scales with SimpleImageWidget
            if hasattr(self, 'overlay_plot') and self.overlay_plot is not None:
                self.overlay_plot.setAxisScale(0, 0, len(x_edges)-1)  # xBottom
                self.overlay_plot.setAxisScale(1, 0, len(y_edges)-1)  # yLeft
                
                # Update margins to match SimpleImageWidget
                margin_left = 50 if self.g_2dplot.axis_enabled('yLeft') else 0
                margin_right = 50 if self.g_2dplot.axis_enabled('yRight') else 0
                margin_top = 30 if self.g_2dplot.axis_enabled('xTop') else 0
                margin_bottom = 30 if self.g_2dplot.axis_enabled('xBottom') else 0
                self.overlay_plot.set_margins(margin_left, margin_right, margin_top, margin_bottom)
                
                self.overlay_plot.replot()
        else:
            _ensure_guiqwt()
            self.g_2dplot.setAxisScale(_QwtPlot.xBottom, 0, len(x_edges)-1)
            self.g_2dplot.setAxisScale(_QwtPlot.yLeft, 0, len(y_edges)-1)

        # Replot
        if getattr(self, '_use_simple_backend', True):
            logging.info(f"[DISPLAY] Calling g_2dplot.replot() (simple backend)")
            self.g_2dplot.replot()
            # Ensure widget is visible and updated
            if hasattr(self.g_2dplot, 'show'):
                self.g_2dplot.show()
            if hasattr(self.g_2dplot, 'update'):
                self.g_2dplot.update()
        else:
            logging.info(f"[DISPLAY] Calling g_2dplot.replot() (guiqwt backend)")
            self.g_2dplot.replot()
        
        logging.info(f"[DISPLAY] Completed 2D plot update: H shape={H.shape}, edges: x={len(x_edges)}, y={len(y_edges)}")

    def bin_to_value(self, bin_idx, edges):
        """Convert a bin index to a value (center of the bin).

        Args:
            bin_idx: The bin index
            edges: The bin edges array

        Returns:
            The center value of the bin, or None if the bin index is invalid
        """
        logging.debug(f"bin_to_value(bin_idx={bin_idx}, edges={edges})")
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
        logging.debug(f"value_to_bin(value={value}, edges={edges})")
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
        """Delegate auto-contrast adjustment to helper module."""
        plot_update_helpers.auto_contrast(self)

    def update_curve_overlays(self):
        """Update the curve overlays on the 2D histogram."""
        logging.debug("update_curve_overlays()")
        try:
            # Initialize histogram metadata cache
            self._histogram_metadata = {}

            # Get the 2D histogram data and edges (handle both old tuple and new clean formats)
            hist_2d = self._histogram.get("2d")
            if hist_2d is None:
                logging.debug("No 2D histogram available yet for curve overlays")
                return
            if hasattr(hist_2d, 'H'):
                # New clean Histogram2D object
                H = hist_2d.H
                x_edges = hist_2d.x_edges
                y_edges = hist_2d.y_edges
            elif isinstance(hist_2d, tuple) and len(hist_2d) == 3:
                # Old tuple format (H, x_edges, y_edges)
                H, x_edges, y_edges = hist_2d
            else:
                logging.error("Invalid 2D histogram format in curve overlays")
                return
                
            histogram_data = (H, x_edges, y_edges)
            
            # Check if histogram data is valid
            if H is None or H.size == 0:
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

    def check_and_set_image_axes(self) -> bool:
        """Delegate image-axis detection to axis_helpers."""
        return axis_helpers.check_and_set_image_axes(self)

    def apply_default_axes_from_settings(self):
        """
        Apply default axis selections from settings via axis_helpers.
        This is used after a dataset is loaded to preselect X/Y/Z/weight axes.
        It will not override image axes (the caller should check first).
        """
        return axis_helpers.apply_default_axes_from_settings(self)

    def on_gaussian_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        """Delegate to GaussianFit."""
        logging.debug(f"on_gaussian_table_item_changed(item={item})")
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit.on_gaussian_table_item_changed(item)

    def eventFilter(self, obj, event):
        """
        Delegate table/overlay events to GaussianFit; do not trigger 2D updates
        from canvas-resize here to avoid race conditions with resizeEvent().
        """
        logging.debug(f"eventFilter(obj={obj}, event={event})")
        handled_by_gaussian = False
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            try:
                handled_by_gaussian = bool(self.gaussian_fit.eventFilter(obj, event))
            except Exception:
                handled_by_gaussian = False

        if handled_by_gaussian:
            return True
        return super(NDXplorer, self).eventFilter(obj, event)

    def _delete_selected_gaussian_rows(self, rows: List[int]):
        """Delegate to GaussianFit."""
        logging.debug(f"_delete_selected_gaussian_rows(rows={rows})")
        if hasattr(self, 'gaussian_fit') and self.gaussian_fit is not None:
            return self.gaussian_fit._delete_selected_gaussian_rows(rows)

    def _on_fit_dock_visibility_changed(self, visible: bool):
        """Delegate to GaussianFit."""
        logging.debug(f"_on_fit_dock_visibility_changed(visible={visible})")
        return self.gaussian_fit.on_fit_dock_visibility_changed(visible)

    def resizeEvent(self, event):
        """
        Trigger a 2D plot update on window resize to prevent orientation issues.
        Use a zero-timeout singleShot to run after layout has applied new sizes.
        Debounce scheduling to avoid flooding during continuous resizing.
        """
        logging.debug("resizeEvent()")
        # First perform the default resize handling
        super(NDXplorer, self).resizeEvent(event)
        # Then schedule an update of the 2D plot
        def _do_update():
            self.update_plots()
        QtCore.QTimer.singleShot(0, _do_update)
