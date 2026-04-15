from __future__ import print_function
from typing import List, Dict, Optional
import json
import os
import pathlib
import math

import numpy as np

from qtpy import QtGui, uic, QtCore, QtWidgets

try:
    from pyqtgraph.widgets.SpinBox import SpinBox
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    SpinBox = None

from ..core.data_source import RectangularDataSelection, Gaussian2DSelection, MaskDataSelection
from ..logging_config import logging
from .background_histograms import HistogramComputationManager, EnhancedHistogramCache
from ..widgets.mask_drawing_widget import MaskDrawingWidget
from .controls import ScaleControlMixin, AxisControlMixin, HistogramControlMixin


def is_background_computation_enabled():
    """Check if background computation is enabled via environment variable or settings."""
    # First check if there's a settings file override
    try:
        from .utils.performance_config import _get_environment_overrides
        settings_env = _get_environment_overrides()
        if "NDXPLORER_ENABLE_BACKGROUND_WORKER" in settings_env and settings_env["NDXPLORER_ENABLE_BACKGROUND_WORKER"] is not None:
            value = str(settings_env["NDXPLORER_ENABLE_BACKGROUND_WORKER"]).lower()
            enabled = value not in ('0', 'false', 'no', 'off')
            logging.debug(f"Background worker enabled via settings: {enabled}")
            return enabled
    except Exception as e:
        logging.debug(f"Failed to check settings for background worker: {e}")
    
    # Fall back to environment variable (default to enabled unless explicitly disabled)
    return os.environ.get('NDXPLORER_ENABLE_BACKGROUND_WORKER', '').lower() not in ('0', 'false', 'no', 'off')


class SurfacePlotWidget(ScaleControlMixin, AxisControlMixin, HistogramControlMixin, QtWidgets.QWidget):
    """
    Surface plot control widget with modular control mixins.
    
    Architecture:
    - ScaleControlMixin: Log/linear scale controls
    - AxisControlMixin: X/Y/Z/Weight parameter selection
    - HistogramControlMixin: Bins and normalization settings
    """

    axis_settings = dict()  # type: Dict[str, Dict[str, float]]

    @property
    def p2(self):
        idx = self.comboBoxSelY.currentIndex()
        name = self.comboBoxSelY.currentText()
        return idx, str(name)
        
    @p2.setter
    def p2(self, value, block_signals=False):
        """
        Set the Y axis parameter.
        
        Args:
            value: Either an index (int) or parameter name (str) or tuple (idx, name)
            block_signals: If True, signals will be blocked during the change
        """
        self._set_combobox_from_tuple_or_value(self.comboBoxSelY, value, block_signals)

    @property
    def p3(self):
        idx = self.comboBoxSelZ.currentIndex()
        name = self.comboBoxSelZ.currentText()
        return idx, str(name)
        
    @property
    def x_label(self):
        """
        Get the X axis label.
        
        Returns:
            str: The name of the parameter selected for the X axis
        """
        return str(self.comboBoxSelX.currentText())
        
    @property
    def y_label(self):
        """
        Get the Y axis label.
        
        Returns:
            str: The name of the parameter selected for the Y axis
        """
        return str(self.comboBoxSelY.currentText())
        
    @property
    def z_label(self):
        """
        Get the Z axis label.
        
        Returns:
            str: The name of the parameter selected for the Z axis
        """
        return str(self.comboBoxSelZ.currentText())
        
    @p3.setter
    def p3(self, value, block_signals=False):
        """
        Set the Z axis parameter.
        
        Args:
            value: Either an index (int) or parameter name (str) or tuple (idx, name)
            block_signals: If True, signals will be blocked during the change
        """
        self._set_combobox_from_tuple_or_value(self.comboBoxSelZ, value, block_signals)
    
    # Note: set_axis_by_name is now provided by AxisControlMixin

    @property
    def binsX(self):
        return int(self.spinBoxBin1DX.value())

    @property
    def binsY(self):
        return int(self.spinBoxBin1DY.value())

    @property
    def binsZ(self):
        return int(self.spinBoxBin1DZ.value())

    @property
    def bins2X(self):
        return int(self.spinBoxBin2DX.value())

    @property
    def bins2Y(self):
        return int(self.spinBoxBin2DY.value())

    @property
    def n_xhist_1d(self):
        return int(self.spinBoxBin1DX.value())

    @n_xhist_1d.setter
    def n_xhist_1d(self, v):
        self.spinBoxBin1DX.setValue(int(v))

    @property
    def n_yhist_1d(self):
        return int(self.spinBoxBin1DY.value())

    @n_yhist_1d.setter
    def n_yhist_1d(self, v):
        self.spinBoxBin1DY.setValue(int(v))

    @property
    def n_zhist_1d(self):
        return int(self.spinBoxBin1DZ.value())

    @n_zhist_1d.setter
    def n_zhist_1d(self, v):
        self.spinBoxBin1DZ.setValue(int(v))

    @property
    def n_xhist_2d(self):
        return int(self.spinBoxBin2DX.value())

    @n_xhist_2d.setter
    def n_xhist_2d(self, v):
        self.spinBoxBin2DX.setValue(int(v))

    @property
    def n_yhist_2d(self):
        return int(self.spinBoxBin2DY.value())

    @n_yhist_2d.setter
    def n_yhist_2d(self, v):
        self.spinBoxBin2DY.setValue(int(v))

    @property
    def selected_cluster(self):
        """
        Returns the currently selected cluster from spinBoxCluster.
        A value of -1 means all clusters should be displayed.
        """
        return int(self.spinBoxCluster.value())

    @property
    def x_range(self):
        return float(self.spinBoxXmin.value()), \
               float(self.spinBoxXmax.value())

    @property
    def xmin(self):
        return self.x_range[0]

    @xmin.setter
    def xmin(self, v):
        self.spinBoxXmin.setValue(v)

    @property
    def xmax(self):
        return self.x_range[1]

    @xmax.setter
    def xmax(self, v):
        self.spinBoxXmax.setValue(v)

    @property
    def y_range(self):
        return float(self.spinBoxYmin.value()), \
               float(self.spinBoxYmax.value())

    @property
    def ymin(self):
        return float(self.spinBoxYmin.value())

    @ymin.setter
    def ymin(self, v):
        self.spinBoxYmin.setValue(v)

    @property
    def ymax(self):
        return float(self.spinBoxYmax.value())

    @ymax.setter
    def ymax(self, v):
        self.spinBoxYmax.setValue(v)

    @property
    def z_range(self):
        return float(self.spinBoxZmin.value()), \
               float(self.spinBoxZmax.value())

    @property
    def zmin(self):
        return float(self.spinBoxZmin.value())

    @zmin.setter
    def zmin(self, v):
        self.spinBoxZmin.setValue(v)

    @property
    def zmax(self):
        return float(self.spinBoxZmax.value())

    @zmax.setter
    def zmax(self, v):
        self.spinBoxZmax.setValue(v)

    def __init__(self, parent=None):
        super(SurfacePlotWidget, self).__init__()
        self.parent = parent
        self._selections = list()  # Instance-level selections list
        self.axis_settings = dict()  # Instance-level axis settings
        logging.log(0, "Initializing SurfacePlotWidget")
        #########################
        # GUI
        #########################
        ui_file = pathlib.Path(__file__).parent / 'plot_control.ui'

        uic.loadUi(str(ui_file.as_posix()), self)

        # Expose the clustering button on the parent (NDXplorer) for legacy wiring.
        clustering_btn = getattr(self, "pushButtonShowClusteringDialog", None)
        if clustering_btn is not None and self.parent is not None:
            self.parent.pushButtonShowClusteringDialog = clustering_btn
        
        # Handle pyqtgraph SpinBox replacement if available
        if PYQTGRAPH_AVAILABLE and SpinBox is not None:
            # Replace QSpinBox widgets with pyqtgraph SpinBox for better functionality
            logging.log(0, "Replacing QSpinBox with pyqtgraph SpinBox widgets")
            
            # Get the current layout positions and widget properties
            x_min_widget = self.spinBoxXmin
            x_max_widget = self.spinBoxXmax
            y_min_widget = self.spinBoxYmin
            y_max_widget = self.spinBoxYmax
            z_min_widget = self.spinBoxZmin
            z_max_widget = self.spinBoxZmax
            
            # Create pyqtgraph SpinBox widgets
            self.spinBoxXmin = SpinBox()
            self.spinBoxXmax = SpinBox()
            self.spinBoxYmin = SpinBox()
            self.spinBoxYmax = SpinBox()
            self.spinBoxZmin = SpinBox()
            self.spinBoxZmax = SpinBox()
            
            # Copy properties from original widgets
            for new_widget, old_widget in [
                (self.spinBoxXmin, x_min_widget),
                (self.spinBoxXmax, x_max_widget),
                (self.spinBoxYmin, y_min_widget),
                (self.spinBoxYmax, y_max_widget),
                (self.spinBoxZmin, z_min_widget),
                (self.spinBoxZmax, z_max_widget)
            ]:
                # Copy size policy
                new_widget.setSizePolicy(old_widget.sizePolicy())
                # Copy tooltip
                new_widget.setToolTip(old_widget.toolTip())
                # Set reasonable range
                new_widget.setRange(-1000000, 1000000)
            
            # Get the grid layout and replace widgets in their positions
            grid_layout = self.gridLayout_6  # This is the main histogram grid layout
            
            # Replace widgets in the grid layout
            # X axis widgets are at row 6
            grid_layout.replaceWidget(x_min_widget, self.spinBoxXmin)
            grid_layout.replaceWidget(x_max_widget, self.spinBoxXmax)
            # Y axis widgets are at row 8
            grid_layout.replaceWidget(y_min_widget, self.spinBoxYmin)
            grid_layout.replaceWidget(y_max_widget, self.spinBoxYmax)
            # Z axis widgets are in a different grid (gridLayout_2)
            z_grid_layout = self.gridLayout_2
            z_grid_layout.replaceWidget(z_min_widget, self.spinBoxZmin)
            z_grid_layout.replaceWidget(z_max_widget, self.spinBoxZmax)
            
            # Delete old widgets
            x_min_widget.deleteLater()
            x_max_widget.deleteLater()
            y_min_widget.deleteLater()
            y_max_widget.deleteLater()
            z_min_widget.deleteLater()
            z_max_widget.deleteLater()
        else:
            # Use QSpinBox widgets from UI file
            logging.log(0, "Using QSpinBox widgets from UI file")
        
        # Initialize mask drawing widget - use UI elements directly
        self.mask_widget = MaskDrawingWidget(self)
        # Store reference to NDXplorer parent for easy access
        self.mask_widget._ndxplorer_parent = self.parent
        
        # Connect UI elements from the .ui file to the mask widget
        self.categorySpinBox.valueChanged.connect(self.mask_widget._on_category_changed)
        self.brushSpinBox.valueChanged.connect(self.mask_widget._on_brush_size_changed)
        self.drawRadio.toggled.connect(self.mask_widget._on_mode_changed)
        self.loadMaskBtn.clicked.connect(self.mask_widget.load_mask)
        self.saveMaskBtn.clicked.connect(self.mask_widget.save_mask)
        self.clearMaskBtn.clicked.connect(self.mask_widget.clear_mask)
        self.applyMaskBtn.clicked.connect(self.mask_widget._on_apply_mask)
        self.groupBoxMaskDrawing.toggled.connect(self.mask_widget._on_drawing_enabled_changed)
        
        # Set up references to UI elements for the mask widget
        self.mask_widget.category_spinbox = self.categorySpinBox
        self.mask_widget.brush_spinbox = self.brushSpinBox
        self.mask_widget.draw_radio = self.drawRadio
        self.mask_widget.erase_radio = self.eraseRadio
        self.mask_widget.load_mask_btn = self.loadMaskBtn
        self.mask_widget.save_mask_btn = self.saveMaskBtn
        self.mask_widget.clear_mask_btn = self.clearMaskBtn
        self.mask_widget.apply_mask_btn = self.applyMaskBtn
        self.mask_widget.enable_drawing_checkbox = self.groupBoxMaskDrawing
        # maskStatsLabel has been removed from UI - set to None
        self.mask_widget.stats_label = None
        
        # Initialize spinbox values to prevent empty fields
        if PYQTGRAPH_AVAILABLE and SpinBox is not None:
            # For pyqtgraph SpinBox, set default values
            self.spinBoxXmin.setValue(0.0)
            self.spinBoxXmax.setValue(1.0)
            self.spinBoxYmin.setValue(0.0)
            self.spinBoxYmax.setValue(1.0)
            self.spinBoxZmin.setValue(0.0)
            self.spinBoxZmax.setValue(1.0)
        else:
            # For QSpinBox fallback, set default values
            # The UI file already has these values, but we set them again for consistency
            self.spinBoxXmin.setValue(0)
            self.spinBoxXmax.setValue(100)
            self.spinBoxYmin.setValue(0)
            self.spinBoxYmax.setValue(100)
            self.spinBoxZmin.setValue(0)
            self.spinBoxZmax.setValue(100)

        # Allow inline editing of selection numeric bounds in the table with single-click
        # Keep double-click deletion as defined in the .ui (cellDoubleClicked -> actionSelectionTableClicked)
        # React to edits in the selection table
        try:
            self.tableWidget.itemChanged.disconnect()
        except Exception:
            pass
        self.tableWidget.itemChanged.connect(self.onSelectionItemChanged)
        # Guard flag to prevent recursive updates during programmatic edits
        self._block_selection_item_changed = False

        # Use our own single-click edit behavior and preserve double-click for delete
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._single_click_edit_timer = QtCore.QTimer(self)
        self._single_click_edit_timer.setSingleShot(True)
        self._single_click_edit_timer.timeout.connect(self._perform_pending_single_click_edit)
        self._pending_edit_index = None
        self.tableWidget.cellClicked.connect(self.onSelectionCellClicked)
        self.tableWidget.cellDoubleClicked.connect(self.onSelectionCellDoubleClicked)

        # Delete key removes selected selection rows
        try:
            shortcut_delete = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self.tableWidget)
            shortcut_delete.activated.connect(self.onDeleteSelectionRows)
        except Exception:
            pass

        # Set up context menu for selection table
        self.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.onSelectionTableContextMenu)

        # Auto complete for selectors
        self.comboBoxSelX.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.comboBoxSelX.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.comboBoxSelY.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.comboBoxSelY.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.comboBoxSelZ.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.comboBoxSelZ.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        #########################
        # Actions
        #########################
        # Generic action
        self.actionUpdatePlots.triggered.connect(self._on_normalize_changed)

        # Auto range
        self.actionAuto_range_x.triggered.connect(self.on_auto_range_x)
        self.actionAuto_range_x.triggered.connect(self.update_axis_scales)
        self.actionAuto_range_y.triggered.connect(self.on_auto_range_y)
        self.actionAuto_range_y.triggered.connect(self.update_axis_scales)
        self.actionAuto_range_z.triggered.connect(self.on_auto_range_z)
        self.actionAuto_range_z.triggered.connect(self.update_axis_scales)
        self.actionAuto_range_z.triggered.connect(self.auto_selection_range)
        self.actionUpdate_axis_scales.triggered.connect(self.update_axis_scales)
        self.actionAuto_range_x.triggered.connect(self.update_axis_scales)

        # Selection table
        self.actionSelectionTableClicked.triggered.connect(self.onSelectionTableClicked)
        self.actionSave_selection.triggered.connect(self.onSave_selection)
        self.actionLoad_selection.triggered.connect(self.onLoad_selection)
        self.actionClear_Selection.triggered.connect(self.onClearSelection)
        self.actionAdd_Selection.triggered.connect(self.onAddSelection)
        self.actionSave_Burst_IDs.triggered.connect(self.parent.onSaveBurstIDs)
        
        # Connect toolButtonClearSelection to clear selection action
        if hasattr(self, 'toolButtonClearSelection'):
            self.toolButtonClearSelection.clicked.connect(self.onClearSelection)

        # Change axis range
        if PYQTGRAPH_AVAILABLE and SpinBox is not None:
            # pyqtgraph SpinBox signals
            self.spinBoxXmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxXmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxYmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxYmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxZmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxZmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        else:
            # QSpinBox signals
            self.spinBoxXmin.valueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxXmax.valueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxYmin.valueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxYmax.valueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxZmin.valueChanged.connect(self.actionUpdate_axis_scales.trigger)
            self.spinBoxZmax.valueChanged.connect(self.actionUpdate_axis_scales.trigger)

        # Change parameter plotted on axis
        self.actionX_axis_changed.triggered.connect(self.on_x_axis_changed)
        self.actionY_axis_changed.triggered.connect(self.on_y_axis_changed)
        self.actionZ_axis_changed.triggered.connect(self.on_z_axis_changed)

        # Update axis settings
        self.actionUpdate_x_axis_settings.triggered.connect(self.update_x_axis_settings)
        self.actionUpdate_y_axis_settings.triggered.connect(self.update_y_axis_settings)
        self.actionUpdate_z_axis_settings.triggered.connect(self.update_z_axis_settings)

        # Connect spinBoxCluster to update plots when value changes
        self.spinBoxCluster.valueChanged.connect(self.onClusterSelectionChanged)
        
        # Connect bin count spinboxes to clear mask when bins change
        self.spinBoxBin2DX.valueChanged.connect(self.on_bin_count_changed)
        self.spinBoxBin2DY.valueChanged.connect(self.on_bin_count_changed)
        
        # Connect 1D bin count spinboxes as well
        self.spinBoxBin1DX.valueChanged.connect(self.on_bin_count_changed)
        self.spinBoxBin1DY.valueChanged.connect(self.on_bin_count_changed)
        self.spinBoxBin1DZ.valueChanged.connect(self.on_bin_count_changed)
        
        # Initialize frame/time series related attributes
        self._frame_param = None
        self._n_frames = 0
        self._frame_histogram_cache = {}
        self._playback_direction = 0
        
        # Initialize background computation (no caching)
        self._histogram_worker = None
        self._background_computation_enabled = is_background_computation_enabled()
        self._background_computation_pending = False
        self._frame_duration_ms = 200  # Default value
        
        # Log background computation status
        if self._background_computation_enabled:
            logging.info("Background histogram computation enabled (use NDXPLORER_ENABLE_BACKGROUND_WORKER=0 to disable)")
        else:
            logging.info("Background histogram computation disabled (NDXPLORER_ENABLE_BACKGROUND_WORKER=0 or false/no/off)")
        
        self._load_playback_settings()
        
        # Setup playback controls (hidden by default)
        self._setup_playback_controls()
        
        # Initialize frame selection widgets as hidden
        self.hide_frame_selection()

    def set_axis_settings(self, name, amin, amax, scale, bins_1d, bins_2d):
        self.axis_settings[str(name)] = {
            "n_bins_1d": float(bins_1d),
            "min": float(amin),
            "max": float(amax),
            "scale": str(scale)
        }
        if bins_2d is not None:
            self.axis_settings[str(name)].update(
                {
                    "n_bins_2d": int(bins_2d)
                }
            )
        logging.log(0, f"Axis settings updated for {name}: {self.axis_settings[str(name)]}")

    def update_axis_settings(self, axis):
        """
        Update settings for the specified axis.

        Args:
            axis (str): The axis to update ('x', 'y', or 'z')
        """
        axis = axis.lower()
        logging.log(0, f"update_axis_settings for {axis} axis")

        if axis == 'x':
            self.set_axis_settings(
                self.p1[1],
                self.xmin, self.xmax,
                self.scale_x,
                self.n_xhist_1d,
                self.n_xhist_2d
            )
        elif axis == 'y':
            self.set_axis_settings(
                self.p2[1],
                self.ymin, self.ymax,
                self.scale_y,
                self.n_yhist_1d,
                self.n_yhist_2d
            )
        elif axis == 'z':
            self.set_axis_settings(
                self.p3[1],
                self.zmin, self.zmax,
                self.scale_z,
                self.n_zhist_1d,
                None
            )
        else:
            logging.log(0, f"Invalid axis: {axis}")

    def update_x_axis_settings(self):
        """Update settings for the X axis"""
        self.update_axis_settings('x')

    # Keep the old method name for backward compatibility
    onUpdate_x_axis_settings = update_x_axis_settings

    def update_y_axis_settings(self):
        """Update settings for the Y axis"""
        self.update_axis_settings('y')

    # Keep the old method name for backward compatibility
    onUpdate_y_axis_settings = update_y_axis_settings

    def update_z_axis_settings(self):
        """Update settings for the Z axis"""
        self.update_axis_settings('z')

    # Keep the old method name for backward compatibility
    onUpdate_z_axis_settings = update_z_axis_settings

    def onClusterSelectionChanged(self, value):
        """
        Handle changes to the cluster selection spinbox.

        Args:
            value: The new value of the spinbox
        """
        logging.log(0, f"Cluster selection changed to {value}")
        # Update plots with skip_clustering=True to avoid re-clustering the data
        self.parent.request_plot_update(skip_clustering=True)

    def on_axis_changed(self, axis):
        """
        Handle changes to any axis (X, Y, or Z).

        This method updates the histogram bins, range, and scale settings for the specified axis
        based on the currently selected parameter. If settings for the parameter exist in 
        axis_settings, those are used; otherwise, auto-range is applied.

        Args:
            axis (str): The axis to update ('x', 'y', or 'z')
        """
        axis = axis.lower()

        # Define mappings for each axis to its properties
        axis_properties = {
            'x': {
                'property': self.p1,
                'hist_1d': 'n_xhist_1d',
                'hist_2d': 'n_xhist_2d',
                'min': 'xmin',
                'max': 'xmax',
                'scale': 'scale_x',
                'auto_range': self.on_auto_range_x,
                'parent_min': 'xmin',
                'parent_max': 'xmax'
            },
            'y': {
                'property': self.p2,
                'hist_1d': 'n_yhist_1d',
                'hist_2d': 'n_yhist_2d',
                'min': 'ymin',
                'max': 'ymax',
                'scale': 'scale_y',
                'auto_range': self.on_auto_range_y,
                'parent_min': 'ymin',
                'parent_max': 'ymax'
            },
            'z': {
                'property': self.p3,
                'hist_1d': 'n_zhist_1d',
                'hist_2d': None,  # Z axis doesn't have 2D histogram bins
                'min': 'zmin',
                'max': 'zmax',
                'scale': 'scale_z',
                'auto_range': self.on_auto_range_z,
                'parent_min': 'zmin',
                'parent_max': 'zmax'
            }
        }

        # Check if the axis is valid
        if axis not in axis_properties:
            logging.log(0, f"Invalid axis: {axis}")
            return

        # Get the properties for this axis
        props = axis_properties[axis]
        _, name = props['property']

        if name in self.axis_settings:
            d = self.axis_settings[name]

            # Set the 1D histogram bins
            setattr(self, props['hist_1d'], d.get('n_bins_1d', 50))

            # Set the 2D histogram bins if applicable
            if props['hist_2d'] is not None:
                # Special handling for pixel - adjust 2d hist bins to max value
                if "pixel" in name.lower():
                    setattr(self, props['hist_2d'], int(d.get('max', 256)))
                    logging.log(0, f"{name} selected: Setting {props['hist_2d']} to {getattr(self, props['hist_2d'])}")
                else:
                    setattr(self, props['hist_2d'], d.get('n_bins_2d', 50))

            # Set the min, max, and scale
            setattr(self, props['min'], d.get('min', getattr(self.parent, props['parent_min'])))
            setattr(self, props['max'], d.get('max', getattr(self.parent, props['parent_max'])))
            setattr(self, props['scale'], d.get('scale', "lin"))

            logging.log(0, f"{axis.upper()} axis changed to settings: {d}")
        else:
            logging.log(0, f"{axis.upper()} axis settings for {name} not found. Using auto range.")
            props['auto_range']()

        # When X or Y axis changes, disable drawing mode and clear current drawing
        # but keep existing bitmap selections (they will be automatically updated)
        if axis in ['x', 'y']:
            logging.log(0, f"{axis.upper()} axis changed - disabling drawing mode and clearing current drawing")
            # Disable drawing mode and clear current drawing only
            if hasattr(self, 'mask_widget') and self.mask_widget is not None:
                self.mask_widget.set_drawing_enabled(False)
                self.mask_widget.clear_mask()
            
            # Note: Existing bitmap selections are kept - they will be automatically
            # updated when the histogram is recomputed with new axis/bin settings

        self.parent.request_plot_update()

    def on_x_axis_changed(self):
        """Call the combined axis change method for X axis"""
        self.on_axis_changed('x')

    # Keep the old method name for backward compatibility
    onX_axis_changed = on_x_axis_changed

    def on_y_axis_changed(self):
        """Call the combined axis change method for Y axis"""
        self.on_axis_changed('y')

    # Keep the old method name for backward compatibility
    onY_axis_changed = on_y_axis_changed

    def on_z_axis_changed(self):
        """Call the combined axis change method for Z axis"""
        self.on_axis_changed('z')

    # Keep the old method name for backward compatibility
    onZ_axis_changed = on_z_axis_changed

    def on_bin_count_changed(self):
        """
        Handle changes to 2D histogram bin counts.
        
        When bin counts change, disable drawing mode, clear current drawing,
        and force immediate histogram recomputation with new bin sizes.
        Bypasses all caching and background computation for immediate response.
        """
        logging.log(0, "Bin count changed - forcing immediate recomputation with no cache")
        
        # Disable drawing mode and clear current drawing only
        if hasattr(self, 'mask_widget') and self.mask_widget is not None:
            self.mask_widget.set_drawing_enabled(False)
            self.mask_widget.clear_mask()
        
        # Force immediate recomputation by bypassing background system entirely
        # This is critical for bin count changes - no cache should be used
        old_pending = getattr(self, '_background_computation_pending', False)
        self._background_computation_pending = False  # Clear any pending flag
        
        # Clear all histogram caches
        self.clear_frame_histogram_cache()
        self.clear_histogram_cache()
        
        # Force immediate histogram computation (no background, no cache)
        try:
            from .plot_update_helpers import _update_histograms_immediate
            _update_histograms_immediate(self.parent)
            logging.log(0, "Bin count change - immediate histogram computation completed")
        except Exception as e:
            logging.error(f"Failed to compute histograms immediately after bin count change: {e}")
            # Fallback to regular update
            self.parent.request_plot_update(skip_clustering=True)
        
        logging.log(0, "Bin count change handled - forced immediate recomputation")

    def _on_normalize_changed(self):
        """Handle normalization (density) checkbox changes."""
        logging.log(0, "Normalization changed - clearing all caches and triggering full plot update")
        
        # Clear all caches to force fresh computation with new normalization settings
        self.clear_frame_histogram_cache()
        self.clear_histogram_cache()
        
        # Clear any pending background computation to ensure immediate update
        self._background_computation_pending = False
        
        # Trigger full update like the main update button
        self.parent.update_plots()
        
        # Add another update click at the end to ensure final refresh
        QtCore.QTimer.singleShot(50, lambda: self.parent.update_plots())
        
        logging.log(0, "Normalization change handled - full plot update triggered with final refresh")

    def update_axis_scales(self):
        """Update all axis scales based on current settings and refresh plots"""
        # Set y-plot axes
        self.parent.g_yplot.set_axis_scale("left", self.scale_y)
        self.parent.g_yplot.set_axis_scale("right", self.scale_y)

        # Set x-plot axes
        self.parent.g_xplot.set_axis_scale("bottom", self.scale_x)
        self.parent.g_xplot.set_axis_scale("top", self.scale_x)

        # Set z-plot axes
        self.parent.g_zplot.set_axis_scale("bottom", self.scale_z)

        # CRITICAL: Log scale changes require full plot update like the main update button
        # because the binning, data representation, and axes all change with scale
        logging.log(0, "Axis scales updated - clearing all caches and triggering full plot update for log scale changes")
        
        # Clear all caches to force fresh computation with new scale settings
        self.clear_frame_histogram_cache()
        self.clear_histogram_cache()
        
        # Clear any pending background computation to ensure immediate update
        self._background_computation_pending = False
        
        # Trigger full update like the main update button
        self.parent.update_plots()
        
        # Add another update click at the end to ensure final refresh
        QtCore.QTimer.singleShot(50, lambda: self.parent.update_plots())
        
        logging.log(0, "Axis scales updated and full plot update triggered with final refresh")

    # Keep the old method name for backward compatibility
    onUpdate_axis_scales = update_axis_scales

    def auto_selection_range(self):
        z = self.parent.z_values
        m = z.mean()
        sd = z.std()
        self.parent.selection_z.set_range(m - 2 * sd, m + 2 * sd)
        logging.log(0, f"Auto selection range set to: {(m - 2 * sd, m + 2 * sd)}")

    def update(self, update_comboboxes=True, update_plots=True, skip_clustering=True):
        """
        Update the plot control widget.
        
        Args:
            update_comboboxes (bool): Whether to refresh the axis selection comboboxes
            update_plots (bool): Whether to trigger plot updates
            skip_clustering (bool): Whether to skip clustering when updating plots
        """
        super(SurfacePlotWidget, self).update()
        
        if update_comboboxes:
            self.actionUpdatePlots.blockSignals(True)
            self.actionUpdate_axis_scales.blockSignals(True)
            
            # Save current selections before updating
            current_x = self.comboBoxSelX.currentText()
            current_y = self.comboBoxSelY.currentText()
            current_z = self.comboBoxSelZ.currentText()
            current_w = self.comboBoxWeight.currentText() if hasattr(self, 'comboBoxWeight') else ''
            
            # Block combobox signals to prevent triggering replots during updates
            self.comboBoxSelX.blockSignals(True)
            self.comboBoxSelY.blockSignals(True)
            self.comboBoxSelZ.blockSignals(True)
            
            try:
                # Prefer dataframe columns, but fall back to parameter_names for loaders
                # that provide names/values before a full dataframe-backed column map exists.
                try:
                    pn = [str(c) for c in list(self.parent.data_source.data.columns)]
                except Exception:
                    pn = []
                if not pn:
                    try:
                        pn = [str(c) for c in list(self.parent.data_source.parameter_names)]
                    except Exception:
                        pn = []
                self.comboBoxSelX.clear()
                self.comboBoxSelY.clear()
                self.comboBoxSelZ.clear()
                if hasattr(self, 'comboBoxWeight'):
                    self.comboBoxWeight.clear()
                self.comboBoxSelX.addItems(pn)
                self.comboBoxSelY.addItems(pn)
                self.comboBoxSelZ.addItems(pn)
                if hasattr(self, 'comboBoxWeight'):
                    self.comboBoxWeight.addItems(pn)
                
                # Restore previous selections if they still exist in the updated list
                if current_x in pn:
                    self.comboBoxSelX.setCurrentText(current_x)
                else:
                    if pn:
                        self.comboBoxSelX.setCurrentIndex(0)
                    logging.info(f"X selection '{current_x}' not available; keeping default")
                if current_y in pn:
                    self.comboBoxSelY.setCurrentText(current_y)
                else:
                    if pn:
                        self.comboBoxSelY.setCurrentIndex(0)
                    logging.info(f"Y selection '{current_y}' not available; keeping default")
                if current_z in pn:
                    self.comboBoxSelZ.setCurrentText(current_z)
                else:
                    if pn:
                        self.comboBoxSelZ.setCurrentIndex(0)
                    logging.info(f"Z selection '{current_z}' not available; keeping default")
                # Restore weight selection
                if hasattr(self, 'comboBoxWeight'):
                    if current_w in pn:
                        self.comboBoxWeight.setCurrentText(current_w)
                    else:
                        if pn:
                            self.comboBoxWeight.setCurrentIndex(0)
                        logging.info(f"Weight selection '{current_w}' not available; keeping default")
            finally:
                # Unblock combobox signals after updates
                self.comboBoxSelX.blockSignals(False)
                self.comboBoxSelY.blockSignals(False)
                self.comboBoxSelZ.blockSignals(False)
                self.actionUpdatePlots.blockSignals(False)
                self.actionUpdate_axis_scales.blockSignals(False)
            logging.log(0, "Updated parameter selectors (preserved existing selections, no replot triggered)")

        if update_plots:
            # Instead of triggering the action, call update_plots (batched) with skip_clustering
            self.parent.request_plot_update(skip_clustering=skip_clustering)
            logging.log(0, f"Triggered plot update with clustering {'skipped' if skip_clustering else 'enabled'}")

    def onClearSelection(self):
        logging.log(0, "onClearSelection")
        self.tableWidget.setRowCount(0)
        self._selections.clear()  # Clear instance-level selections
        # Clear frame histogram cache when selections change
        self.clear_frame_histogram_cache()
        self.clear_histogram_cache()
        # Preserve contrast during selection operations
        self.parent._preserve_contrast = True
        self.parent.request_plot_update(skip_clustering=True)
        self.parent._preserve_contrast = False
        
        # Add delayed marginal plot update to ensure proper rendering
        QtCore.QTimer.singleShot(100, self._delayed_marginal_update_after_clear)

    def _delayed_marginal_update_after_clear(self):
        """Delayed update of marginal plots after clearing selections."""
        try:
            # Import the marginal update function
            from .plot_update_helpers import _update_marginal_plots_from_cache
            _update_marginal_plots_from_cache(self.parent)
            logging.log(0, "Delayed marginal plot update completed after clear selection")
        except Exception as e:
            logging.warning(f"Failed to update marginal plots after clear selection: {e}")

    def onSave_selection(self):
        logging.log(0, "onSave_selection")
        l = [s.__dict__ for s in self.get_selections()]
        fn = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Selection JSON",
            self.parent.working_path,
            'All files (*.selection.json)'
        )[0]
        with open(fn, "w") as fp:
            json.dump(l, fp=fp, indent=4)
        logging.log(0, f"Selection saved to file: {fn}")

    def onLoad_selection(self):
        fn = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Selection JSON",
            self.parent.working_path,
            'All files (*.selection.json)'
        )[0]
        with open(fn, "r") as fp:
            d = json.load(fp)
            for selection in d:
                self.addSelection(
                    selection['parameter_idx'],
                    selection['lower'],
                    selection['upper'],
                    selection['invert'],
                    selection['enabled'],
                    selection['name']
                )
        logging.log(0, f"Selections loaded from file: {fn}")

    def on_auto_range_x(self):
        """Set X axis range to auto values from parent"""
        logging.log(0, "on_auto_range_x")
        self.spinBoxXmin.blockSignals(True)
        self.spinBoxXmax.blockSignals(True)
        self.spinBoxXmin.setValue(self.parent.xmin)
        self.spinBoxXmax.setValue(self.parent.xmax)
        self.spinBoxXmin.blockSignals(False)
        self.spinBoxXmax.blockSignals(False)

    # Keep the old method name for backward compatibility
    onAutoRangeX = on_auto_range_x

    def on_auto_range_y(self):
        """Set Y axis range to auto values from parent"""
        logging.log(0, "on_auto_range_y")
        self.spinBoxYmin.blockSignals(True)
        self.spinBoxYmax.blockSignals(True)
        self.spinBoxYmin.setValue(self.parent.ymin)
        self.spinBoxYmax.setValue(self.parent.ymax)
        self.spinBoxYmin.blockSignals(False)
        self.spinBoxYmax.blockSignals(False)

    # Keep the old method name for backward compatibility
    onAutoRangeY = on_auto_range_y

    def on_auto_range_z(self):
        """Set Z axis range to auto values from parent"""
        logging.log(0, "on_auto_range_z")
        self.spinBoxZmin.blockSignals(True)
        self.spinBoxZmax.blockSignals(True)
        self.spinBoxZmin.setValue(self.parent.zmin)
        self.spinBoxZmax.setValue(self.parent.zmax)
        self.spinBoxZmin.blockSignals(False)
        self.spinBoxZmax.blockSignals(False)

    # Keep the old method name for backward compatibility
    onAutoRangeZ = on_auto_range_z

    def onSelectionTableClicked(self):
        logging.log(0, "onSelectionTableClicked")
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        # Clear frame histogram cache when selections change
        self.clear_frame_histogram_cache()
        self.clear_histogram_cache()
        self.parent.request_plot_update(skip_clustering=True)

    def addSelection(self, idx, xmin, xmax, invert=False, enabled=True, name=""):
        # Clear frame histogram cache when selections change
        self.clear_frame_histogram_cache()
        self.clear_histogram_cache()
        
        # Ensure xmin < xmax
        if xmin > xmax:
            xmin, xmax = xmax, xmin
            logging.log(0, f"Swapped xmin and xmax to ensure min-max ordering: ({xmin}, {xmax})")

        table = self.tableWidget
        row = table.rowCount()
        table.setRowCount(row + 1)

        tmp = QtWidgets.QTableWidgetItem("%s" % name)
        tmp.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
        tmp.setData(1, idx)
        table.setItem(row, 0, tmp)

        tmp = QtWidgets.QTableWidgetItem()
        tmp.setText(str(xmin))
        tmp.setData(0, float(xmin))
        tmp.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
        tmp.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        tmp.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        font = QtGui.QFont()
        font.setPointSize(10)
        tmp.setFont(font)
        tmp.setTextAlignment(QtCore.Qt.AlignCenter)
        table.setItem(row, 1, tmp)

        tmp = QtWidgets.QTableWidgetItem()
        tmp.setText(str(xmax))
        tmp.setData(0, float(xmax))
        tmp.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
        tmp.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        tmp.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        font = QtGui.QFont()
        font.setPointSize(10)
        tmp.setFont(font)
        tmp.setTextAlignment(QtCore.Qt.AlignCenter)
        table.setItem(row, 2, tmp)

        cb_invert_x = QtWidgets.QCheckBox(table)
        table.setCellWidget(row, 3, cb_invert_x)
        cb_invert_x.setChecked(invert)

        cb_enable_x = QtWidgets.QCheckBox(table)
        table.setCellWidget(row, 4, cb_enable_x)
        cb_enable_x.setChecked(enabled)
        # Fast path for 2D rectangle selection: use batched update to keep UI snappy
        # - request_plot_update() batches rapid selections (40ms timer)
        # - skip_clustering=True avoids expensive clustering recomputation
        # - Cache system naturally detects selection changes and recomputes only when needed
        self.parent.request_plot_update(skip_clustering=True)

        # Actions for selection checkbox
        cb_enable_x.stateChanged.connect(self.actionUpdatePlots.trigger)
        cb_invert_x.stateChanged.connect(self.actionUpdatePlots.trigger)
        logging.log(0, f"Added selection for parameter index {idx} with range ({xmin}, {xmax}), invert={invert}, enabled={enabled}")

    def addGaussianSelection(self, idx1, idx2, mu, cov, sigma=1.0, invert=False, enabled=True, name="", log_x=False, log_y=False):
        table = self.tableWidget
        row = table.rowCount()
        table.setRowCount(row + 1)

        # Column 0: name with metadata
        meta = {
            "type": "G2D",
            "idx1": int(idx1),
            "idx2": int(idx2),
            "mu": [float(mu[0]), float(mu[1])],
            "cov": [
                [float(cov[0][0]), float(cov[0][1])],
                [float(cov[1][0]), float(cov[1][1])]
            ],
            "sigma": float(sigma),
            "log_x": bool(log_x),
            "log_y": bool(log_y)
        }
        item0 = QtWidgets.QTableWidgetItem("%s" % name)
        item0.setFlags(QtCore.Qt.ItemIsEnabled)
        # Keep legacy index role for compatibility (store idx1)
        item0.setData(1, int(idx1))
        try:
            item0.setData(32, json.dumps(meta))  # Qt.UserRole
        except Exception:
            item0.setData(1, int(idx1))
        table.setItem(row, 0, item0)

        # Columns 1 and 2: placeholders (not used by G2D), keep numeric values to avoid parsing errors
        it1 = QtWidgets.QTableWidgetItem()
        it1.setText(str(0.0))
        it1.setData(0, float(0.0))
        it1.setFlags(QtCore.Qt.ItemIsEnabled)
        it1.setTextAlignment(QtCore.Qt.AlignCenter)
        table.setItem(row, 1, it1)

        it2 = QtWidgets.QTableWidgetItem()
        it2.setText(str(0.0))
        it2.setData(0, float(0.0))
        it2.setFlags(QtCore.Qt.ItemIsEnabled)
        it2.setTextAlignment(QtCore.Qt.AlignCenter)
        table.setItem(row, 2, it2)

        # Invert and Enabled checkboxes
        cb_invert = QtWidgets.QCheckBox(table)
        table.setCellWidget(row, 3, cb_invert)
        cb_invert.setChecked(bool(invert))

        cb_enable = QtWidgets.QCheckBox(table)
        table.setCellWidget(row, 4, cb_enable)
        cb_enable.setChecked(bool(enabled))

        # Fast path for 2D Gaussian selection: use batched update to keep UI snappy
        # - request_plot_update() batches rapid selections (40ms timer)
        # - skip_clustering=True avoids expensive clustering recomputation
        # - Cache system naturally detects selection changes and recomputes only when needed
        self.parent.request_plot_update(skip_clustering=True)
        cb_enable.stateChanged.connect(self.actionUpdatePlots.trigger)
        cb_invert.stateChanged.connect(self.actionUpdatePlots.trigger)
        logging.log(0, f"Added G2D selection for idxs ({idx1}, {idx2}) with sigma={sigma}, invert={invert}, enabled={enabled}, log_x={log_x}, log_y={log_y}")

    def addMaskSelection(self, name, mask, edges1, edges2, idx1, idx2, invert=False, enabled=True, selection_id=None):
        """Add a mask-based selection to the table."""
        table = self.tableWidget
        row = table.rowCount()
        
        try:
            self._block_selection_item_changed = True
            table.setRowCount(row + 1)

            # Metadata for MaskDataSelection
            meta = {
                "type": "Mask",
                "name": str(name),
                "idx1": int(idx1),
                "idx2": int(idx2),
                "invert": bool(invert),
                "enabled": bool(enabled),
                "selection_id": str(selection_id) if selection_id else None
            }

            item0 = QtWidgets.QTableWidgetItem(str(name))
            item0.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)
            # Store index for compatibility
            item0.setData(1, int(idx1))
            try:
                # Store metadata redundantly in UserRole AND role 32
                meta_json = json.dumps(meta)
                item0.setData(QtCore.Qt.UserRole, meta_json)
                item0.setData(32, meta_json)
            except Exception as e:
                logging.debug(f"Failed to set metadata for mask selection: {e}")
            table.setItem(row, 0, item0)

            # Placeholders for columns 1 and 2
            for col in [1, 2]:
                tmp = QtWidgets.QTableWidgetItem("Bitmap")
                # Do NOT set data role 0 to 0.0, so onSelectionItemChanged knows it's text
                tmp.setFlags(QtCore.Qt.ItemIsEnabled)
                tmp.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(row, col, tmp)

            # Invert and Enabled checkboxes
            cb_invert = QtWidgets.QCheckBox(table)
            table.setCellWidget(row, 3, cb_invert)
            cb_invert.setChecked(bool(invert))

            cb_enable = QtWidgets.QCheckBox(table)
            table.setCellWidget(row, 4, cb_enable)
            cb_enable.setChecked(bool(enabled))

            # Re-trigger plot update
            self.parent.request_plot_update(skip_clustering=True)
            cb_enable.stateChanged.connect(self.actionUpdatePlots.trigger)
            cb_invert.stateChanged.connect(self.actionUpdatePlots.trigger)
            logging.info(f"Added mask selection UI row: {name} (id={selection_id})")
        finally:
            self._block_selection_item_changed = False

    def onAddSelection(self):
        idx, name = self.p3
        xsel = self.parent.selection_z.get_range()
        xmin = float(min(xsel))
        xmax = float(max(xsel))
        self.addSelection(idx, xmin, xmax, False, True, name)
        logging.log(0, f"onAddSelection: Added selection for {name} with range ({xmin}, {xmax})")
        
        # If in single frame mode, also add frame selection
        self._add_frame_selection_if_needed()
        
        # Preserve contrast during selection operations
        self.parent._preserve_contrast = True
        self.parent.update_plots()
        self.parent._preserve_contrast = False
    def get_selections(self):
        selections = list()
        table = self.tableWidget
        n_rows = int(table.rowCount())
        
        # Log the internal selections list state for debugging
        logging.debug(f"get_selections: table_rows={n_rows}, internal_list_size={len(self._selections)}")
        
        for r in range(n_rows):
            item0 = table.item(r, 0)
            if item0 is None:
                continue
                
            idx = int(item0.data(1))
            name = str(item0.text()) # Use text() directly for comparison
            lower_item = table.item(r, 1)
            upper_item = table.item(r, 2)
            
            # Use safe conversion for lower/upper bounds
            def safe_float(item):
                if item is None:
                    return 0.0
                try:
                    # Try data role 0 first (stored numeric value)
                    data_val = item.data(0)
                    if data_val is not None:
                        try:
                            return float(data_val)
                        except (ValueError, TypeError):
                            pass
                    
                    # Fallback to display text
                    txt = item.text().strip()
                    if txt in ("Bitmap", "Mask", "G2D", "---"):
                        return 0.0
                    return float(txt)
                except (ValueError, TypeError):
                    return 0.0

            lower = safe_float(lower_item)
            upper = safe_float(upper_item)
            
            # Get invert and enabled states from checkboxes
            cb_invert = table.cellWidget(r, 3)
            cb_enable = table.cellWidget(r, 4)
            invert = cb_invert.isChecked() if cb_invert else False
            enabled = cb_enable.isChecked() if cb_enable else True
            logging.debug(f"Row {r} ({name}): invert={invert}, enabled={enabled}")
            
            # Try to decode metadata
            meta = None
            try:
                # Try multiple roles for metadata
                # Use a larger set of roles to be safe
                for role in [QtCore.Qt.UserRole, 32, QtCore.Qt.UserRole + 10, QtCore.Qt.UserRole + 100]:
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
                meta = None
            
            # Identify selection type with heavy fallback
            sel_type = None
            if isinstance(meta, dict):
                sel_type = meta.get("type")
            
            # Fallback identification based on cell text if metadata missing or corrupted
            if sel_type is None:
                txt1 = lower_item.text() if lower_item else ""
                txt2 = upper_item.text() if upper_item else ""
                if "Bitmap" in txt1 or "Bitmap" in txt2 or "Mask" in name:
                    sel_type = "Mask"
                    logging.info(f"Row {r}: Identified as 'Mask' via text fallback (name='{name}')")
                elif "G2D" in name:
                    sel_type = "G2D"
                    logging.info(f"Row {r}: Identified as 'G2D' via text fallback (name='{name}')")

            # RECONSTRUCTION
            if sel_type == "G2D":
                try:
                    idx1 = int(meta.get("idx1", idx)) if meta else idx
                    idx2 = int(meta.get("idx2", idx)) if meta else idx
                    mu = meta.get("mu", [0.0, 0.0]) if meta else [0.0, 0.0]
                    cov = meta.get("cov", [[1.0, 0.0], [0.0, 1.0]]) if meta else [[1.0, 0.0], [0.0, 1.0]]
                    sigma = float(meta.get("sigma", 1.0)) if meta else 1.0
                    log_x = bool(meta.get("log_x", False)) if meta else False
                    log_y = bool(meta.get("log_y", False)) if meta else False
                    
                    selections.append(
                        Gaussian2DSelection(
                            parameter_idx1=idx1,
                            parameter_idx2=idx2,
                            mu=mu,
                            cov=cov,
                            sigma=sigma,
                            invert=invert,
                            enabled=enabled,
                            name=name,
                            log_x=log_x,
                            log_y=log_y
                        )
                    )
                    continue # Success
                except Exception as e:
                    logging.error(f"Error recreating G2D selection '{name}': {e}")
                    # NEVER fall through to rectangular for suspected G2D
                    continue

            elif sel_type == "Mask":
                try:
                    mask_found = False
                    meta_idx1 = int(meta.get('idx1', -1)) if meta else -1
                    meta_idx2 = int(meta.get('idx2', -1)) if meta else -1
                    sel_id = meta.get('selection_id') if meta else None
                    
                    # Recover the MaskDataSelection object from the internal list
                    for s in self._selections:
                        if isinstance(s, MaskDataSelection):
                            # 1. Try matching by selection_id
                            if sel_id and hasattr(s, 'selection_id') and s.selection_id == sel_id:
                                mask_found = True
                            
                            # 2. Fallback to indices and name matching
                            if not mask_found:
                                # If meta is missing, idx matching will use -1, so we rely on name or single-mask assumption
                                idx_match = (s.idx1 == meta_idx1 and s.idx2 == meta_idx2) or (meta is None)
                                # Compare against current table text 'name' and original 'meta_name'
                                name_match = (s.name == name or name.startswith(s.name) or s.name.startswith(name))
                                
                                if idx_match and (name_match or len([x for x in self._selections if isinstance(x, MaskDataSelection)]) == 1):
                                    mask_found = True
                            
                            if mask_found:
                                s.enabled = enabled
                                s.invert = invert
                                s.name = name # Keep in sync with UI
                                selections.append(s)
                                logging.debug(f"Recovered MaskDataSelection object for '{name}' (id={getattr(s, 'selection_id', 'None')[:8]})")
                                break
                    
                    if not mask_found:
                        available = [f"{x.name}(id={getattr(x, 'selection_id', 'None')[:8]}, idxs={x.idx1},{x.idx2})" for x in self._selections if isinstance(x, MaskDataSelection)]
                        logging.warning(f"MaskDataSelection object NOT FOUND for '{name}' (id={sel_id}, idxs={meta_idx1},{meta_idx2}). Available: {available}")
                    
                    continue # CRITICAL: never fall through to rectangular for Mask type
                except Exception as e:
                    logging.error(f"Error retrieving MaskDataSelection '{name}': {e}")
                    continue

            # Default rectangular selection
            selections.append(
                RectangularDataSelection(
                    parameter_idx=idx,
                    lower=lower,
                    upper=upper,
                    invert=invert,
                    enabled=enabled,
                    name=name
                )
            )
        
        logging.debug(f"get_selections: Returning {len(selections)} valid selection objects")
        return selections

    def onSelectionItemChanged(self, item: QtWidgets.QTableWidgetItem):
        """Allow inline editing of rectangular selection bounds and names.
        - Column 0: name (editable)
        - Column 1: lower bound (editable for rectangular selections)
        - Column 2: upper bound (editable for rectangular selections)
        Changing values triggers plot updates.
        """
        if getattr(self, "_block_selection_item_changed", False):
            return
        table = self.tableWidget
        row = item.row()
        col = item.column()
        
        # Guard against invalid row/col or missing items
        if row < 0 or col < 0:
            return
            
        # Get metadata from column 0 to check if this is a special selection type
        item0 = table.item(row, 0)
        if item0 is None:
            return
            
        name = item0.text()
        meta = None
        try:
            # Try multiple roles for metadata
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
            meta = None
        
        sel_type = meta.get("type") if isinstance(meta, dict) else None
        
        # Fallback identification based on name if metadata missing
        if sel_type is None:
            if name.startswith("Mask") or "Bitmap" in name:
                sel_type = "Mask"
            elif name.startswith("G2D"):
                sel_type = "G2D"
                
        is_special = sel_type in ("G2D", "Mask")

        # Name edits: trigger update only
        if col == 0:
            # Preserve contrast during selection operations
            self.parent._preserve_contrast = True
            self.parent.update_plots()
            self.parent._preserve_contrast = False
            return

        # Only columns 1 and 2 are numeric bounds for rectangular selections
        if col not in (1, 2):
            return
            
        if is_special:
            # Revert to stored value or placeholder if accidentally made editable
            try:
                self._block_selection_item_changed = True
                if sel_type == "Mask":
                    item.setText("Bitmap")
                    item.setData(0, None) # Clear any accidental numeric data
                elif sel_type == "G2D":
                    # For G2D, we typically show 0.0 as placeholder
                    val_data = item.data(0)
                    try:
                        val = float(val_data) if val_data is not None else 0.0
                        item.setText(str(val))
                    except (ValueError, TypeError):
                        item.setText("0.0")
                else:
                    item.setText(str(item.data(0) or "---"))
            finally:
                self._block_selection_item_changed = False
            return

        # Parse the edited text as float for normal rectangular selections
        txt = item.text().strip()
        try:
            val = float(txt)
        except Exception:
            # Revert to previous value stored in data role 0
            try:
                self._block_selection_item_changed = True
                prev_data = item.data(0)
                if prev_data is not None:
                    try:
                        prev = float(prev_data)
                        item.setText(str(prev))
                    except (ValueError, TypeError):
                        item.setText(str(prev_data))
                else:
                    item.setText("0.0")
            finally:
                self._block_selection_item_changed = False
            return

        # Commit the numeric value
        try:
            self._block_selection_item_changed = True
            item.setData(0, float(val))
            # Enforce ordering lower <= upper by adjusting the sibling cell
            lower_item = table.item(row, 1)
            upper_item = table.item(row, 2)
            try:
                lower = float(lower_item.data(0)) if lower_item is not None else float("nan")
            except Exception:
                lower = float("nan")
            try:
                upper = float(upper_item.data(0)) if upper_item is not None else float("nan")
            except Exception:
                upper = float("nan")

            if col == 1 and not math.isnan(upper) and val > upper:
                upper_item.setData(0, float(val))
                upper_item.setText(str(float(val)))
            elif col == 2 and not math.isnan(lower) and val < lower:
                lower_item.setData(0, float(val))
                lower_item.setText(str(float(val)))
        finally:
            self._block_selection_item_changed = False

        # Trigger plot update (only when data is ready)
        if hasattr(self.parent, 'is_data_ready') and not self.parent.is_data_ready():
            # If data isn't ready, revert numeric edits and skip replot
            if col in (1, 2):
                try:
                    self._block_selection_item_changed = True
                    prev = float(item.data(0)) if item.data(0) is not None else 0.0
                    item.setText(str(prev))
                finally:
                    self._block_selection_item_changed = False
                logging.info("Selection edit ignored: load data before editing selections.")
            # For name edits (col 0), accept but skip replot
            return
        # Preserve contrast during selection operations
        self.parent._preserve_contrast = True
        self.parent.update_plots()
        self.parent._preserve_contrast = False

    def onDeleteSelectionRows(self):
        """Delete selected selection rows using the Delete key."""
        table = self.tableWidget
        sel_model = table.selectionModel()
        if sel_model is None:
            return
        rows = sorted({idx.row() for idx in sel_model.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < table.rowCount():
                table.removeRow(r)
        # Preserve contrast during selection operations
        self.parent._preserve_contrast = True
        self.parent.update_plots()
        self.parent._preserve_contrast = False

    def onSelectionTableContextMenu(self, position):
        """Show context menu for selection table."""
        table = self.tableWidget
        # Create context menu
        menu = QtWidgets.QMenu(self)
        
        # Add actions
        select_all_action = menu.addAction("Select All")
        clear_action = menu.addAction("Clear")
        delete_action = menu.addAction("Delete")
        
        # Show menu and get action
        action = menu.exec_(table.mapToGlobal(position))
        
        # Handle action
        if action == select_all_action:
            self.onSelectAllSelections()
        elif action == clear_action:
            self.onClearSelection()
        elif action == delete_action:
            self.onDeleteSelectionRows()

    def onSelectAllSelections(self):
        """Select all rows in the selection table."""
        table = self.tableWidget
        table.selectAll()
        logging.log(0, "All selection rows selected")

    def onSelectionCellClicked(self, row: int, col: int):
        """Handle single-click on a cell to start inline editing after the
        double-click interval has elapsed (so double-click can still delete).
        """
        # Store pending index for single-click editing
        try:
            model = self.tableWidget.model()
            if model is None:
                return
            self._pending_edit_index = model.index(row, col)
            # Start a timer equal to the system double-click interval
            app = QtWidgets.QApplication.instance()
            dci = app.doubleClickInterval() if app is not None else 250
            self._single_click_edit_timer.start(int(dci))
        except Exception:
            # Fallback: start soon
            self._single_click_edit_timer.start(200)

    def onSelectionCellDoubleClicked(self, row: int, col: int):
        """Cancel pending single-click edit when a double-click occurs.
        The actual deletion is handled by the .ui connection to
        actionSelectionTableClicked.
        """
        try:
            if self._single_click_edit_timer.isActive():
                self._single_click_edit_timer.stop()
        except Exception:
            pass
        self._pending_edit_index = None

    def _perform_pending_single_click_edit(self):
        """If there is a pending single-click index, start editing it.
        Only allow editing of rectangular selection fields:
        - Column 0 (name) editable
        - Columns 1 and 2 (lower/upper) editable
        - For Gaussian2D rows, columns 1 and 2 are not editable.
        """
        index = getattr(self, "_pending_edit_index", None)
        self._pending_edit_index = None
        if index is None or not index.isValid():
            return
        row = index.row()
        col = index.column()

        # Determine if this row is G2D
        item0 = self.tableWidget.item(row, 0)
        meta = None
        try:
            meta_raw = item0.data(32)
            if meta_raw:
                meta = json.loads(meta_raw)
        except Exception:
            meta = None
        is_g2d = isinstance(meta, dict) and meta.get("type") == "G2D"

        # Permissions: name (col 0) always allowed for rectangular; G2D name not editable per flags
        if col == 0:
            # Try to edit if the item is editable by flags
            it = self.tableWidget.item(row, col)
            if it is not None and (it.flags() & QtCore.Qt.ItemIsEditable):
                self.tableWidget.edit(index)
            return

        # Bounds columns 1 and 2: only for rectangular selections
        if col in (1, 2) and not is_g2d:
            it = self.tableWidget.item(row, col)
            if it is not None and (it.flags() & QtCore.Qt.ItemIsEditable):
                self.tableWidget.edit(index)
            return
        # Otherwise, do nothing (non-editable)
        return

    def setup_frame_selection(self, frame_param: str, n_frames: int):
        """
        Setup frame selection UI for time series or z-stack images.
        
        Args:
            frame_param: Name of the frame parameter (T pixel or Z pixel)
            n_frames: Total number of frames in the stack
        """
        self._frame_param = frame_param
        self._n_frames = n_frames
        
        # Initialize per-frame histogram cache for time series playback
        self._frame_histogram_cache = {}
        
        self.labelFrameInfo.setText(f"{frame_param}: ")
        self.spinBoxFrameNumber.setMaximum(n_frames - 1)
        self.spinBoxFrameNumber.setValue(0)
        
        # Update frame count label
        if hasattr(self, 'label_frame_nbr'):
            self.label_frame_nbr.setText(f"/{n_frames}")
        
        try:
            self.checkBoxStackFrames.toggled.disconnect()
        except Exception:
            pass
        try:
            self.spinBoxFrameNumber.valueChanged.disconnect()
        except Exception:
            pass
            
        self.checkBoxStackFrames.toggled.connect(self.on_frame_selection_changed)
        self.spinBoxFrameNumber.valueChanged.connect(self.on_frame_selection_changed)
        
        # Show image/frame controls as a single group for image datasets
        if hasattr(self, 'groupBoxImage'):
            self.groupBoxImage.setVisible(True)
        
        logging.info(f"Frame selection setup: {frame_param} with {n_frames} frames")

    def hide_frame_selection(self):
        """Hide frame selection UI when no frame stack is detected."""
        self._frame_param = None
        self._n_frames = 0
        self._frame_histogram_cache = {}
        self._stop_playback()

        if hasattr(self, 'checkBoxStackFrames'):
            self.checkBoxStackFrames.setChecked(True)
        if hasattr(self, 'spinBoxFrameNumber'):
            self.spinBoxFrameNumber.setValue(0)

        # Hide image/frame controls as a single group for non-image datasets
        if hasattr(self, 'groupBoxImage'):
            self.groupBoxImage.setVisible(False)

    def on_frame_selection_changed(self):
        """Handle frame selection changes and trigger plot update."""
        if not hasattr(self, '_frame_param') or self._frame_param is None:
            return
            
        if self.checkBoxStackFrames.isChecked():
            logging.debug("Stack frames enabled (showing all frames)")
            # Clear cache when switching to stacked mode
            self._frame_histogram_cache = {}
            self._stop_playback()
        else:
            frame_num = self.spinBoxFrameNumber.value()
            logging.debug(f"Single frame mode: showing frame {frame_num}")
            # Always compute live histograms (no caching)
            
        self.parent.request_plot_update()

    def _add_frame_selection_if_needed(self):
        """
        Add frame selection to the selection table when in single frame mode.
        This ensures that when users make selections, the current frame is included.
        """
        if not hasattr(self, '_frame_param') or self._frame_param is None:
            return
            
        if self.checkBoxStackFrames.isChecked():
            return
            
        try:
            # Check if frame selection already exists
            param_names = self.parent.data_source.parameter_names
            if self._frame_param not in param_names:
                return
                
            frame_idx = param_names.index(self._frame_param)
            frame_num = self.spinBoxFrameNumber.value()
            
            # Check if this selection already exists
            for sel in self.get_selections():
                if hasattr(sel, 'idx') and sel.idx == frame_idx:
                    # Frame selection already exists, update it
                    if hasattr(sel, 'lower') and hasattr(sel, 'upper'):
                        if sel.lower == frame_num and sel.upper == frame_num:
                            return
            
            # Add frame selection
            self.addSelection(frame_idx, frame_num, frame_num, False, True, self._frame_param)
            logging.info(f"Added frame selection: {self._frame_param} = {frame_num}")
        except Exception as e:
            logging.warning(f"Failed to add frame selection: {e}")

    def get_frame_filter_mask(self, data_source):
        """
        Get a boolean mask for filtering data by selected frame.
        
        Args:
            data_source: The data source containing parameter values
            
        Returns:
            numpy array of boolean values or None if frame filtering is disabled
        """
        if not hasattr(self, '_frame_param') or self._frame_param is None:
            return None
            
        if self.checkBoxStackFrames.isChecked():
            return None
            
        try:
            param_names = data_source.parameter_names
            if self._frame_param not in param_names:
                return None
                
            frame_idx = param_names.index(self._frame_param)
            frame_values = data_source.values[frame_idx, :]
            selected_frame = self.spinBoxFrameNumber.value()
            
            import numpy as np
            mask = frame_values == selected_frame
            logging.debug(f"Frame filter mask: {mask.sum()} events in frame {selected_frame}")
            return mask
        except Exception as e:
            logging.warning(f"Failed to create frame filter mask: {e}")
            return None

    # ==================== Settings and Configuration ====================
    
    def _load_playback_settings(self):
        """Load playback settings from settings file."""
        try:
            settings_path = pathlib.Path(__file__).parent.parent / 'settings' / 'mfd.settings.json'
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                
                playback_settings = settings.get('playback', {})
                self._frame_duration_ms = playback_settings.get('frame_duration_ms', 200)
                # Only enable background computation if both env var and settings allow it
                env_enabled = is_background_computation_enabled()
                settings_enabled = playback_settings.get('enable_background_computation', True)
                self._background_computation_enabled = env_enabled and settings_enabled
                
                # Configure cache based on settings
                cache_size_mb = playback_settings.get('cache_size_mb', 100)
                cache_max_entries = playback_settings.get('cache_max_entries', 50)
                self._histogram_cache = EnhancedHistogramCache(
                    max_size=cache_max_entries,
                    max_memory_mb=cache_size_mb
                )
                
                logging.info(f"Loaded playback settings: duration={self._frame_duration_ms}ms, "
                           f"background={self._background_computation_enabled}, "
                           f"cache={cache_size_mb}MB")
            else:
                # Default settings
                self._frame_duration_ms = 200
                self._background_computation_enabled = is_background_computation_enabled()
                logging.warning("Playback settings file not found, using defaults")
        except Exception as e:
            logging.warning(f"Failed to load playback settings: {e}")
            # Fallback to defaults
            self._frame_duration_ms = 200
            self._background_computation_enabled = is_background_computation_enabled()
    
    def update_playback_settings(self, frame_duration_ms: int = None, 
                                enable_background: bool = None,
                                cache_size_mb: int = None,
                                cache_max_entries: int = None):
        """Update playback settings and reconfigure components."""
        if frame_duration_ms is not None:
            self._frame_duration_ms = frame_duration_ms
            if hasattr(self, '_playback_timer'):
                self._playback_timer.setInterval(self._frame_duration_ms)
        
        if enable_background is not None:
            # Only enable if both env var and parameter allow it
            env_enabled = is_background_computation_enabled()
            self._background_computation_enabled = enable_background and env_enabled
        
        if cache_size_mb is not None or cache_max_entries is not None:
            # Recreate cache with new settings
            old_cache = self._histogram_cache
            self._histogram_cache = EnhancedHistogramCache(
                max_size=cache_max_entries or old_cache.max_size,
                max_memory_mb=cache_size_mb or (old_cache.max_memory_bytes / 1024 / 1024)
            )
        
        logging.info(f"Updated playback settings: duration={self._frame_duration_ms}ms, "
                   f"background={self._background_computation_enabled}")
    
    # ==================== Time Series Playback Controls ====================
    
    def _setup_playback_controls(self):
        """Setup playback control buttons - connect signals and create timer."""
        self.toolButtonStepBackward.clicked.connect(self._on_step_backward)
        self.toolButtonPlayBackward.clicked.connect(self._on_play_backward)
        self.toolButtonPause.clicked.connect(self._on_pause)
        self.toolButtonPlayForward.clicked.connect(self._on_play_forward)
        self.toolButtonStepForward.clicked.connect(self._on_step_forward)
        
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(self._frame_duration_ms)  # Use settings value
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_direction = 0
        
    def _on_play_backward(self):
        """Start playing backward through frames."""
        if self.toolButtonPlayBackward.isChecked():
            self._playback_direction = -1
            self.toolButtonPlayForward.setChecked(False)
            self._playback_timer.start()
            logging.debug("Started backward playback")
        else:
            self._stop_playback()
            
    def _on_play_forward(self):
        """Start playing forward through frames."""
        if self.toolButtonPlayForward.isChecked():
            self._playback_direction = 1
            self.toolButtonPlayBackward.setChecked(False)
            self._playback_timer.start()
            logging.debug("Started forward playback")
        else:
            self._stop_playback()
            
    def _on_pause(self):
        """Pause playback."""
        self._stop_playback()
        
    def _on_step_backward(self):
        """Step one frame backward."""
        if not hasattr(self, '_frame_param') or self._frame_param is None:
            return
            
        if self.checkBoxStackFrames.isChecked():
            return
            
        current = self.spinBoxFrameNumber.value()
        n_frames = getattr(self, '_n_frames', 0)
        
        # Step backward with loop
        new_frame = current - 1
        if new_frame < 0:
            new_frame = n_frames - 1
            
        self.spinBoxFrameNumber.setValue(new_frame)
        logging.debug(f"Stepped backward to frame {new_frame}")
        
    def _on_step_forward(self):
        """Step one frame forward."""
        if not hasattr(self, '_frame_param') or self._frame_param is None:
            return
            
        if self.checkBoxStackFrames.isChecked():
            return
            
        current = self.spinBoxFrameNumber.value()
        n_frames = getattr(self, '_n_frames', 0)
        
        # Step forward with loop
        new_frame = current + 1
        if new_frame >= n_frames:
            new_frame = 0
            
        self.spinBoxFrameNumber.setValue(new_frame)
        logging.debug(f"Stepped forward to frame {new_frame}")
        
    def _stop_playback(self):
        """Stop any active playback."""
        if hasattr(self, '_playback_timer'):
            self._playback_timer.stop()
        self._playback_direction = 0
        self.toolButtonPlayBackward.setChecked(False)
        self.toolButtonPlayForward.setChecked(False)
        
    def _on_playback_tick(self):
        """Handle playback timer tick - advance to next/previous frame."""
        if not hasattr(self, '_frame_param') or self._frame_param is None:
            self._stop_playback()
            return
            
        if self.checkBoxStackFrames.isChecked():
            self._stop_playback()
            return
            
        current = self.spinBoxFrameNumber.value()
        n_frames = getattr(self, '_n_frames', 0)
        
        if self._playback_direction > 0:
            # Forward
            new_frame = current + 1
            if new_frame >= n_frames:
                new_frame = 0  # Loop
        elif self._playback_direction < 0:
            # Backward
            new_frame = current - 1
            if new_frame < 0:
                new_frame = n_frames - 1  # Loop
        else:
            return
            
        self.spinBoxFrameNumber.setValue(new_frame)
        
    # ==================== Background Histogram Computation ====================
    
    def _initialize_histogram_worker(self):
        """Initialize the background histogram computation worker."""
        if self._histogram_worker is None:
            self._histogram_worker = HistogramComputationManager(self)
            self._histogram_worker.computation_complete.connect(self._on_histograms_computed)
            self._histogram_worker.computation_failed.connect(self._on_histogram_computation_failed)
            self._histogram_worker.computation_started.connect(self._on_computation_started)
            self._histogram_worker.progress_update.connect(self._on_computation_progress)
    
    def compute_histograms_background(self, histogram_params: dict, weights: Optional[np.ndarray] = None):
        """Compute histograms in background thread if enabled, otherwise compute immediately."""
        logging.info(f"compute_histograms_background called, enabled={self._background_computation_enabled}")
        if not self._background_computation_enabled:
            logging.debug("Background computation disabled, using immediate computation")
            return self._compute_histograms_immediate(histogram_params, weights)
            
        if not hasattr(self.parent, 'data_source'):
            logging.warning("No data source available, cannot compute histograms")
            return
        
        # Check if computation is already pending
        if self._background_computation_pending:
            logging.info("Background computation already pending, skipping duplicate request")
            return
        
        # Initialize worker if needed
        self._initialize_histogram_worker()
        
        # Always do live computation (no caching)
        # Schedule background computation
        try:
            logging.info("Starting background histogram computation")
            logging.info(f"  data_source: {self.parent.data_source}")
            logging.info(f"  histogram_params keys: {list(histogram_params.keys())}")
            logging.info(f"  weights shape: {weights.shape if weights is not None else None}")
            logging.info(f"  BIN SETTINGS: x_bins_2d={histogram_params.get('x_bins_2d')}, y_bins_2d={histogram_params.get('y_bins_2d')}")
            logging.info(f"  BIN ARRAYS: x_bins_2d_arr length={len(histogram_params.get('x_bins_2d_arr', []))}, y_bins_2d_arr length={len(histogram_params.get('y_bins_2d_arr', []))}")
            # Set flag to prevent immediate computation fallback
            self._background_computation_pending = True
            self._histogram_worker.compute_histograms(
                self.parent.data_source,
                histogram_params,
                weights
            )
            logging.info("Background computation scheduled successfully")
        except Exception as e:
            logging.error(f"Failed to schedule background histogram computation: {e}")
            logging.info("Falling back to immediate computation")
            self._background_computation_pending = False
            self._compute_histograms_immediate(histogram_params, weights)
    
    def _compute_histograms_immediate(self, histogram_params: dict, weights: Optional[np.ndarray] = None):
        """Compute histograms immediately in the main thread."""
        try:
            # Import here to avoid circular imports
            from ..utils.histogram_computation import compute_histograms_sync
            
            result = compute_histograms_sync(
                self.parent.data_source,
                histogram_params,
                weights
            )
            self._on_histograms_computed(result)
        except Exception as e:
            logging.error(f"Immediate histogram computation failed: {e}")
            self._on_histogram_computation_failed(str(e))
    
    def _on_histograms_computed(self, histogram_data: dict):
        """Handle completion of histogram computation (background or immediate)."""
        try:
            logging.info(f"[UI HANDLER] _on_histograms_computed called with {len(histogram_data)} items")
            logging.info("Background histogram computation completed - updating UI")
            
            # Log 2D histogram shape from background
            if '2d' in histogram_data:
                hist_2d = histogram_data['2d']
                if isinstance(hist_2d, tuple) and len(hist_2d) == 3:
                    H, x_edges, y_edges = hist_2d
                    logging.info(f"[UI] Received 2D histogram from background: H shape={H.shape}, x_edges={len(x_edges)}, y_edges={len(y_edges)}")
                    logging.info(f"[UI] H dtype={H.dtype}, contiguous={H.flags['C_CONTIGUOUS']}, min={np.min(H)}, max={np.max(H)}, sum={np.sum(H)}")
            
            # Clear background computation pending flag
            self._background_computation_pending = False
            
            # Update the parent's histogram data
            self.parent._histogram = histogram_data
            logging.info(f"[UI] Stored histogram in parent._histogram")
            
            # Update the UI
            if '_count' in histogram_data:
                self.parent.lineEditCountCurrent.setText(str(histogram_data['_count']))
            
            # Clear progress indicator
            if hasattr(self.parent, 'statusBar') and self.parent.statusBar():
                self.parent.statusBar().showMessage("Ready", 2000)

            # Update 2D plot and apply auto-contrast
            if hasattr(self.parent, 'update_2d_plot'):
                logging.info("[UI] Calling update_2d_plot to display histogram")
                self.parent.update_2d_plot()
            
            if hasattr(self.parent, 'on_auto_contrast'):
                logging.info("[UI] Applying auto-contrast")
                self.parent.on_auto_contrast()
            
            # Ensure plots repaint immediately (background path otherwise may only refresh on resize)
            try:
                from . import plot_update_helpers

                if 'x' in histogram_data and hasattr(self.parent, 'g_xplot'):
                    x_edges, x_counts = histogram_data['x']
                    plot_update_helpers._autoscale_horizontal_hist(self.parent.g_xplot, x_edges, x_counts)
                if 'y' in histogram_data and hasattr(self.parent, 'g_yplot'):
                    y_edges, y_counts = histogram_data['y']
                    plot_update_helpers._autoscale_vertical_hist(self.parent.g_yplot, y_edges, y_counts)
                if 'z' in histogram_data and hasattr(self.parent, 'g_zplot'):
                    z_edges, z_counts = histogram_data['z']
                    plot_update_helpers._autoscale_horizontal_hist(self.parent.g_zplot, z_edges, z_counts)

                # Refresh 2D image and replot all
                self.parent.update_2d_plot()
                self.parent.g_xplot.replot()
                self.parent.g_yplot.replot()
                if hasattr(self.parent, 'g_zplot'):
                    self.parent.g_zplot.replot()
                self.parent.g_2dplot.replot()
            except Exception as exc:
                logging.debug("Could not force replot after histogram computation: %s", exc)
            
            # No frame caching - always compute live
            
            logging.debug(f"Updated histogram displays (computation time: {histogram_data.get('_computation_time', 'N/A'):.3f}s)")
            
        except Exception as e:
            logging.error(f"Failed to update histogram displays: {e}")
    
    def _on_histogram_computation_failed(self, error_message: str):
        """Handle failure of histogram computation."""
        logging.error(f"Histogram computation failed: {error_message}")
        # Clear progress indicator
        if hasattr(self.parent, 'statusBar') and self.parent.statusBar():
            self.parent.statusBar().clearMessage()
    
    def _on_computation_started(self):
        """Handle start of background computation."""
        if hasattr(self.parent, 'statusBar') and self.parent.statusBar():
            self.parent.statusBar().showMessage("Computing histograms...")
    
    def _on_computation_progress(self, message: str):
        """Handle progress update from background computation."""
        if hasattr(self.parent, 'statusBar') and self.parent.statusBar():
            self.parent.statusBar().showMessage(message)
    
    def _update_histogram_displays_from_data(self, histogram_data: dict):
        """Update histogram plot widgets from computed data."""
        try:
            # Update 2D histogram
            if '2d' in histogram_data:
                logging.info("[UI] Updating 2D histogram display from background computation")
                # The 2D histogram is already stored in parent._histogram by _on_histograms_computed
                # Call update_2d_plot to properly render it
                if hasattr(self.parent, 'update_2d_plot'):
                    try:
                        self.parent.update_2d_plot()
                        logging.info("[UI] Called update_2d_plot to render 2D histogram")
                    except Exception as e:
                        logging.debug(f"Could not update 2D plot: {e}")
                
                # Also replot the 2D plot to ensure it's displayed
                if hasattr(self.parent, 'g_2dplot'):
                    try:
                        self.parent.g_2dplot.replot()
                        logging.info("[UI] Replotted 2D histogram")
                    except Exception as e:
                        logging.debug(f"Could not replot 2D histogram: {e}")
            
            # Update X histogram
            if 'x' in histogram_data and hasattr(self.parent, 'g_xhist_m'):
                x_bin_edges, x_counts = histogram_data['x']
                x_edges_for_plot = x_bin_edges[1:]
                self.parent.g_xhist_m.set_data(x_edges_for_plot, x_counts)
                if hasattr(self.parent, 'g_xplot'):
                    self.parent.g_xplot.replot()
            
            # Update Y histogram
            if 'y' in histogram_data and hasattr(self.parent, 'g_yhist_m'):
                y_bin_edges, y_counts = histogram_data['y']
                y_edges_for_plot = y_bin_edges[1:]
                # For Y marginal: counts on X-axis (horizontal), edges on Y-axis (vertical)
                self.parent.g_yhist_m.set_data(y_counts, y_edges_for_plot)
                if hasattr(self.parent, 'g_yplot'):
                    self.parent.g_yplot.replot()
            
            # Update Z histogram
            if ('z' in histogram_data and hasattr(self.parent, 'g_zhist_m') and 
                hasattr(self.parent, 'groupBox_3') and self.parent.groupBox_3.isChecked()):
                z_bin_edges, z_counts = histogram_data['z']
                z_edges_for_plot = z_bin_edges[1:]
                self.parent.g_zhist_m.set_data(z_edges_for_plot, z_counts)
                if hasattr(self.parent, 'g_zplot'):
                    self.parent.g_zplot.replot()
                
        except Exception as e:
            logging.error(f"Failed to update histogram displays: {e}")

    def clear_frame_histogram_cache(self):
        """Clear the frame-specific histogram cache."""
        if hasattr(self, '_frame_histogram_cache'):
            self._frame_histogram_cache.clear()
            logging.debug("Cleared frame histogram cache")

    def clear_histogram_cache(self):
        """Clear the main histogram cache."""
        if hasattr(self, '_histogram_cache') and self._histogram_cache is not None:
            self._histogram_cache.clear()
            logging.debug("Cleared main histogram cache")
