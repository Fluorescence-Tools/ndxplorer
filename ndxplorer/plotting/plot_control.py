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

from ..core.data_source import RectangularDataSelection, Gaussian2DSelection
from ..logging_config import logging
from .background_histograms import HistogramComputeWorker, EnhancedHistogramCache


class SurfacePlotWidget(QtWidgets.QWidget):

    _selections = list()  # type: List[RectangularDataSelection]
    axis_settings = dict()  # type: Dict[str, Dict[str, float]]

    @property
    def scale_x(self):
        if bool(self.checkBoxLogX.isChecked()):
            return "log"
        else:
            return "lin"

    @scale_x.setter
    def scale_x(self, v):
        if v == "log":
            self.checkBoxLogX.setChecked(True)
        else:
            self.checkBoxLogX.setChecked(False)

    @property
    def scale_y(self):
        if bool(self.checkBoxLogY.isChecked()):
            return "log"
        else:
            return "lin"

    @scale_y.setter
    def scale_y(self, v):
        if v == "log":
            self.checkBoxLogY.setChecked(True)
        else:
            self.checkBoxLogY.setChecked(False)

    @property
    def scale_z(self):
        if self.checkBoxLogZ.isChecked():
            return "log"
        else:
            return "lin"

    @scale_z.setter
    def scale_z(self, v):
        if v == "log":
            self.checkBoxLogZ.setChecked(True)
        else:
            self.checkBoxLogZ.setChecked(False)

    @property
    def normed_hist_x(self):
        return bool(self.checkBoxNormX.isChecked())

    @property
    def normed_hist_y(self):
        return bool(self.checkBoxNormY.isChecked())

    @property
    def normed_hist_z(self):
        return bool(self.checkBoxNormZ.isChecked())
        
    @property
    def weight_enabled(self):
        """Returns whether weighting is enabled."""
        return bool(self.checkBoxWeight.isChecked())
        
    @weight_enabled.setter
    def weight_enabled(self, value, block_signals=False):
        """
        Set whether weighting is enabled.
        
        Args:
            value: Boolean indicating whether weighting should be enabled
            block_signals: If True, signals will be blocked during the change
        """
        # Handle the block_signals parameter
        was_blocked = self.checkBoxWeight.signalsBlocked()
        if block_signals and not was_blocked:
            self.checkBoxWeight.blockSignals(True)
            
        try:
            self.checkBoxWeight.setChecked(bool(value))
        finally:
            # Restore previous signal blocking state
            if block_signals and not was_blocked:
                self.checkBoxWeight.blockSignals(was_blocked)
                
    @property
    def weight_parameter(self):
        """Returns the currently selected weight parameter."""
        return str(self.comboBoxWeight.currentText())
        
    @weight_parameter.setter
    def weight_parameter(self, value, block_signals=False):
        """
        Set the weight parameter.
        
        Args:
            value: Parameter name (str) to use for weighting or index (int)
            block_signals: If True, signals will be blocked during the change
        """
        # Handle the block_signals parameter
        was_blocked = self.comboBoxWeight.signalsBlocked()
        if block_signals and not was_blocked:
            self.comboBoxWeight.blockSignals(True)
            
        try:
            if isinstance(value, str):
                index = self.comboBoxWeight.findText(value)
                if index >= 0:
                    self.comboBoxWeight.setCurrentIndex(index)
            elif isinstance(value, int) and value >= 0:
                self.comboBoxWeight.setCurrentIndex(value)
        finally:
            # Restore previous signal blocking state
            if block_signals and not was_blocked:
                self.comboBoxWeight.blockSignals(was_blocked)

    @property
    def p1(self):
        idx = self.comboBoxSelX.currentIndex()
        name = self.comboBoxSelX.currentText()
        return idx, str(name)
        
    @p1.setter
    def p1(self, value, block_signals=False):
        """
        Set the X axis parameter.
        
        Args:
            value: Either an index (int) or parameter name (str) or tuple (idx, name)
            block_signals: If True, signals will be blocked during the change
        """
        # Handle the block_signals parameter
        was_blocked = self.comboBoxSelX.signalsBlocked()
        if block_signals and not was_blocked:
            self.comboBoxSelX.blockSignals(True)
            
        try:
            if isinstance(value, tuple) and len(value) == 2:
                # If a tuple (idx, name) is provided
                idx, name = value
                if isinstance(idx, int) and idx >= 0:
                    self.comboBoxSelX.setCurrentIndex(idx)
                elif isinstance(name, str):
                    index = self.comboBoxSelX.findText(name)
                    if index >= 0:
                        self.comboBoxSelX.setCurrentIndex(index)
            elif isinstance(value, int) and value >= 0:
                # If just an index is provided
                self.comboBoxSelX.setCurrentIndex(value)
            elif isinstance(value, str):
                # If just a name is provided
                index = self.comboBoxSelX.findText(value)
                if index >= 0:
                    self.comboBoxSelX.setCurrentIndex(index)
        finally:
            # Restore previous signal blocking state
            if block_signals and not was_blocked:
                self.comboBoxSelX.blockSignals(was_blocked)

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
        # Handle the block_signals parameter
        was_blocked = self.comboBoxSelY.signalsBlocked()
        if block_signals and not was_blocked:
            self.comboBoxSelY.blockSignals(True)
            
        try:
            if isinstance(value, tuple) and len(value) == 2:
                # If a tuple (idx, name) is provided
                idx, name = value
                if isinstance(idx, int) and idx >= 0:
                    self.comboBoxSelY.setCurrentIndex(idx)
                elif isinstance(name, str):
                    index = self.comboBoxSelY.findText(name)
                    if index >= 0:
                        self.comboBoxSelY.setCurrentIndex(index)
            elif isinstance(value, int) and value >= 0:
                # If just an index is provided
                self.comboBoxSelY.setCurrentIndex(value)
            elif isinstance(value, str):
                # If just a name is provided
                index = self.comboBoxSelY.findText(value)
                if index >= 0:
                    self.comboBoxSelY.setCurrentIndex(index)
        finally:
            # Restore previous signal blocking state
            if block_signals and not was_blocked:
                self.comboBoxSelY.blockSignals(was_blocked)

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
        # Handle the block_signals parameter
        was_blocked = self.comboBoxSelZ.signalsBlocked()
        if block_signals and not was_blocked:
            self.comboBoxSelZ.blockSignals(True)
            
        try:
            if isinstance(value, tuple) and len(value) == 2:
                # If a tuple (idx, name) is provided
                idx, name = value
                if isinstance(idx, int) and idx >= 0:
                    self.comboBoxSelZ.setCurrentIndex(idx)
                elif isinstance(name, str):
                    index = self.comboBoxSelZ.findText(name)
                    if index >= 0:
                        self.comboBoxSelZ.setCurrentIndex(index)
            elif isinstance(value, int) and value >= 0:
                # If just an index is provided
                self.comboBoxSelZ.setCurrentIndex(value)
            elif isinstance(value, str):
                # If just a name is provided
                index = self.comboBoxSelZ.findText(value)
                if index >= 0:
                    self.comboBoxSelZ.setCurrentIndex(index)
        finally:
            # Restore previous signal blocking state
            if block_signals and not was_blocked:
                self.comboBoxSelZ.blockSignals(was_blocked)
                
    def set_axis_by_name(self, axis, name, match_contains=True, block_signals=False):
        """
        Set an axis by parameter name with optional substring matching.
        
        Args:
            axis: String indicating which axis to set ('x', 'y', or 'z')
            name: Parameter name to set
            match_contains: If True, use Qt.MatchContains to find partial matches
            block_signals: If True, signals will be blocked during the change
            
        Returns:
            bool: True if the axis was successfully set, False otherwise
        """

        if not isinstance(name, str) or not name:
            return False
            
        # Determine which combo box to use based on the axis
        combo_box = None
        if axis.lower() == 'x':
            combo_box = self.comboBoxSelX
        elif axis.lower() == 'y':
            combo_box = self.comboBoxSelY
        elif axis.lower() == 'z':
            combo_box = self.comboBoxSelZ
        elif axis.lower() == 'weight':
            combo_box = self.comboBoxWeight
        else:
            return False
            
        # Find the index of the parameter name
        match_flag = QtCore.Qt.MatchContains if match_contains else QtCore.Qt.MatchExactly
        index = combo_box.findText(name, match_flag)
        
        if index < 0:
            return False
            
        # Block signals if requested
        was_blocked = combo_box.signalsBlocked()
        if block_signals and not was_blocked:
            combo_box.blockSignals(True)
            
        try:
            # Set the combo box to the found index
            combo_box.setCurrentIndex(index)
            return True
        finally:
            # Restore previous signal blocking state
            if block_signals and not was_blocked:
                combo_box.blockSignals(was_blocked)

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
        logging.log(0, "Initializing SurfacePlotWidget")
        #########################
        # GUI
        #########################
        ui_file = pathlib.Path(__file__).parent / 'plot_control.ui'
        
        if PYQTGRAPH_AVAILABLE and SpinBox is not None:
            self.spinBoxXmin = SpinBox()
            self.spinBoxXmax = SpinBox()
            self.spinBoxYmin = SpinBox()
            self.spinBoxYmax = SpinBox()
            self.spinBoxZmin = SpinBox()
            self.spinBoxZmax = SpinBox()
        else:
            # Fallback to regular QSpinBox if pyqtgraph is not available
            logging.warning("pyqtgraph SpinBox not available, using QSpinBox fallback")
            self.spinBoxXmin = QtWidgets.QSpinBox()
            self.spinBoxXmax = QtWidgets.QSpinBox()
            self.spinBoxYmin = QtWidgets.QSpinBox()
            self.spinBoxYmax = QtWidgets.QSpinBox()
            self.spinBoxZmin = QtWidgets.QSpinBox()
            self.spinBoxZmax = QtWidgets.QSpinBox()
            
            # Configure fallback spinboxes
            for sb in [self.spinBoxXmin, self.spinBoxXmax, self.spinBoxYmin, 
                      self.spinBoxYmax, self.spinBoxZmin, self.spinBoxZmax]:
                sb.setRange(-1000000, 1000000)

        uic.loadUi(str(ui_file.as_posix()), self)
        self.horizontalLayout.addWidget(self.spinBoxXmin)
        self.horizontalLayout.addWidget(self.spinBoxXmax)
        self.horizontalLayout_2.addWidget(self.spinBoxYmin)
        self.horizontalLayout_2.addWidget(self.spinBoxYmax)
        self.horizontalLayout_3.addWidget(self.spinBoxZmin)
        self.horizontalLayout_3.addWidget(self.spinBoxZmax)

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
        self.actionUpdatePlots.triggered.connect(self.parent.request_plot_update)

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
        
        # Initialize frame/time series related attributes
        self._frame_param = None
        self._n_frames = 0
        self._frame_histogram_cache = {}
        self._playback_direction = 0
        
        # Initialize background computation and enhanced caching
        self._histogram_worker = None
        self._histogram_cache = EnhancedHistogramCache()
        self._background_computation_enabled = True
        self._frame_duration_ms = 200  # Default value
        self._load_playback_settings()
        
        # Setup playback controls (hidden by default)
        self._setup_playback_controls()

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

        self.parent.update_plots()
        logging.log(0, "Axis scales updated for all plot axes")

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
                # Only show columns that actually exist (computed successfully or present in the dataframe)
                try:
                    pn = [str(c) for c in list(self.parent.data_source.data.columns)]
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
        # Clear frame histogram cache when selections change
        self.clear_frame_histogram_cache()
        # Preserve contrast during selection operations
        self.parent._preserve_contrast = True
        self.parent.update_plots()
        self.parent._preserve_contrast = False

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
        self.parent.update_plots()

    def addSelection(self, idx, xmin, xmax, invert=False, enabled=True, name=""):
        # Clear frame histogram cache when selections change
        self.clear_frame_histogram_cache()
        
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
        for r in range(n_rows):
            item0 = table.item(r, 0)
            idx = int(item0.data(1)) if item0 is not None else 0
            name = str(item0.data(0)) if item0 is not None else ""
            lower_item = table.item(r, 1)
            upper_item = table.item(r, 2)
            lower = float(lower_item.data(0)) if lower_item is not None else 0.0
            upper = float(upper_item.data(0)) if upper_item is not None else 0.0
            invert = bool(table.cellWidget(r, 3).checkState())
            enabled = bool(table.cellWidget(r, 4).checkState())

            # Try to decode Gaussian2D metadata
            meta_raw = None
            try:
                meta_raw = item0.data(32)
            except Exception:
                meta_raw = None
            meta = None
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = None

            if isinstance(meta, dict) and meta.get("type") == "G2D":
                try:
                    idx1 = int(meta.get("idx1", idx))
                    idx2 = int(meta.get("idx2", idx))
                    mu = meta.get("mu", [0.0, 0.0])
                    cov = meta.get("cov", [[1.0, 0.0], [0.0, 1.0]])
                    sigma = float(meta.get("sigma", 1.0))
                    log_x = bool(meta.get("log_x", False))
                    log_y = bool(meta.get("log_y", False))
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
                    continue
                except Exception:
                    # Fallback to rectangular if decoding fails
                    pass

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
        logging.log(0, f"get_selections: Retrieved {len(selections)} selections")
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
        # If this row encodes a Gaussian2D selection, ignore bound edits (placeholders)
        item0 = table.item(row, 0)
        meta = None
        try:
            meta_raw = item0.data(32)
            if meta_raw:
                meta = json.loads(meta_raw)
        except Exception:
            meta = None
        is_g2d = isinstance(meta, dict) and meta.get("type") == "G2D"

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
        if is_g2d:
            # Revert to stored numeric (keep placeholders) if accidentally made editable
            try:
                self._block_selection_item_changed = True
                val = float(item.data(0) or 0.0)
                item.setText(str(val))
            finally:
                self._block_selection_item_changed = False
            return

        # Parse the edited text as float
        txt = item.text().strip()
        try:
            val = float(txt)
        except Exception:
            # Revert to previous value stored in data role 0
            try:
                self._block_selection_item_changed = True
                prev = float(item.data(0)) if item.data(0) is not None else 0.0
                item.setText(str(prev))
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
        self.checkBoxStackFrames.setChecked(True)
        
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
        
        logging.info(f"Frame selection setup: {frame_param} with {n_frames} frames")

    def hide_frame_selection(self):
        """Hide frame selection UI when no frame stack is detected."""
        self._frame_param = None
        self._n_frames = 0
        self._frame_histogram_cache = {}
        self._stop_playback()

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
            # Try to use cached histogram for this frame
            if self._use_cached_frame_histogram(frame_num):
                return
            
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
                self._background_computation_enabled = playback_settings.get('enable_background_computation', True)
                
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
                self._background_computation_enabled = True
                logging.warning("Playback settings file not found, using defaults")
        except Exception as e:
            logging.warning(f"Failed to load playback settings: {e}")
            # Fallback to defaults
            self._frame_duration_ms = 200
            self._background_computation_enabled = True
    
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
            self._background_computation_enabled = enable_background
        
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
        self.toolButtonPlayBackward.clicked.connect(self._on_play_backward)
        self.toolButtonPause.clicked.connect(self._on_pause)
        self.toolButtonPlayForward.clicked.connect(self._on_play_forward)
        
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
            self._histogram_worker = HistogramComputeWorker(self)
            self._histogram_worker.computation_complete.connect(self._on_histograms_computed)
            self._histogram_worker.computation_failed.connect(self._on_histogram_computation_failed)
    
    def compute_histograms_background(self, histogram_params: dict, weights: Optional[np.ndarray] = None):
        """Compute histograms in background thread if enabled, otherwise compute immediately."""
        if not self._background_computation_enabled or not hasattr(self.parent, 'data_source'):
            # Fall back to immediate computation
            return self._compute_histograms_immediate(histogram_params, weights)
        
        self._initialize_histogram_worker()
        
        # Check cache first
        cached_result = self._histogram_cache.get(histogram_params, weights)
        if cached_result is not None:
            logging.debug("Using cached histogram data")
            self._on_histograms_computed(cached_result)
            return
        
        # Schedule background computation
        try:
            self._histogram_worker.compute_histograms(
                self.parent.data_source,
                histogram_params,
                weights
            )
            logging.debug("Scheduled background histogram computation")
        except Exception as e:
            logging.error(f"Failed to schedule histogram computation: {e}")
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
            # Cache the result
            if '_params' in histogram_data:
                self._histogram_cache.put(
                    histogram_data['_params'],
                    histogram_data,
                    # We'll need to extract weights from the parent context
                )
            
            # Update the parent's histogram data
            self.parent._histogram = histogram_data
            
            # Update the UI
            if '_count' in histogram_data:
                self.parent.lineEditCountCurrent.setText(str(histogram_data['_count']))
            
            # Update histogram displays
            self._update_histogram_displays_from_data(histogram_data)
            
            # Cache for frame if we're in frame mode
            if hasattr(self, '_frame_param') and self._frame_param and not self.checkBoxStackFrames.isChecked():
                self.cache_current_frame_histogram(histogram_data)
            
            logging.debug(f"Updated histogram displays (computation time: {histogram_data.get('_computation_time', 'N/A'):.3f}s)")
            
        except Exception as e:
            logging.error(f"Failed to update histogram displays: {e}")
    
    def _on_histogram_computation_failed(self, error_message: str):
        """Handle failure of histogram computation."""
        logging.error(f"Histogram computation failed: {error_message}")
        # Could show user notification here
    
    def _update_histogram_displays_from_data(self, histogram_data: dict):
        """Update histogram plot widgets from computed data."""
        try:
            # Update X histogram
            if 'x' in histogram_data and hasattr(self.parent, 'g_xhist_m'):
                x_bin_edges, x_counts = histogram_data['x']
                self.parent.g_xhist_m.set_data(x_bin_edges[1:], x_counts)
            
            # Update Y histogram
            if 'y' in histogram_data and hasattr(self.parent, 'g_yhist_m'):
                y_bin_edges, y_counts = histogram_data['y']
                self.parent.g_yhist_m.set_data(y_counts, y_bin_edges[1:])
            
            # Update Z histogram
            if ('z' in histogram_data and hasattr(self.parent, 'g_zhist_m') and 
                hasattr(self.parent, 'checkBoxEnableZ') and self.parent.checkBoxEnableZ.isChecked()):
                z_bin_edges, z_counts = histogram_data['z']
                self.parent.g_zhist_m.set_data(z_bin_edges[1:], z_counts)
                
        except Exception as e:
            logging.error(f"Failed to update histogram displays: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get histogram cache statistics for debugging."""
        return self._histogram_cache.get_stats()
    
    def clear_histogram_cache(self):
        """Clear the histogram cache."""
        self._histogram_cache.clear()
        logging.info("Cleared histogram cache")
    
    # ==================== Frame Histogram Caching ====================
    
    def cache_current_frame_histogram(self, histogram_data: dict):
        """
        Cache the histogram data for the current frame.
        
        Args:
            histogram_data: Dictionary containing histogram data (x, y, z, 2d)
        """
        if not hasattr(self, '_frame_histogram_cache'):
            self._frame_histogram_cache = {}
            
        if self.checkBoxStackFrames.isChecked():
            return  # Don't cache in stacked mode
            
        frame_num = self.spinBoxFrameNumber.value()
        # Deep copy the histogram data to avoid reference issues
        import copy
        self._frame_histogram_cache[frame_num] = copy.deepcopy(histogram_data)
        logging.debug(f"Cached histogram for frame {frame_num}")
        
    def get_cached_frame_histogram(self, frame_num: int) -> dict:
        """
        Get cached histogram data for a specific frame.
        
        Args:
            frame_num: Frame number to retrieve
            
        Returns:
            Cached histogram dict or None if not cached
        """
        if not hasattr(self, '_frame_histogram_cache'):
            return None
        return self._frame_histogram_cache.get(frame_num)
        
    def _use_cached_frame_histogram(self, frame_num: int) -> bool:
        """
        Try to use cached histogram for the given frame.
        
        Returns:
            True if cache was used and plots updated, False otherwise
        """
        cached = self.get_cached_frame_histogram(frame_num)
        if cached is None:
            return False
            
        try:
            # Update parent's histogram data directly from cache
            self.parent._histogram = cached
            # Trigger plot update without recomputing histograms
            from . import plot_update_helpers
            # Update the plots using cached data
            self.parent.lineEditCountCurrent.setText(str(cached.get('_count', '?')))
            
            # Update histogram displays
            if 'x' in cached:
                x_bin_edges, x_counts = cached['x']
                self.parent.g_xhist_m.set_data(x_bin_edges[1:], x_counts)
            if 'y' in cached:
                y_bin_edges, y_counts = cached['y']
                self.parent.g_yhist_m.set_data(y_counts, y_bin_edges[1:])
            if 'z' in cached and hasattr(self.parent, 'checkBoxEnableZ') and self.parent.checkBoxEnableZ.isChecked():
                z_bin_edges, z_counts = cached['z']
                self.parent.g_zhist_m.set_data(z_bin_edges[1:], z_counts)
                
            # Update 2D plot
            self.parent.update_2d_plot()
            
            # Replot all
            self.parent.g_xplot.replot()
            self.parent.g_yplot.replot()
            self.parent.g_zplot.replot()
            self.parent.g_2dplot.replot()
            
            logging.debug(f"Used cached histogram for frame {frame_num}")
            return True
        except Exception as e:
            logging.warning(f"Failed to use cached histogram: {e}")
            return False
            
    def clear_frame_histogram_cache(self):
        """Clear all cached frame histograms."""
        self._frame_histogram_cache = {}
        logging.debug("Cleared frame histogram cache")
