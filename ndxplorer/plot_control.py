from __future__ import print_function
from typing import List, Dict
import json
import os
import pathlib

from qtpy import QtGui, uic, QtCore, QtWidgets
from pyqtgraph.widgets.SpinBox import SpinBox

from . data_source import RectangularDataSelection

try:
    from chisurf import logging
except:
    import logging
    logging.basicConfig()


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
    def p1(self):
        idx = self.comboBoxSelX.currentIndex()
        name = self.comboBoxSelX.currentText()
        return idx, str(name)

    @property
    def p2(self):
        idx = self.comboBoxSelY.currentIndex()
        name = self.comboBoxSelY.currentText()
        return idx, str(name)

    @property
    def p3(self):
        idx = self.comboBoxSelZ.currentIndex()
        name = self.comboBoxSelZ.currentText()
        return idx, str(name)

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
        self.spinBoxXmin = SpinBox()
        self.spinBoxXmax = SpinBox()
        self.spinBoxYmin = SpinBox()
        self.spinBoxYmax = SpinBox()
        self.spinBoxZmin = SpinBox()
        self.spinBoxZmax = SpinBox()

        uic.loadUi(str(ui_file.as_posix()), self)
        self.horizontalLayout.addWidget(self.spinBoxXmin)
        self.horizontalLayout.addWidget(self.spinBoxXmax)
        self.horizontalLayout_2.addWidget(self.spinBoxYmin)
        self.horizontalLayout_2.addWidget(self.spinBoxYmax)
        self.horizontalLayout_3.addWidget(self.spinBoxZmin)
        self.horizontalLayout_3.addWidget(self.spinBoxZmax)

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
        self.actionUpdatePlots.triggered.connect(self.parent.update_plots)

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
        self.spinBoxXmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxXmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxYmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxYmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxZmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxZmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)

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
        self.parent.update_plots(skip_clustering=True)

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

        self.parent.update_plots()

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

    def update(self):
        super(SurfacePlotWidget, self).update()
        self.actionUpdatePlots.blockSignals(True)
        self.actionUpdate_axis_scales.blockSignals(True)
        pn = self.parent.data_source.parameter_names
        self.comboBoxSelX.clear()
        self.comboBoxSelY.clear()
        self.comboBoxSelZ.clear()
        self.comboBoxSelX.addItems(pn)
        self.comboBoxSelY.addItems(pn)
        self.comboBoxSelZ.addItems(pn)
        self.actionUpdatePlots.blockSignals(False)
        self.actionUpdate_axis_scales.blockSignals(False)

        # Instead of triggering the action, call update_plots directly with skip_clustering=True
        # This ensures clustering is not applied automatically after loading data
        self.parent.update_plots(skip_clustering=True)
        logging.log(0, "Updated parameter selectors and triggered plot update with clustering skipped")

    def onClearSelection(self):
        logging.log(0, "onClearSelection")
        self.tableWidget.setRowCount(0)
        self.parent.update_plots()

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
        self.parent.update_plots()

    def addSelection(self, idx, xmin, xmax, invert=False, enabled=True, name=""):
        # Ensure xmin < xmax
        if xmin > xmax:
            xmin, xmax = xmax, xmin
            logging.log(0, f"Swapped xmin and xmax to ensure min-max ordering: ({xmin}, {xmax})")

        table = self.tableWidget
        row = table.rowCount()
        table.setRowCount(row + 1)

        tmp = QtWidgets.QTableWidgetItem("%s" % name)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setData(1, idx)
        table.setItem(row, 0, tmp)

        tmp = QtWidgets.QTableWidgetItem()
        tmp.setText(str(xmin))
        tmp.setData(0, xmin)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))  # Set text color to black
        tmp.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 255)))  # Set background color to white
        font = QtGui.QFont()
        font.setPointSize(10)  # Set font size
        tmp.setFont(font)
        tmp.setTextAlignment(QtCore.Qt.AlignCenter)  # Center align the text
        table.setItem(row, 1, tmp)

        tmp = QtWidgets.QTableWidgetItem()
        tmp.setText(str(xmax))
        tmp.setData(0, xmax)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))  # Set text color to black
        tmp.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 255)))  # Set background color to white
        font = QtGui.QFont()
        font.setPointSize(10)  # Set font size
        tmp.setFont(font)
        tmp.setTextAlignment(QtCore.Qt.AlignCenter)  # Center align the text
        table.setItem(row, 2, tmp)

        cb_invert_x = QtWidgets.QCheckBox(table)
        table.setCellWidget(row, 3, cb_invert_x)
        cb_invert_x.setChecked(invert)

        cb_enable_x = QtWidgets.QCheckBox(table)
        table.setCellWidget(row, 4, cb_enable_x)
        cb_enable_x.setChecked(enabled)
        self.parent.update_plots()

        # Actions for selection checkbox
        cb_enable_x.stateChanged.connect(self.actionUpdatePlots.trigger)
        cb_invert_x.stateChanged.connect(self.actionUpdatePlots.trigger)
        logging.log(0, f"Added selection for parameter index {idx} with range ({xmin}, {xmax}), invert={invert}, enabled={enabled}")

    def onAddSelection(self):
        idx, name = self.p3
        xsel = self.parent.selection_z.get_range()
        xmin = float(min(xsel))
        xmax = float(max(xsel))
        self.addSelection(idx, xmin, xmax, False, True, name)
        logging.log(0, f"onAddSelection: Added selection for {name} with range ({xmin}, {xmax})")

    def get_selections(self):
        selections = list()
        table = self.tableWidget
        n_rows = int(table.rowCount())
        for r in range(n_rows):
            idx = int(table.item(r, 0).data(1))
            name = str(table.item(r, 0).data(0))
            lower = float(table.item(r, 1).data(0))
            upper = float(table.item(r, 2).data(0))
            invert = bool(table.cellWidget(r, 3).checkState())
            enabled = bool(table.cellWidget(r, 4).checkState())
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
