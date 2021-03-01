from __future__ import print_function
from typing import List, Dict
import json
import os

from qtpy import QtGui, uic, QtCore, QtWidgets
from pyqtgraph.widgets.SpinBox import SpinBox

from plot_main import USE_GUIQWT
from data_selection import RectangularDataSelection


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
        if bool(self.checkBoxLogZ.isChecked()):
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
        self.spinBoxBin1DX.setValue(v)

    @property
    def n_yhist_1d(self):
        return int(self.spinBoxBin1DY.value())

    @n_yhist_1d.setter
    def n_yhist_1d(self, v):
        self.spinBoxBin1DY.setValue(v)

    @property
    def n_zhist_1d(self):
        return int(self.spinBoxBin1DZ.value())

    @n_zhist_1d.setter
    def n_zhist_1d(self, v):
        self.spinBoxBin1DZ.setValue(v)

    @property
    def n_xhist_2d(self):
        return int(self.spinBoxBin2DX.value())

    @n_xhist_2d.setter
    def n_xhist_2d(self, v):
        self.spinBoxBin2DX.setValue(v)

    @property
    def n_yhist_2d(self):
        return int(self.spinBoxBin2DY.value())

    @n_yhist_2d.setter
    def n_yhist_2d(self, v):
        self.spinBoxBin2DY.setValue(v)

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
        #########################
        # GUI
        #########################
        ui_file = os.path.dirname(__file__) + '/plot_control.ui'
        print(ui_file)
        self.spinBoxXmin = SpinBox()
        self.spinBoxXmax = SpinBox()
        self.spinBoxYmin = SpinBox()
        self.spinBoxYmax = SpinBox()
        self.spinBoxZmin = SpinBox()
        self.spinBoxZmax = SpinBox()
        uic.loadUi(ui_file, self)
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
        self.connect(self.actionUpdatePlots, QtCore.SIGNAL("triggered()"), self.parent.update_plots)

        # Auto range
        self.connect(self.actionAuto_range_x, QtCore.SIGNAL("triggered()"), self.onAutoRangeX)
        self.connect(self.actionAuto_range_x, QtCore.SIGNAL("triggered()"), self.onUpdate_axis_scales)
        self.connect(self.actionAuto_range_y, QtCore.SIGNAL("triggered()"), self.onAutoRangeY)
        self.connect(self.actionAuto_range_y, QtCore.SIGNAL("triggered()"), self.onUpdate_axis_scales)
        self.connect(self.actionAuto_range_z, QtCore.SIGNAL("triggered()"), self.onAutoRangeZ)
        self.connect(self.actionAuto_range_z, QtCore.SIGNAL("triggered()"), self.onUpdate_axis_scales)
        self.connect(self.actionAuto_range_z, QtCore.SIGNAL("triggered()"), self.auto_selection_range)
        self.connect(self.actionUpdate_axis_scales, QtCore.SIGNAL("triggered()"), self.onUpdate_axis_scales)

        # Selection table
        self.connect(self.actionSelectionTableClicked, QtCore.SIGNAL("triggered()"), self.onSelectionTableClicked)
        self.connect(self.actionSave_selection, QtCore.SIGNAL("triggered()"), self.onSave_selection)
        self.connect(self.actionLoad_selection, QtCore.SIGNAL("triggered()"), self.onLoad_selection)
        self.connect(self.actionClear_Selection, QtCore.SIGNAL("triggered()"), self.onClearSelection)
        self.connect(self.actionAdd_Selection, QtCore.SIGNAL("triggered()"), self.onAddSelection)

        # Change axis range
        self.spinBoxXmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxXmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxYmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxYmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxZmin.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)
        self.spinBoxZmax.sigValueChanged.connect(self.actionUpdate_axis_scales.trigger)

        # Change parameter plotted on axis
        self.connect(self.actionX_axis_changed, QtCore.SIGNAL("triggered()"), self.onX_axis_changed)
        self.connect(self.actionY_axis_changed, QtCore.SIGNAL("triggered()"), self.onY_axis_changed)
        self.connect(self.actionZ_axis_changed, QtCore.SIGNAL("triggered()"), self.onZ_axis_changed)

        # Update axis settings
        self.connect(self.actionUpdate_x_axis_settings, QtCore.SIGNAL("triggered()"), self.onUpdate_x_axis_settings)
        self.connect(self.actionUpdate_y_axis_settings, QtCore.SIGNAL("triggered()"), self.onUpdate_y_axis_settings)
        self.connect(self.actionUpdate_z_axis_settings, QtCore.SIGNAL("triggered()"), self.onUpdate_z_axis_settings)

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

    def onUpdate_x_axis_settings(self):
        print("onUpdate_x_axis_settings")
        self.set_axis_settings(
            self.p1[1],
            self.xmin, self.xmax,
            self.scale_x,
            self.n_xhist_1d,
            self.n_xhist_2d
        )

    def onUpdate_y_axis_settings(self):
        print("onUpdate_y_axis_settings")
        self.set_axis_settings(
            self.p2[1],
            self.ymin, self.ymax,
            self.scale_y,
            self.n_yhist_1d,
            self.n_yhist_2d
        )

    def onUpdate_z_axis_settings(self):
        print("onUpdate_z_axis_settings")
        self.set_axis_settings(
            self.p3[1],
            self.zmin, self.zmax,
            self.scale_z,
            self.n_zhist_1d,
            None
        )

    def onX_axis_changed(self):
        _, name = self.p1
        try:
            d = self.axis_settings[name]
            self.n_xhist_1d = d['n_bins_1d']
            self.n_xhist_2d = d['n_bins_2d']
            self.xmin = d['min']
            self.xmax = d['max']
            self.scale_x = d['scale']
        except KeyError:
            if self.checkBoxAutoScaleX.isChecked():
                self.onAutoRangeX()
        self.parent.update_plots()

    def onY_axis_changed(self):
        _, name = self.p2
        try:
            d = self.axis_settings[name]
            self.n_yhist_1d = d['n_bins_1d']
            self.n_yhist_2d = d['n_bins_2d']
            self.ymin = d['min']
            self.ymax = d['max']
            self.scale_y = d['scale']
        except KeyError:
            if self.checkBoxAutoScaleY.isChecked():
                self.onAutoRangeY()
        self.parent.update_plots()

    def onZ_axis_changed(self):
        _, name = self.p3
        try:
            d = self.axis_settings[name]
            self.n_zhist_1d = d['n_bins_1d']
            self.zmin = d['min']
            self.zmax = d['max']
            self.scale_z = d['scale']
        except KeyError:
            if self.checkBoxAutoScaleZ.isChecked():
                self.onAutoRangeZ()
        self.parent.update_plots()

    def onUpdate_axis_scales(self):
        if USE_GUIQWT:
            self.parent.g_yplot.set_axis_scale("left", self.scale_y)
            self.parent.g_xplot.set_axis_scale("bottom", self.scale_x)
            self.parent.g_zplot.set_axis_scale("bottom", self.scale_z)
            self.parent.g_xyplot.set_axis_scale("bottom", self.scale_x)
            self.parent.g_xyplot.set_axis_scale("left", self.scale_y)
        else:
            self.parent.g_xplot.setLogMode(self.scale_x == "log", None)
            self.parent.g_yplot.setLogMode(None, "log" == self.scale_y)
            # self.parent.g_zplot.setLogMode(self.scale_z == "log", None)
            self.parent.g_xyplot.setLogMode(
                self.scale_x == "log",
                self.scale_y == "log"
            )
        self.parent.update_plots()

    def auto_selection_range(self):
        z = self.parent.z_values
        m = z.mean()
        sd = z.std()
        if USE_GUIQWT:
            self.parent.selection_z.set_range(m - 2 * sd, m + 2 * sd)
        else:
            self.parent.selection_z.setRegion(m - 2 * sd, m + 2 * sd)

    def update(self):
        super(SurfacePlotWidget, self).update()
        self.actionUpdatePlots.blockSignals(True)
        pn = self.parent.data_source.parameter_names
        self.comboBoxSelX.clear()
        self.comboBoxSelY.clear()
        self.comboBoxSelZ.clear()
        self.comboBoxSelX.addItems(pn)
        self.comboBoxSelY.addItems(pn)
        self.comboBoxSelZ.addItems(pn)
        self.actionUpdatePlots.blockSignals(False)
        self.actionUpdatePlots.trigger()

    def onClearSelection(self):
        print("onClearSelection")
        self.tableWidget.setRowCount(0)
        self.parent.update_plots()

    def onSave_selection(self):
        print("onSave_selection")
        l = [s.__dict__ for s in self.get_selections()]
        fn = QtGui.QFileDialog.getSaveFileName(
            None,
            "Selection JSON",
            os.path.dirname(__file__),
            'All files (*.selection.json)'
        )
        with open(fn, "w") as fp:
            json.dump(l, fp=fp, indent=4)

    def onLoad_selection(self):
        fn = QtGui.QFileDialog.getOpenFileName(
            None,
            "Selection JSON",
            os.path.dirname(__file__),
            'All files (*.selection.json)'
        )
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

    def onAutoRangeX(self):
        print("onAutoRangeX")
        self.spinBoxXmin.blockSignals(True)
        self.spinBoxXmax.blockSignals(True)
        self.spinBoxXmin.setValue(self.parent.xmin)
        self.spinBoxXmax.setValue(self.parent.xmax)
        self.spinBoxXmin.blockSignals(False)
        self.spinBoxXmax.blockSignals(False)

    def onAutoRangeY(self):
        print("onAutoRangeY")
        self.spinBoxYmin.blockSignals(True)
        self.spinBoxYmax.blockSignals(True)
        self.spinBoxYmin.setValue(self.parent.ymin)
        self.spinBoxYmax.setValue(self.parent.ymax)
        self.spinBoxYmin.blockSignals(False)
        self.spinBoxYmax.blockSignals(False)

    def onAutoRangeZ(self):
        print("onAutoRangeZ")
        self.spinBoxZmin.blockSignals(True)
        self.spinBoxZmax.blockSignals(True)
        self.spinBoxZmin.setValue(self.parent.zmin)
        self.spinBoxZmax.setValue(self.parent.zmax)
        self.spinBoxZmin.blockSignals(False)
        self.spinBoxZmax.blockSignals(False)

    def onSelectionTableClicked(self):
        print("onSelectionTableClicked")
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        self.parent.update_plots()

    def addSelection(self, idx, xmin, xmax, invert=False, enabled=True, name=""):

        table = self.tableWidget
        row = table.rowCount()
        table.setRowCount(row + 1)

        tmp = QtGui.QTableWidgetItem("%s" % name)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setData(1, idx)
        table.setItem(row, 0, tmp)

        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, xmin)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 1, tmp)

        tmp = QtGui.QTableWidgetItem()
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setData(0, xmax)
        table.setItem(row, 2, tmp)

        cb_invert_x = QtGui.QCheckBox(table)
        table.setCellWidget(row, 3, cb_invert_x)
        cb_invert_x.setChecked(invert)

        cb_enable_x = QtGui.QCheckBox(table)
        table.setCellWidget(row, 4, cb_enable_x)
        cb_enable_x.setChecked(enabled)
        self.parent.update_plots()

        # Actions for selection checkbox
        cb_enable_x.connect(cb_enable_x, QtCore.SIGNAL("stateChanged(int)"), self.actionUpdatePlots.trigger)
        cb_invert_x.connect(cb_invert_x, QtCore.SIGNAL("stateChanged(int)"), self.actionUpdatePlots.trigger)

    def onAddSelection(self):
        idx, name = self.p3
        if USE_GUIQWT:
            xsel = self.parent.selection_z.get_range()
        else:
            xsel = self.parent.selection_z.getRegion()
        xmin = float(min(xsel))
        xmax = float(max(xsel))
        self.addSelection(idx, xmin, xmax, False, True, name)

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
        return selections
