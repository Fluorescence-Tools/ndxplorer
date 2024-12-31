from __future__ import print_function
from typing import Dict, List

import sys
import os
import json
import yaml
import typing

from . plot_control import SurfacePlotWidget
from . parameter_editor import ParameterEditor
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

if sys.version_info.major > 2:
    import pathlib
else:
    import pathlib2 as pathlib

try:
    from chisurf.gui import QtGui, QtCore, uic, QtWidgets
except ImportError:
    from qtpy import QtCore, uic
    from qtpy import QtGui, QtWidgets

from qwt.plot import QwtPlot

import guiqwt.signals
import guiqwt.plot
import guiqwt.image
import guiqwt.curve
import guiqwt.styles
from guiqwt.plot import CurveDialog, ImageDialog
from guiqwt.builder import make

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from . import reader
from . import writer


class NDXplorer(QtWidgets.QMainWindow):

    settings = dict()  # type: Dict
    equations = list()  # type: List[Dict[str, str]]
    constants = dict()  # type: Dict[str, float]
    _histogram = {
        "x": (),
        "y": (),
        "z": (),
        "2d": ()
    }
    _mask_inf = True  # type: bool
    _mask_nan = True  # type: bool
    _data_source = DataSource()  # type: DataSource
    _default_data_source = DataSource(
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

    @property
    def data_source(self) ->DataSource:
        if self._data_source.empty:
            values = self._default_data_source
        else:
            values = self._data_source
        return values

    @data_source.setter
    def data_source(self, v):
        #  type: (DataSource)->()
        self._data_source = v
        self._data_source.compute_columns(
            constants=self.constants,
            equations=self.equations
        )

    @property
    def x_values(self) -> np.ndarray:
        return self.values[self.plot_control.p1[0]].astype('float64')

    @property
    def y_values(self) -> np.ndarray:
        return self.values[self.plot_control.p2[0]].astype('float64')

    @property
    def z_values(self)-> np.ndarray:
        return self.values[self.plot_control.p3[0]].astype('float64')

    @property
    def values(self) -> np.ndarray:
        values = self.data_source.values
        selections = self.plot_control.get_selections()
        mask = self.data_source.get_mask(
            selections=selections,
            idxs=[
                self.plot_control.p1[0],
                self.plot_control.p2[0],
                self.plot_control.p3[0]
            ],
            mask_inf=self._mask_inf,
            mask_nan=self._mask_nan
        )
        x = np.ma.array(values, mask=mask)
        oCol, oRow = x.shape
        re = np.ma.compressed(x)
        nD = re.shape[0]
        re = re.reshape((oCol, int(nD /oCol)))
        return re

    @property
    def ymax(self) -> float:
        return max(self.y_values)

    @property
    def zmin(self):
        v = self.z_values[self.z_values > -np.inf]
        if self.plot_control.scale_z == "log":
            v = v[np.where(v > 0)[0]]
        return min(v)

    @property
    def zmax(self) -> float:
        return max(self.z_values)

    @property
    def working_path(self):
        return self.lineEditWorkingPath.text()

    @working_path.setter
    def working_path(self, v):
        if pathlib.Path(v).is_dir():
            self.lineEditWorkingPath.setText(v)

    @property
    def xmin(self) -> float:
        v = self.x_values[self.x_values > -np.inf]
        if self.plot_control.scale_x == "log":
            v = v[np.where(v > 0)[0]]
        return min(v)

    @property
    def xmax(self) -> float:
        return max(self.x_values)

    @property
    def ymin(self) -> float:
        v = self.y_values[self.y_values > -np.inf]
        if self.plot_control.scale_y == "log":
            v = v[np.where(v > 0)[0]]
        return min(v)

    def __init__(
            self,
            data_source=None,  # type: DataSource
            settings_json_fn=None,  # type: str
            parent=None
    ):
        # type: (DataSource, QtWidgets.QWidget) -> ()
        if isinstance(data_source, DataSource):
            self._data_source = data_source

        super(NDXplorer, self).__init__(parent=parent)
        self.plot_control = SurfacePlotWidget(self)
        self.equation_editor = CodeEditor(parent=self)
        uic.loadUi(os.path.dirname(__file__) + '/plot_main.ui', self)
        self.verticalLayout_3.addWidget(self.plot_control)
        self.verticalLayout_15.addWidget(self.equation_editor)

        def save_cb():
            logging.log(0, "Save CB")
            json_str = self.equation_editor.text()
            self.equations = yaml.load(json_str)
        self.equation_editor.save_callback = save_cb

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

        # 2D-Histogram
        # Create a matplotlib figure and axis
        fig, ax = plt.subplots()
        self.fig = fig
        d = np.ones((200, 200))
        d[120, 120] = 250
        d[150, 120] = 250
        # Display the data as an image using imshow
        cax = ax.imshow(d, cmap='hot', interpolation='nearest')
        self.cax = cax

        # Create a FigureCanvas object to integrate matplotlib with PyQt
        self.canvas = FigureCanvas(fig)

        # Remove axes
        ax.axis('off')

        # Set aspect ratio to 'auto' to allow resizing
        ax.set_aspect('auto')

        # Adjust the layout to fill the widget
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Show the initial plot
        self.canvas.draw()

        # Add the widget to your PyQt layout
        self.verticalLayout_11.addWidget(self.canvas)

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
        self.actionUpdate_plot.triggered.connect(self.update_plots)
        self.actionClear_plot.triggered.connect(self.clear_plots)
        self.actionMask_toggle_changed.triggered.connect(self.onMaskChanged)
        # Axis range

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

    def clear_plots(self):
        logging.log(0, "clearing plots")
        self._data_source.clear()
        self.update()

    def onMaskChanged(self):
        self._mask_inf = self.checkBoxMaskInf.isChecked()
        self._mask_nan = self.checkBoxMaskNaN.isChecked()
        self.update_plots()

    def onSelectWorkingPath(self):
        working_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select current path', self.working_path)
        self.lineEditWorkingPath.blockSignals(True)
        self.lineEditWorkingPath.setText(working_path)
        self.lineEditWorkingPath.blockSignals(False)

    def onSaveBurstIDs(self, folder=None):
        logging.log(0, "saving burst")
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Folder for Burst IDs', self.working_path
            )
        writer.save_burst_ids(
            folder_name=folder,
            selections=self.plot_control.get_selections(),
            data_source=self.data_source
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
        elif file_type in ["paris_dir"]:
            file_handles = QtWidgets.QFileDialog.getExistingDirectory(None, 'Open MFD analysis folder', self.working_path)
            data_reader = reader.read_paris_analysis
        else: #if file_type in [None, "csv"]:
            if file_handles is None:
                file_handles = QtWidgets.QFileDialog.getOpenFileNames(None, 'Comma separated value files', self.working_path, 'Text files (*.*)')
            data_reader = reader.read_csv
        if file_handles:
            self.working_path = str(pathlib.Path(file_handles[0]).parent)
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
        self.open_files(file_type="paris_dir")

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
        self._histogram["x"] = np.histogram(d1, bins=x_bins_1d, density=self.plot_control.normed_hist_x)[::-1]
        self._histogram["y"] = np.histogram(d2, bins=y_bins_1d, density=self.plot_control.normed_hist_y)[::-1]
        self._histogram["z"] = np.histogram(d3, bins=z_bins_1d, density=self.plot_control.normed_hist_z)[::-1]

        # 2D Histogram
        ####################
        try:
            H, x_edges, y_edges = np.histogram2d(x=d1, y=d2, bins=[x_bins_2d, y_bins_2d], density=True)
            self._histogram["2d"] = H, x_edges, y_edges
        except ValueError:
            logging.log(1, "Did not compute 2D histogram")

    def update_plots(self):
        self.update_parameter_names()
        self.update_histograms()
        self.g_xhist_m.set_data(self._histogram["x"][0][1:], self._histogram["x"][1])
        self.g_yhist_m.set_data(self._histogram["y"][1], self._histogram["y"][0][1:])
        self.g_zhist_m.set_data(self._histogram["z"][0][1:], self._histogram["z"][1])

        self.g_xplot.do_autoscale()
        self.g_yplot.do_autoscale()
        self.g_zplot.do_autoscale()
        self.update_2d_plot()

    def update_2d_plot(self):
        try:
            new_data, x_edges, y_edges = self._histogram["2d"]
        except ValueError:
            return None

        # Update the data of the displayed image
        self.cax.set_data(np.rot90(new_data, k=1))

        # Autoscale the intensity (set vmin and vmax)
        self.cax.autoscale()

        # Redraw the canvas to update the display
        self.canvas.draw_idle()

