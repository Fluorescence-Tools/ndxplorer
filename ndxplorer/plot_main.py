from __future__ import print_function
from typing import Dict, List
USE_GUIQWT = True

import sys
import os
import json
import yaml

from . plot_control import SurfacePlotWidget
from . parameter_editor import ParameterEditor
from . qsci_editor import CodeEditor
from . data_source import DataSource

if sys.version_info.major > 2:
    import pathlib
else:
    import pathlib2 as pathlib

from qtpy import QtCore, uic
from qtpy import QtGui, QtWidgets

if USE_GUIQWT:
    import guiqwt.signals
    import guiqwt.plot
    import guiqwt.image
    import guiqwt.curve
    import guiqwt.styles
    from guiqwt.plot import CurveDialog, ImageDialog
    from guiqwt.builder import make
else:
    import pyqtgraph as pg

import numpy as np
import pyqtgraph as pg

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
    def data_source(self):
        #  type: ()->DataSource
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
    def x_values(self):
        # type: () -> np.ndarray
        return self.values[self.plot_control.p1[0]].astype('float64')

    @property
    def y_values(self):
        # type: () -> np.ndarray
        return self.values[self.plot_control.p2[0]].astype('float64')

    @property
    def z_values(self):
        # type: () -> np.ndarray
        return self.values[self.plot_control.p3[0]].astype('float64')

    @property
    def values(self):
        # type: () -> np.ndarray
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
    def ymax(self):
        # type: () -> float
        return max(self.y_values)

    @property
    def zmin(self):
        # type: () -> float
        return min(self.z_values)

    @property
    def zmax(self):
        # type: () -> float
        return max(self.z_values)

    @property
    def working_path(self):
        return self.lineEditWorkingPath.text()

    @working_path.setter
    def working_path(self, v):
        if pathlib.Path(v).is_dir():
            self.lineEditWorkingPath.setText(v)

    @property
    def xmin(self):
        # type: () -> float
        v = self.x_values[self.x_values > -np.inf]
        return min(v)

    @property
    def xmax(self):
        # type: () -> float
        return max(self.x_values)

    @property
    def ymin(self):
        # type: () -> float
        return min(self.y_values)

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
            print("Save CB")
            json_str = self.equation_editor.text()
            self.equations = yaml.load(json_str)
        self.equation_editor.save_callback = save_cb

        # Plots
        #############
        # z-axis
        if USE_GUIQWT:
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
        else:
            self.g_zplot = pg.PlotWidget()
            self.g_zhist_m = pg.PlotCurveItem(
                [0, 1], [0, 1],
                # stepMode=True,
                fillLevel=0,
                brush=(255, 255, 0, 255)
            )
            self.g_zplot.addItem(self.g_zhist_m)
            self.selection_z = pg.LinearRegionItem()
            self.g_zplot.addItem(self.selection_z)
        # self.selectors.append(
        #     ('z', self.selection_z)
        # )

        self.plot_control.verticalLayout_4.addWidget(self.g_zplot)

        # x-axis
        if USE_GUIQWT:
            win_x = CurveDialog()
            self.g_xplot = win_x.get_plot()
            curveparam = guiqwt.styles.CurveParam("Curve", icon='curve.png')
            curveparam.curvestyle = "Steps"
            curveparam.line.color = '#0066cc'
            curveparam.shade = 0.5
            curveparam.line.width = 2.0
            self.g_xhist_m = guiqwt.curve.CurveItem(curveparam=curveparam)
            self.g_xplot.add_item(self.g_xhist_m)
        else:
            self.g_xplot = pg.PlotWidget()
            self.g_xhist_m = pg.PlotCurveItem(
                [0, 1], [0, 1],
                # stepMode=True,
                fillLevel=0,
                brush=(0, 0, 255, 255)
            )
            self.g_xplot.addItem(self.g_xhist_m)
        self.verticalLayout_5.addWidget(self.g_xplot)

        # y-axis
        if USE_GUIQWT:
            win_y = CurveDialog()
            self.g_yplot = win_y.get_plot()
            curveparam = guiqwt.styles.CurveParam()
            curveparam.curvestyle = "Steps"
            curveparam.line.color = '#00ff00'
            curveparam.shade = 0.1
            curveparam.line.width = 2.0
            self.g_yhist_m = guiqwt.curve.CurveItem(curveparam=curveparam)
            self.g_yplot.add_item(self.g_yhist_m)
        else:
            self.g_yplot = pg.PlotWidget()
            self.g_yhist_m = pg.PlotCurveItem(
                [0, 1], [0, 1],
                # stepMode=True,
                fillLevel=0,
                brush=(255, 0, 255, 255)
            )
            self.g_yplot.addItem(self.g_yhist_m)
        self.verticalLayout_7.addWidget(self.g_yplot)

        # 2D-Histogram
        if USE_GUIQWT:
            image_param = guiqwt.styles.ImageParam()
            image_param.alpha_mask = False
            image_param.colormap = "hot"
            self.g_hist2d_m = guiqwt.image.ImageItem(param=image_param)
            win_xy = ImageDialog(edit=False, toolbar=False)
            self.g_xyplot = win_xy.get_plot()
            self.g_xyplot.set_aspect_ratio(1., False)
            self.g_xyplot.set_axis_direction('left', reverse=False)
            self.g_xyplot.add_item(self.g_hist2d_m)
        else:
            self.g_xyplot = pg.PlotWidget(parent=self)
            image = pg.ImageItem()
            self.g_xyplot.addItem(image)
            self.g_hist2d_m = image
            self.g_hist2d_m.getViewBox().invertY(False)
        self.verticalLayout_11.addWidget(self.g_xyplot)
        if not USE_GUIQWT:
            self.g_xyplot.setXLink(self.g_xplot)
            self.g_xyplot.setYLink(self.g_yplot)
            # # 2D Selection
            # self.selection_2d_y = pg.LinearRegionItem(orientation='horizontal', movable=True)
            # self.selection_2d_x = pg.LinearRegionItem(orientation='vertical', movable=True)
            # self.g_xyplot.addItem(self.selection_2d_y)
            # self.g_xyplot.addItem(self.selection_2d_x)

        self.g_xplot.setMaximumHeight(150)
        self.g_yplot.setMaximumWidth(150)
        self.g_zplot.setMaximumHeight(150)

        # Clean plots
        ################
        if USE_GUIQWT:
            pass
        else:

            self.g_xplot.showGrid(True, True)
            self.g_yplot.showGrid(True, True)
            self.g_zplot.showGrid(True, True)

            self.g_xyplot.getPlotItem().showAxis('top')
            self.g_xyplot.getPlotItem().showAxis('bottom')
            self.g_xyplot.getPlotItem().showAxis('right')
            self.g_xyplot.getPlotItem().showAxis('left')

            self.g_zplot.getPlotItem().showAxis('right')
            self.g_xplot.getPlotItem().showAxis('bottom')
            self.g_xplot.getPlotItem().showAxis('right')
            self.g_yplot.getPlotItem().hideAxis('left')
            self.g_yplot.getPlotItem().showAxis('right')
            self.g_yplot.getPlotItem().showAxis('top')
            self.g_yplot.getPlotItem().showAxis('bottom')

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
        print("clear_plots")
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
        print("onSaveBurstIDs")
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
                None, 'Axis settings file', self.working_path, 'Axis file (*.axis.json)'
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
            d = yaml.load(fp)
            self.equations = d
        fn_constants = pathlib.Path(settings_json_fn).parent / self.settings["constants"]
        with open(str(fn_constants), "r") as fp:
            d = json.load(fp)
            self.constants.update(d)
        self.equation_editor.load_file(str(fn_equations))

    def open_files(
            self,
            file_handles=None,  # type: List[str]
            file_type=None  # type: str
    ):
        if file_type in ["cs_sampling", "er4"]:
            if file_handles is None:
                file_handles = QtWidgets.QFileDialog.getOpenFileNames(
                    None, 'ChiSurf sampling files', self.working_path, 'All files (*.er4)'
                )
            self.working_path = str(pathlib.Path(file_handles[0]).parent)
            data_reader = reader.read_csv_sampling
        elif file_type in ["paris_dir"]:
            file_handles = QtWidgets.QFileDialog.getExistingDirectory(
                None, 'Open MFD analysis folder', self.working_path)
            self.working_path = str(pathlib.Path(file_handles))
            data_reader = reader.read_paris_analysis
        else: #if file_type in [None, "csv"]:
            if file_handles is None:
                file_handles = QtWidgets.QFileDialog.getOpenFileNames(
                    None, 'Comma separated value files', self.working_path, 'Text files (*.csv;*.dat;*.txt)'
                )
            self.working_path = str(pathlib.Path(file_handles[0]).parent)
            data_reader = reader.read_csv
        self._data_source = data_reader(file_handles)
        self.update()

    def onOpenCsv(
            self,
            filenames=None  # type: List[str]
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
        if USE_GUIQWT:
            self.g_yplot.set_titles(xlabel="counts", ylabel=p2_name)
            self.g_xplot.set_titles(xlabel=p1_name, ylabel="counts")
            self.g_xyplot.set_titles(ylabel=p2_name, xlabel=p1_name)
        else:
            # self.g_yplot.setLabel('bottom', "counts")
            self.g_xplot.setLabel('top', p1_name)
            # self.g_xplot.setLabel('left', "counts")
            self.g_yplot.setLabel('right', p2_name)
            # self.g_xplot.setLabel('left', p2_name)

    def get_bins(self, arange, scale, n_1d, n_2d):
        xmin, xmax = arange
        # set log scales
        if scale == "log":
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
        return self.get_bins(
            self.plot_control.z_range,
            self.plot_control.scale_z,
            self.plot_control.n_zhist_1d,
            10
        )

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
        self._histogram["x"] = np.histogram(d1, bins=x_bins_1d, normed=self.plot_control.normed_hist_x)[::-1]
        self._histogram["y"] = np.histogram(d2, bins=y_bins_1d, normed=self.plot_control.normed_hist_y)[::-1]
        self._histogram["z"] = np.histogram(d3, bins=z_bins_1d, normed=self.plot_control.normed_hist_z)[::-1]
        # 2D Histogram
        ####################
        try:
            H, x_edges, y_edges = np.histogram2d(x=d1, y=d2, bins=[x_bins_2d, y_bins_2d])
            self._histogram["2d"] = H, x_edges, y_edges
        except ValueError:
            print("Did not compute 2D histogram", file=sys.stderr)

    def update_plots(self):
        self.update_parameter_names()
        self.update_histograms()
        if USE_GUIQWT:
            self.g_xhist_m.set_data(self._histogram["x"][0][1:], self._histogram["x"][1])
            self.g_yhist_m.set_data(self._histogram["y"][1], self._histogram["y"][0][1:])
            self.g_zhist_m.set_data(self._histogram["z"][0][1:], self._histogram["z"][1])
            self.g_xplot.do_autoscale()
            self.g_yplot.do_autoscale()
            self.g_zplot.do_autoscale()
        else:
            self.g_xhist_m.setData(self._histogram["x"][0][1:], self._histogram["x"][1])
            self.g_yhist_m.setData(self._histogram["y"][1], self._histogram["y"][0][1:])
            self.g_zhist_m.setData(self._histogram["z"][0][1:], self._histogram["z"][1])
        self.update_2d_plot()

    def update_2d_plot(self):
        try:
            H, x_edges, y_edges = self._histogram["2d"]
        except ValueError:
            return None
        if USE_GUIQWT:
            self.g_hist2d_m.set_data(H.T, lut_range=None)
            self.g_hist2d_m.set_xdata(x_edges[1], x_edges[-1])
            self.g_hist2d_m.set_ydata(y_edges[1], y_edges[-1])
            self.g_hist2d_m.update_bounds()
            self.g_xyplot.do_autoscale()
        else:
            img = H
            x0, x1 = (x_edges[0], x_edges[-1])
            y0, y1 = (y_edges[0], y_edges[-1])
            if self.plot_control.scale_x == "log":
                x0, x1 = np.log10(x0), np.log10(x1)
            if self.plot_control.scale_y == "log":
                y0, y1 = np.log10(y0), np.log10(y1)
            xscale, yscale = (x1 - x0) / img.shape[0], (y1 - y0) / img.shape[1]

            self.g_xyplot.removeItem(self.g_hist2d_m)
            self.g_hist2d_m = pg.ImageItem(image=H)
            self.g_hist2d_m.setCompositionMode(QtWidgets.QPainter.CompositionMode_Plus)
            #self.g_hist2d_m.translate(x0, y0)
            self.g_hist2d_m.scale(xscale, yscale)
            self.g_xyplot.addItem(self.g_hist2d_m)

