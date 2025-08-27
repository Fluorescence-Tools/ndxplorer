"""
Gaussian fitting UI attachment and logic for ndxplorer.

This module encapsulates the construction of the Gaussian Fit controls and
attaches them to the main NDXplorer window, keeping plot_main.py cleaner.
It wires all Gaussian-related signal handlers here and implements the logic
(fitting, overlays, table I/O, marginals, and Delete-key row removal).
"""
from typing import Iterator, Optional, Tuple, List

import json
import csv
import os
from datetime import datetime

import numpy as np
from qtpy import QtCore, QtWidgets


class GaussianFit(QtCore.QObject):
    """
    Helper that constructs and wires the Gaussian Fit UI into the provided
    main window (NDXplorer instance). It creates the widgets directly in
    verticalLayout_18 as requested and owns all Gaussian-related logic.

    UI elements are stored on `main` for consistency with existing code.
    Plot/marginal state lists (gaussian_items, etc.) and palette are also
    initialized on `main` to minimize changes elsewhere.
    """

    @property
    def is_log_x(self) -> bool:
        """True if current x-axis is in log scale.
        Uses main.plot_control.scale_x if available; defaults to False on error."""
        try:
            m = self.main
            return str(m.plot_control.scale_x).lower() == "log"
        except Exception:
            return False

    @property
    def is_log_y(self) -> bool:
        """True if current y-axis is in log scale.
        Uses main.plot_control.scale_y if available; defaults to False on error."""
        try:
            m = self.main
            return str(m.plot_control.scale_y).lower() == "log"
        except Exception:
            return False

    @property
    def fit_in_log(self):
        try:
            m = self.main
            fit_in_log = bool(m.checkBoxGaussFitLog.isChecked())
        except Exception:
            fit_in_log = False
        return fit_in_log

    @property
    def log_axes(self) -> (bool, bool):
        # Determine log fitting mode from the UI checkbox (overrides axis scales)
        return np.array([self.is_log_x, self.is_log_y], dtype=bool)

    def __init__(self, main: QtWidgets.QMainWindow):
        super().__init__(main)
        self.main = main
        self._build_ui()
        self._connect_signals()

    # ------------------------------ UI ---------------------------------
    def _build_ui(self):
        m = self.main
        # Button row: Fit + Clear + Select + Selection σ + Settings
        btn_row = QtWidgets.QHBoxLayout()
        m.btnFit2DGauss = QtWidgets.QToolButton(m); m.btnFit2DGauss.setText("Fit")
        m.btnClearGaussians = QtWidgets.QToolButton(m); m.btnClearGaussians.setText("Clear")
        m.btnSelectGaussian = QtWidgets.QToolButton(m); m.btnSelectGaussian.setText("Select")
        # Selection height (number of sigmas) next to Select button
        lblSigma = QtWidgets.QLabel("Selection σ:", m)
        m.spinSelectionSigma = QtWidgets.QDoubleSpinBox(m)
        m.spinSelectionSigma.setRange(0.1, 4.0)
        m.spinSelectionSigma.setSingleStep(0.2)
        m.spinSelectionSigma.setDecimals(2)
        m.spinSelectionSigma.setValue(1.0)
        m.spinSelectionSigma.setToolTip("Number of sigmas used for Gaussian burst selection (1.0 = 1σ).")
        m.btnSelectPoint = QtWidgets.QCheckBox("Select point", m)
        m.checkBoxShowMarginals = QtWidgets.QCheckBox("Marginals", m)
        m.checkBoxShowMarginals.setChecked(True)
        # New: toggle to choose fitting in log-space vs normal
        m.checkBoxGaussFitLog = QtWidgets.QCheckBox("Log Gauss", m)
        m.checkBoxGaussFitLog.setChecked(False)
        m.checkBoxGaussFitLog.setToolTip("Log Gauss: when enabled, fit Gaussians in log scale (both X and Y; positive values only). When disabled, fit in linear scale.")
        # New: GMM Settings button
        m.btnGMMSettings = QtWidgets.QToolButton(m); m.btnGMMSettings.setText("Settings")
        m.btnGMMSettings.setToolTip("Configure GaussianMixture (sklearn) parameters and save them to your user settings.")
        btn_row.addWidget(m.btnFit2DGauss)
        btn_row.addWidget(m.btnClearGaussians)
        btn_row.addWidget(m.btnSelectGaussian)
        btn_row.addWidget(lblSigma)
        btn_row.addWidget(m.spinSelectionSigma)
        btn_row.addWidget(m.btnGMMSettings)
        # Place the button row directly into the target layout
        m.verticalLayout_18.addLayout(btn_row)


        # Table of gaussians (x, y, cov, weight)
        m.tableGaussians = QtWidgets.QTableWidget(m)
        m.tableGaussians.setColumnCount(6)
        m.tableGaussians.setHorizontalHeaderLabels(["x", "y", "cov_xx", "cov_xy", "cov_yy", "w"])
        # Make columns compact: do not stretch and size to contents
        header = m.tableGaussians.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        # Hide row numbers to save space
        m.tableGaussians.verticalHeader().setVisible(False)
        # Adjust to contents to keep the table compact
        m.tableGaussians.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        m.tableGaussians.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        m.tableGaussians.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        # Add table directly into the target layout
        m.verticalLayout_18.addWidget(m.tableGaussians)

        # Options row placed below the table: checkboxes + Save/Load
        options_row = QtWidgets.QHBoxLayout()
        options_row.addWidget(m.btnSelectPoint)
        options_row.addWidget(m.checkBoxShowMarginals)
        options_row.addWidget(m.checkBoxGaussFitLog)
        # Save/Load in same row as checkboxes
        m.btnSaveGaussians = QtWidgets.QToolButton(m); m.btnSaveGaussians.setText("Save")
        m.btnLoadGaussians = QtWidgets.QToolButton(m); m.btnLoadGaussians.setText("Load")
        options_row.addWidget(m.btnSaveGaussians)
        options_row.addWidget(m.btnLoadGaussians)
        options_row.addStretch(1)
        m.verticalLayout_18.addLayout(options_row)

        # Guard flag to avoid recursive redraws during programmatic updates
        m._updating_gaussian_table = False

        # Storage for gaussian overlay items and color cycle
        m.gaussian_items = []
        # Stable palette and legacy cycle (kept for compatibility)
        m._gaussian_palette = ["#ff0000", "#00aa00", "#0000ff", "#aa00aa", "#00aaaa", "#ffaa00"]
        m._gaussian_color_cycle = iter(m._gaussian_palette)  # type: Iterator[str]
        # Storage for marginal overlay curve items
        m.gaussian_marginal_items_x = []
        m.gaussian_marginal_items_y = []

    def _connect_signals(self):
        m = self.main
        # Wire actions to handlers in this class
        m.btnFit2DGauss.clicked.connect(self.on_fit_2d_gaussian)
        m.btnSelectPoint.toggled.connect(self.on_select_point_toggled)
        m.btnClearGaussians.clicked.connect(self.on_clear_gaussians)
        m.btnSelectGaussian.clicked.connect(self.on_select_gaussian)
        # Open GMM settings dialog
        try:
            m.btnGMMSettings.clicked.connect(self.on_open_gmm_settings)
        except Exception:
            pass
        # Table edits update overlays
        m.tableGaussians.itemChanged.connect(self.on_gaussian_table_item_changed)

        # Highlight selected gaussians in overlay when selection changes
        sel_model = m.tableGaussians.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(self.on_gaussian_table_selection_changed)
        # Also connect the generic itemSelectionChanged signal to ensure updates
        m.tableGaussians.itemSelectionChanged.connect(lambda: self.on_gaussian_table_selection_changed(None, None))
        # Marginals toggle
        m.checkBoxShowMarginals.toggled.connect(self.on_toggle_gaussian_marginals)
        # Save/Load buttons
        m.btnSaveGaussians.clicked.connect(self.on_save_gaussians)
        m.btnLoadGaussians.clicked.connect(self.on_load_gaussians)

        # Install event filter on table to allow Delete key to remove rows
        m.tableGaussians.installEventFilter(self)
        
    # ---------------------------- Handlers ------------------------------
    def on_gaussian_table_selection_changed(self, selected, deselected):
        """Highlight selected Gaussian overlays by increasing line width."""
        m = self.main
        items = getattr(m, 'gaussian_items', [])
        table = getattr(m, 'tableGaussians', None)
        if table is None or not items:
            return
        # Build set of selected row indices
        sel = set()
        sel_model = table.selectionModel()
        if sel_model is not None:
            sel = {ix.row() for ix in sel_model.selectedRows()}

        # Update line widths
        for i, it in enumerate(items):
            pen = it.pen()
            pen.setWidth(8 if i in sel else 2)
            it.setPen(pen)
        m.overlay_plot.replot()

    def on_select_gaussian(self):
        """Add a Gaussian 2D selection (1σ) for the selected Gaussian rows.
        The selection is created for the currently selected X/Y parameters
        and labeled as G2D(ParamX, ParamY).
        """
        m = self.main
        table = getattr(m, 'tableGaussians', None)
        if table is None:
            return
        # Determine current X/Y parameter indices and names
        try:
            idx1, name1 = m.plot_control.p1
            idx2, name2 = m.plot_control.p2
        except Exception:
            QtWidgets.QMessageBox.warning(m, "No Parameters", "Could not determine current X/Y parameters.")
            return
        label = f"G2D({name1}, {name2})"
        # Determine axis log state for selection parameters
        is_log_x = self.is_log_x
        is_log_y = self.is_log_y
        # Collect selected rows; if none selected, use all rows if exactly one exists
        selected = [ix.row() for ix in table.selectionModel().selectedRows()] if table.selectionModel() else []
        if not selected:
            if table.rowCount() == 1:
                selected = [0]
            else:
                QtWidgets.QMessageBox.information(m, "Select Gaussian", "Please select one or more Gaussian rows in the table.")
                return
        # For each selected row, read mu and cov and add selection
        for r in selected:
            try:
                def getf(c):
                    it = table.item(r, c)
                    return float(it.text()) if it is not None else None
                x = getf(0); y = getf(1); cxx = getf(2); cxy = getf(3); cyy = getf(4)
                if None in (x, y, cxx, cxy, cyy):
                    continue
                mu_v = np.array([x, y], dtype=float)
                cov_v = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)
                # minimal regularization if needed
                try:
                    eig = np.linalg.eigvalsh(cov_v)
                    if np.any(eig <= 0):
                        cov_v = cov_v + 1e-9 * np.eye(2)
                except Exception:
                    cov_v = cov_v + 1e-9 * np.eye(2)
                # Transform to log space for axes that are log so that selection matches displayed Gaussian
                mu_s = mu_v.copy()
                cov_s = cov_v.copy()
                if self.is_log_x or self.is_log_y:
                    mu_s, cov_s = self._transform_params_for_axes(mu_v, cov_v)
                # Add selection to the selection table with log flags
                try:
                    try:
                        sigma_val = float(m.spinSelectionSigma.value())
                    except Exception:
                        sigma_val = 1.0
                    if not np.isfinite(sigma_val) or sigma_val <= 0:
                        sigma_val = 1.0
                    m.plot_control.addGaussianSelection(idx1, idx2, mu_s, cov_s, sigma=sigma_val, invert=False, enabled=True, name=label, log_x=is_log_x, log_y=is_log_y)
                except Exception:
                    # Fallback: show warning
                    QtWidgets.QMessageBox.warning(m, "Selection Error", "Could not add Gaussian selection to the selection table.")
                    return
            except Exception:
                continue
        # Trigger update (addGaussianSelection already triggers update)

    def on_open_gmm_settings(self):
        """Open the GMM settings dialog and refresh cached settings if accepted."""
        try:
            from .gaussian_settings_dialog import GMMSettingsDialog, load_gmm_settings
        except Exception:
            return
        dlg = GMMSettingsDialog(parent=self.main)
        if dlg.exec_():
            # Reload settings into cache and refresh UI elements relying on settings
            try:
                cfg = load_gmm_settings()
                self._gmm_settings = cfg
                m = self.main
                if hasattr(m, 'spinLocalWindow') and m.spinLocalWindow is not None:
                    lw = int(cfg.get("local_window_bins", 10))
                    lw = max(1, min(lw, 200))
                    try:
                        m.spinLocalWindow.blockSignals(True)
                        m.spinLocalWindow.setValue(lw)
                    finally:
                        m.spinLocalWindow.blockSignals(False)
            except Exception:
                pass

    def _get_gmm_settings(self):
        """Load or return cached GMM settings dict."""
        if not hasattr(self, "_gmm_settings") or not isinstance(self._gmm_settings, dict):
            try:
                from .gaussian_settings_dialog import load_gmm_settings
                self._gmm_settings = load_gmm_settings()
            except Exception:
                self._gmm_settings = {}
        return dict(self._gmm_settings)

    def on_fit_2d_gaussian(self):
        """
        Optimize the parameters (means, covariances, weights) of the Gaussians listed
        in the table directly against the currently selected raw data points
        (not the histogram) using a Gaussian Mixture fit.
        """
        m = self.main
        rows = self._read_gaussian_table()
        if len(rows) == 0:
            QtWidgets.QMessageBox.warning(m, "No Gaussians", "Add one or more Gaussians (click on the histogram) before fitting.")
            return

        # Retrieve filtered X/Y data directly from the data source
        try:
            d1 = np.asarray(m.x_values, dtype=float)
            d2 = np.asarray(m.y_values, dtype=float)
        except Exception:
            QtWidgets.QMessageBox.warning(m, "No data", "Unable to retrieve selected data for fitting.")
            return
        if d1.size == 0 or d2.size == 0 or len(d1) != len(d2):
            QtWidgets.QMessageBox.warning(m, "No data", "No data available for fitting.")
            return

        X = np.column_stack([d1, d2])

        # Keep only points within the currently visible histogram range (value space)
        try:
            _, x_edges, y_edges = m._histogram["2d"]
            x_min_vis = float(x_edges[0]); x_max_vis = float(x_edges[-1])
            y_min_vis = float(y_edges[0]); y_max_vis = float(y_edges[-1])
        except Exception:
            QtWidgets.QMessageBox.warning(m, "No histogram", "No 2D histogram available. Fit is restricted to visible data; please update histogram first.")
            return

        vis_mask = (
            (X[:, 0] >= x_min_vis) & (X[:, 0] <= x_max_vis) &
            (X[:, 1] >= y_min_vis) & (X[:, 1] <= y_max_vis)
        )
        if not np.any(vis_mask):
            QtWidgets.QMessageBox.warning(m, "No data", "No data within the visible histogram range to fit.")
            return
        X = X[vis_mask]

        # Filter to finite rows in value space
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.any(finite_mask):
            QtWidgets.QMessageBox.warning(m, "No data", "Selected data contains no finite values for fitting.")
            return
        X = X[finite_mask]

        log_axes = self.log_axes
        is_log_x, is_log_y = log_axes

        # Prepare data in fitting space (Z-space): log-transform axes on log scale
        X_fit = X
        if np.any(log_axes):
            # Remove non-positive values for log-transformed axes
            pos_mask = np.ones(X.shape[0], dtype=bool)
            if is_log_x:
                pos_mask &= X[:, 0] > 0.0
            if is_log_y:
                pos_mask &= X[:, 1] > 0.0
            if not np.any(pos_mask):
                QtWidgets.QMessageBox.warning(m, "No data", "No positive data available on log-scaled axis for fitting.")
                return
            X_pos = X[pos_mask].copy()
            # Apply log to required columns
            if is_log_x:
                X_pos[:, 0] = np.log(X_pos[:, 0])
            if is_log_y:
                X_pos[:, 1] = np.log(X_pos[:, 1])
            # Replace with transformed subset
            X_fit = X_pos

        # Lazy import of sklearn GaussianMixture via documented API
        try:
            from sklearn.mixture import GaussianMixture
        except Exception:
            QtWidgets.QMessageBox.warning(m, "scikit-learn Not Available", "GaussianMixture (scikit-learn) is not installed.")
            return

        n_components = len(rows)

        # Helper to transform (mu, cov) between value space and fitting space
        eps = 1e-12
        def to_fit_space(mu_v: np.ndarray, cov_v: np.ndarray) -> (np.ndarray, np.ndarray):
            mu_v = np.asarray(mu_v, dtype=float).reshape(2)
            cov_v = np.asarray(cov_v, dtype=float).reshape(2, 2)
            mu_z = mu_v.copy()
            J = np.eye(2, dtype=float)
            if is_log_x:
                mu_safe = mu_v[0] if mu_v[0] > eps else eps
                mu_z[0] = np.log(mu_safe)
                J[0, 0] = 1.0 / mu_safe
            if is_log_y:
                mu_safe = mu_v[1] if mu_v[1] > eps else eps
                mu_z[1] = np.log(mu_safe)
                J[1, 1] = 1.0 / mu_safe
            cov_z = J @ cov_v @ J.T
            # Minimal regularization to ensure positive semidefinite
            eig = np.linalg.eigvalsh(cov_z)
            if np.any(eig <= 0):
                cov_z = cov_z + 1e-9 * np.eye(2)
            return mu_z, cov_z

        def to_value_space(mu_z: np.ndarray, cov_z: np.ndarray) -> (np.ndarray, np.ndarray):
            mu_z = np.asarray(mu_z, dtype=float).reshape(2)
            cov_z = np.asarray(cov_z, dtype=float).reshape(2, 2)
            mu_v = mu_z.copy()
            G = np.eye(2, dtype=float)
            if is_log_x:
                mu_v[0] = np.exp(mu_z[0])
                G[0, 0] = mu_v[0]
            if is_log_y:
                mu_v[1] = np.exp(mu_z[1])
                G[1, 1] = mu_v[1]
            cov_v = G @ cov_z @ G.T
            # Minimal regularization
            try:
                eig = np.linalg.eigvalsh(cov_v)
                if np.any(eig <= 0):
                    cov_v = cov_v + 1e-9 * np.eye(2)
            except Exception:
                cov_v = cov_v + 1e-9 * np.eye(2)
            return mu_v, cov_v

        # Build initialization arrays in fitting space according to sklearn API
        means_init_fit = []
        covs_fit = []
        w_init = []
        for (mu_v, cov_v, w) in rows:
            mu_z, cov_z = (mu_v, cov_v) if not np.any(log_axes) else to_fit_space(mu_v, cov_v)
            means_init_fit.append(mu_z)
            covs_fit.append(cov_z)
            w_init.append(max(0.0, float(w)))
        means_init_fit = np.array(means_init_fit, dtype=float)
        covs_fit = np.array(covs_fit, dtype=float)
        precs_fit = np.array([np.linalg.pinv(c) for c in covs_fit])
        w_init = np.array(w_init, dtype=float)
        s = np.sum(w_init)
        if not np.isfinite(s) or s <= 0:
            w_init = np.ones(n_components, dtype=float) / n_components
        else:
            w_init = w_init / s

        # Load user-configured GMM settings
        cfg = self._get_gmm_settings()
        covariance_type = str(cfg.get('covariance_type', 'full'))
        params = dict(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=float(cfg.get('tol', 1e-3)),
            reg_covar=float(cfg.get('reg_covar', 1e-6)),
            max_iter=int(cfg.get('max_iter', 200)),
            n_init=int(cfg.get('n_init', 1)),
            init_params=str(cfg.get('init_params', 'kmeans')),
            warm_start=bool(cfg.get('warm_start', False)),
            verbose=int(cfg.get('verbose', 0)),
            verbose_interval=int(cfg.get('verbose_interval', 10)),
        )
        rs = cfg.get('random_state', None)
        if rs is not None:
            try:
                params['random_state'] = int(rs)
            except Exception:
                params['random_state'] = None
        # Only pass precisions_init if covariance_type is 'full' to match shapes
        if covariance_type == 'full':
            params['precisions_init'] = precs_fit
        # We can safely pass means_init and weights_init for all types
        params['means_init'] = means_init_fit
        params['weights_init'] = w_init

        gm = GaussianMixture(**params)

        # Fit to data in fitting space
        try:
            gm.fit(X_fit)
        except Exception as e:
            QtWidgets.QMessageBox.warning(m, "Fit failed", f"GMM fit failed: {e}")
            return

        # Retrieve fitted parameters (in fitting space)
        means_fit = np.array(gm.means_, dtype=float)
        try:
            covariances_fit = np.array(gm.covariances_, dtype=float)
        except Exception:
            try:
                precisions = np.array(gm.precisions_, dtype=float)
                covariances_fit = np.array([np.linalg.pinv(P) for P in precisions])
            except Exception:
                QtWidgets.QMessageBox.warning(m, "Fit failed", "Could not retrieve covariances from the fitted model.")
                return
        try:
            weights_fitted = np.array(getattr(gm, 'weights_', None), dtype=float)
        except Exception:
            weights_fitted = None

        # Transform fitted parameters back to value space if needed and update table
        for i in range(n_components):
            mu_i = means_fit[i]
            cov_i = covariances_fit[i]
            if np.any(log_axes):
                mu_v_i, cov_v_i = to_value_space(mu_i, cov_i)
            else:
                mu_v_i, cov_v_i = mu_i, cov_i
            wi = None
            try:
                if weights_fitted is not None and i < len(weights_fitted):
                    wi = float(weights_fitted[i])
            except Exception:
                wi = None
            self._update_gaussian_row(i, mu_v_i, cov_v_i, wi)
        # Redraw overlays from the updated table
        self._redraw_gaussian_overlays_from_table()

    def on_select_point_toggled(self, checked: bool):
        """Toggle point selection mode for clicking on the histogram."""
        m = self.main
        if hasattr(m, 'mouse_event_filter') and m.mouse_event_filter is not None:
            m.mouse_event_filter.set_point_mode(checked, callback=self.on_point_selected if checked else None)

    def on_point_selected(self, pos):
        """Handle a point click on the overlay canvas; compute a local Gaussian around the clicked bin.
        Also append it to the Gaussians table.
        """
        m = self.main
        try:
            H, x_edges, y_edges = m._histogram["2d"]
        except Exception:
            return
        if H is None or H.size == 0:
            return
        # Convert canvas position to normalized coordinates [0,1]
        canvas = m.overlay_plot.canvas()
        w = max(1, canvas.width())
        h = max(1, canvas.height())
        x_norm = pos.x() / w
        y_norm = 1.0 - (pos.y() / h)  # invert Y
        # Clamp
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        # Convert to bin indices
        ix = int(x_norm * (len(x_edges) - 1))
        iy = int(y_norm * (len(y_edges) - 1))
        # Compute local Gaussian using a small window around (ix, iy)
        # Requirement: only the width (covariance) should change, not the position (mean)
        # Therefore, fix the mean to the clicked bin center and estimate covariance locally.
        x_c = m.bin_to_x_value(ix, x_edges)
        y_c = m.bin_to_y_value(iy, y_edges)
        # Read local window half-size from UI control if available; fallback to settings
        try:
            window_size = int(m.spinLocalWindow.value())
        except Exception:
            try:
                from .gaussian_settings_dialog import load_gmm_settings
                cfg = load_gmm_settings()
                window_size = int(cfg.get("local_window_bins", 10))
            except Exception:
                window_size = 10
        window_size = max(1, min(window_size, 200))
        mu_local, cov = self._compute_local_moments(H, x_edges, y_edges, ix, iy, window=window_size)
        if cov is None:
            # fallback: use a modest default width if local covariance cannot be estimated
            cov = np.diag([((x_edges[-1]-x_edges[0])/20.0)**2, ((y_edges[-1]-y_edges[0])/20.0)**2])
        # Fix the mean to the clicked center regardless of local mean
        mu = (x_c, y_c)
        # Append to table first to get stable row index
        row_index = self._append_gaussian_row(mu, cov)
        # Determine stable color by row index
        try:
            palette = getattr(m, '_gaussian_palette', ["#ff0000", "#00aa00", "#0000ff", "#aa00aa", "#00aaaa", "#ffaa00"])
            color = palette[row_index % len(palette)] if row_index >= 0 and len(palette) else None
        except Exception:
            color = None
        # Draw overlay with stable color
        self._add_gaussian_overlay(mu, cov, label=f"({mu[0]:.3g},{mu[1]:.3g})", color=color)
        # If marginals are enabled, redraw overlays and marginals immediately
        try:
            if hasattr(m, 'checkBoxShowMarginals') and m.checkBoxShowMarginals.isChecked():
                self._redraw_gaussian_overlays_from_table()
        except Exception:
            pass

    def on_clear_gaussians(self):
        """Remove all Gaussian overlays and clear the table and marginals."""
        m = self.main
        if hasattr(m, 'gaussian_items'):
            for item in m.gaussian_items:
                try:
                    m.overlay_plot.del_item(item)
                except Exception:
                    pass
            m.gaussian_items.clear()
        # Clear marginal items too
        try:
            self._clear_gaussian_marginal_items()
        except Exception:
            pass
        if hasattr(m, 'tableGaussians'):
            try:
                m._updating_gaussian_table = True
                m.tableGaussians.setRowCount(0)
            except Exception:
                pass
            finally:
                m._updating_gaussian_table = False
        m.overlay_plot.replot()

    # ---------------------------- Helpers -------------------------------
    def _transform_params_for_axes(self, mu: Tuple[float, float], cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform mean/covariance to construction space depending on log axes.
        If an axis is log, we linearize around mu using J=diag(1/mu) for that axis.
        Returns (mu_s, cov_s). Safe against non-positive means by clamping with eps.
        """
        mx, my = float(mu[0]), float(mu[1])
        cov = np.asarray(cov, dtype=float).reshape(2, 2)
        mu_s = np.array([mx, my], dtype=float)
        cov_s = cov.copy()
        if self.is_log_x or self.is_log_y:
            eps = 1e-12
            J = np.eye(2, dtype=float)
            if self.is_log_x:
                mx_safe = mx if mx > eps else eps
                mu_s[0] = np.log(mx_safe)
                J[0, 0] = 1.0 / mx_safe
            if self.is_log_y:
                my_safe = my if my > eps else eps
                mu_s[1] = np.log(my_safe)
                J[1, 1] = 1.0 / my_safe
            cov_s = J @ cov @ J.T
        # Minimal regularization if needed
        try:
            eig = np.linalg.eigvalsh(cov_s)
            if np.any(eig <= 0):
                cov_s = cov_s + 1e-9 * np.eye(2)
        except Exception:
            cov_s = cov_s + 1e-9 * np.eye(2)
        return mu_s, cov_s

    def _component_marginal_pdf(self, centers: np.ndarray, mean: float, var: float, is_log_axis: bool) -> np.ndarray:
        """Return 1D marginal density for a single Gaussian component on given centers.
        - If is_log_axis: use log-normal with parameters from linearization at mean.
        - Else: use normal N(mean, var).
        """
        centers = np.asarray(centers, dtype=float)
        var = float(var)
        if not np.isfinite(var) or var <= 0:
            return np.zeros_like(centers, dtype=float)
        s = np.sqrt(var)
        eps = 1e-12
        if is_log_axis:
            m_safe = mean if mean > eps else eps
            var_log = max(var / (m_safe * m_safe), 1e-12)
            s_log = np.sqrt(var_log)
            out = np.zeros_like(centers, dtype=float)
            pos = centers > 0.0
            z = (np.log(centers[pos]) - np.log(m_safe)) / s_log
            out[pos] = np.exp(-0.5 * z * z) / (centers[pos] * s_log * np.sqrt(2 * np.pi))
            return out
        else:
            z = (centers - mean) / s
            return np.exp(-0.5 * z * z) / (s * np.sqrt(2 * np.pi))

    def _compute_moments(self, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray):
        """Compute weighted mean (mu) and covariance (cov) from histogram H."""
        S = float(np.sum(H))
        if not np.isfinite(S) or S <= 0:
            return None, None
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
        W = H.astype(float)
        mx = np.sum(W * X) / S
        my = np.sum(W * Y) / S
        dx = X - mx
        dy = Y - my
        var_x = np.sum(W * dx * dx) / S
        var_y = np.sum(W * dy * dy) / S
        cov_xy = np.sum(W * dx * dy) / S
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=float)
        if not np.all(np.isfinite(cov)):
            return None, None
        return (mx, my), cov

    def _compute_local_moments(self, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, ix: int, iy: int, window: int = 5):
        """Compute moments in a local window of size (2*window+1)^2 around (ix,iy)."""
        nx, ny = H.shape
        x0 = max(0, ix - window)
        x1 = min(nx, ix + window + 1)
        y0 = max(0, iy - window)
        y1 = min(ny, iy + window + 1)
        subH = H[x0:x1, y0:y1]
        if subH.size == 0 or np.sum(subH) <= 0:
            return None, None
        sub_x_edges = x_edges[x0:x1+1]
        sub_y_edges = y_edges[y0:y1+1]
        return self._compute_moments(subH, sub_x_edges, sub_y_edges)

    def _add_gaussian_overlay(self, mu: Tuple[float, float], cov: np.ndarray, label: str = "", color: Optional[str] = None):
        """Create and add a Gaussian ellipse overlay to overlay_plot.
        If an axis is log-scaled, construct the ellipse in log space so it appears
        as a true ellipse on the log-spaced histogram grid, then map back to value
        space for bin-index conversion.
        If `color` is provided, use it; otherwise fall back to the legacy color cycle.
        """
        m = self.main
        try:
            from qwt.plot import QwtPlot
            import guiqwt.styles
            import guiqwt.curve
        except Exception:
            return
        # Ensure overlay axes match histogram
        try:
            _, x_edges, y_edges = m._histogram["2d"]
            m.overlay_plot.setAxisScale(QwtPlot.xBottom, 0, len(x_edges) - 1)
            m.overlay_plot.setAxisScale(QwtPlot.yLeft, 0, len(y_edges) - 1)
        except Exception:
            return
        # Detect log axes
        is_log_x = self.is_log_x
        is_log_y = self.is_log_y
        # Transform parameters to the construction space (log for log-axes)
        mx, my = float(mu[0]), float(mu[1])
        cov = np.asarray(cov, dtype=float).reshape(2, 2)
        # Transform parameters to the construction space (log for log-axes)
        mu_s, cov_s = self._transform_params_for_axes((mx, my), cov)
        # Build ellipse points in construction space for c=1 contour
        vals, vecs = np.linalg.eigh(cov_s)
        vals = np.maximum(vals, 1e-12)
        t = np.linspace(0, 2*np.pi, 200)
        circ = np.vstack([np.cos(t), np.sin(t)])  # 2 x N
        L = np.diag(np.sqrt(vals))
        pts = (vecs @ L @ circ)
        xs_s = pts[0, :] + mu_s[0]
        ys_s = pts[1, :] + mu_s[1]
        # Map construction-space points back to value space for bin conversion
        xs_v = np.array(xs_s, dtype=float)
        ys_v = np.array(ys_s, dtype=float)
        if is_log_x:
            xs_v = np.exp(xs_v)
        if is_log_y:
            ys_v = np.exp(ys_v)
        # Map to bin coordinates
        x_coords = []
        y_coords = []
        for xv, yv in zip(xs_v, ys_v):
            xb = m.value_to_bin(xv, x_edges)
            yb = m.value_to_bin(yv, y_edges)
            if xb is None or yb is None:
                continue
            x_coords.append(xb)
            y_coords.append(yb)
        if not x_coords:
            return
        # Determine color
        if color is None:
            try:
                color = next(m._gaussian_color_cycle)
            except Exception:
                palette = getattr(m, '_gaussian_palette', ["#ff0000", "#00aa00", "#0000ff", "#aa00aa", "#00aaaa", "#ffaa00"])
                m._gaussian_color_cycle = iter(palette)
                color = next(m._gaussian_color_cycle)
        # If collecting colors for marginals, store this color in order
        try:
            if getattr(m, '_collect_gaussian_colors', False):
                if not hasattr(m, '_last_gaussian_draw_colors'):
                    m._last_gaussian_draw_colors = []
                m._last_gaussian_draw_colors.append(color)
        except Exception:
            pass
        curveparam = guiqwt.styles.CurveParam()
        curveparam.line.color = color
        curveparam.line.width = 2.0
        curve_item = guiqwt.curve.CurveItem(curveparam=curveparam)
        curve_item.set_data(x_coords, y_coords)
        m.overlay_plot.add_item(curve_item)
        m.gaussian_items.append(curve_item)
        m.overlay_plot.replot()

    def _append_gaussian_row(self, mu: Tuple[float, float], cov: np.ndarray, w: float = 1.0):
        """Append a Gaussian (mu, cov, w) as a new row in the table."""
        m = self.main
        try:
            m._updating_gaussian_table = True
            row = m.tableGaussians.rowCount()
            m.tableGaussians.insertRow(row)
            vals = [float(mu[0]), float(mu[1]), float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1]), float(w)]
            for col, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(f"{v:.6g}")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                m.tableGaussians.setItem(row, col, item)
            return row
        except Exception:
            return -1
        finally:
            m._updating_gaussian_table = False

    def _update_gaussian_row(self, row: int, mu: np.ndarray, cov: np.ndarray, w: float = None):
        """Update an existing row with new mu, cov (and optionally weight) values."""
        m = self.main
        if not hasattr(m, 'tableGaussians'):
            return
        if row < 0 or row >= m.tableGaussians.rowCount():
            return
        try:
            m._updating_gaussian_table = True
            vals = [float(mu[0]), float(mu[1]), float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])]
            for col, v in enumerate(vals):
                item = m.tableGaussians.item(row, col)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    m.tableGaussians.setItem(row, col, item)
                item.setText(f"{v:.6g}")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            # Optionally update weight in last column if provided and column exists
            if w is not None and m.tableGaussians.columnCount() >= 6:
                item_w = m.tableGaussians.item(row, 5)
                if item_w is None:
                    item_w = QtWidgets.QTableWidgetItem()
                    m.tableGaussians.setItem(row, 5, item_w)
                item_w.setText(f"{float(w):.6g}")
                item_w.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        except Exception:
            pass
        finally:
            m._updating_gaussian_table = False

    def _read_gaussian_table(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Read all rows from the table and return list of (mu(2,), cov(2,2), w)."""
        m = self.main
        out = []
        if not hasattr(m, 'tableGaussians'):
            return out
        try:
            n = m.tableGaussians.rowCount()
            for r in range(n):
                def getf(c):
                    it = m.tableGaussians.item(r, c)
                    if it is None:
                        return None
                    try:
                        return float(it.text())
                    except Exception:
                        return None
                x = getf(0); y = getf(1); cxx = getf(2); cxy = getf(3); cyy = getf(4)
                w = getf(5) if m.tableGaussians.columnCount() >= 6 else 1.0
                if None in (x, y, cxx, cxy, cyy):
                    continue
                if w is None or not np.isfinite(w) or w < 0:
                    w = 1.0
                mu = np.array([x, y], dtype=float)
                cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)
                # Ensure positive semi-definite by minimal regularization
                try:
                    eigvals = np.linalg.eigvalsh(cov)
                    if np.any(eigvals <= 0):
                        cov = cov + 1e-9 * np.eye(2)
                except Exception:
                    cov = cov + 1e-9 * np.eye(2)
                out.append((mu, cov, float(w)))
        except Exception:
            return out
        return out

    def _redraw_gaussian_overlays_from_table(self):
        """Clear and redraw Gaussian overlays from the current table rows."""
        m = self.main
        # Clear current overlays
        if hasattr(m, 'gaussian_items'):
            for item in m.gaussian_items:
                try:
                    m.overlay_plot.del_item(item)
                except Exception:
                    pass
            m.gaussian_items.clear()
        # Also clear marginals prior to redraw
        try:
            self._clear_gaussian_marginal_items()
        except Exception:
            pass
        # Draw again
        try:
            H, x_edges, y_edges = m._histogram["2d"]
        except Exception:
            return
        rows = self._read_gaussian_table()
        # Collect overlay colors in the same order with stable mapping by row index
        m._last_gaussian_draw_colors = []
        m._collect_gaussian_colors = True
        palette = getattr(m, '_gaussian_palette', ["#ff0000", "#00aa00", "#0000ff", "#aa00aa", "#00aaaa", "#ffaa00"])
        for idx, (mu, cov, w) in enumerate(rows):
            color = palette[idx % len(palette)] if len(palette) else "#ff0000"
            self._add_gaussian_overlay((float(mu[0]), float(mu[1])), np.array(cov, dtype=float), color=color)
        m._collect_gaussian_colors = False
        m.overlay_plot.replot()
        # Re-apply selection highlighting after redraw
        try:
            self.on_gaussian_table_selection_changed(None, None)
        except Exception:
            pass
        # Draw marginals if toggled on
        try:
            if hasattr(m, 'checkBoxShowMarginals') and m.checkBoxShowMarginals.isChecked():
                self._draw_gaussian_marginals_from_table(rows, getattr(m, '_last_gaussian_draw_colors', None))
        except Exception:
            pass

    def _clear_gaussian_marginal_items(self):
        """Remove marginal overlay curves from x and y plots."""
        m = self.main
        # X marginals
        try:
            if hasattr(m, 'gaussian_marginal_items_x') and hasattr(m, 'g_xplot'):
                for item in m.gaussian_marginal_items_x:
                    try:
                        m.g_xplot.del_item(item)
                    except Exception:
                        pass
                m.gaussian_marginal_items_x.clear()
                try:
                    m.g_xplot.replot()
                except Exception:
                    pass
        except Exception:
            pass
        # Y marginals
        try:
            if hasattr(m, 'gaussian_marginal_items_y') and hasattr(m, 'g_yplot'):
                for item in m.gaussian_marginal_items_y:
                    try:
                        m.g_yplot.del_item(item)
                    except Exception:
                        pass
                m.gaussian_marginal_items_y.clear()
                try:
                    m.g_yplot.replot()
                except Exception:
                    pass
        except Exception:
            pass

    def _draw_gaussian_marginals_from_table(self, rows, colors=None):
        """Draw 1D marginal distributions for each Gaussian in rows on x and y plots.
        rows: list of (mu, cov, w)
        colors: optional list of color strings matching rows
        """
        m = self.main
        try:
            import guiqwt.styles
            import guiqwt.curve
        except Exception:
            return
        # Clear existing items first
        self._clear_gaussian_marginal_items()
        # Access histograms
        try:
            x_edges, x_counts = m._histogram["x"]
            y_edges, y_counts = m._histogram["y"]
        except Exception:
            return
        # Build centers
        try:
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        except Exception:
            return
        # Detect log axes to match 2D Gaussian handling
        x_max = float(np.nanmax(x_counts)) if len(x_counts) else 1.0
        y_max = float(np.nanmax(y_counts)) if len(y_counts) else 1.0
        x_scale = 0.9 * x_max if x_max > 0 else 1.0
        y_scale = 0.9 * y_max if y_max > 0 else 1.0
        # Prepare storage
        if not hasattr(m, 'gaussian_marginal_items_x'):
            m.gaussian_marginal_items_x = []
        if not hasattr(m, 'gaussian_marginal_items_y'):
            m.gaussian_marginal_items_y = []
        # Default color cycle fallback
        default_colors = ["#ff0000", "#00aa00", "#0000ff", "#aa00aa", "#00aaaa", "#ffaa00"]
        # Normalize weights for visualization
        try:
            w_list = [max(0.0, float(w)) for (mu, cov, w) in rows]
            w_sum = float(np.sum(w_list)) if len(w_list) else 0.0
            if not np.isfinite(w_sum) or w_sum <= 0:
                w_norm = [1.0/len(rows)]*len(rows) if rows else []
            else:
                w_norm = [w/w_sum for w in w_list]
        except Exception:
            w_norm = [1.0/len(rows)]*len(rows) if rows else []
        # Precompute weighted PDFs and colors to get global maxima
        comp_gx = []
        comp_gy = []
        col_list = []
        max_x_comp = 0.0
        max_y_comp = 0.0
        for idx, (row, wn) in enumerate(zip(rows, w_norm)):
            try:
                mu, cov, _w = row
                mx, my = float(mu[0]), float(mu[1])
                varx = float(cov[0, 0])
                vary = float(cov[1, 1])
                sx = np.sqrt(max(varx, 1e-12))
                sy = np.sqrt(max(vary, 1e-12))
            except Exception:
                comp_gx.append(None); comp_gy.append(None); col_list.append(None)
                continue
            # Colors
            color = None
            try:
                if colors is not None and idx < len(colors):
                    color = colors[idx]
            except Exception:
                color = None
            if color is None:
                color = default_colors[idx % len(default_colors)]
            # Compute normalized marginal pdfs matching 2D handling:
            # - linear axis: Normal in value space N(mu, var)
            # - log axis: Log-normal in value space with log-parameters from linearization at mu
            try:
                gx0 = self._component_marginal_pdf(x_centers, mx, varx, self.is_log_x)
                gy0 = self._component_marginal_pdf(y_centers, my, vary, self.is_log_y)
                gxw = gx0 * wn
                gyw = gy0 * wn
                mxx = float(np.nanmax(gxw)) if gxw.size else 0.0
                myy = float(np.nanmax(gyw)) if gyw.size else 0.0
                if np.isfinite(mxx) and mxx > max_x_comp:
                    max_x_comp = mxx
                if np.isfinite(myy) and myy > max_y_comp:
                    max_y_comp = myy
            except Exception:
                gxw = None; gyw = None
            comp_gx.append(gxw)
            comp_gy.append(gyw)
            col_list.append(color)
        # Compute global scale factors so relative amplitudes are preserved
        fac_x = (x_scale / max_x_comp) if max_x_comp > 0 else 1.0
        fac_y = (y_scale / max_y_comp) if max_y_comp > 0 else 1.0
        # Now draw each component with global scaling
        for gxw, gyw, color in zip(comp_gx, comp_gy, col_list):
            if gxw is not None:
                try:
                    gx_plot = gxw * fac_x
                    curveparam_x = guiqwt.styles.CurveParam()
                    curveparam_x.line.color = color
                    curveparam_x.line.width = 2.0
                    item_x = guiqwt.curve.CurveItem(curveparam=curveparam_x)
                    item_x.set_data(list(x_centers), list(gx_plot))
                    m.g_xplot.add_item(item_x)
                    m.gaussian_marginal_items_x.append(item_x)
                except Exception:
                    pass
            if gyw is not None:
                try:
                    gy_plot = gyw * fac_y
                    curveparam_y = guiqwt.styles.CurveParam()
                    curveparam_y.line.color = color
                    curveparam_y.line.width = 2.0
                    item_y = guiqwt.curve.CurveItem(curveparam=curveparam_y)
                    # For Y marginal: counts on X-axis, y-centers on Y-axis
                    item_y.set_data(list(gy_plot), list(y_centers))
                    m.g_yplot.add_item(item_y)
                    m.gaussian_marginal_items_y.append(item_y)
                except Exception:
                    pass
        # Replot 1D plots
        try:
            m.g_xplot.replot()
        except Exception:
            pass
        try:
            m.g_yplot.replot()
        except Exception:
            pass

    def on_toggle_gaussian_marginals(self, checked: bool):
        """Handle toggling of marginal plots visibility."""
        m = self.main
        try:
            if checked:
                rows = self._read_gaussian_table()
                self._draw_gaussian_marginals_from_table(rows, getattr(m, '_last_gaussian_draw_colors', None))
            else:
                self._clear_gaussian_marginal_items()
        except Exception:
            pass

    def on_gaussian_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        """Redraw Gaussian overlays when the user edits any cell in the table."""
        m = self.main
        try:
            if getattr(m, '_updating_gaussian_table', False):
                return
        except Exception:
            pass
        try:
            self._redraw_gaussian_overlays_from_table()
        except Exception:
            pass

    # ----------------------------- Save/Load -----------------------------
    def _rows_to_dicts(self):
        rows = []
        for mu, cov, w in self._read_gaussian_table():
            rows.append({
                "x": float(mu[0]),
                "y": float(mu[1]),
                "cov_xx": float(cov[0, 0]),
                "cov_xy": float(cov[0, 1]),
                "cov_yy": float(cov[1, 1]),
                "w": float(w)
            })
        return rows

    def _current_axes_info(self):
        """Return a dict with axis info: index, name, and scale (linear/log) for x and y.
        Falls back gracefully if plot_control does not provide expected attributes.
        """
        m = self.main
        try:
            idx1, name1 = m.plot_control.p1
        except Exception:
            idx1, name1 = None, None
        try:
            idx2, name2 = m.plot_control.p2
        except Exception:
            idx2, name2 = None, None
        scale_x = "log" if self.is_log_x else "linear"
        scale_y = "log" if self.is_log_y else "linear"
        return {
            "x": {"index": idx1, "name": name1, "scale": scale_x},
            "y": {"index": idx2, "name": name2, "scale": scale_y},
            "fit_in_log": bool(getattr(self, "fit_in_log", False)),
        }

    def _compute_data_marginals(self, H, x_edges, y_edges):
        try:
            x_centers = 0.5 * (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:]))
            y_centers = 0.5 * (np.asarray(y_edges[:-1]) + np.asarray(y_edges[1:]))
            data_x = np.sum(H, axis=1).astype(float)
            data_y = np.sum(H, axis=0).astype(float)
            return x_centers.tolist(), y_centers.tolist(), data_x.tolist(), data_y.tolist()
        except Exception:
            return None, None, None, None

    def _compute_model_marginals(self, rows, x_edges, y_edges, total_counts: float):
        """Compute combined 1D model marginals on centers to match data domain.
        Scales each component by its weight w and overall so that the model area matches data counts.
        """
        try:
            x_centers = 0.5 * (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:]))
            y_centers = 0.5 * (np.asarray(y_edges[:-1]) + np.asarray(y_edges[1:]))
            dx = float(np.mean(np.diff(x_centers))) if len(x_centers) > 1 else 1.0
            dy = float(np.mean(np.diff(y_centers))) if len(y_centers) > 1 else 1.0
            model_x = np.zeros_like(x_centers, dtype=float)
            model_y = np.zeros_like(y_centers, dtype=float)
            # normalize weights
            wsum = sum(max(0.0, float(w)) for (_, _, w) in rows) or 1.0
            for (mu, cov, w) in rows:
                try:
                    w = max(0.0, float(w)) / wsum
                    mx, my = float(mu[0]), float(mu[1])
                    varx = float(cov[0, 0]); vary = float(cov[1, 1])
                    if not np.isfinite(varx) or varx <= 0: continue
                    if not np.isfinite(vary) or vary <= 0: continue
                    gx = self._component_marginal_pdf(x_centers, mx, varx, self.is_log_x)
                    gy = self._component_marginal_pdf(y_centers, my, vary, self.is_log_y)
                    model_x += w * gx
                    model_y += w * gy
                except Exception:
                    continue
            # convert densities to counts roughly by multiplying bin width and total counts
            model_x_counts = (model_x * dx * total_counts).tolist()
            model_y_counts = (model_y * dy * total_counts).tolist()
            return x_centers.tolist(), y_centers.tolist(), model_x_counts, model_y_counts
        except Exception:
            return None, None, None, None

    def _compute_model_grid_2d(self, rows, x_edges, y_edges, total_counts: float):
        """Compute 2D model counts grid aligned with data histogram bins.
        - Evaluate mixture density at bin centers in value space.
        - Account for log axes by transforming to log space for Gaussian evaluation and applying Jacobian factors.
        - Convert densities to counts by multiplying by bin areas and total_counts.
        Returns (x_centers, y_centers, M) where M shape matches (nx, ny).
        """
        try:
            x_edges = np.asarray(x_edges, dtype=float)
            y_edges = np.asarray(y_edges, dtype=float)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            nx = x_centers.size
            ny = y_centers.size
            if nx == 0 or ny == 0:
                return x_centers.tolist(), y_centers.tolist(), None
            # Bin widths (value space)
            dx = np.diff(x_edges)
            dy = np.diff(y_edges)
            # Precompute transformed centers and jacobian factors
            Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')
            # Mask invalid for log axes
            valid = np.ones_like(Xc, dtype=bool)
            if self.is_log_x:
                valid &= (Xc > 0.0)
            if self.is_log_y:
                valid &= (Yc > 0.0)
            # Transformed coordinates
            Sx = np.log(Xc, where=(Xc>0.0), out=np.zeros_like(Xc)) if self.is_log_x else Xc
            Sy = np.log(Yc, where=(Yc>0.0), out=np.zeros_like(Yc)) if self.is_log_y else Yc
            # Jacobian determinant factor |d(s)/d(v)| = 1/(x^alpha y^beta)
            jac = np.ones_like(Xc, dtype=float)
            if self.is_log_x:
                jac = jac / np.maximum(Xc, 1e-300)
            if self.is_log_y:
                jac = jac / np.maximum(Yc, 1e-300)
            # Normalize weights
            rows_full = rows
            if not rows_full:
                return x_centers.tolist(), y_centers.tolist(), None
            wsum = sum(max(0.0, float(w)) for (_, _, w) in rows_full) or 1.0
            density = np.zeros((nx, ny), dtype=float)
            for (mu, cov, w) in rows_full:
                try:
                    w = max(0.0, float(w)) / wsum
                    mx, my = float(mu[0]), float(mu[1])
                    cov = np.asarray(cov, dtype=float).reshape(2, 2)
                    # Transform mean/cov to construction space
                    mu_s, cov_s = self._transform_params_for_axes((mx, my), cov)
                    # Evaluate 2D normal in transformed space
                    try:
                        inv = np.linalg.inv(cov_s)
                        det = float(np.linalg.det(cov_s))
                        if not np.isfinite(det) or det <= 0:
                            continue
                        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
                    except Exception:
                        continue
                    # Quadratic form for all grid points
                    dxs = Sx - mu_s[0]
                    dys = Sy - mu_s[1]
                    Q = inv[0,0]*dxs*dxs + 2.0*inv[0,1]*dxs*dys + inv[1,1]*dys*dys
                    comp = norm * np.exp(-0.5 * Q)
                    comp = comp * jac
                    comp[~valid] = 0.0
                    density += w * comp
                except Exception:
                    continue
            # Multiply by bin areas (outer product of dx and dy) and total counts
            A = np.outer(dx, dy)
            M = density * A * float(total_counts)
            return x_centers.tolist(), y_centers.tolist(), M
        except Exception:
            return None, None, None

    def on_save_gaussians(self):
        m = self.main
        rows = self._rows_to_dicts()
        if not rows:
            QtWidgets.QMessageBox.information(m, "Save Gaussians", "There are no Gaussian rows to save.")
            return
        # Ask for base filename
        default_dir = os.path.expanduser("~")
        base_path, _ = QtWidgets.QFileDialog.getSaveFileName(m, "Save Gaussian Fits", default_dir, "Gaussian Files (*.json *.csv);;All Files (*.*)")
        if not base_path:
            return
        root, ext = os.path.splitext(base_path)
        if ext.lower() in (".json", ".csv"):
            base = root
        else:
            base = base_path
        json_path = base + ".json"
        gauss_csv_path = base + "_gaussians.csv"
        hist_csv_path = base + "_hist2d.csv"
        model_csv_path = base + "_model2d.csv"
        margx_csv_path = base + "_marginal_x.csv"
        margy_csv_path = base + "_marginal_y.csv"
        axes_info = self._current_axes_info()
        # Collect histogram and compute marginals
        marg = {}
        try:
            H, x_edges, y_edges = m._histogram["2d"]
            if H is not None and H.size:
                x_centers, y_centers, data_x, data_y = self._compute_data_marginals(H, x_edges, y_edges)
                total_counts = float(np.sum(H)) if np.isfinite(np.sum(H)) else 0.0
                # rows for model from table reader
                rows_full = self._read_gaussian_table()
                # Convert to same tuple layout (mu,cov,w)
                model_xc, model_yc, model_x, model_y = self._compute_model_marginals(rows_full, x_edges, y_edges, total_counts)
                # Also compute individual component marginals
                comps_x = []
                comps_y = []
                # normalize weights
                wsum = sum(max(0.0, float(w)) for (_, _, w) in rows_full) or 1.0
                for (mu, cov, w) in rows_full:
                    try:
                        wn = max(0.0, float(w)) / wsum
                        mx, my = float(mu[0]), float(mu[1])
                        varx = float(cov[0, 0]); vary = float(cov[1, 1])
                        gx = self._component_marginal_pdf(x_centers, mx, varx, self.is_log_x)
                        gy = self._component_marginal_pdf(y_centers, my, vary, self.is_log_y)
                        # convert to counts similar to model scaling
                        dxw = float(np.mean(np.diff(x_centers))) if len(x_centers) > 1 else 1.0
                        dyw = float(np.mean(np.diff(y_centers))) if len(y_centers) > 1 else 1.0
                        comps_x.append((wn * gx * dxw * total_counts).tolist())
                        comps_y.append((wn * gy * dyw * total_counts).tolist())
                    except Exception:
                        comps_x.append([])
                        comps_y.append([])
                marg = {
                    "x": {"centers": x_centers, "data": data_x, "model": model_x, "components": comps_x},
                    "y": {"centers": y_centers, "data": data_y, "model": model_y, "components": comps_y},
                }
        except Exception:
            marg = {}
        # Save JSON (single file including marginals and axes)
        try:
            payload = {
                "type": "ndxplorer.gaussians",
                "version": 3,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "columns": ["x", "y", "cov_xx", "cov_xy", "cov_yy", "w"],
                "axes": axes_info,
                "rows": rows,
                "marginals": marg,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Save Error", f"Failed to save JSON:\n{e}")
            return
        # Save Gaussians CSV (separate file)
        try:
            with open(gauss_csv_path, "w", newline="", encoding="utf-8") as f:
                try:
                    xinfo = axes_info.get("x", {})
                    yinfo = axes_info.get("y", {})
                    f.write(f"# ndxplorer.gaussians version=3\n")
                    f.write(f"# created={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# x_axis index={xinfo.get('index')} name={xinfo.get('name')} scale={xinfo.get('scale')}\n")
                    f.write(f"# y_axis index={yinfo.get('index')} name={yinfo.get('name')} scale={yinfo.get('scale')}\n")
                    f.write(f"# fit_in_log={axes_info.get('fit_in_log', False)}\n")
                except Exception:
                    pass
                writer = csv.DictWriter(f, fieldnames=["x", "y", "cov_xx", "cov_xy", "cov_yy", "w"]) 
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Save Error", f"Failed to save Gaussians CSV:\n{e}")
            return
        # Save 2D histogram CSV (separate file)
        try:
            with open(hist_csv_path, "w", newline="", encoding="utf-8") as f:
                try:
                    xinfo = axes_info.get("x", {})
                    yinfo = axes_info.get("y", {})
                    f.write(f"# ndxplorer.hist2d version=1\n")
                    f.write(f"# created={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# x_axis index={xinfo.get('index')} name={xinfo.get('name')} scale={xinfo.get('scale')}\n")
                    f.write(f"# y_axis index={yinfo.get('index')} name={yinfo.get('name')} scale={yinfo.get('scale')}\n")
                    f.write(f"# shape={list(H.shape) if 'H' in locals() and H is not None else None}\n")
                except Exception:
                    pass
                if 'H' in locals() and H is not None and H.size:
                    # header row: first empty cell then y-centers
                    y_centers = 0.5 * (np.asarray(y_edges[:-1]) + np.asarray(y_edges[1:]))
                    header = ["x\\y"] + [float(v) for v in y_centers]
                    writer = csv.writer(f)
                    writer.writerow(header)
                    x_centers = 0.5 * (np.asarray(x_edges[:-1]) + np.asarray(x_edges[1:]))
                    for i, xv in enumerate(x_centers):
                        row = [float(xv)] + [float(v) for v in H[i, :].tolist()]
                        writer.writerow(row)
                else:
                    f.write("# No histogram available\n")
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Save Error", f"Failed to save 2D histogram CSV:\n{e}")
            return
        # Save 2D model CSV (mixture and individual components)
        try:
            with open(model_csv_path, "w", newline="", encoding="utf-8") as f:
                try:
                    xinfo = axes_info.get("x", {})
                    yinfo = axes_info.get("y", {})
                    f.write(f"# ndxplorer.model2d version=2\n")
                    f.write(f"# created={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# x_axis index={xinfo.get('index')} name={xinfo.get('name')} scale={xinfo.get('scale')}\n")
                    f.write(f"# y_axis index={yinfo.get('index')} name={yinfo.get('name')} scale={yinfo.get('scale')}\n")
                except Exception:
                    pass
                if 'H' in locals() and H is not None and H.size:
                    rows_full = self._read_gaussian_table()
                    # Mixture grid
                    xc, yc, M = self._compute_model_grid_2d(rows_full, x_edges, y_edges, float(np.sum(H)))
                    writer = csv.writer(f)
                    if M is not None:
                        f.write("# section=mixture\n")
                        header = ["x\\y"] + [float(v) for v in yc]
                        writer.writerow(header)
                        for i, xv in enumerate(xc):
                            row = [float(xv)] + [float(v) for v in M[i, :].tolist()]
                            writer.writerow(row)
                    else:
                        f.write("# No model available\n")
                    # Individual component grids
                    if rows_full:
                        for idx, comp in enumerate(rows_full):
                            xc1, yc1, M1 = self._compute_model_grid_2d([comp], x_edges, y_edges, float(np.sum(H)))
                            f.write(f"# section=component index={idx}\n")
                            if M1 is not None:
                                header1 = ["x\\y"] + [float(v) for v in yc1]
                                writer.writerow(header1)
                                for i, xv in enumerate(xc1):
                                    row1 = [float(xv)] + [float(v) for v in M1[i, :].tolist()]
                                    writer.writerow(row1)
                            else:
                                f.write("# component empty\n")
                else:
                    f.write("# No histogram available (cannot compute model grid)\n")
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Save Error", f"Failed to save 2D model CSV:\n{e}")
            return
        # Save 1D marginals CSV (X)
        try:
            with open(margx_csv_path, "w", newline="", encoding="utf-8") as f:
                try:
                    xinfo = axes_info.get("x", {})
                    yinfo = axes_info.get("y", {})
                    f.write(f"# ndxplorer.marginal_x version=1\n")
                    f.write(f"# created={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# x_axis index={xinfo.get('index')} name={xinfo.get('name')} scale={xinfo.get('scale')}\n")
                    f.write(f"# y_axis index={yinfo.get('index')} name={yinfo.get('name')} scale={yinfo.get('scale')}\n")
                except Exception:
                    pass
                writer = csv.writer(f)
                # dynamic header includes component columns
                comp_count = 0
                comps = []
                if marg and "x" in marg:
                    comps = marg["x"].get("components") or []
                    comp_count = len(comps)
                header = ["center", "data", "model"] + [f"comp_{i}" for i in range(comp_count)]
                writer.writerow(header)
                if marg and "x" in marg:
                    xc = marg["x"].get("centers") or []
                    dx = marg["x"].get("data") or []
                    mx = marg["x"].get("model") or []
                    n = min(len(xc), len(dx), len(mx))
                    for i in range(n):
                        row = [float(xc[i]), float(dx[i]), float(mx[i])]
                        for k in range(comp_count):
                            try:
                                row.append(float(comps[k][i]))
                            except Exception:
                                row.append(0.0)
                        writer.writerow(row)
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Save Error", f"Failed to save X marginal CSV:\n{e}")
            return
        # Save 1D marginals CSV (Y)
        try:
            with open(margy_csv_path, "w", newline="", encoding="utf-8") as f:
                try:
                    xinfo = axes_info.get("x", {})
                    yinfo = axes_info.get("y", {})
                    f.write(f"# ndxplorer.marginal_y version=1\n")
                    f.write(f"# created={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# x_axis index={xinfo.get('index')} name={xinfo.get('name')} scale={xinfo.get('scale')}\n")
                    f.write(f"# y_axis index={yinfo.get('index')} name={yinfo.get('name')} scale={yinfo.get('scale')}\n")
                except Exception:
                    pass
                writer = csv.writer(f)
                comp_count = 0
                comps = []
                if marg and "y" in marg:
                    comps = marg["y"].get("components") or []
                    comp_count = len(comps)
                header = ["center", "data", "model"] + [f"comp_{i}" for i in range(comp_count)]
                writer.writerow(header)
                if marg and "y" in marg:
                    yc = marg["y"].get("centers") or []
                    dy_ = marg["y"].get("data") or []
                    my_ = marg["y"].get("model") or []
                    n = min(len(yc), len(dy_), len(my_))
                    for i in range(n):
                        row = [float(yc[i]), float(dy_[i]), float(my_[i])]
                        for k in range(comp_count):
                            try:
                                row.append(float(comps[k][i]))
                            except Exception:
                                row.append(0.0)
                        writer.writerow(row)
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Save Error", f"Failed to save Y marginal CSV:\n{e}")
            return
        QtWidgets.QMessageBox.information(m, "Saved", (
            "Saved files:\n"
            + os.path.basename(json_path) + "\n"
            + os.path.basename(gauss_csv_path) + "\n"
            + os.path.basename(hist_csv_path) + "\n"
            + os.path.basename(model_csv_path) + "\n"
            + os.path.basename(margx_csv_path) + "\n"
            + os.path.basename(margy_csv_path)
        ))

    def _load_from_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "rows" in data:
            rows = data.get("rows", [])
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        out = []
        for r in rows:
            try:
                x = float(r["x"]) ; y = float(r["y"]) ; cxx = float(r["cov_xx"]) ; cxy = float(r["cov_xy"]) ; cyy = float(r["cov_yy"]) ; w = float(r.get("w", 1.0))
            except Exception:
                continue
            out.append((np.array([x, y], dtype=float), np.array([[cxx, cxy],[cxy, cyy]], dtype=float), float(w)))
        return out, (data.get("axes") if isinstance(data, dict) else None)

    def _load_from_csv(self, path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    x = float(r.get("x")) ; y = float(r.get("y"))
                    cxx = float(r.get("cov_xx")) ; cxy = float(r.get("cov_xy")) ; cyy = float(r.get("cov_yy"))
                    w = float(r.get("w", 1.0))
                except Exception:
                    continue
                out.append((np.array([x, y], dtype=float), np.array([[cxx, cxy],[cxy, cyy]], dtype=float), float(w)))
        return out

    def on_load_gaussians(self):
        m = self.main
        start_dir = os.path.expanduser("~")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(m, "Load Gaussian Fits", start_dir, "Gaussian Files (*.json *.csv);;All Files (*.*)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        axes_meta = None
        try:
            if ext == ".json":
                rows, axes_meta = self._load_from_json(path)
            elif ext == ".csv":
                rows = self._load_from_csv(path)
            else:
                # Try JSON first, then CSV
                try:
                    rows, axes_meta = self._load_from_json(path)
                except Exception:
                    rows = self._load_from_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(m, "Load Error", f"Failed to load file:\n{e}")
            return
        if not rows:
            QtWidgets.QMessageBox.warning(m, "Load Gaussians", "No valid Gaussian rows found in the selected file.")
            return
        # Optional: warn if axes differ from current configuration
        try:
            if axes_meta:
                cur = self._current_axes_info()
                def s(ax):
                    a = axes_meta.get(ax, {}) ; b = cur.get(ax, {})
                    return f"{a.get('name')} ({a.get('scale')})" , f"{b.get('name')} ({b.get('scale')})"
                (sx_old, sx_cur), (sy_old, sy_cur) = (s('x'), s('y'))
                if sx_old != sx_cur or sy_old != sy_cur:
                    QtWidgets.QMessageBox.information(m, "Axis Mismatch", f"File axes:\nX: {sx_old}\nY: {sy_old}\nCurrent axes:\nX: {sx_cur}\nY: {sy_cur}\n\nGaussians were loaded regardless.")
        except Exception:
            pass
        # Clear existing table and populate
        try:
            m._updating_gaussian_table = True
            m.tableGaussians.setRowCount(0)
            for mu, cov, w in rows:
                self._append_gaussian_row((float(mu[0]), float(mu[1])), np.array(cov, dtype=float), float(w))
        finally:
            m._updating_gaussian_table = False
        # Redraw overlays from the table
        try:
            self._redraw_gaussian_overlays_from_table()
        except Exception:
            pass

    # ------------------------- Event filtering ---------------------------
    def eventFilter(self, obj, event):
        """Intercept Delete key presses on the Gaussians table to delete selected rows."""
        try:
            m = self.main
            if obj is getattr(m, 'tableGaussians', None) and event.type() == QtCore.QEvent.KeyPress:
                if event.key() in (QtCore.Qt.Key_Delete,):
                    sel_model = m.tableGaussians.selectionModel()
                    if sel_model is not None:
                        rows = [idx.row() for idx in sel_model.selectedRows()]
                        if rows:
                            self._delete_selected_gaussian_rows(rows)
                            return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _delete_selected_gaussian_rows(self, rows: List[int]):
        """Delete the given selected rows from the Gaussians table and refresh overlays."""
        m = self.main
        if not hasattr(m, 'tableGaussians') or not rows:
            return
        try:
            m._updating_gaussian_table = True
            for r in sorted(set(rows), reverse=True):
                if 0 <= r < m.tableGaussians.rowCount():
                    m.tableGaussians.removeRow(r)
        except Exception:
            pass
        finally:
            m._updating_gaussian_table = False
        try:
            self._redraw_gaussian_overlays_from_table()
        except Exception:
            pass

    # ----------------------- Optional visibility hook --------------------
    def on_fit_dock_visibility_changed(self, visible: bool = False):
        """When the Fit dock visibility changes, keep UX consistent:
        - If becoming visible, automatically enable 'Select point' for seamless interaction.
        - If becoming hidden, disable select mode and clear point mode in the mouse filter.
        """
        print("on_fit_dock_visibility_changed", visible)
        m = self.main
        m.btnSelectPoint.setChecked(visible)
        if visible:
            m.mouse_event_filter.set_point_mode(visible, callback=self.on_point_selected)
        else:
            m.mouse_event_filter.set_point_mode(False, callback=None)
