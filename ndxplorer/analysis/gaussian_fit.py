"""
Gaussian fitting UI attachment and logic for ndxplorer.

This module encapsulates the construction of the Gaussian Fit controls and
attaches them to the main NDXplorer window, keeping plot_main.py cleaner.
It wires all Gaussian-related signal handlers here and implements the logic
(fitting, overlays, table I/O, marginals, and Delete-key row removal).
"""
from ..logging_config import logging
from typing import Iterator, Optional, Tuple, List

import json
import csv
import os
from datetime import datetime

import numpy as np
from qtpy import QtCore, QtWidgets



class GaussianMixtureFixedEM:
    """
    Minimal 2D GMM EM implementation that supports fixing subset of means/covariances per component.
    Work in 'fit space' (i.e., already log-transformed axes if needed).
    """
    def __init__(self, means_init, covs_init, weights_init=None,
                 reg_covar=1e-6, max_iter=200, tol=1e-3, verbose=0, weight_floor=0.0):
        self.means_ = np.array(means_init, dtype=float)         # (K,2)
        self.covs_  = np.array(covs_init, dtype=float)          # (K,2,2)
        K = self.means_.shape[0]
        if weights_init is None:
            self.weights_ = np.ones(K, dtype=float) / K
        else:
            w = np.array(weights_init, dtype=float)
            s = float(np.sum(w)); self.weights_ = (w/s) if s > 0 else (np.ones(K)/K)
        self.reg_covar   = float(reg_covar)
        self.max_iter    = int(max_iter)
        self.tol         = float(tol)
        self.verbose     = int(verbose)
        self.weight_floor= float(max(0.0, weight_floor))
        self.converged_  = False
        self.n_iter_     = 0
        self.lower_bound_= -np.inf

    @staticmethod
    def _log_gaussian_2d(X, mu, cov):
        # X: (N,2), mu: (2,), cov: (2,2)
        # return log N(x|mu,cov) for each row
        try:
            L = np.linalg.cholesky(cov)
        except Exception:
            # fallback via eig-clip
            ev, V = np.linalg.eigh(cov)
            ev = np.maximum(ev, 1e-12)
            cov = (V @ np.diag(ev) @ V.T)
            L = np.linalg.cholesky(cov)
        diff = X - mu[None, :]
        # solve L y = diff^T  -> y^T = L^{-1} diff
        y = np.linalg.solve(L, diff.T)  # (2,N)
        maha = np.sum(y*y, axis=0)      # (N,)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5*(maha + log_det + 2*np.log(2*np.pi))

    @staticmethod
    def _nearest_psd(M, eps=1e-12):
        M = 0.5*(M + M.T)
        ev, V = np.linalg.eigh(M)
        ev = np.maximum(ev, eps)
        return (V @ np.diag(ev) @ V.T)

    def fit(self, X, fix_mu_mask, fix_cov_mask, mu_fixed_vals, cov_fixed_vals):
        """
        X: (N,2)
        fix_mu_mask: (K,2) bool   (True=fix that mean element)
        fix_cov_mask:(K,2,2) bool (True=fix that cov element; symmetric)
        mu_fixed_vals: (K,2) float
        cov_fixed_vals:(K,2,2) float
        """
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        K = self.means_.shape[0]

        # regularize initial covs
        for k in range(K):
            self.covs_[k] = self._nearest_psd(self.covs_[k]) + self.reg_covar*np.eye(2)

        def e_step():
            # compute responsibilities (N,K)
            log_prob = np.empty((N, K), dtype=float)
            for k in range(K):
                log_prob[:, k] = (np.log(self.weights_[k]+1e-300) +
                                  self._log_gaussian_2d(X, self.means_[k], self.covs_[k]))
            # log-sum-exp
            m = np.max(log_prob, axis=1, keepdims=True)
            lse = m + np.log(np.sum(np.exp(log_prob - m), axis=1, keepdims=True))
            log_resp = log_prob - lse
            resp = np.exp(log_resp)
            lower_bound = float(np.sum(lse))
            return resp, lower_bound

        def m_step(resp):
            Nk = np.clip(np.sum(resp, axis=0), 1e-12, np.inf)  # (K,)
            self.weights_ = Nk / float(N)

            # means
            new_means = (resp.T @ X) / Nk[:, None]  # (K,2)
            # apply mean constraints
            for k in range(K):
                if fix_mu_mask[k, 0]: new_means[k, 0] = mu_fixed_vals[k, 0]
                if fix_mu_mask[k, 1]: new_means[k, 1] = mu_fixed_vals[k, 1]
            self.means_ = new_means

            # covariances
            new_covs = np.zeros_like(self.covs_)
            for k in range(K):
                diff = X - self.means_[k][None, :]
                Sk = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
                Sk = self._nearest_psd(Sk) + self.reg_covar*np.eye(2)
                # apply element-wise constraints (keep symmetry)
                Cfix = cov_fixed_vals[k]
                Mfix = fix_cov_mask[k]
                if np.any(Mfix):
                    Sk[Mfix] = Cfix[Mfix]
                    Sk = 0.5*(Sk + Sk.T)
                    Sk = self._nearest_psd(Sk) + self.reg_covar*np.eye(2)
                new_covs[k] = Sk
            self.covs_ = new_covs

        # EM loop
        prev_lb = -np.inf
        for it in range(1, self.max_iter+1):
            resp, lb = e_step()
            m_step(resp)
            improve = lb - prev_lb
            if self.verbose and (it % 10 == 0 or it == 1):
                print(f"[EM] iter={it}  lower_bound={lb:.6f}  +{improve:.6f}")
            if improve < self.tol:
                self.converged_ = True
                self.lower_bound_ = lb
                self.n_iter_ = it
                return self
            prev_lb = lb
        self.lower_bound_ = prev_lb
        self.n_iter_ = self.max_iter
        return self


class GaussianFit(QtCore.QObject):
    """
    Helper that constructs and wires the Gaussian Fit UI into the provided
    main window (NDXplorer instance). It creates the widgets directly in
    verticalLayout_18 as requested and owns all Gaussian-related logic.

    UI elements are stored on `main` for consistency with existing code.
    Plot/marginal state lists (gaussian_items, etc.) and palette are also
    initialized on `main` to minimize changes elsewhere.
    """

    # ---- Table column indices (6 columns) ----
    COL_X = 0
    COL_Y = 1
    COL_CXX = 2
    COL_CYY = 3
    COL_CXY = 4
    COL_W = 5

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
    def _num_item(self, val: float) -> QtWidgets.QTableWidgetItem:
        # Display numbers in scientific notation with 2 decimals; align left so checkbox appears in front of text
        it = QtWidgets.QTableWidgetItem(f"{float(val):.2e}")
        it.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        return it

    def _make_fixable(self, item: QtWidgets.QTableWidgetItem, checked: bool = False):
        # Add an in-cell checkbox without losing editability
        flags = (item.flags()
                 | QtCore.Qt.ItemIsUserCheckable
                 | QtCore.Qt.ItemIsEditable
                 | QtCore.Qt.ItemIsEnabled
                 | QtCore.Qt.ItemIsSelectable)
        item.setFlags(flags)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def _is_fixed_item(self, item: QtWidgets.QTableWidgetItem) -> bool:
        return item is not None and item.checkState() == QtCore.Qt.Checked

    def _make_check_item(self, checked: bool = False) -> QtWidgets.QTableWidgetItem:
        it = QtWidgets.QTableWidgetItem("")
        it.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        return it

    def _is_checked(self, item: Optional[QtWidgets.QTableWidgetItem]) -> bool:
        return (item is not None) and (item.checkState() == QtCore.Qt.Checked)

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
        m.btnGMMSettings.setToolTip("Configure built-in GMM (EM) parameters and save them to your user settings.")
        btn_row.addWidget(m.btnFit2DGauss)
        btn_row.addWidget(m.btnClearGaussians)
        btn_row.addWidget(m.btnSelectGaussian)
        btn_row.addWidget(lblSigma)
        btn_row.addWidget(m.spinSelectionSigma)
        btn_row.addWidget(m.btnGMMSettings)
        # Place the button row directly into the target layout
        m.verticalLayout_18.addLayout(btn_row)

        m.tableGaussians = QtWidgets.QTableWidget(m)
        m.tableGaussians.setColumnCount(6)
        m.tableGaussians.setHorizontalHeaderLabels(["x", "y", "sd_x", "sd_y", "rho", "w"])
        header = m.tableGaussians.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setMinimumSectionSize(20)
        # Reduce vertical spacing between rows
        vheader = m.tableGaussians.verticalHeader()
        vheader.setVisible(False)
        try:
            # Compute a compact default row height based on current font
            fm = m.tableGaussians.fontMetrics()
            row_h = max(16, fm.height() + 2)  # minimal padding
            vheader.setDefaultSectionSize(row_h)
        except Exception:
            pass
        # Remove extra item padding; keep a couple of horizontal pixels for readability
        m.tableGaussians.setStyleSheet("QTableWidget::item{padding:0px 2px;} QTableWidget{gridline-color: palette(mid);} ")
        m.tableGaussians.setWordWrap(False)
        m.tableGaussians.setShowGrid(True)
        m.tableGaussians.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        m.tableGaussians.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        m.tableGaussians.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
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

        # Add context menu on the Gaussians table to remove selected rows
        try:
            m.tableGaussians.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            m.tableGaussians.customContextMenuRequested.connect(self._on_gaussian_table_context_menu)
        except Exception:
            pass
        
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
            # Skip tuple placeholders from simple backend - they don't have pen() method
            if isinstance(it, tuple):
                continue
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
                x = getf(self.COL_X); y = getf(self.COL_Y); sd_x = getf(self.COL_CXX); sd_y = getf(self.COL_CYY); rho = getf(self.COL_CXY)
                if None in (x, y, sd_x, rho, sd_y):
                    continue
                mu_v = np.array([x, y], dtype=float)
                try:
                    sdx = max(0.0, float(sd_x)); sdy = max(0.0, float(sd_y)); rh = float(rho)
                    rh = np.clip(rh, -1.0, 1.0)
                    cov_xy = rh * sdx * sdy
                    cov_v = np.array([[sdx*sdx, cov_xy], [cov_xy, sdy*sdy]], dtype=float)
                except Exception:
                    cov_v = np.array([[0.0, 0.0],[0.0, 0.0]], dtype=float)
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
            H, x_edges, y_edges = m._histogram["2d"]
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

        # Build init arrays (fit space) and FIX masks/values
        rows_full = self._read_gaussian_table_with_fixed()
        n_components = len(rows_full)

        means_init_fit = np.zeros((n_components, 2), dtype=float)
        covs_fit       = np.zeros((n_components, 2, 2), dtype=float)
        weights_init   = np.zeros((n_components,), dtype=float)
        fix_mu_mask    = np.zeros((n_components, 2), dtype=bool)
        fix_cov_mask   = np.zeros((n_components, 2, 2), dtype=bool)
        mu_fixed_vals  = np.zeros((n_components, 2), dtype=float)
        cov_fixed_vals = np.zeros((n_components, 2, 2), dtype=float)

        # --- helpers to map between value space and fit space (log axes supported) ---
        eps = 1e-12

        def to_fit_space(mu_v: np.ndarray, cov_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Map (mu, cov) from value space → fit space.
               For log-axes we linearize log() around mu via J = diag(1/mu)."""
            mu_v = np.asarray(mu_v, dtype=float).reshape(2)
            cov_v = np.asarray(cov_v, dtype=float).reshape(2, 2)

            mu_z = mu_v.copy()
            J = np.eye(2, dtype=float)

            if is_log_x:
                mx = mu_v[0] if mu_v[0] > eps else eps
                mu_z[0] = np.log(mx)
                J[0, 0] = 1.0 / mx
            if is_log_y:
                my = mu_v[1] if mu_v[1] > eps else eps
                mu_z[1] = np.log(my)
                J[1, 1] = 1.0 / my

            cov_z = J @ cov_v @ J.T
            try:
                if np.any(np.linalg.eigvalsh(cov_z) <= 0):
                    cov_z = cov_z + 1e-9 * np.eye(2)
            except Exception:
                cov_z = cov_z + 1e-9 * np.eye(2)
            return mu_z, cov_z

        def to_value_space(mu_z: np.ndarray, cov_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Map (mu, cov) from fit space → value space (inverse of above)."""
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
            try:
                if np.any(np.linalg.eigvalsh(cov_v) <= 0):
                    cov_v = cov_v + 1e-9 * np.eye(2)
            except Exception:
                cov_v = cov_v + 1e-9 * np.eye(2)
            return mu_v, cov_v

        # --- end helpers ---

        for k, r in enumerate(rows_full):
            mu_v, cov_v, w = r["mu"], r["cov"], r["w"]
            mu_z, cov_z = (mu_v, cov_v)
            if np.any(log_axes):
                mu_z, cov_z = to_fit_space(mu_v, cov_v)
            means_init_fit[k] = mu_z
            covs_fit[k]       = cov_z
            weights_init[k]   = max(0.0, float(w))
            # fix masks
            fix_mu_mask[k]    = r["fix_mu"]
            fix_cov_mask[k]   = r["fix_cov"]
            # fixed values (in FIT space!)
            mu_fv, cov_fv = mu_z.copy(), cov_z.copy()
            # ensure if fixed, values come from the current row
            mu_fixed_vals[k]  = mu_fv
            cov_fixed_vals[k] = cov_fv

        s = float(np.sum(weights_init))
        if not np.isfinite(s) or s <= 0:
            weights_init[:] = 1.0 / n_components
        else:
            weights_init /= s

        # Load settings
        cfg = self._get_gmm_settings()
        reg_covar = float(cfg.get('reg_covar', 1e-6))
        tol       = float(cfg.get('tol', 1e-3))
        max_iter  = int(cfg.get('max_iter', 200))
        verbose   = int(cfg.get('verbose', 0))
        weight_floor = float(cfg.get('weight_floor', 0.0))

        # Run EM with constraints in FIT space
        em = GaussianMixtureFixedEM(
            means_init=means_init_fit,
            covs_init=covs_fit,
            weights_init=weights_init,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            weight_floor=weight_floor,
        ).fit(
            X_fit,
            fix_mu_mask=fix_mu_mask,
            fix_cov_mask=fix_cov_mask,
            mu_fixed_vals=mu_fixed_vals,
            cov_fixed_vals=cov_fixed_vals
        )

        means_fit = np.array(em.means_, dtype=float)
        covariances_fit = np.array(em.covs_, dtype=float)
        weights_fitted = np.array(em.weights_, dtype=float)

        # Transform back to VALUE space and update table
        for i in range(n_components):
            mu_i = means_fit[i]
            cov_i = covariances_fit[i]
            if np.any(log_axes):
                mu_v_i, cov_v_i = to_value_space(mu_i, cov_i)
            else:
                mu_v_i, cov_v_i = mu_i, cov_i
            self._update_gaussian_row(i, mu_v_i, cov_v_i, float(weights_fitted[i]))

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
        # Append to table first to get stable row index, honoring default fix flags
        try:
            cfg = self._get_gmm_settings()
            fix_new = bool(cfg.get("fix_new_means", True))
        except Exception:
            fix_new = True
        row_index = self._append_gaussian_row(mu, cov, fix_x=fix_new, fix_y=fix_new)
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
        
        # Handle simple backend (DrawingOverlayWidget)
        try:
            m.overlay_plot.clear_curves()
        except Exception:
            pass
        
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
        
        # Update display
        try:
            if hasattr(m, '_use_simple_backend') and m._use_simple_backend:
                m.overlay_plot.update()
            else:
                m.overlay_plot.replot()
        except Exception:
            pass

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
        m = self.main
        self._add_gaussian_overlay_simple(mu, cov, label, color)

    def _add_gaussian_overlay_simple(self, mu: Tuple[float, float], cov: np.ndarray, label: str = "", color: Optional[str] = None):
        """Add Gaussian ellipse overlay for simple backend using DrawingOverlayWidget."""
        m = self.main
        try:
            from qtpy.QtGui import QColor
        except Exception:
            return
        
        # Get histogram data for coordinate conversion
        try:
            H, x_edges, y_edges = m._histogram["2d"]
        except Exception:
            return
        
        # Detect log axes
        is_log_x = self.is_log_x
        is_log_y = self.is_log_y
        
        # Transform parameters to the construction space (log for log-axes)
        mx, my = float(mu[0]), float(mu[1])
        cov = np.asarray(cov, dtype=float).reshape(2, 2)
        mu_s, cov_s = self._transform_params_for_axes((mx, my), cov)
        
        # Build ellipse points in construction space for multiple contour levels (1σ, 2σ, 3σ)
        vals, vecs = np.linalg.eigh(cov_s)
        vals = np.maximum(vals, 1e-12)
        sigma_levels = [1.0, 2.0, 3.0]  # 1σ, 2σ, 3σ contours
        t = np.linspace(0, 2*np.pi, 200)
        circ = np.vstack([np.cos(t), np.sin(t)])  # 2 x N
        
        for sigma_idx, sigma_level in enumerate(sigma_levels):
            L = np.diag(np.sqrt(vals) * sigma_level)
            pts = (vecs @ L @ circ)
            xs_s = pts[0, :] + mu_s[0]
            ys_s = pts[1, :] + mu_s[1]
            
            # Map construction-space points back to value space
            xs_v = np.array(xs_s, dtype=float)
            ys_v = np.array(ys_s, dtype=float)
            if is_log_x:
                xs_v = np.exp(xs_v)
            if is_log_y:
                ys_v = np.exp(ys_v)

            # Convert value-space coordinates to bin/pixel coordinates
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
                continue

            # Determine color (only once for the first sigma level)
            if sigma_idx == 0:
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

            # Convert color string to QColor
            if isinstance(color, str):
                qcolor = QColor(color)
            else:
                qcolor = color

            # Make 1σ thickest, 2σ medium, 3σ thinnest
            line_width = int(4.0 - sigma_idx)  # 4, 3, 2 for 1σ, 2σ, 3σ
            line_width = max(1, line_width)

            # Add curve to overlay widget
            try:
                m.overlay_plot.add_curve(np.array(x_coords), np.array(y_coords), color=qcolor, width=line_width)
                if not hasattr(m, 'gaussian_items'):
                    m.gaussian_items = []
                m.gaussian_items.append(('curve', len(x_coords), len(y_coords)))
            except Exception:
                pass
        
        # Trigger overlay update
        try:
            m.overlay_plot.update()
        except Exception:
            pass

    def _append_gaussian_row(self, mu, cov, w: float = 1.0,
                             fix_x=False, fix_y=False, fix_cxx=False, fix_cxy=False, fix_cyy=False):
        m = self.main
        try:
            m._updating_gaussian_table = True
            row = m.tableGaussians.rowCount()
            m.tableGaussians.insertRow(row)

            # numeric items
            ix = self._num_item(mu[0]);
            m.tableGaussians.setItem(row, self.COL_X, ix)
            iy = self._num_item(mu[1]);
            m.tableGaussians.setItem(row, self.COL_Y, iy)
            # store standard deviations and correlation instead of raw covariances
            var_x = float(cov[0, 0]); var_y = float(cov[1, 1]); cov_xy = float(cov[0, 1])
            sd_x = np.sqrt(max(var_x, 0.0)); sd_y = np.sqrt(max(var_y, 0.0))
            denom = sd_x*sd_y if sd_x>0 and sd_y>0 else 0.0
            rho = cov_xy/denom if denom>0 else 0.0
            icxx = self._num_item(sd_x); m.tableGaussians.setItem(row, self.COL_CXX, icxx)
            icxy = self._num_item(rho);  m.tableGaussians.setItem(row, self.COL_CXY, icxy)
            icyy = self._num_item(sd_y); m.tableGaussians.setItem(row, self.COL_CYY, icyy)
            iw = self._num_item(w);
            # Override weight formatting to normal notation with two decimals
            iw.setText(f"{float(w):.2f}")
            iw.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            m.tableGaussians.setItem(row, self.COL_W, iw)

            # make fixable (checkbox in same cell). Weight is NOT fixable.
            self._make_fixable(ix, fix_x)
            self._make_fixable(iy, fix_y)
            self._make_fixable(icxx, fix_cxx)
            self._make_fixable(icxy, fix_cxy)
            self._make_fixable(icyy, fix_cyy)

            return row
        finally:
            m._updating_gaussian_table = False

    def _update_gaussian_row(self, row: int, mu: np.ndarray, cov: np.ndarray, w: float = None):
        m = self.main
        if not hasattr(m, 'tableGaussians'): return
        if row < 0 or row >= m.tableGaussians.rowCount(): return
        try:
            m._updating_gaussian_table = True

            def _set(col, val, fixable=False):
                it = m.tableGaussians.item(row, col)
                if it is None:
                    it = self._num_item(val)
                    if col == self.COL_W:
                        it.setText(f"{float(val):.2f}")
                        it.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                    m.tableGaussians.setItem(row, col, it)
                    if fixable: self._make_fixable(it, False)
                else:
                    # keep the current check state
                    cs = it.checkState()
                    if col == self.COL_W:
                        it.setText(f"{float(val):.2f}")
                    else:
                        it.setText(f"{float(val):.2e}")
                    it.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                    if fixable:
                        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                        it.setCheckState(cs)

            _set(self.COL_X, float(mu[0]), fixable=True)
            _set(self.COL_Y, float(mu[1]), fixable=True)
            # update from covariance but display as sd and rho
            var_x = float(cov[0, 0]); var_y = float(cov[1, 1]); cov_xy = float(cov[0, 1])
            sd_x = np.sqrt(max(var_x, 0.0)); sd_y = np.sqrt(max(var_y, 0.0))
            denom = sd_x*sd_y if sd_x>0 and sd_y>0 else 0.0
            rho = cov_xy/denom if denom>0 else 0.0
            _set(self.COL_CXX, sd_x, fixable=True)
            _set(self.COL_CXY, rho,  fixable=True)
            _set(self.COL_CYY, sd_y, fixable=True)
            if w is not None:
                _set(self.COL_W, float(w), fixable=False)
        finally:
            m._updating_gaussian_table = False

    def _read_gaussian_table_with_fixed(self):
        m = self.main
        out = []
        n = m.tableGaussians.rowCount()
        for r in range(n):
            def getf(c):
                it = m.tableGaussians.item(r, c)
                if it is None: return None
                try:
                    return float(it.text())
                except Exception:
                    return None

            x = getf(self.COL_X);
            y = getf(self.COL_Y)
            sd_x = getf(self.COL_CXX);
            rho  = getf(self.COL_CXY);
            sd_y = getf(self.COL_CYY)
            w = getf(self.COL_W) or 1.0
            if None in (x, y, sd_x, rho, sd_y): continue

            mu = np.array([x, y], dtype=float)
            try:
                sdx = max(0.0, float(sd_x)); sdy = max(0.0, float(sd_y)); rh = float(rho)
                rh = np.clip(rh, -1.0, 1.0)
                cov_xy = rh * sdx * sdy
                cov = np.array([[sdx*sdx, cov_xy], [cov_xy, sdy*sdy]], dtype=float)
            except Exception:
                cov = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
            try:
                ev = np.linalg.eigvalsh(cov)
                if np.any(ev <= 0): cov = cov + 1e-9 * np.eye(2)
            except Exception:
                cov = cov + 1e-9 * np.eye(2)

            fix_mu = np.array([
                self._is_fixed_item(m.tableGaussians.item(r, self.COL_X)),
                self._is_fixed_item(m.tableGaussians.item(r, self.COL_Y)),
            ], dtype=bool)
            fix_cov = np.array([
                [self._is_fixed_item(m.tableGaussians.item(r, self.COL_CXX)),
                 self._is_fixed_item(m.tableGaussians.item(r, self.COL_CXY))],
                [self._is_fixed_item(m.tableGaussians.item(r, self.COL_CXY)),
                 self._is_fixed_item(m.tableGaussians.item(r, self.COL_CYY))]
            ], dtype=bool)

            out.append({"mu": mu, "cov": cov, "w": float(w),
                        "fix_mu": fix_mu, "fix_cov": fix_cov})
        return out

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
                x = getf(self.COL_X); y = getf(self.COL_Y); sd_x = getf(self.COL_CXX); sd_y = getf(self.COL_CYY); rho = getf(self.COL_CXY)
                w = getf(self.COL_W) if m.tableGaussians.columnCount() >= 6 else 1.0
                if None in (x, y, sd_x, rho, sd_y):
                    continue
                if w is None or not np.isfinite(w) or w < 0:
                    w = 1.0
                mu = np.array([x, y], dtype=float)
                try:
                    sdx = max(0.0, float(sd_x)); sdy = max(0.0, float(sd_y)); rh = float(rho)
                    rh = np.clip(rh, -1.0, 1.0)
                    cov_xy = rh * sdx * sdy
                    cov = np.array([[sdx*sdx, cov_xy], [cov_xy, sdy*sdy]], dtype=float)
                except Exception:
                    cov = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
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
        try:
            m.overlay_plot.clear_curves()
        except Exception:
            pass
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

        # Trigger resize / otherwise plot flipped ? - Ugly fix
        def _do_update():
            m.update_plots()
        QtCore.QTimer.singleShot(1, _do_update)

    def _clear_gaussian_marginal_items(self):
        m = self.main
        try:
            if hasattr(m, 'gaussian_marginal_items_x') and hasattr(m, 'g_xplot'):
                for item in m.gaussian_marginal_items_x:
                    try:
                        m.g_xplot.getPlotItem().removeItem(item)
                    except Exception:
                        pass
                m.gaussian_marginal_items_x.clear()
                m.g_xplot.replot()
        except Exception:
            pass
        try:
            if hasattr(m, 'gaussian_marginal_items_y') and hasattr(m, 'g_yplot'):
                for item in m.gaussian_marginal_items_y:
                    try:
                        m.g_yplot.getPlotItem().removeItem(item)
                    except Exception:
                        pass
                m.gaussian_marginal_items_y.clear()
                m.g_yplot.replot()
        except Exception:
            pass

    def _draw_gaussian_marginals_from_table(self, rows, colors=None):
        m = self.main
        self._clear_gaussian_marginal_items()
        try:
            x_edges, x_vals = m._histogram["x"]
            y_edges, y_vals = m._histogram["y"]
        except Exception:
            return
        try:
            x_edges = np.asarray(x_edges, dtype=float)
            y_edges = np.asarray(y_edges, dtype=float)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            x_bw = np.diff(x_edges)
            y_bw = np.diff(y_edges)
        except Exception:
            return
        norm_x = bool(getattr(m.plot_control, 'normed_hist_x', False)) if hasattr(m, 'plot_control') else False
        norm_y = bool(getattr(m.plot_control, 'normed_hist_y', False)) if hasattr(m, 'plot_control') else False
        Nx = float(np.nansum(x_vals)) if (x_vals is not None and len(x_vals)) and not norm_x else 1.0
        Ny = float(np.nansum(y_vals)) if (y_vals is not None and len(y_vals)) and not norm_y else 1.0
        if not hasattr(m, 'gaussian_marginal_items_x'):
            m.gaussian_marginal_items_x = []
        if not hasattr(m, 'gaussian_marginal_items_y'):
            m.gaussian_marginal_items_y = []
        default_colors = ["#ff0000", "#00aa00", "#0000ff", "#aa00aa", "#00aaaa", "#ffaa00"]
        try:
            w_list = [max(0.0, float(w)) for (_mu, _cov, w) in rows]
            w_sum = float(np.sum(w_list)) if len(w_list) else 0.0
            w_norm = [w / w_sum for w in w_list] if (np.isfinite(w_sum) and w_sum > 0) else ([1.0/len(rows)]*len(rows) if rows else [])
        except Exception:
            w_norm = [1.0/len(rows)]*len(rows) if rows else []
        for idx, (row, wn) in enumerate(zip(rows, w_norm)):
            try:
                mu, cov, _w = row
                mx, my = float(mu[0]), float(mu[1])
                varx = float(cov[0, 0])
                vary = float(cov[1, 1])
                if not (np.isfinite(varx) and varx > 0 and np.isfinite(vary) and vary > 0):
                    continue
            except Exception:
                continue
            color = None
            try:
                if colors is not None and idx < len(colors):
                    color = colors[idx]
            except Exception:
                color = None
            if color is None:
                color = default_colors[idx % len(default_colors)]
            gx_pdf = self._component_marginal_pdf(x_centers, mx, varx, self.is_log_x)
            gy_pdf = self._component_marginal_pdf(y_centers, my, vary, self.is_log_y)
            if norm_x:
                gx_plot = wn * gx_pdf
            else:
                gx_plot = wn * gx_pdf * x_bw * Nx
            if norm_y:
                gy_plot = wn * gy_pdf
            else:
                gy_plot = wn * gy_pdf * y_bw * Ny
            try:
                item_x = m.g_xplot.getPlotItem().plot(list(x_centers), list(gx_plot), pen=color)
                m.gaussian_marginal_items_x.append(item_x)
            except Exception:
                pass
            try:
                item_y = m.g_yplot.getPlotItem().plot(list(gy_plot), list(y_centers), pen=color)
                m.gaussian_marginal_items_y.append(item_y)
            except Exception:
                pass
        m.g_xplot.replot()
        m.g_yplot.replot()

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
        m = self.main
        n = m.tableGaussians.rowCount()
        for r in range(n):
            def getf(c):
                it = m.tableGaussians.item(r, c)
                if it is None: return None
                try: return float(it.text())
                except Exception: return None
            x   = getf(self.COL_X)
            y   = getf(self.COL_Y)
            cxx = getf(self.COL_CXX)
            cxy = getf(self.COL_CXY)
            cyy = getf(self.COL_CYY)
            w   = getf(self.COL_W) or 1.0
            if None in (x, y, cxx, cxy, cyy):
                continue
            rows.append({
                "x": x, "y": y, "sd_x": cxx, "rho": cxy, "sd_y": cyy, "w": w,
                "fix_x": self._is_fixed_item(m.tableGaussians.item(r, self.COL_X)),
                "fix_y": self._is_fixed_item(m.tableGaussians.item(r, self.COL_Y)),
                "fix_sd_x": self._is_fixed_item(m.tableGaussians.item(r, self.COL_CXX)),
                "fix_rho": self._is_fixed_item(m.tableGaussians.item(r, self.COL_CXY)),
                "fix_sd_y": self._is_fixed_item(m.tableGaussians.item(r, self.COL_CYY)),
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
                "version": 4,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "columns": ["x","y","sd_x","sd_y","rho","w",
                            "fix_x","fix_y","fix_sd_x","fix_sd_y","fix_rho"],
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
                    f.write(f"# ndxplorer.gaussians version=4\n")
                    f.write(f"# created={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# x_axis index={xinfo.get('index')} name={xinfo.get('name')} scale={xinfo.get('scale')}\n")
                    f.write(f"# y_axis index={yinfo.get('index')} name={yinfo.get('name')} scale={yinfo.get('scale')}\n")
                    f.write(f"# fit_in_log={axes_info.get('fit_in_log', False)}\n")
                except Exception:
                    pass
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "x","y","sd_x","sd_y","rho","w",
                        "fix_x","fix_y","fix_sd_x","fix_sd_y","fix_rho"
                    ]
                )
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
        rows = data.get("rows", data if isinstance(data, list) else [])
        out = []
        for r in rows:
            try:
                x = float(r["x"]); y = float(r["y"])
                # support both old cov_* and new sd/rho
                if "sd_x" in r or "rho" in r or "sd_y" in r:
                    sd_x = float(r.get("sd_x", 0.0)); rho = float(r.get("rho", 0.0)); sd_y = float(r.get("sd_y", 0.0))
                    rho = float(np.clip(rho, -1.0, 1.0))
                    cov_xy = rho * sd_x * sd_y
                    cxx = sd_x*sd_x; cxy = cov_xy; cyy = sd_y*sd_y
                    fx  = bool(r.get("fix_x", False))
                    fy  = bool(r.get("fix_y", False))
                    fcx = bool(r.get("fix_sd_x", False))
                    fcy = bool(r.get("fix_rho", False))
                    fcyy= bool(r.get("fix_sd_y", False))
                else:
                    cxx = float(r["cov_xx"]); cxy = float(r["cov_xy"]); cyy = float(r["cov_yy"]) 
                    fx  = bool(r.get("fix_x", False))
                    fy  = bool(r.get("fix_y", False))
                    fcx = bool(r.get("fix_cxx", False))
                    fcy = bool(r.get("fix_cxy", False))
                    fcyy= bool(r.get("fix_cyy", False))
                w = float(r.get("w", 1.0))
            except Exception:
                continue
            out.append((
                np.array([x, y], dtype=float),
                np.array([[cxx, cxy],[cxy, cyy]], dtype=float),
                float(w),
                fx, fy, fcx, fcy, fcyy
            ))
        return out, (data.get("axes") if isinstance(data, dict) else None)

    def _load_from_csv(self, path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    x   = float(r.get("x"))
                    y   = float(r.get("y"))
                    if r.get("sd_x") is not None or r.get("rho") is not None or r.get("sd_y") is not None:
                        sd_x = float(r.get("sd_x", 0.0))
                        rho  = float(r.get("rho", 0.0))
                        sd_y = float(r.get("sd_y", 0.0))
                        rho = float(np.clip(rho, -1.0, 1.0))
                        cov_xy = rho * sd_x * sd_y
                        cxx = sd_x*sd_x; cxy = cov_xy; cyy = sd_y*sd_y
                        fx  = r.get("fix_x", "False").strip().lower() in ("1","true","yes","y")
                        fy  = r.get("fix_y", "False").strip().lower() in ("1","true","yes","y")
                        fcx = r.get("fix_sd_x", "False").strip().lower() in ("1","true","yes","y")
                        fcy = r.get("fix_rho", "False").strip().lower() in ("1","true","yes","y")
                        fcyy= r.get("fix_sd_y", "False").strip().lower() in ("1","true","yes","y")
                    else:
                        cxx = float(r.get("cov_xx"))
                        cxy = float(r.get("cov_xy"))
                        cyy = float(r.get("cov_yy"))
                        fx  = r.get("fix_x", "False").strip().lower() in ("1","true","yes","y")
                        fy  = r.get("fix_y", "False").strip().lower() in ("1","true","yes","y")
                        fcx = r.get("fix_cxx", "False").strip().lower() in ("1","true","yes","y")
                        fcy = r.get("fix_cxy", "False").strip().lower() in ("1","true","yes","y")
                        fcyy= r.get("fix_cyy", "False").strip().lower() in ("1","true","yes","y")
                    w   = float(r.get("w", 1.0))
                except Exception:
                    continue
                out.append((
                    np.array([x, y], dtype=float),
                    np.array([[cxx, cxy],[cxy, cyy]], dtype=float),
                    float(w),
                    fx, fy, fcx, fcy, fcyy
                ))
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

        # Clear existing and populate
        try:
            m._updating_gaussian_table = True
            m.tableGaussians.setRowCount(0)
            for row in rows:
                if len(row) == 8:  # from JSON/CSV with fix flags
                    mu, cov, w, fx, fy, fcx, fcy, fcyy = row
                    self._append_gaussian_row((float(mu[0]), float(mu[1])), np.array(cov, dtype=float), float(w),
                                              fix_x=fx, fix_y=fy, fix_cxx=fcx, fix_cxy=fcy, fix_cyy=fcyy)
                else:  # legacy (no fix flags)
                    mu, cov, w = row
                    self._append_gaussian_row((float(mu[0]), float(mu[1])), np.array(cov, dtype=float), float(w))
        finally:
            m._updating_gaussian_table = False

        # Redraw overlays from the table
        try:
            self._redraw_gaussian_overlays_from_table()
        except Exception:
            pass

    # ------------------------- Event filtering ---------------------------
    def _on_gaussian_table_context_menu(self, pos: QtCore.QPoint):
        """Show context menu on the Gaussians table to remove selected gaussians.
        Right-click -> Remove Gaussian(s) removes all selected rows.
        """
        m = self.main
        try:
            table = m.tableGaussians
            if table is None:
                return
            # Map the point to global for the popup menu
            global_pos = table.viewport().mapToGlobal(pos)
            sel_model = table.selectionModel()
            selected_rows = []
            if sel_model is not None:
                selected_rows = [idx.row() for idx in sel_model.selectedRows()]
            menu = QtWidgets.QMenu(table)
            act_remove = QtWidgets.QAction("Remove Gaussian(s)", menu)
            act_remove.setEnabled(len(selected_rows) > 0)
            menu.addAction(act_remove)
            action = menu.exec_(global_pos)
            if action == act_remove and selected_rows:
                self._delete_selected_gaussian_rows(selected_rows)
        except Exception:
            pass

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
