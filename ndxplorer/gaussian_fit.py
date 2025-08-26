"""
Gaussian fitting UI attachment and logic for ndxplorer.

This module encapsulates the construction of the Gaussian Fit controls and
attaches them to the main NDXplorer window, keeping plot_main.py cleaner.
It wires all Gaussian-related signal handlers here and implements the logic
(fitting, overlays, table I/O, marginals, and Delete-key row removal).
"""
from typing import Iterator, Optional, Tuple, List

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

    def __init__(self, main: QtWidgets.QMainWindow):
        super().__init__(main)
        self.main = main
        self._build_ui()
        self._connect_signals()

    # ------------------------------ UI ---------------------------------
    def _build_ui(self):
        m = self.main
        # Button row: Fit + Clear + Select + Marginals
        btn_row = QtWidgets.QHBoxLayout()
        m.btnFit2DGauss = QtWidgets.QPushButton("Fit", m)
        m.btnClearGaussians = QtWidgets.QPushButton("Clear", m)
        m.btnSelectPoint = QtWidgets.QCheckBox("Select point", m)
        m.checkBoxShowMarginals = QtWidgets.QCheckBox("Show marginals", m)
        m.checkBoxShowMarginals.setChecked(True)
        btn_row.addWidget(m.btnFit2DGauss)
        btn_row.addWidget(m.btnClearGaussians)
        btn_row.addWidget(m.btnSelectPoint)
        btn_row.addWidget(m.checkBoxShowMarginals)
        # Place the button row directly into the target layout
        m.verticalLayout_18.addLayout(btn_row)

        # Local window controls for click-based covariance estimation
        m.lblLocalWindow = QtWidgets.QLabel("Local window (bins):", m)
        m.spinLocalWindow = QtWidgets.QSpinBox(m)
        m.spinLocalWindow.setRange(1, 200)
        m.spinLocalWindow.setSingleStep(1)
        m.spinLocalWindow.setValue(10)
        m.spinLocalWindow.setToolTip("Size of half-window in bins for local covariance estimation around the clicked point.\nEffective window size is (2*value+1) in each dimension.")
        window_row = QtWidgets.QHBoxLayout()
        window_row.addWidget(m.lblLocalWindow)
        window_row.addWidget(m.spinLocalWindow)
        window_row.addStretch(1)
        m.verticalLayout_18.addLayout(window_row)

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
        # Table edits update overlays
        m.tableGaussians.itemChanged.connect(self.on_gaussian_table_item_changed)
        # Marginals toggle
        m.checkBoxShowMarginals.toggled.connect(self.on_toggle_gaussian_marginals)
        # Install event filter on table to allow Delete key to remove rows
        m.tableGaussians.installEventFilter(self)

    # ---------------------------- Handlers ------------------------------
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

        # Keep only points within the currently visible histogram range
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

        # Filter to finite rows
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.any(finite_mask):
            QtWidgets.QMessageBox.warning(m, "No data", "Selected data contains no finite values for fitting.")
            return
        X = X[finite_mask]

        # Lazy import of sklearn GaussianMixture via documented API
        try:
            from sklearn.mixture import GaussianMixture
        except Exception:
            QtWidgets.QMessageBox.warning(m, "scikit-learn Not Available", "GaussianMixture (scikit-learn) is not installed.")
            return

        n_components = len(rows)
        # Build initialization arrays according to sklearn GaussianMixture API
        means_init = np.array([mu for (mu, cov, w) in rows], dtype=float)
        covs = np.array([cov for (mu, cov, w) in rows], dtype=float)
        precs = np.array([np.linalg.pinv(c) for c in covs])
        w_init = np.array([max(0.0, float(w)) for (mu, cov, w) in rows], dtype=float)
        s = np.sum(w_init)
        if not np.isfinite(s) or s <= 0:
            w_init = np.ones(n_components, dtype=float) / n_components
        else:
            w_init = w_init / s

        gm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=0,
            init_params='kmeans',
            max_iter=200,
            means_init=means_init,
            precisions_init=precs,
            weights_init=w_init
        )

        # Fit directly to visible raw data without weighting
        try:
            gm.fit(X)
        except Exception as e:
            QtWidgets.QMessageBox.warning(m, "Fit failed", f"GMM fit failed: {e}")
            return

        # Retrieve fitted parameters
        means = np.array(gm.means_, dtype=float)
        try:
            covariances = np.array(gm.covariances_, dtype=float)
        except Exception:
            try:
                precisions = np.array(gm.precisions_, dtype=float)
                covariances = np.array([np.linalg.pinv(P) for P in precisions])
            except Exception:
                QtWidgets.QMessageBox.warning(m, "Fit failed", "Could not retrieve covariances from the fitted model.")
                return
        try:
            weights_fitted = np.array(getattr(gm, 'weights_', None), dtype=float)
        except Exception:
            weights_fitted = None

        for i in range(n_components):
            wi = None
            try:
                if weights_fitted is not None and i < len(weights_fitted):
                    wi = float(weights_fitted[i])
            except Exception:
                wi = None
            self._update_gaussian_row(i, means[i], covariances[i], wi)
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
        # Read local window half-size from UI control if available
        try:
            window_size = int(m.spinLocalWindow.value())
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
            pass
        # Build ellipse points in value space for c=1 contour
        mx, my = mu
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-12)
        t = np.linspace(0, 2*np.pi, 200)
        circ = np.vstack([np.cos(t), np.sin(t)])  # 2 x N
        L = np.diag(np.sqrt(vals))
        pts = (vecs @ L @ circ)
        xs = pts[0, :] + mx
        ys = pts[1, :] + my
        # Map to bin coordinates
        x_coords = []
        y_coords = []
        for xv, yv in zip(xs, ys):
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
        # Determine scaling based on current counts
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
            # Compute normalized gaussian pdfs (area=1), then weight by wn
            try:
                gx0 = np.exp(-0.5 * ((x_centers - mx) / sx) ** 2) / (sx * np.sqrt(2 * np.pi))
                gy0 = np.exp(-0.5 * ((y_centers - my) / sy) ** 2) / (sy * np.sqrt(2 * np.pi))
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

    # ------------------------- Event filtering ---------------------------
    def eventFilter(self, obj, event):
        """Intercept Delete key presses on the Gaussians table to delete selected rows."""
        m = self.main
        try:
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
    def on_fit_dock_visibility_changed(self, visible: bool):
        """Ensure select mode is disabled when the Fit dock is hidden."""
        m = self.main
        try:
            if not visible:
                if hasattr(m, 'btnSelectPoint') and m.btnSelectPoint is not None:
                    try:
                        if m.btnSelectPoint.isChecked():
                            m.btnSelectPoint.setChecked(False)
                    except Exception:
                        pass
                if hasattr(m, 'mouse_event_filter') and m.mouse_event_filter is not None:
                    try:
                        m.mouse_event_filter.set_point_mode(False, callback=None)
                    except Exception:
                        pass
        except Exception:
            pass
