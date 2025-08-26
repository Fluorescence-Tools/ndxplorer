"""
GMM Settings dialog for ndxplorer.
Allows users to configure sklearn.mixture.GaussianMixture parameters
and persists them in the ndxplorer user settings folder.
"""
from typing import Optional, Dict, Any

try:
    from chisurf.gui import QtWidgets
except ImportError:
    from qtpy import QtWidgets

import json
from .settings import get_settings_path, ensure_default_settings

_DEFAULTS: Dict[str, Any] = {
    "covariance_type": "full",          # {'full', 'tied', 'diag', 'spherical'}
    "tol": 1e-3,                         # float > 0
    "reg_covar": 1e-6,                   # float >= 0
    "max_iter": 200,                     # int > 0
    "n_init": 1,                         # int >= 1
    "init_params": "kmeans",            # {'kmeans', 'random'}
    "random_state": None,                # int or None
    "warm_start": False,                 # bool
    "verbose": 0,                        # int >= 0
    "verbose_interval": 10,              # int >= 1 (used if verbose > 0)
    "local_window_bins": 10              # int >= 1, half-window size in bins for local covariance
}

_SETTINGS_FILENAME = "gmm_settings.json"


def load_gmm_settings() -> Dict[str, Any]:
    """Load user GMM settings or return defaults if missing/corrupt."""
    ensure_default_settings()
    settings_path = get_settings_path()
    fn = settings_path / _SETTINGS_FILENAME
    if not fn.exists():
        # write defaults
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(_DEFAULTS, f, indent=2)
        except Exception:
            return dict(_DEFAULTS)
        return dict(_DEFAULTS)
    try:
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        # merge with defaults to keep compatibility
        merged = dict(_DEFAULTS)
        merged.update({k: data.get(k, v) for k, v in _DEFAULTS.items()})
        return merged
    except Exception:
        return dict(_DEFAULTS)


def save_gmm_settings(cfg: Dict[str, Any]) -> None:
    """Persist GMM settings to the user settings folder."""
    ensure_default_settings()
    settings_path = get_settings_path()
    fn = settings_path / _SETTINGS_FILENAME
    # sanitize types
    out = dict(_DEFAULTS)
    out.update({k: cfg.get(k, v) for k, v in _DEFAULTS.items()})
    try:
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass


class GMMSettingsDialog(QtWidgets.QDialog):
    """Qt dialog exposing GaussianMixture configuration parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GMM Settings")
        self._cfg = load_gmm_settings()
        self._build_ui()
        self._load_to_widgets()

    def _build_ui(self):
        layout = QtWidgets.QFormLayout(self)

        # covariance_type
        self.combo_cov = QtWidgets.QComboBox(self)
        self.combo_cov.addItems(["full", "tied", "diag", "spherical"])
        layout.addRow("Covariance type:", self.combo_cov)

        # tol
        self.spin_tol = QtWidgets.QDoubleSpinBox(self)
        self.spin_tol.setDecimals(8)
        self.spin_tol.setRange(1e-12, 1.0)
        self.spin_tol.setSingleStep(1e-3)
        layout.addRow("Tolerance (tol):", self.spin_tol)

        # reg_covar
        self.spin_reg = QtWidgets.QDoubleSpinBox(self)
        self.spin_reg.setDecimals(12)
        self.spin_reg.setRange(0.0, 1.0)
        self.spin_reg.setSingleStep(1e-6)
        layout.addRow("Reg. covar:", self.spin_reg)

        # max_iter
        self.spin_max_iter = QtWidgets.QSpinBox(self)
        self.spin_max_iter.setRange(1, 10000)
        layout.addRow("Max iterations:", self.spin_max_iter)

        # n_init
        self.spin_n_init = QtWidgets.QSpinBox(self)
        self.spin_n_init.setRange(1, 1000)
        layout.addRow("n_init:", self.spin_n_init)

        # init_params
        self.combo_init = QtWidgets.QComboBox(self)
        self.combo_init.addItems(["kmeans", "random"])
        layout.addRow("Init params:", self.combo_init)

        # random_state
        rs_layout = QtWidgets.QHBoxLayout()
        self.check_rs_enable = QtWidgets.QCheckBox("Use fixed random_state")
        self.spin_rs = QtWidgets.QSpinBox(self)
        self.spin_rs.setRange(-2**31, 2**31-1)
        rs_layout.addWidget(self.check_rs_enable)
        rs_layout.addWidget(self.spin_rs)
        layout.addRow("Random state:", rs_layout)

        # warm_start
        self.check_warm = QtWidgets.QCheckBox(self)
        layout.addRow("Warm start:", self.check_warm)

        # verbose
        self.spin_verbose = QtWidgets.QSpinBox(self)
        self.spin_verbose.setRange(0, 10)
        layout.addRow("Verbose:", self.spin_verbose)

        # verbose_interval
        self.spin_verbose_int = QtWidgets.QSpinBox(self)
        self.spin_verbose_int.setRange(1, 1000)
        layout.addRow("Verbose interval:", self.spin_verbose_int)

        # local window (bins) for local covariance estimation in Gaussian Fit
        self.spin_local_window = QtWidgets.QSpinBox(self)
        self.spin_local_window.setRange(1, 200)
        self.spin_local_window.setSingleStep(1)
        layout.addRow("Local window (bins):", self.spin_local_window)

        # buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self
        )
        # Add explicit Save button (saves without closing)
        self.btn_save = QtWidgets.QPushButton("Save", self)
        btn_box.addButton(self.btn_save, QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_save.setToolTip("Save settings to your user folder without closing this dialog")
        self.btn_save.clicked.connect(self.on_save_clicked)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addRow(btn_box)

        # react to verbose
        self.spin_verbose.valueChanged.connect(self._on_verbose_changed)

    def _on_verbose_changed(self, v: int):
        self.spin_verbose_int.setEnabled(v > 0)

    def _load_to_widgets(self):
        cfg = self._cfg
        self.combo_cov.setCurrentText(str(cfg.get("covariance_type", _DEFAULTS["covariance_type"])) )
        self.spin_tol.setValue(float(cfg.get("tol", _DEFAULTS["tol"])) )
        self.spin_reg.setValue(float(cfg.get("reg_covar", _DEFAULTS["reg_covar"])) )
        self.spin_max_iter.setValue(int(cfg.get("max_iter", _DEFAULTS["max_iter"])) )
        self.spin_n_init.setValue(int(cfg.get("n_init", _DEFAULTS["n_init"])) )
        self.combo_init.setCurrentText(str(cfg.get("init_params", _DEFAULTS["init_params"])) )
        rs = cfg.get("random_state", _DEFAULTS["random_state"]) 
        if rs is None:
            self.check_rs_enable.setChecked(False)
            self.spin_rs.setEnabled(False)
            self.spin_rs.setValue(0)
        else:
            self.check_rs_enable.setChecked(True)
            self.spin_rs.setEnabled(True)
            try:
                self.spin_rs.setValue(int(rs))
            except Exception:
                self.spin_rs.setValue(0)
        self.check_warm.setChecked(bool(cfg.get("warm_start", _DEFAULTS["warm_start"])) )
        self.spin_verbose.setValue(int(cfg.get("verbose", _DEFAULTS["verbose"])) )
        self.spin_verbose_int.setValue(int(cfg.get("verbose_interval", _DEFAULTS["verbose_interval"])) )
        # local window bins
        try:
            self.spin_local_window.setValue(int(cfg.get("local_window_bins", _DEFAULTS["local_window_bins"])) )
        except Exception:
            self.spin_local_window.setValue(_DEFAULTS["local_window_bins"])
        self._on_verbose_changed(self.spin_verbose.value())

        # connect enabling toggle
        self.check_rs_enable.toggled.connect(self.spin_rs.setEnabled)

    def get_settings(self) -> Dict[str, Any]:
        cfg = {
            "covariance_type": self.combo_cov.currentText(),
            "tol": float(self.spin_tol.value()),
            "reg_covar": float(self.spin_reg.value()),
            "max_iter": int(self.spin_max_iter.value()),
            "n_init": int(self.spin_n_init.value()),
            "init_params": self.combo_init.currentText(),
            "random_state": int(self.spin_rs.value()) if self.check_rs_enable.isChecked() else None,
            "warm_start": bool(self.check_warm.isChecked()),
            "verbose": int(self.spin_verbose.value()),
            "verbose_interval": int(self.spin_verbose_int.value()),
            "local_window_bins": int(self.spin_local_window.value()),
        }
        return cfg

    def on_save_clicked(self):
        cfg = self.get_settings()
        save_gmm_settings(cfg)
        try:
            QtWidgets.QMessageBox.information(self, "GMM Settings", "Settings saved to user folder.")
        except Exception:
            pass

    def accept(self):
        cfg = self.get_settings()
        save_gmm_settings(cfg)
        super().accept()
