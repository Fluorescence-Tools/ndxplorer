"""
GMM Settings dialog for ndxplorer.
Configure parameters for the built-in Gaussian Mixture (EM) implementation
and persist them in the ndxplorer user settings folder.
"""
from typing import Optional, Dict, Any

from qtpy import QtWidgets, QtCore

import json
from ..logging_config import logging
from ..settings import get_settings_path, ensure_default_settings
from .feedback import FriendlyErrorPresenter

_DEFAULTS: Dict[str, Any] = {
    "tol": 1e-3,                         # float > 0
    "reg_covar": 1e-6,                   # float >= 0
    "max_iter": 200,                     # int > 0
    "verbose": 0,                        # int >= 0
    "local_window_bins": 10,             # int >= 1, half-window size in bins for local covariance
    "weight_floor": 0.0,                 # float >= 0, minimum component weight (0 disables clamping)
    "fix_new_means": True                # bool, fix mean when adding new Gaussians from point selection
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


class GaussianSettingsDialog(QtWidgets.QDialog):
    """Qt dialog exposing built-in GMM (EM) configuration parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GMM Settings")
        self.setSizeGripEnabled(True)
        self.setMinimumWidth(420)
        self._cfg = load_gmm_settings()
        self._dialogs = FriendlyErrorPresenter(self)
        self._build_ui()
        self._load_to_widgets()

    def _build_ui(self):
        layout = QtWidgets.QFormLayout(self)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

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

        # verbose
        self.spin_verbose = QtWidgets.QSpinBox(self)
        self.spin_verbose.setRange(0, 10)
        layout.addRow("Verbose:", self.spin_verbose)

        # weight floor
        self.spin_weight_floor = QtWidgets.QDoubleSpinBox(self)
        self.spin_weight_floor.setDecimals(8)
        self.spin_weight_floor.setRange(0.0, 1.0)
        self.spin_weight_floor.setSingleStep(1e-4)
        layout.addRow("Weight floor:", self.spin_weight_floor)

        # local window (bins) for local covariance estimation in Gaussian Fit
        self.spin_local_window = QtWidgets.QSpinBox(self)
        self.spin_local_window.setRange(1, 200)
        self.spin_local_window.setSingleStep(1)
        layout.addRow("Local window (bins):", self.spin_local_window)

        # Fix means for new Gaussians (when added via point selection)
        self.check_fix_new_means = QtWidgets.QCheckBox("Fix new means by default", self)
        layout.addRow(self.check_fix_new_means)

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


    def _load_to_widgets(self):
        cfg = self._cfg
        self.spin_tol.setValue(float(cfg.get("tol", _DEFAULTS["tol"])) )
        self.spin_reg.setValue(float(cfg.get("reg_covar", _DEFAULTS["reg_covar"])) )
        self.spin_max_iter.setValue(int(cfg.get("max_iter", _DEFAULTS["max_iter"])) )
        self.spin_verbose.setValue(int(cfg.get("verbose", _DEFAULTS["verbose"])) )
        try:
            self.spin_weight_floor.setValue(float(cfg.get("weight_floor", _DEFAULTS["weight_floor"])) )
        except Exception:
            self.spin_weight_floor.setValue(_DEFAULTS["weight_floor"])
        try:
            self.spin_local_window.setValue(int(cfg.get("local_window_bins", _DEFAULTS["local_window_bins"])) )
        except Exception:
            self.spin_local_window.setValue(_DEFAULTS["local_window_bins"])
        self.check_fix_new_means.setChecked(bool(cfg.get("fix_new_means", _DEFAULTS["fix_new_means"])) )

    def get_settings(self) -> Dict[str, Any]:
        cfg = {
            "tol": float(self.spin_tol.value()),
            "reg_covar": float(self.spin_reg.value()),
            "max_iter": int(self.spin_max_iter.value()),
            "verbose": int(self.spin_verbose.value()),
            "local_window_bins": int(self.spin_local_window.value()),
            "weight_floor": float(self.spin_weight_floor.value()),
            "fix_new_means": bool(self.check_fix_new_means.isChecked()),
        }
        return cfg

    def on_save_clicked(self):
        cfg = self.get_settings()
        save_gmm_settings(cfg)
        self._dialogs.info("GMM Settings", "Settings saved to your NDxplorer user folder.")

    def accept(self):
        cfg = self.get_settings()
        save_gmm_settings(cfg)
        super().accept()


# Backward compatibility for older imports
GMMSettingsDialog = GaussianSettingsDialog
