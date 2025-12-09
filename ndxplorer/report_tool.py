from __future__ import annotations
from typing import List, Dict, Optional
import os
import json
import yaml
from pathlib import Path
import re
import shutil
import stat
import time

from qtpy import QtWidgets, QtCore
from qtpy.QtGui import QIcon, QPixmap, QColor, QBrush

from . import reader
from .plot_main import NDXplorer
from .settings import get_settings_path, ensure_default_settings

try:
    from docx import Document  # python-docx
    from docx.shared import Inches
except Exception:
    Document = None  # type: ignore

# Use a non-interactive backend for matplotlib and import pyplot
try:
    import matplotlib as _mpl
    _mpl.use("Agg")  # Ensure headless rendering
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Fallback: will check at runtime


def _safe_token(s: str) -> str:
    try:
        t = str(s).strip()
    except Exception:
        t = ""
    if not t:
        return "unknown"
    t = t.replace(' ', '_')
    try:
        t = re.sub(r'[^A-Za-z0-9._-]+', '-', t)
    except Exception:
        pass
    # Limit length to keep filenames manageable
    return t[:80]


def _default_config() -> Dict:
    return {
        "plots": [
            {
                "type": "2d",
                "title": "2D Histogram (basic)",
                "template": "2d_basic"
                # x and y can be specified by the user in YAML; template defines only style
            },
            {
                "type": "2d",
                "title": "2D Histogram with Marginals",
                "template": "2d_marginals"
                # x and y can be specified by the user in YAML; template defines only style
            },
            {
                "type": "1d",
                "title": "1D Histogram",
                "axis": "x",  # x or y
                "template": "1d_basic"
                # x or y selection is outside of the template
            }
        ]
    }


class FolderDropList(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setAlternatingRowColors(False)
        # Enable standard custom context menu handling
        try:
            self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self._on_context_menu)
        except Exception:
            pass

    def _copy_selected_paths(self) -> None:
        try:
            items = self.selectedItems()
            if not items and self.count() > 0:
                # If nothing selected, copy the item under the cursor (fallback)
                idx = self.currentRow()
                if idx >= 0:
                    items = [self.item(idx)]
            paths = [it.text() for it in items] if items else []
            if not paths:
                return
            txt = "\n".join(paths)
            cb = QtWidgets.QApplication.clipboard()
            cb.setText(txt)
        except Exception:
            pass

    def _on_context_menu(self, pos):
        try:
            menu = QtWidgets.QMenu(self)
            act_copy = menu.addAction("Copy Path(s)")
            global_pos = self.mapToGlobal(pos)
            action = menu.exec_(global_pos)
            if action == act_copy:
                self._copy_selected_paths()
        except Exception:
            pass

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md and md.hasUrls():
            for url in md.urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        md = event.mimeData()
        if md and md.hasUrls():
            for url in md.urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if not md or not md.hasUrls():
            event.ignore(); return
        # Collect dropped roots
        roots = []
        for url in md.urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.is_dir():
                roots.append(p)
        # Use ReportWizard discovery helper from the top-level dialog (no guard)
        wizard = self.window()
        added_any = False
        discovered = wizard._discover_analysis_folders(roots)  # type: ignore[attr-defined]
        for ap in discovered:
            s = str(ap)
            if not any(self.item(i).text() == s for i in range(self.count())):
                self.addItem(s)
                added_any = True
        # If no analysis folders were found, inform the user and do not add raw folders
        if not added_any:
            try:
                QtWidgets.QMessageBox.information(self, "No analysis folders found", "No analysis folders were found in the dropped directory/directories.")
            except Exception:
                pass
        # Refresh highlight colors after adding
        try:
            if hasattr(wizard, '_refresh_folder_highlights'):
                wizard._refresh_folder_highlights()
        except Exception:
            pass
        event.acceptProposedAction()

    def filenames(self) -> List[str]:
        return [self.item(i).text() for i in range(self.count())]


class ReportWizard(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NDXplorer Report Tool")
        self.resize(900, 600)

        self._config: Dict = _default_config()

        # Layouts
        main_layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(self)
        split.setOrientation(QtCore.Qt.Horizontal)
        main_layout.addWidget(split)

        # Left: folders list and buttons
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self.folder_list = FolderDropList(self)
        left_layout.addWidget(QtWidgets.QLabel("Analysis folders (drop here):"))
        left_layout.addWidget(self.folder_list)
        btns_left = QtWidgets.QHBoxLayout()
        self.btn_add_folder = QtWidgets.QPushButton("Add…")
        self.btn_remove_folder = QtWidgets.QPushButton("Remove")
        self.btn_clear_folders = QtWidgets.QPushButton("Clear")
        btns_left.addWidget(self.btn_add_folder)
        btns_left.addWidget(self.btn_remove_folder)
        btns_left.addWidget(self.btn_clear_folders)
        left_layout.addLayout(btns_left)

        split.addWidget(left)

        # Right: simple YAML text editor and load/save buttons
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("Report YAML (report.yaml):"))
        self.editor = QtWidgets.QPlainTextEdit()
        # Monospace font for readability
        font = self.editor.font()
        font.setFamily("Courier New")
        font.setPointSize(10)
        self.editor.setFont(font)
        self.editor.setTabStopDistance(4 * self.editor.fontMetrics().width(' ')) if hasattr(self.editor, 'setTabStopDistance') else None
        right_layout.addWidget(self.editor)

        btns_right = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load…")
        self.btn_save = QtWidgets.QPushButton("Save…")
        btns_right.addStretch(1)
        btns_right.addWidget(self.btn_load)
        btns_right.addWidget(self.btn_save)
        right_layout.addLayout(btns_right)

        # --- Images browser section ---
        images_box = QtWidgets.QGroupBox("Browse generated images")
        images_layout = QtWidgets.QVBoxLayout(images_box)
        # Selected dataset path display (read-only) shown below the group label
        self.selected_path_edit = QtWidgets.QLineEdit()
        try:
            self.selected_path_edit.setReadOnly(True)
            self.selected_path_edit.setPlaceholderText("No folder selected")
            self.selected_path_edit.setToolTip("Shows the full path of the currently selected dataset")
        except Exception:
            pass
        images_layout.addWidget(self.selected_path_edit)
        # Row: plot combobox (to pick among images)
        combo_row = QtWidgets.QHBoxLayout()
        combo_row.addWidget(QtWidgets.QLabel("Plot:"))
        self.image_combo = QtWidgets.QComboBox()
        combo_row.addWidget(self.image_combo, 1)
        images_layout.addLayout(combo_row)
        # Preview area only
        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setMinimumSize(200, 200)
        self.image_preview.setStyleSheet("QLabel { background: #222; color: #ccc; border: 1px solid #444; }")
        images_layout.addWidget(self.image_preview, 1)
        right_layout.addWidget(images_box, 1)

        split.addWidget(right)
        split.setStretchFactor(1, 2)

        # Bottom buttons
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        self.btn_clear_reports = QtWidgets.QPushButton("Clear Reports")
        self.btn_generate = QtWidgets.QPushButton("Generate Reports")
        bottom.addWidget(self.btn_clear_reports)
        bottom.addWidget(self.btn_generate)
        main_layout.addLayout(bottom)

        # Connections
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_remove_folder.clicked.connect(self._on_remove_folder)
        self.btn_clear_folders.clicked.connect(self._on_clear_folders)
        self.btn_load.clicked.connect(self._on_load_cfg)
        self.btn_save.clicked.connect(self._on_save_cfg)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_clear_reports.clicked.connect(self._on_clear_reports)

        # Initialize default report config in USER settings folder
        user_settings_path = get_settings_path()
        self._default_cfg_path = user_settings_path / 'report.yaml'
        try:
            user_settings_path.mkdir(parents=True, exist_ok=True)
            # If no user report.yaml exists, try to seed it from the bundled module default
            if not self._default_cfg_path.exists():
                bundled = Path(__file__).parent / 'settings' / 'report.yaml'
                try:
                    if bundled.exists():
                        with open(bundled, 'r', encoding='utf-8') as f_in:
                            bundled_cfg = yaml.safe_load(f_in)
                        if isinstance(bundled_cfg, dict) and 'plots' in bundled_cfg:
                            with open(self._default_cfg_path, 'w', encoding='utf-8') as f_out:
                                yaml.safe_dump(bundled_cfg, f_out, sort_keys=False)
                        else:
                            raise ValueError('Bundled report.yaml invalid')
                    else:
                        # Fallback to the built-in default config
                        with open(self._default_cfg_path, 'w', encoding='utf-8') as f_out:
                            yaml.safe_dump(self._config, f_out, sort_keys=False)
                except Exception:
                    # On any error, write the in-memory default
                    with open(self._default_cfg_path, 'w', encoding='utf-8') as f_out:
                        yaml.safe_dump(self._config, f_out, sort_keys=False)
        except Exception:
            # If creating file fails, continue with in-memory defaults
            pass
        # Load user settings default if present and show in editor
        try:
            if self._default_cfg_path.exists():
                with open(self._default_cfg_path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict) and 'plots' in loaded:
                    self._config = loaded
        except Exception:
            pass
        # Seed editor content from current config
        try:
            self.editor.setPlainText(yaml.safe_dump(self._config, sort_keys=False))
        except Exception:
            self.editor.setPlainText("# report.yaml could not be loaded; edit and save here\n" + yaml.safe_dump(_default_config(), sort_keys=False))

        # Ensure default settings (including templates) are available and load templates
        try:
            ensure_default_settings()
        except Exception:
            pass
        self._templates: Dict[str, Dict] = {}
        try:
            self._templates = self._load_templates()
        except Exception:
            self._templates = {}

        # Highlight folders initially (safe even if list is empty)
        try:
            self._refresh_folder_highlights()
        except Exception:
            pass

        # Prepare holder for axis settings (loaded on demand)
        self._axis_settings: Optional[Dict] = None

        # Image browsing state
        self._image_paths: List[Path] = []
        self._current_image_path: Optional[Path] = None

        # Wire image browsing signals
        try:
            self.image_combo.currentIndexChanged.connect(self._on_image_combo_changed)
            # Context menu to copy current image path
            self.image_combo.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.image_combo.customContextMenuRequested.connect(self._on_image_combo_context_menu)
        except Exception:
            pass

        # Load images automatically when a folder is selected/clicked
        try:
            self.folder_list.itemClicked.connect(self._on_folder_item_clicked)
            self.folder_list.itemSelectionChanged.connect(self._on_folder_selection_changed)
        except Exception:
            pass

    def _load_axis_settings(self) -> Dict:
        """
        Load axis settings from mfd.axis.json. Preference order:
        1) User/chisurf ndxplorer settings folder
        2) Module settings folder
        Returns an empty dict if not found or on error.
        """
        # Lazy import to avoid circulars
        try:
            from .settings import get_settings_path
        except Exception:
            get_settings_path = None  # type: ignore
        candidates = []
        try:
            if get_settings_path is not None:
                candidates.append(get_settings_path() / 'mfd.axis.json')
        except Exception:
            pass
        try:
            candidates.append(Path(__file__).parent / 'settings' / 'mfd.axis.json')
        except Exception:
            pass
        for fn in candidates:
            try:
                if fn.exists():
                    with open(fn, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        if isinstance(d, dict):
                            return d
            except Exception:
                continue
        return {}

    # Helper widgets for table cells
    def _make_axis_combo(self, current: str = "x") -> QtWidgets.QComboBox:
        cb = QtWidgets.QComboBox(self)
        cb.addItems(["x", "y"])
        idx = cb.findText(current.lower())
        if idx >= 0:
            cb.setCurrentIndex(idx)
        return cb

    def _make_column_cell(self, text: str = "") -> None:
        """
        Return either a QComboBox populated with column names (if available) or a plain QTableWidgetItem.
        This method returns a tuple (is_widget, widget_or_item).
        """
        if getattr(self, "_column_names", None):
            cb = QtWidgets.QComboBox(self)
            cb.setEditable(False)
            cb.addItems(self._column_names)
            # Try exact match first, then contains
            idx = cb.findText(text)
            if idx < 0 and text:
                idx = cb.findText(text, QtCore.Qt.MatchContains)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            return True, cb
        # Fallback: free text item
        return False, QtWidgets.QTableWidgetItem(text)

    def _setup_row_widgets(self, row: int, type_: str, title: str, axis: str, x: str, y: str,
                           xmin: str, xmax: str, ymin: str, ymax: str, bins1d: str, bins2d: str) -> None:
        # type
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(type_))
        # title
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(title))
        # axis (only meaningful for 1d)
        if type_.lower() == "1d":
            self.table.setCellWidget(row, 2, self._make_axis_combo(axis or "x"))
        else:
            # keep cell empty for 2d
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
        # x column selector
        is_wx, wx = self._make_column_cell(x)
        if is_wx:
            self.table.setCellWidget(row, 3, wx)
        else:
            self.table.setItem(row, 3, wx)
        # y column selector (only used for 2d but allow value anyway)
        is_wy, wy = self._make_column_cell(y)
        if is_wy:
            self.table.setCellWidget(row, 4, wy)
        else:
            self.table.setItem(row, 4, wy)
        # numeric cells
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(xmin))
        self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(xmax))
        self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(ymin))
        self.table.setItem(row, 8, QtWidgets.QTableWidgetItem(ymax))
        self.table.setItem(row, 9, QtWidgets.QTableWidgetItem(bins1d))
        self.table.setItem(row, 10, QtWidgets.QTableWidgetItem(bins2d))

    # UI helpers
    def _on_add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select analysis folder")
        if d:
            if not any(self.folder_list.item(i).text() == d for i in range(self.folder_list.count())):
                self.folder_list.addItem(d)
                try:
                    # Select the newly added item
                    self.folder_list.setCurrentRow(self.folder_list.count() - 1)
                    # Update selected path display
                    if hasattr(self, 'selected_path_edit') and self.selected_path_edit is not None:
                        self.selected_path_edit.setText(str(Path(d).resolve()))
                except Exception:
                    pass
                try:
                    self._refresh_folder_highlights()
                except Exception:
                    pass

    def _on_remove_folder(self):
        for it in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(it))
        try:
            # Update selected path display based on new current selection
            if hasattr(self, 'selected_path_edit') and self.selected_path_edit is not None:
                item = self.folder_list.currentItem()
                if item is None:
                    self.selected_path_edit.clear()
                else:
                    try:
                        self.selected_path_edit.setText(str(Path(item.text()).resolve()))
                    except Exception:
                        self.selected_path_edit.setText(item.text())
        except Exception:
            pass
        try:
            self._refresh_folder_highlights()
        except Exception:
            pass

    def _on_clear_folders(self):
        """Clear all folders from the list (with confirmation)."""
        try:
            if self.folder_list.count() == 0:
                return
            res = QtWidgets.QMessageBox.question(
                self,
                "Clear list",
                "Remove all folders from the list?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            if res != QtWidgets.QMessageBox.Yes:
                return
        except Exception:
            # If message box fails for any reason, proceed to clear
            pass
        self.folder_list.clear()
        try:
            # Clear selected path display
            if hasattr(self, 'selected_path_edit') and self.selected_path_edit is not None:
                self.selected_path_edit.clear()
        except Exception:
            pass
        try:
            self._refresh_folder_highlights()
        except Exception:
            pass

    def _on_clear_reports(self):
        """Delete the generated 'report' folders for the selected (or all) analysis folders."""
        # Determine targets: selected items or all items if none selected
        try:
            selected = self.folder_list.selectedItems()
            if selected:
                targets = [Path(it.text()) for it in selected]
            else:
                targets = [Path(self.folder_list.item(i).text()) for i in range(self.folder_list.count())]
        except Exception:
            targets = []
        if not targets:
            try:
                QtWidgets.QMessageBox.information(self, "No folders", "No analysis folders to clear.")
            except Exception:
                pass
            return
        # Confirm
        try:
            msg = f"This will permanently delete report folders for {len(targets)} analysis folder(s). Continue?"
            res = QtWidgets.QMessageBox.question(
                self,
                "Clear Reports",
                msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if res != QtWidgets.QMessageBox.Yes:
                return
        except Exception:
            pass
        cleared = 0
        # Remember current selection to refresh image browser if needed
        try:
            current_item = self.folder_list.currentItem()
            current_path = Path(current_item.text()) if current_item else None
        except Exception:
            current_path = None
        for folder in targets:
            try:
                # Remove typical report directory names
                removed_any = False
                for name in ("report", "Report", "REPORT"):
                    rd = folder / name
                    if rd.exists() and rd.is_dir():
                        success = self._robust_rmtree(rd)
                        if success:
                            removed_any = True
                if removed_any:
                    cleared += 1
            except Exception:
                continue
        # Refresh image browser if the current folder was affected
        try:
            if current_path is not None:
                self._open_report_images_for_folder(current_path)
        except Exception:
            pass
        # Inform user
        try:
            if cleared > 0:
                QtWidgets.QMessageBox.information(self, "Reports cleared", f"Cleared reports for {cleared} folder(s).")
            else:
                QtWidgets.QMessageBox.information(self, "Nothing to clear", "No report folders were found to delete.")
        except Exception:
            pass
        # Refresh folder highlights to reflect cleared reports
        try:
            self._refresh_folder_highlights()
        except Exception:
            pass

    def _robust_rmtree(self, path: Path, max_retries: int = 3) -> bool:
        """Attempt to remove a directory tree robustly on Windows.
        Returns True if the directory is gone, False otherwise.
        """
        def _onerror(func, p, exc_info):
            # Try to make the path writable and retry
            try:
                os.chmod(p, stat.S_IWRITE)
            except Exception:
                pass
            try:
                func(p)
            except Exception:
                pass
        # Try rmtree with retries
        for _ in range(max_retries):
            try:
                shutil.rmtree(str(path), onerror=_onerror)
            except Exception:
                pass
            if not path.exists():
                return True
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
            time.sleep(0.05)
        # Fallback: manual walk
        try:
            for root, dirs, files in os.walk(str(path), topdown=False):
                for name in files:
                    fp = Path(root) / name
                    try:
                        os.chmod(str(fp), stat.S_IWRITE)
                    except Exception:
                        pass
                    try:
                        os.remove(str(fp))
                    except Exception:
                        pass
                for name in dirs:
                    dp = Path(root) / name
                    try:
                        os.rmdir(str(dp))
                    except Exception:
                        pass
            os.rmdir(str(path))
        except Exception:
            pass
        return not path.exists()

    def _on_add_1d(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        # type, title, axis, x, y, xmin, xmax, ymin, ymax, bins1d, bins2d
        self._setup_row_widgets(
            row,
            "1d", "1D Histogram", "x",
            "Proximity ratio", "",
            "0.0", "1.0", "", "",
            "100", ""
        )

    def _on_add_2d(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._setup_row_widgets(
            row,
            "2d", "2D Histogram", "",
            "Proximity ratio", "Tau (green)",
            "0.0", "1.0", "0.0", "5.0",
            "", "64,64"
        )

    def _on_remove_plot(self):
        for it in self.table.selectedItems():
            self.table.removeRow(it.row())
            break

    def _on_load_cfg(self):
        initial = str(getattr(self, '_default_cfg_path', (Path(__file__).parent / 'settings' / 'report.yaml')))
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open config", initial, "YAML/JSON (*.yaml *.yml *.json)")
        if not fn:
            return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                text = f.read()
            # Optionally validate YAML/JSON before setting
            try:
                if fn.lower().endswith((".yaml", ".yml")):
                    loaded = yaml.safe_load(text)
                else:
                    loaded = json.loads(text)
                if not isinstance(loaded, dict) or "plots" not in loaded:
                    raise ValueError("Invalid config: missing 'plots'")
                self._config = loaded
            except Exception as ve:
                res = QtWidgets.QMessageBox.question(
                    self,
                    "Parse warning",
                    f"File read, but failed to parse config (will still load text):\n{ve}\n\nLoad as-is?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                if res != QtWidgets.QMessageBox.Yes:
                    return
            self.editor.setPlainText(text)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _on_save_cfg(self):
        initial = str(getattr(self, '_default_cfg_path', (Path(__file__).parent / 'settings' / 'report.yaml')))
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config", initial, "YAML (*.yaml);;JSON (*.json)")
        if not fn:
            return
        try:
            text = self.editor.toPlainText()
            # Validate before saving
            if fn.lower().endswith(".json"):
                # Convert YAML text to JSON structure then dump JSON
                data = yaml.safe_load(text)
                if not isinstance(data, dict) or "plots" not in data:
                    raise ValueError("Invalid config: missing 'plots'")
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                if not fn.lower().endswith((".yaml", ".yml")):
                    fn += ".yaml"
                data = yaml.safe_load(text)
                if not isinstance(data, dict) or "plots" not in data:
                    raise ValueError("Invalid config: missing 'plots'")
                with open(fn, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False)
            # Update internal config if valid
            self._config = data
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _load_config_into_table(self, cfg: Dict):
        self.table.setRowCount(0)
        for p in cfg.get("plots", []):
            row = self.table.rowCount()
            self.table.insertRow(row)
            type_ = p.get("type", "1d")
            title = p.get("title", "")
            axis = p.get("axis", "x" if type_ == "1d" else "")
            x = p.get("x", "")
            y = p.get("y", "")
            xmin = str(p.get("xmin", ""))
            xmax = str(p.get("xmax", ""))
            ymin = str(p.get("ymin", ""))
            ymax = str(p.get("ymax", ""))
            bins1d = str(p.get("bins1d", ""))
            if type_ == "2d":
                bx = p.get("bins2d_x", 64)
                by = p.get("bins2d_y", 64)
                bins2d = f"{bx},{by}"
            else:
                bins2d = ""
            self._setup_row_widgets(row, type_, title, axis, x, y, xmin, xmax, ymin, ymax, bins1d, bins2d)

    def _config_from_table(self) -> Dict:
        plots: List[Dict] = []
        for r in range(self.table.rowCount()):
            type_ = (self.table.item(r, 0).text() if self.table.item(r, 0) else "1d").strip().lower()
            title = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            # axis may be a combobox or an item
            axis_widget = self.table.cellWidget(r, 2)
            if isinstance(axis_widget, QtWidgets.QComboBox):
                axis = axis_widget.currentText().strip().lower()
            else:
                axis = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip().lower()
            # x may be a combobox or an item
            x_widget = self.table.cellWidget(r, 3)
            if isinstance(x_widget, QtWidgets.QComboBox):
                x = x_widget.currentText().strip()
            else:
                x = (self.table.item(r, 3).text() if self.table.item(r, 3) else "").strip()
            # y may be a combobox or an item
            y_widget = self.table.cellWidget(r, 4)
            if isinstance(y_widget, QtWidgets.QComboBox):
                y = y_widget.currentText().strip()
            else:
                y = (self.table.item(r, 4).text() if self.table.item(r, 4) else "").strip()
            xmin = self._to_float(self.table.item(r, 5))
            xmax = self._to_float(self.table.item(r, 6))
            ymin = self._to_float(self.table.item(r, 7))
            ymax = self._to_float(self.table.item(r, 8))
            bins1d = self._to_int(self.table.item(r, 9))
            b2 = (self.table.item(r, 10).text() if self.table.item(r, 10) else "").strip()
            bins2d_x = bins2d_y = None
            if b2:
                try:
                    parts = [int(s) for s in b2.split(',')]
                    if len(parts) == 2:
                        bins2d_x, bins2d_y = parts
                except Exception:
                    bins2d_x = bins2d_y = None
            item: Dict = {
                "type": type_,
                "title": title,
                "x": x,
                "y": y,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
            }
            if type_ == "1d":
                item["axis"] = axis or "x"
                if bins1d:
                    item["bins1d"] = bins1d
            else:
                if bins2d_x is not None:
                    item["bins2d_x"] = bins2d_x
                if bins2d_y is not None:
                    item["bins2d_y"] = bins2d_y
            plots.append(item)
        return {"plots": plots}

    @staticmethod
    def _to_float(item: Optional[QtWidgets.QTableWidgetItem]) -> Optional[float]:
        try:
            if item and item.text().strip() != "":
                return float(item.text())
        except Exception:
            return None
        return None

    @staticmethod
    def _to_int(item: Optional[QtWidgets.QTableWidgetItem]) -> Optional[int]:
        try:
            if item and item.text().strip() != "":
                return int(item.text())
        except Exception:
            return None
        return None

    # ---- Images browsing helpers ----

    def _load_images_from_dir(self, folder: Path):
        try:
            folder = Path(folder)
            if not folder.exists() or not folder.is_dir():
                self._image_paths = []
            else:
                # Collect PNG images (generated plots)
                pngs = []
                for name in os.listdir(str(folder)):
                    if name.lower().endswith('.png'):
                        pngs.append(folder / name)
                # Sort by filename
                self._image_paths = sorted(pngs, key=lambda p: p.name.lower())
        except Exception:
            self._image_paths = []
        # Populate combobox only, preserving previous selection index
        prev_idx = self.image_combo.currentIndex() if hasattr(self, 'image_combo') else -1
        try:
            self.image_combo.blockSignals(True)
            self.image_combo.clear()
            for p in self._image_paths:
                label = p.name
                self.image_combo.addItem(label, str(p))
        finally:
            try:
                self.image_combo.blockSignals(False)
            except Exception:
                pass
        # Restore previous index if possible, else select first
        if self._image_paths:
            new_idx = prev_idx if 0 <= prev_idx < len(self._image_paths) else 0
            self.image_combo.setCurrentIndex(new_idx)
            self._display_image_at_index(new_idx)
        else:
            self._current_image_path = None
            self.image_preview.setText("No images found")
            self.image_preview.setPixmap(QPixmap())

    def _on_image_list_changed(self, row: int):
        # List widget removed; no-op to maintain backward compatibility
        return

    def _on_image_combo_changed(self, idx: int):
        if idx is None or idx < 0:
            return
        # Update preview based on combobox selection
        try:
            self._display_image_at_index(idx, sync_combo=False)
        except Exception:
            pass

    def _on_image_combo_context_menu(self, pos):
        try:
            if self.image_combo.count() <= 0:
                return
            idx = self.image_combo.currentIndex()
            if idx < 0:
                return
            # Retrieve stored path from item data; fallback to displayed text
            path_str = self.image_combo.itemData(idx) or self.image_combo.itemText(idx)
            menu = QtWidgets.QMenu(self.image_combo)
            act_copy = menu.addAction("Copy Image Path")
            global_pos = self.image_combo.mapToGlobal(pos)
            action = menu.exec_(global_pos)
            if action == act_copy and path_str:
                QtWidgets.QApplication.clipboard().setText(str(path_str))
        except Exception:
            pass

    def _display_image_at_index(self, index: int, sync_combo: bool = False):
        if index < 0 or index >= len(self._image_paths):
            return
        path = self._image_paths[index]
        self._current_image_path = path
        # Sync combobox selection by matching data
        if sync_combo:
            try:
                path_str = str(path)
                for i in range(self.image_combo.count()):
                    if self.image_combo.itemData(i) == path_str:
                        if self.image_combo.currentIndex() != i:
                            self.image_combo.blockSignals(True)
                            self.image_combo.setCurrentIndex(i)
                            self.image_combo.blockSignals(False)
                        break
            except Exception:
                pass
        self._update_image_preview()

    def _update_image_preview(self):
        if not self._current_image_path:
            self.image_preview.setText("No image selected")
            self.image_preview.setPixmap(QPixmap())
            return
        try:
            pix = QPixmap(str(self._current_image_path))
            if pix.isNull():
                self.image_preview.setText(f"Cannot load image\n{self._current_image_path}")
                self.image_preview.setPixmap(QPixmap())
                return
            # Scale to fit while keeping aspect ratio and some padding
            size = self.image_preview.size()
            scaled = pix.scaled(size.width()-8, size.height()-8, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_preview.setPixmap(scaled)
            self.image_preview.setText("")
        except Exception as e:
            self.image_preview.setText(f"Error loading image: {e}")
            self.image_preview.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale image on resize
        try:
            self._update_image_preview()
        except Exception:
            pass

    # ---- Auto-open report images when selecting folders ----
    def _on_folder_item_clicked(self, item: QtWidgets.QListWidgetItem):
        try:
            p = Path(item.text())
            # Update selected path display
            try:
                if hasattr(self, 'selected_path_edit') and self.selected_path_edit is not None:
                    self.selected_path_edit.setText(str(p.resolve()))
            except Exception:
                pass
            self._open_report_images_for_folder(p)
        except Exception:
            pass

    def _on_folder_selection_changed(self):
        try:
            item = self.folder_list.currentItem()
            # Update selected path display
            try:
                if hasattr(self, 'selected_path_edit') and self.selected_path_edit is not None:
                    if item is None:
                        self.selected_path_edit.clear()
                    else:
                        p_disp = Path(item.text())
                        self.selected_path_edit.setText(str(p_disp.resolve()))
            except Exception:
                pass
            if not item:
                return
            p = Path(item.text())
            self._open_report_images_for_folder(p)
        except Exception:
            pass

    def _open_report_images_for_folder(self, folder: Path):
        # Determine report directory inside the analysis folder and load images
        try:
            folder = Path(folder)
            report_dir = folder / 'report'
            if report_dir.exists() and report_dir.is_dir():
                self._load_images_from_dir(report_dir)
                return
            # If no direct 'report' dir, try to find one-level down
            for name in ('Report', 'REPORT'):  # common variants
                alt = folder / name
                if alt.exists() and alt.is_dir():
                    self._load_images_from_dir(alt)
                    return
            # Fallback: try to load images directly from the folder (if user placed PNGs there)
            self._load_images_from_dir(folder)
        except Exception:
            # As a last resort, clear the combobox and preview
            self._image_paths = []
            self.image_combo.clear()
            self.image_preview.setText('No images found')
            self.image_preview.setPixmap(QPixmap())

    # ---- Processed-folder highlighting helpers ----
    def _is_folder_processed(self, folder: Path) -> bool:
        """Return True if the analysis folder appears to have reports generated.
        Criteria: presence of a report directory containing PNGs or axes_info.yaml,
        or a DOCX report inside the report directory.
        """
        try:
            folder = Path(folder)
            if not folder.exists() or not folder.is_dir():
                return False
            # Look for common report directory names
            report_dir = None
            for name in ("report", "Report", "REPORT"):
                p = folder / name
                if p.exists() and p.is_dir():
                    report_dir = p
                    break
            if report_dir is None:
                return False
            # Check for typical outputs
            try:
                for name in os.listdir(str(report_dir)):
                    low = name.lower()
                    if low.endswith('.png'):
                        return True
                    if low == 'axes_info.yaml':
                        return True
                    if low.endswith('.docx'):
                        return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    def _set_item_processed_style(self, item: QtWidgets.QListWidgetItem, processed: bool) -> None:
        try:
            if processed:
                # light green background, dark text
                bg = QBrush(QColor(212, 237, 218))  # #d4edda
                fg = QBrush(QColor(33, 37, 41))     # #212529
                item.setBackground(bg)
                item.setForeground(fg)
            else:
                # Reset to default
                item.setBackground(QBrush())
                item.setForeground(QBrush())
        except Exception:
            pass

    def _refresh_folder_highlights(self) -> None:
        try:
            for i in range(self.folder_list.count()):
                it = self.folder_list.item(i)
                p = Path(it.text())
                processed = self._is_folder_processed(p)
                self._set_item_processed_style(it, processed)
                try:
                    tip = f"Processed: {'Yes' if processed else 'No'}\n{str(p)}"
                    it.setToolTip(tip)
                except Exception:
                    pass
        except Exception:
            pass

    # Analysis folder discovery helpers
    def _is_analysis_folder(self, p: Path) -> bool:
        """Return True if the given path looks like a valid analysis folder.
        Criteria:
        - contains 'hdf5' subfolder with .h5/.hdf5, or
        - contains 'bi4_bur' with .bur files, or
        - contains 'bur' with .bur files
        """
        try:
            if not p.exists() or not p.is_dir():
                return False
            hdf5_dir = p / 'hdf5'
            if hdf5_dir.is_dir():
                for f in hdf5_dir.iterdir():
                    suf = f.suffix.lower()
                    if f.is_file() and (suf == '.h5' or suf == '.hdf5'):
                        return True
            for sub in ('bi4_bur', 'bur'):
                sd = p / sub
                if sd.is_dir():
                    for f in sd.iterdir():
                        if f.is_file() and f.suffix.lower() == '.bur':
                            return True
            return False
        except Exception:
            return False

    def _discover_analysis_folders(self, roots: List[Path]) -> List[Path]:
        """Recurse into subfolders of roots and collect analysis folders.
        - If a root itself is an analysis folder, include it.
        - Otherwise, walk its subdirectories and include those that qualify.
        - Avoid hidden directories and typical VCS directories.
        - Deduplicate and preserve discovery order.
        """
        discovered: List[Path] = []
        seen = set()
        skip_names = {'.git', '.hg', '.svn', '__pycache__', '.idea', '.vscode'}
        for root in roots:
            try:
                root = Path(root)
                if self._is_analysis_folder(root):
                    key = str(root.resolve()) if hasattr(root, 'resolve') else str(root)
                    if key not in seen:
                        discovered.append(root)
                        seen.add(key)
                    # do not descend further into this root
                    continue
                # Walk the tree
                for curr_dir, dirnames, _filenames in os.walk(str(root)):
                    # prune hidden and skipped directories
                    pruned = []
                    for d in list(dirnames):
                        if d in skip_names or d.startswith('.'):
                            continue
                        pruned.append(d)
                    dirnames[:] = pruned
                    p = Path(curr_dir)
                    if self._is_analysis_folder(p):
                        key = str(p.resolve()) if hasattr(p, 'resolve') else str(p)
                        if key not in seen:
                            discovered.append(p)
                            seen.add(key)
                            # If we found an analysis folder, we can avoid descending further within it
                            dirnames[:] = []
            except Exception:
                continue
        return discovered

    def _load_templates(self) -> Dict[str, Dict]:
        """Load plot templates from settings/templates directory.
        Returns a mapping: template_name -> template_dict
        """
        templates: Dict[str, Dict] = {}
        try:
            base = get_settings_path() / 'templates'
        except Exception:
            base = Path(__file__).parent / 'settings' / 'templates'
        if base.exists() and base.is_dir():
            for f in base.iterdir():
                if not f.is_file():
                    continue
                name = f.stem.lower()
                try:
                    if f.suffix.lower() in ('.yaml', '.yml'):
                        with open(f, 'r', encoding='utf-8') as fh:
                            data = yaml.safe_load(fh)
                    elif f.suffix.lower() == '.json':
                        with open(f, 'r', encoding='utf-8') as fh:
                            data = json.load(fh)
                    else:
                        continue
                    if isinstance(data, dict) and data.get('renderer'):
                        templates[name] = data
                except Exception:
                    continue
        return templates

    # Generation
    def _on_generate(self):
        # Read config from YAML editor
        text = self.editor.toPlainText()
        try:
            cfg = yaml.safe_load(text) if text.strip() else {}
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "YAML error", f"Could not parse YAML: {e}")
            return
        if not isinstance(cfg, dict) or 'plots' not in cfg:
            QtWidgets.QMessageBox.warning(self, "Invalid config", "The YAML must define a top-level 'plots' list.")
            return
        roots = [Path(p) for p in self.folder_list.filenames()]
        if not roots:
            QtWidgets.QMessageBox.warning(self, "No folders", "Please add analysis folders.")
            return
        # Expand by discovering analysis folders recursively
        targets = self._discover_analysis_folders(roots)
        # Case 1: none discovered at all
        if not targets:
            QtWidgets.QMessageBox.warning(self, "No analysis folders found", "No valid analysis subfolders were found. Please check your selection.")
            return
        # Skip folders that already have a report
        try:
            targets = [t for t in targets if not self._is_folder_processed(t)]
        except Exception:
            # If any error occurs during check, proceed without filtering
            targets = targets
        # Case 2: all discovered folders already processed
        if not targets:
            QtWidgets.QMessageBox.information(self, "Nothing to do", "All selected analysis folders already have a report. Nothing to generate.")
            return
        # Start cooperative generation to keep UI responsive
        self._start_generation(cfg, targets)

    def _start_generation(self, cfg: Dict, targets: List[Path]) -> None:
        # Guard against re-entry
        if getattr(self, '_gen_running', False):
            return
        self._gen_running = True
        # Create one hidden NDXplorer instance for reuse (must live on GUI thread)
        self._gen_ndx = NDXplorer(parent=None)
        try:
            self._gen_ndx.hide()
        except Exception:
            pass
        # Initialize state
        self._gen_cfg = cfg
        self._gen_targets = list(targets)
        self._gen_total = len(self._gen_targets)
        self._gen_index = 0
        self._gen_results = []
        self._gen_canceled = False
        # Setup progress dialog
        self._gen_progress = None
        try:
            self._gen_progress = QtWidgets.QProgressDialog("Generating reports...", "Cancel", 0, self._gen_total, self)
            self._gen_progress.setWindowTitle("Generating")
            self._gen_progress.setWindowModality(QtCore.Qt.WindowModal)
            self._gen_progress.setAutoClose(False)
            self._gen_progress.setAutoReset(False)
            self._gen_progress.show()
        except Exception:
            self._gen_progress = None
        # Disable buttons during generation
        try:
            self.btn_generate.setEnabled(False)
            self.btn_clear_reports.setEnabled(False)
            self.btn_add_folder.setEnabled(False)
            self.btn_remove_folder.setEnabled(False)
            self.btn_clear_folders.setEnabled(False)
        except Exception:
            pass
        # Kick off first slice on the event loop
        QtCore.QTimer.singleShot(0, self._process_next_target)

    def _process_next_target(self) -> None:
        # If canceled by user
        try:
            if self._gen_progress and self._gen_progress.wasCanceled():
                self._gen_canceled = True
        except Exception:
            pass
        # If finished or canceled
        if self._gen_canceled or self._gen_index >= self._gen_total:
            # Close progress
            try:
                if self._gen_progress:
                    self._gen_progress.close()
            except Exception:
                pass
            # Re-enable buttons
            try:
                self.btn_generate.setEnabled(True)
                self.btn_clear_reports.setEnabled(True)
                self.btn_add_folder.setEnabled(True)
                self.btn_remove_folder.setEnabled(True)
                self.btn_clear_folders.setEnabled(True)
            except Exception:
                pass
            # If multiple folders, offer to create a combined final report
            results = getattr(self, '_gen_results', []) or []
            if len(results) > 1:
                if Document is None:
                    try:
                        QtWidgets.QMessageBox.warning(self, "python-docx missing",
                                                      "python-docx is not installed. Skipping combined final report.")
                    except Exception:
                        pass
                else:
                    try:
                        suggested = str(Path(results[0]['folder']).parent / "NDXplorer_Batch_Report.docx")
                        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                            self,
                            "Save Final Combined Report",
                            suggested,
                            "DOCX (*.docx)"
                        )
                        if fn:
                            final_doc = Document()
                            final_doc.add_heading("NDXplorer Batch Report", level=1)
                            for res in results:
                                # Section heading per folder (use parent folder name)
                                folder_path = Path(res['folder'])
                                sample_name = folder_path.parent.name if folder_path.parent else folder_path.name
                                final_doc.add_heading(f"Sample: {sample_name}", level=2)
                                # Place the full analysis folder path below the title
                                final_doc.add_paragraph(str(folder_path.resolve()))
                                entries = res.get('entries', [])
                                # Make 2-column tables of plots (title, image, filepaths)
                                for i in range(0, len(entries), 2):
                                    table = final_doc.add_table(rows=3, cols=2)
                                    for col in range(2):
                                        idx = i + col
                                        top_cell = table.cell(0, col)
                                        mid_cell = table.cell(1, col)
                                        bot_cell = table.cell(2, col)
                                        if idx < len(entries):
                                            entry = entries[idx]
                                            top_cell.text = entry.get("title", "")
                                            p = mid_cell.paragraphs[0]
                                            r = p.add_run()
                                            img_path = entry.get("img", "")
                                            try:
                                                r.add_picture(img_path, width=Inches(3.0))
                                            except Exception:
                                                p.add_run(img_path)
                                            # Back-references: image and CSV filepaths
                                            csv_path = entry.get("csv", "")
                                            bot_lines = []
                                            if csv_path:
                                                bot_lines.append(f"CSV:   {csv_path}")
                                            bot_cell.text = "\n".join(bot_lines) if bot_lines else ""
                                        else:
                                            top_cell.text = ""
                                            try:
                                                mid_cell.text = ""
                                            except Exception:
                                                pass
                                            bot_cell.text = ""
                                    final_doc.add_paragraph("")
                            # Ensure .docx extension
                            out_fn = fn if fn.lower().endswith('.docx') else fn + '.docx'
                            final_doc.save(out_fn)
                    except Exception as e:
                        try:
                            QtWidgets.QMessageBox.critical(self, "Final Report Error", str(e))
                        except Exception:
                            pass
            # Refresh highlights
            try:
                self._refresh_folder_highlights()
            except Exception:
                pass
            # Inform user
            try:
                if self._gen_canceled:
                    QtWidgets.QMessageBox.information(self, "Canceled", "Generation canceled.")
                else:
                    QtWidgets.QMessageBox.information(self, "Done", "Reports generated.")
            except Exception:
                pass
            # Cleanup state
            self._gen_running = False
            self._gen_targets = []
            self._gen_index = 0
            self._gen_total = 0
            self._gen_cfg = {}
            self._gen_progress = None
            try:
                self._gen_ndx = None
            except Exception:
                pass
            return
        # Process one folder slice
        folder = self._gen_targets[self._gen_index]
        # Update progress pre-label and bar
        try:
            if self._gen_progress:
                self._gen_progress.setLabelText(f"Processing: {folder}")
                self._gen_progress.setValue(self._gen_index)
                QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        # Do the work for this folder
        try:
            res = self._generate_for_folder(self._gen_ndx, folder, self._gen_cfg)
            if res:
                self._gen_results.append(res)
            # Update highlight immediately for this item
            try:
                for idx in range(self.folder_list.count()):
                    it = self.folder_list.item(idx)
                    if it and it.text() == str(folder):
                        self._set_item_processed_style(it, True)
                        try:
                            it.setToolTip(f"Processed: Yes\n{it.text()}")
                        except Exception:
                            pass
                        break
            except Exception:
                pass
        except Exception as e:
            try:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error for {folder}: {e}")
            except Exception:
                pass
        # Advance index and update progress bar to reflect completion of this item
        self._gen_index += 1
        try:
            if self._gen_progress:
                self._gen_progress.setValue(self._gen_index)
                QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        # Schedule the next folder to keep UI responsive
        QtCore.QTimer.singleShot(0, self._process_next_target)

    def _generate_for_folder(self, ndx: NDXplorer, folder: Path, cfg: Dict):
        if not folder.exists():
            raise FileNotFoundError(str(folder))
        # Load data
        ds = reader.read_burst_analysis(base_path=str(folder))
        # Guard: skip if no data
        try:
            if ds is None or ds.empty or getattr(ds, 'data', None) is None or ds.data.empty:
                QtWidgets.QMessageBox.warning(self, "Empty dataset", f"No data found in folder: {folder}\nSkipping.")
                return
        except Exception:
            pass
        # Assign via property to compute columns and invalidate caches
        try:
            ndx.data_source = ds
        except Exception:
            # Fallback (should not be needed): direct assign and manual compute
            ndx._data_source = ds
            try:
                ndx._data_source.compute_columns(constants=ndx.constants, equations=ndx.equations)
            except Exception:
                pass
        # Clear any prior selections and refresh plot controls for new columns
        try:
            ndx.plot_control.onClearSelection()
        except Exception:
            pass
        try:
            ndx.plot_control.update()
        except Exception:
            ndx.update_plots(skip_clustering=True)

        # Prepare outputs: single 'report' folder inside the analysis folder
        report_dir = folder / "report"
        report_dir.mkdir(exist_ok=True, parents=True)
        out_docx = str(report_dir / f"{folder.name}_report.docx")

        doc = None
        entries: List[Dict[str, str]] = []
        axes_meta: List[Dict[str, object]] = []
        if Document is not None:
            doc = Document()
            # Use parent folder name as the sample title, as requested
            sample_name = folder.parent.name if folder.parent else folder.name
            doc.add_heading(f"NDXplorer Report: {sample_name}", level=1)
            try:
                # Place the full path below the title as normal text
                doc.add_paragraph(str(folder.resolve()))
            except Exception:
                doc.add_paragraph(str(folder))
        else:
            # If python-docx missing, we still generate PNGs/CSVs and warn once
            pass

        idx_plot = 1
        for p in cfg.get("plots", []):
            p_type = p.get("type", "1d").lower()
            title = p.get("title", f"Plot {idx_plot}")

            # Apply axes from YAML and ranges/bins/scales from mfd.axis.json
            # 1) Ensure axis settings are loaded once and merged into plot_control
            if self._axis_settings is None:
                self._axis_settings = self._load_axis_settings()
            try:
                if isinstance(self._axis_settings, dict):
                    ndx.plot_control.axis_settings.update(self._axis_settings)
            except Exception:
                pass

            # 2) Set requested axes by name
            x_name = p.get('x')
            y_name = p.get('y')
            if isinstance(x_name, str) and x_name:
                ndx.plot_control.set_axis_by_name('x', x_name)
            if p_type == '2d' and isinstance(y_name, str) and y_name:
                ndx.plot_control.set_axis_by_name('y', y_name)

            # 3) Trigger axis-changed handlers so mfd.axis.json settings take effect
            try:
                ndx.plot_control.on_x_axis_changed()
            except Exception:
                pass
            if p_type == '2d':
                try:
                    ndx.plot_control.on_y_axis_changed()
                except Exception:
                    pass
            elif p_type == '1d':
                axis_1d = (p.get('axis', 'x') or 'x').lower()
                if axis_1d == 'y':
                    # Ensure Y axis selection handler is applied if 1D Y is requested
                    try:
                        ndx.plot_control.on_y_axis_changed()
                    except Exception:
                        pass

            # 4) Update plots so histograms reflect these settings
            ndx.update_plots(skip_clustering=True)

            # Cache current labels for figure annotations
            try:
                x_label = getattr(ndx.plot_control, 'x_label', 'X')
                y_label = getattr(ndx.plot_control, 'y_label', 'Y')
            except Exception:
                x_label, y_label = 'X', 'Y'

            # Save CSVs and screenshots
            safe_title = _safe_token(title)
            if p_type == "2d":
                # CSV
                H, x_edges, y_edges = ndx._histogram["2d"]
                x_tok = _safe_token(x_label)
                y_tok = _safe_token(y_label)
                csv_path = report_dir / f"{idx_plot:02d}_2d_{x_tok}_{y_tok}.csv"
                self._save_2d_csv(csv_path, H, x_edges, y_edges, xlabel=x_label, ylabel=y_label)
                # Image via matplotlib
                img_path = report_dir / f"{idx_plot:02d}_2d_{x_tok}_{y_tok}.png"
                try:
                    template_name = (p.get("template") or "2d_basic").lower()
                    tpl = self._templates.get(template_name)
                    renderer = None
                    style = {}
                    if isinstance(tpl, dict):
                        renderer = tpl.get('renderer')
                        style = tpl.get('style') or {}
                    # Fallback to template_name when renderer missing
                    renderer = (renderer or template_name or '2d_basic').lower()
                    if renderer in ("2d_marginals", "hist2d_marginals", "2d_with_marginals"):
                        self._save_2d_marginals_png(
                            img_path,
                            H, x_edges, y_edges,
                            title=title,
                            xlabel=x_label,
                            ylabel=y_label,
                            cmap_2d=style.get('cmap_2d', 'Greys'),
                            color_x=style.get('color_x', 'dimgrey'),
                            color_y=style.get('color_y', 'darkorange'),
                            figsize=style.get('figsize'),
                            dpi=style.get('dpi')
                        )
                    else:
                        self._save_2d_png(
                            img_path,
                            H, x_edges, y_edges,
                            title=title,
                            xlabel=x_label,
                            ylabel=y_label,
                            cmap=style.get('cmap', 'viridis'),
                            figsize=style.get('figsize'),
                            dpi=style.get('dpi')
                        )
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Plot error", f"Failed to render 2D plot with matplotlib: {e}")
                # Queue this plot for table assembly later
                try:
                    img_abs = str(Path(img_path).resolve())
                    csv_abs = str(Path(csv_path).resolve())
                except Exception:
                    img_abs = str(img_path)
                    csv_abs = str(csv_path)
                entries.append({"title": title, "img": img_abs, "csv": csv_abs})
                # Record axes details for info file
                try:
                    axes_meta.append({
                        "index": idx_plot,
                        "type": "2d",
                        "title": title,
                        "x": {
                            "label": x_label,
                            "min": float(x_edges[0]) if len(x_edges) > 0 else None,
                            "max": float(x_edges[-1]) if len(x_edges) > 0 else None,
                            "bins": int(len(x_edges) - 1) if len(x_edges) > 0 else None,
                            "settings": (ndx.plot_control.axis_settings.get(x_label) if hasattr(ndx.plot_control, 'axis_settings') else None)
                        },
                        "y": {
                            "label": y_label,
                            "min": float(y_edges[0]) if len(y_edges) > 0 else None,
                            "max": float(y_edges[-1]) if len(y_edges) > 0 else None,
                            "bins": int(len(y_edges) - 1) if len(y_edges) > 0 else None,
                            "settings": (ndx.plot_control.axis_settings.get(y_label) if hasattr(ndx.plot_control, 'axis_settings') else None)
                        }
                    })
                except Exception:
                    pass
            else:
                axis = p.get("axis", "x").lower()
                if axis == "y":
                    edges, counts = ndx._histogram["y"]
                    a_label = y_label
                else:
                    edges, counts = ndx._histogram["x"]
                    a_label = x_label
                a_tok = _safe_token(a_label)
                csv_path = report_dir / f"{idx_plot:02d}_{axis}-{a_tok}.csv"
                self._save_1d_csv(csv_path, edges, counts, xlabel=a_label)
                img_path = report_dir / f"{idx_plot:02d}_1d_{axis}-{a_tok}.png"
                try:
                    template_name = (p.get("template") or "1d_basic").lower()
                    tpl = self._templates.get(template_name)
                    style = {}
                    if isinstance(tpl, dict):
                        style = tpl.get('style') or {}
                    self._save_1d_png(
                        img_path,
                        edges, counts,
                        title=title,
                        xlabel=a_label,
                        ylabel=style.get('ylabel', 'Count'),
                        color=style.get('color'),
                        edgecolor=style.get('edgecolor'),
                        linewidth=style.get('linewidth'),
                        figsize=style.get('figsize'),
                        dpi=style.get('dpi')
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Plot error", f"Failed to render 1D plot with matplotlib: {e}")
                # Queue this plot for table assembly later
                try:
                    img_abs = str(Path(img_path).resolve())
                    csv_abs = str(Path(csv_path).resolve())
                except Exception:
                    img_abs = str(img_path)
                    csv_abs = str(csv_path)
                entries.append({"title": title, "img": img_abs, "csv": csv_abs})
                # Record 1D axis details for info file
                try:
                    axes_meta.append({
                        "index": idx_plot,
                        "type": "1d",
                        "title": title,
                        "axis": axis,
                        "label": a_label,
                        "min": float(edges[0]) if len(edges) > 0 else None,
                        "max": float(edges[-1]) if len(edges) > 0 else None,
                        "bins": int(len(edges) - 1) if len(edges) > 0 else None,
                        "settings": (ndx.plot_control.axis_settings.get(a_label) if hasattr(ndx.plot_control, 'axis_settings') else None)
                    })
                except Exception:
                    pass

            idx_plot += 1

        if doc is not None:
            try:
                # Assemble plots into 2-column tables with 3 rows: Title, Image, Filepaths (image + CSV)
                for i in range(0, len(entries), 2):
                    table = doc.add_table(rows=3, cols=2)
                    for col in range(2):
                        idx = i + col
                        top_cell = table.cell(0, col)
                        mid_cell = table.cell(1, col)
                        bot_cell = table.cell(2, col)
                        if idx < len(entries):
                            entry = entries[idx]
                            # Top: title
                            top_cell.text = entry.get("title", "")
                            # Middle: image
                            p = mid_cell.paragraphs[0]
                            r = p.add_run()
                            img_path = entry.get("img", "")
                            try:
                                r.add_picture(img_path, width=Inches(3.0))
                            except Exception:
                                # Fallback: write the path if picture cannot be added
                                p.add_run(img_path)
                            # Bottom: full filepaths back-references
                            csv_path = entry.get("csv", "")
                            bot_lines = []
                            if csv_path:
                                bot_lines.append(f"CSV:   {csv_path}")
                            bot_cell.text = "\n".join(bot_lines) if bot_lines else ""
                        else:
                            top_cell.text = ""
                            # ensure empty middle/bottom cells
                            try:
                                mid_cell.text = ""
                            except Exception:
                                pass
                            bot_cell.text = ""
                    # Spacer between tables
                    doc.add_paragraph("")
                doc.save(out_docx)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "DOCX save error", str(e))
        else:
            QtWidgets.QMessageBox.warning(self, "python-docx missing",
                                          "python-docx is not installed. PNGs and CSVs were created; no DOCX.")
        # Write detailed axes info file in the report folder
        try:
            info_obj = {
                "analysis_folder": (str(folder.resolve()) if hasattr(folder, 'resolve') else str(folder)),
                "plots": axes_meta
            }
            with open(report_dir / "axes_info.yaml", 'w', encoding='utf-8') as f:
                yaml.safe_dump(info_obj, f, sort_keys=False, allow_unicode=True)
        except Exception:
            pass
        # Return a summary for potential final combined report
        try:
            folder_path_str = str(folder.resolve())
        except Exception:
            folder_path_str = str(folder)
        return {"folder": folder_path_str, "entries": entries}

    @staticmethod
    def _save_1d_csv(csv_path: Path, edges, counts, xlabel: str = "X"):
        import numpy as np
        edges = edges.astype(float)
        counts = counts.astype(float)
        # Sanitize
        edges = np.nan_to_num(edges, nan=0.0, posinf=0.0, neginf=0.0)
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Use axis label in the header; do not include any plot title
            safe_label = str(xlabel) if isinstance(xlabel, str) else "X"
            f.write(f"{safe_label}Start,{safe_label}End,Count\n")
            for i in range(len(counts)):
                f.write(f"{edges[i]},{edges[i+1]},{counts[i]}\n")

    @staticmethod
    def _save_2d_csv(csv_path: Path, H, x_edges, y_edges, xlabel: str = "X", ylabel: str = "Y"):
        import numpy as np
        H = np.array(H)
        x_edges = np.array(x_edges)
        y_edges = np.array(y_edges)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        # Sanitize
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        x_centers = np.nan_to_num(x_centers, nan=0.0, posinf=0.0, neginf=0.0)
        y_centers = np.nan_to_num(y_centers, nan=0.0, posinf=0.0, neginf=0.0)
        with open(csv_path, 'w', encoding='utf-8') as f:
            # header uses axis labels only; do not include any plot title
            safe_x = str(xlabel) if isinstance(xlabel, str) else "X"
            safe_y = str(ylabel) if isinstance(ylabel, str) else "Y"
            f.write(f"{safe_y}/{safe_x}," + ','.join([str(v) for v in x_centers]) + '\n')
            for j, y in enumerate(y_centers):
                row = [str(y)] + [str(H[i, j]) for i in range(len(x_centers))]
                f.write(','.join(row) + '\n')

    @staticmethod
    def _save_1d_png(img_path: Path, edges, counts, title: str, xlabel: str, ylabel: str = "Count",
                     xmin: Optional[float] = None, xmax: Optional[float] = None,
                     color: Optional[str] = None, edgecolor: Optional[str] = None, linewidth: Optional[float] = None,
                     figsize: Optional[List[float]] = None, dpi: Optional[int] = None) -> None:
        """Render a 1D histogram figure using matplotlib and save as PNG."""
        if plt is None:
            raise RuntimeError("matplotlib is not available for plotting")
        import numpy as np
        edges = np.asarray(edges, dtype=float)
        counts = np.asarray(counts, dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        fig_size = tuple(figsize) if isinstance(figsize, (list, tuple)) and len(figsize) == 2 else (6, 4)
        fig_dpi = int(dpi) if dpi else 150
        ec = edgecolor if edgecolor is not None else 'black'
        lw = float(linewidth) if linewidth is not None else 0.2
        col = color if color is not None else '#4477aa'
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
        # Sanitize data to avoid warnings
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        widths = np.nan_to_num(widths, nan=0.0, posinf=0.0, neginf=0.0)
        # Replace non-positive widths with a small epsilon to avoid invalid bar widths
        if widths.size:
            min_pos = widths[widths > 0].min() if np.any(widths > 0) else 1e-9
            widths = np.where(widths > 0, widths, min_pos)
        ax.bar(centers, counts, width=widths, align='center', edgecolor=ec, linewidth=lw, color=col)
        ax.set_title(title)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel)
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin if xmin is not None else ax.get_xlim()[0],
                        right=xmax if xmax is not None else ax.get_xlim()[1])
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(str(img_path), bbox_inches='tight', facecolor='white')
        plt.close(fig)

    @staticmethod
    def _save_2d_png(img_path: Path, H, x_edges, y_edges, title: str, xlabel: str, ylabel: str,
                     xmin: Optional[float] = None, xmax: Optional[float] = None,
                     ymin: Optional[float] = None, ymax: Optional[float] = None,
                     cmap: str = 'viridis', figsize: Optional[List[float]] = None, dpi: Optional[int] = None) -> None:
        """Render a 2D histogram density figure using matplotlib and save as PNG."""
        if plt is None:
            raise RuntimeError("matplotlib is not available for plotting")
        import numpy as np
        H = np.asarray(H, dtype=float)
        x_edges = np.asarray(x_edges, dtype=float)
        y_edges = np.asarray(y_edges, dtype=float)
        # Sanitize to avoid NaNs/Infs in image and extent
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]] if (x_edges.size and y_edges.size) else [0, 1, 0, 1]
        fig_size = tuple(figsize) if isinstance(figsize, (list, tuple)) and len(figsize) == 2 else (6, 5)
        fig_dpi = int(dpi) if dpi else 150
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
        im = ax.imshow(H.T, origin='lower', aspect='auto', extent=extent, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Density')
        ax.set_title(title)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin if xmin is not None else ax.get_xlim()[0],
                        right=xmax if xmax is not None else ax.get_xlim()[1])
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin if ymin is not None else ax.get_ylim()[0],
                        top=ymax if ymax is not None else ax.get_ylim()[1])
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(str(img_path), bbox_inches='tight', facecolor='white')
        plt.close(fig)

    @staticmethod
    def _save_2d_marginals_png(img_path: Path, H, x_edges, y_edges, title: str, xlabel: str, ylabel: str,
                               cmap_2d: str = 'Greys', color_x: str = 'dimgrey', color_y: str = 'darkorange',
                               figsize: Optional[List[float]] = None, dpi: Optional[int] = None) -> None:
        """Render a 2D histogram with top/right marginal histograms and save as PNG.
        This template is range-agnostic and data-agnostic; it uses provided H and edges only.
        """
        if plt is None:
            raise RuntimeError("matplotlib is not available for plotting")
        import numpy as np
        H = np.asarray(H, dtype=float)
        x_edges = np.asarray(x_edges, dtype=float)
        y_edges = np.asarray(y_edges, dtype=float)
        # Sanitize H to avoid NaNs/Infs
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0 if x_edges.size > 1 else np.array([0.5])
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0 if y_edges.size > 1 else np.array([0.5])
        # Marginals: sum over the other axis
        # H shape is (len(x_bins), len(y_bins)) based on CSV saver orientation
        x_marg = H.sum(axis=1) if H.size else np.array([0.0])
        y_marg = H.sum(axis=0) if H.size else np.array([0.0])
        # Layout via GridSpec
        from matplotlib.gridspec import GridSpec
        fig_size = tuple(figsize) if isinstance(figsize, (list, tuple)) and len(figsize) == 2 else (8, 8)
        fig_dpi = int(dpi) if dpi else 150
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        gs = GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.05, wspace=0.05)
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
        # 2D image in the main axes
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]] if (x_edges.size and y_edges.size) else [0, 1, 0, 1]
        im = ax_scatter.imshow(H.T, origin='lower', aspect='auto', extent=extent, cmap=cmap_2d)
        # Marginals as bar plots with sanitized widths/heights
        x_widths = np.nan_to_num(np.diff(x_edges), nan=0.0, posinf=0.0, neginf=0.0)
        if x_widths.size:
            x_min_pos = x_widths[x_widths > 0].min() if np.any(x_widths > 0) else 1e-9
            x_widths = np.where(x_widths > 0, x_widths, x_min_pos)
        y_heights = np.nan_to_num(np.diff(y_edges), nan=0.0, posinf=0.0, neginf=0.0)
        if y_heights.size:
            y_min_pos = y_heights[y_heights > 0].min() if np.any(y_heights > 0) else 1e-9
            y_heights = np.where(y_heights > 0, y_heights, y_min_pos)
        ax_histx.bar(x_centers, x_marg, width=x_widths, color=color_x, edgecolor='black', linewidth=0.2)
        ax_histy.barh(y_centers, y_marg, height=y_heights, color=color_y, edgecolor='black', linewidth=0.2)
        # Labels and ticks
        ax_scatter.set_xlabel(xlabel or "")
        ax_scatter.set_ylabel(ylabel or "")
        ax_histx.tick_params(axis='x', labelbottom=False)
        ax_histy.tick_params(axis='y', labelleft=False)
        # Clean grids
        ax_histx.grid(False)
        ax_histy.grid(False)
        ax_scatter.grid(False)
        # Colorbar for 2D
        cbar = fig.add_axes([0.92, 0.1, 0.02, 0.55])  # manual position to the right
        cb = fig.colorbar(im, cax=cbar)
        cb.set_label('Counts')
        # Title at top
        fig.suptitle(title, y=0.98)
        # Adjust layout manually to avoid tight_layout warnings with additional axes (colorbar)
        fig.subplots_adjust(left=0.10, bottom=0.08, right=0.88, top=0.94, wspace=0.05, hspace=0.05)
        fig.savefig(str(img_path), bbox_inches='tight', facecolor='white')
        plt.close(fig)


# Convenience function to open dialog from NDXplorer

def open_report_wizard(parent=None):
    dlg = ReportWizard(parent=parent)
    dlg.exec_()
