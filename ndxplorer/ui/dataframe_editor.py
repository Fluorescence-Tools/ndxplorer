"""
DataFrameEditor — Feature-rich table editor for pandas DataFrames.

Provides sortable columns, type-aware editing, search, copy/paste,
and keyboard navigation, matching the usability of guidata's DataFrameEditor.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from qtpy import QtCore, QtGui, QtWidgets


class DataFrameEditor(QtWidgets.QDialog):
    """Modal dialog for viewing and editing a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._original = df.copy()
        self._df = df
        self._clipboard: Optional[List[List[str]]] = None
        self._sort_column: Optional[int] = None
        self._sort_order: QtCore.Qt.SortOrder = QtCore.Qt.AscendingOrder
        self._setup_ui()
        self._populate()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        self.setWindowTitle("DataFrame Editor")
        self.resize(900, 600)
        self.setMinimumSize(500, 300)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Search bar ---
        search_layout = QtWidgets.QHBoxLayout()
        search_layout.setSpacing(4)
        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText("Search...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search_edit)

        self._search_label = QtWidgets.QLabel()
        search_layout.addWidget(self._search_label)
        layout.addLayout(search_layout)

        # --- Table ---
        self._table = QtWidgets.QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectItems
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.ContiguousSelection
        )
        self._table.horizontalHeader().setSectionsMovable(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(24)
        self._table.horizontalHeader().setDefaultSectionSize(100)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.cellChanged.connect(self._on_cell_changed)

        # Keyboard shortcuts
        self._table.installEventFilter(self)

        layout.addWidget(self._table)

        # --- Stats bar ---
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.setSpacing(12)

        self._stats_label = QtWidgets.QLabel()
        stats_layout.addWidget(self._stats_label)
        stats_layout.addStretch()

        self._modified_label = QtWidgets.QLabel()
        self._modified_label.setStyleSheet("color: #cc7000;")
        self._modified_label.setVisible(False)
        stats_layout.addWidget(self._modified_label)

        layout.addLayout(stats_layout)

        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self._btn_reset = QtWidgets.QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self._btn_reset)

        self._btn_apply = QtWidgets.QPushButton("Apply")
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_apply.setDefault(True)
        btn_layout.addWidget(self._btn_apply)

        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)

        layout.addLayout(btn_layout)

        self._update_stats()

    # ------------------------------------------------------------------ Populate

    def _populate(self, df: Optional[pd.DataFrame] = None):
        """Fill the table widget from a DataFrame."""
        df = df if df is not None else self._df
        self._table.blockSignals(True)
        self._table.setSortingEnabled(False)

        nrows, ncols = df.shape
        self._table.setRowCount(nrows)
        self._table.setColumnCount(ncols)

        # Headers
        self._table.setHorizontalHeaderLabels(list(df.columns))
        self._table.setVerticalHeaderLabels([str(i) for i in df.index])

        # Column types for cell rendering
        self._col_dtypes = [df.iloc[:, j].dtype for j in range(ncols)]

        for j in range(ncols):
            dtype = self._col_dtypes[j]
            is_numeric = np.issubdtype(dtype, np.number) if hasattr(dtype, 'kind') else False

            for i in range(nrows):
                val = df.iloc[i, j]
                item = QtWidgets.QTableWidgetItem()
                if pd.isna(val):
                    item.setText("")
                    item.setForeground(QtGui.QColor("#999999"))
                    item.setToolTip("NaN")
                elif is_numeric:
                    if isinstance(val, (float, np.floating)):
                        item.setText(f"{val:.6g}")
                    else:
                        item.setText(str(val))
                    item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                    )
                    item.setFlags(
                        item.flags() | QtCore.Qt.ItemIsEditable
                    )
                else:
                    item.setText(str(val))
                    item.setFlags(
                        item.flags() | QtCore.Qt.ItemIsEditable
                    )
                self._table.setItem(i, j, item)

        self._table.setSortingEnabled(True)
        self._table.blockSignals(False)
        self._resize_columns()
        self._update_stats()

    def _resize_columns(self):
        """Smart column sizing — content-based with sensible limits."""
        header = self._table.horizontalHeader()
        for j in range(self._table.columnCount()):
            header.resizeSections(QtWidgets.QHeaderView.ResizeToContents)
        # Enforce min/max
        for j in range(self._table.columnCount()):
            w = self._table.columnWidth(j)
            w = max(60, min(w + 20, 350))
            self._table.setColumnWidth(j, w)
        self._table.horizontalHeader().setStretchLastSection(True)

    # ------------------------------------------------------------------ Search

    def _on_search(self, text: str):
        """Filter rows to those matching the search text."""
        if not text:
            self._populate()
            return

        text_lower = text.lower()
        df = self._original if hasattr(self, '_original') else self._df

        mask = df.map(
            lambda v: text_lower in str(v).lower() if pd.notna(v) else False
        ).any(axis=1)

        filtered = df[mask]
        self._search_label.setText(
            f"{len(filtered)} / {len(df)} rows"
            if len(filtered) < len(df) else ""
        )
        self._populate(filtered)

    # ------------------------------------------------------------------ Editing

    def _on_cell_changed(self, row: int, col: int):
        """Validate and apply cell edits."""
        item = self._table.item(row, col)
        if item is None:
            return

        raw = item.text().strip()
        dtype = self._col_dtypes[col]

        # Validate numeric columns
        if np.issubdtype(dtype, np.number) if hasattr(dtype, 'kind') else False:
            try:
                if raw == "":
                    val = np.nan
                elif "." in raw or "e" in raw.lower() or "nan" in raw.lower():
                    val = float(raw)
                else:
                    val = int(raw)
            except (ValueError, TypeError):
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Value",
                    f"Column '{self._df.columns[col]}' expects a numeric value."
                )
                # Restore original value
                orig = self._df.iloc[row, col]
                item.setText(f"{orig:.6g}" if isinstance(orig, float) else str(orig))
                return

            item.setText(f"{val:.6g}" if isinstance(val, (float, np.floating)) else str(val))

        self._df.iloc[row, col] = self._parse_cell(raw, dtype)
        self._mark_modified()

    @staticmethod
    def _parse_cell(raw: str, dtype) -> Any:
        """Parse a string cell value to the appropriate type."""
        raw = raw.strip()
        if raw == "" or raw.lower() == "nan":
            return np.nan
        if np.issubdtype(dtype, np.number) if hasattr(dtype, 'kind') else False:
            if "." in raw or "e" in raw.lower():
                return float(raw)
            return int(raw)
        return raw

    def _mark_modified(self):
        self._modified_label.setText("Modified")
        self._modified_label.setVisible(True)

    def _on_reset(self):
        """Reset to original data."""
        self._df = self._original.copy()
        self._modified_label.setVisible(False)
        self._search_edit.clear()
        self._populate()

    def _on_apply(self):
        """Accept changes."""
        self.accept()

    # ------------------------------------------------------------------ Context menu

    def _on_context_menu(self, pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)

        copy_action = menu.addAction("Copy")
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(self._copy_selection)

        paste_action = menu.addAction("Paste")
        paste_action.setShortcut(QtGui.QKeySequence.Paste)
        paste_action.triggered.connect(self._paste_selection)

        menu.addSeparator()

        select_all_action = menu.addAction("Select All")
        select_all_action.setShortcut(QtGui.QKeySequence.SelectAll)
        select_all_action.triggered.connect(self._table.selectAll)

        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _copy_selection(self):
        """Copy selected cells as TSV."""
        selected = self._table.selectedRanges()
        if not selected:
            return

        rng = selected[0]
        rows = []
        for i in range(rng.topRow(), rng.bottomRow() + 1):
            row_vals = []
            for j in range(rng.leftColumn(), rng.rightColumn() + 1):
                item = self._table.item(i, j)
                row_vals.append(item.text() if item else "")
            rows.append("\t".join(row_vals))

        QtWidgets.QApplication.clipboard().setText("\n".join(rows))

    def _paste_selection(self):
        """Paste TSV data from clipboard into selected cells."""
        text = QtWidgets.QApplication.clipboard().text()
        if not text:
            return

        rows_data = [line.split("\t") for line in text.splitlines()]
        if not rows_data:
            return

        selected = self._table.selectedRanges()
        if not selected:
            return

        rng = selected[0]
        start_row = rng.topRow()
        start_col = rng.leftColumn()

        for i, row_vals in enumerate(rows_data):
            for j, val in enumerate(row_vals):
                row = start_row + i
                col = start_col + j
                if row >= self._table.rowCount() or col >= self._table.columnCount():
                    break
                item = self._table.item(row, col)
                if item is not None and item.flags() & QtCore.Qt.ItemIsEditable:
                    item.setText(val.strip())
                    self._on_cell_changed(row, col)

    # ------------------------------------------------------------------ Event filter (keyboard shortcuts)

    def eventFilter(self, obj, event):
        if obj is self._table and event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            mod = event.modifiers()

            if mod == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_C:
                self._copy_selection()
                return True
            if mod == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_V:
                self._paste_selection()
                return True
            if mod == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_A:
                self._table.selectAll()
                return True
            if mod == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_Z:
                self._on_reset()
                return True

            # Arrow-key navigation across cells
            if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                       QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                       QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab):
                return False  # Let QTableWidget handle normally

            # Enter to edit current cell
            if key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter:
                self._table.edit(self._table.currentIndex())
                return True

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ Stats

    def _update_stats(self):
        nrows = self._table.rowCount()
        ncols = self._table.columnCount()
        self._stats_label.setText(f"{nrows} rows × {ncols} columns")

    # ------------------------------------------------------------------ Access

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @staticmethod
    def edit_dataframe(df: pd.DataFrame, parent=None) -> Optional[pd.DataFrame]:
        """Convenience: show editor, return edited copy or None if cancelled."""
        dlg = DataFrameEditor(df, parent)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            return dlg.dataframe
        return None
