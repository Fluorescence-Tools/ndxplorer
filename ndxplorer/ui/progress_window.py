"""Shared progress window UI component."""

from __future__ import annotations

from qtpy.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel
from qtpy.QtCore import Qt


class ProgressWindow(QDialog):
    """Reusable progress dialog with label and progress bar."""

    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(self)
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def set_value(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)

    def set_message(self, message: str):
        """Update progress message."""
        self.label.setText(message)
