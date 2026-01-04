"""Shared progress window UI component."""

from __future__ import annotations

from qtpy.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QPushButton
from qtpy.QtCore import Qt


class ProgressWindow(QDialog):
    """Reusable progress dialog with label and progress bar."""

    def __init__(self, title="Progress", message="Processing...", max_value=100, cancelable=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(self)
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        
        # Add cancel button if cancelable
        self._cancelled = False
        if cancelable:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._on_cancel)
            layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)

    def set_value(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)

    def set_message(self, message: str):
        """Update progress message."""
        self.label.setText(message)
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.label.setText("Cancelling...")
        self.cancel_button.setEnabled(False)
    
    def was_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self._cancelled
