"""Reusable UMAP worker + progress dialog extracted from plot_main."""

from __future__ import annotations

from ..logging_config import logging
from io import StringIO
import sys
import threading
import time

import numpy as np

from qtpy import QtCore, QtGui, QtWidgets


class UMAPWorker(QtCore.QObject):
    """Worker object that runs UMAP in a background thread."""

    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str)

    def __init__(self, clean_data: np.ndarray, umap_params):
        super().__init__()
        self.clean_data = clean_data
        self.umap_params = umap_params

    def run(self) -> None:
        """Perform the embedding while streaming tqdm output."""
        try:
            import umap as _umap

            reducer = _umap.UMAP(**self.umap_params)
            output_buffer = StringIO()

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            def emit_progress():
                last_content = ""
                while not getattr(threading.current_thread(), "stop_flag", False):
                    current_content = output_buffer.getvalue()
                    if current_content != last_content:
                        self.progress.emit(current_content)
                        last_content = current_content
                    time.sleep(0.1)

            progress_thread = threading.Thread(target=emit_progress, daemon=True)

            try:
                sys.stdout = output_buffer
                sys.stderr = output_buffer
                progress_thread.start()
                umap_embedding = reducer.fit_transform(self.clean_data)
                progress_thread.stop_flag = True
                self.progress.emit(output_buffer.getvalue())
                self.finished.emit(umap_embedding)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        except Exception as exc:  # pragma: no cover - GUI path
            self.error.emit(str(exc))


class UMAPProgressDialog(QtWidgets.QDialog):
    """Progress dialog that hosts the worker thread and streams updates."""

    def __init__(self, parent=None, title: str = "UMAP Progress"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setMinimumSize(600, 400)
        self.resize(700, 500)

        layout = QtWidgets.QVBoxLayout(self)

        title_label = QtWidgets.QLabel("UMAP Computation Progress")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title_label)

        self.text_area = QtWidgets.QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QtGui.QFont("Consolas", 10))
        self.text_area.setStyleSheet(
            """
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 10px;
            }
            """
        )
        layout.addWidget(self.text_area)

        self.status_label = QtWidgets.QLabel("Initializing UMAP computation...")
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self.status_label)

        self.button_box = QtWidgets.QDialogButtonBox()
        self.close_button = self.button_box.addButton("Close", QtWidgets.QDialogButtonBox.AcceptRole)
        self.close_button.clicked.connect(self.accept)
        self.button_box.setVisible(False)
        layout.addWidget(self.button_box)

        self.worker = None
        self.thread = None
        self.result = None
        self.error_message = None

        self.auto_close_timer = QtCore.QTimer()
        self.auto_close_timer.setSingleShot(True)
        self.auto_close_timer.timeout.connect(self.accept)

    def run_umap_computation(self, clean_data: np.ndarray, umap_params) -> None:
        """Spin up the worker thread and begin the embedding."""
        self.worker = UMAPWorker(clean_data, umap_params)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_computation_finished)
        self.worker.error.connect(self.on_computation_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.status_label.setText("UMAP computation in progress...")
        self.thread.start()

    def on_progress_update(self, content: str) -> None:
        """Display tqdm output in the dialog."""
        self.text_area.setPlainText(content)
        cursor = self.text_area.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.text_area.setTextCursor(cursor)

    def on_computation_finished(self, result) -> None:
        """Handle a successful run."""
        self.result = result
        completion_message = (
            f"\n\n{'='*60}\n✅ UMAP COMPUTATION COMPLETED SUCCESSFULLY!\n{'='*60}\n"
            f"• Embedding shape: {result.shape}\n"
            f"• Components: {result.shape[1]}\n"
            f"• Data points processed: {result.shape[0]}\n"
            "\nDialog will close automatically in 3 seconds...\n"
        )
        current_content = self.text_area.toPlainText()
        self.text_area.setPlainText(current_content + completion_message)
        cursor = self.text_area.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.text_area.setTextCursor(cursor)

        self.status_label.setText("✅ UMAP computation completed successfully! Auto-closing in 3 seconds...")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.auto_close_timer.start(3000)

    def on_computation_error(self, error_message: str) -> None:
        """Handle worker failures."""
        self.error_message = error_message
        error_msg = (
            f"\n\n{'='*60}\n❌ UMAP COMPUTATION FAILED\n{'='*60}\n"
            f"Error: {error_message}\n\nPlease check your parameters and try again.\n"
        )
        current_content = self.text_area.toPlainText()
        self.text_area.setPlainText(current_content + error_msg)
        cursor = self.text_area.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.text_area.setTextCursor(cursor)
        self.status_label.setText(f"❌ UMAP computation failed: {error_message}")
        self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        self.button_box.setVisible(True)

    def get_result(self):
        """Return embedding array (if any)."""
        return self.result

    def get_error(self):
        """Return error string (if any)."""
        return self.error_message
