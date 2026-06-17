from __future__ import annotations

from typing import Optional

from qtpy import QtCore, QtWidgets


class ProgressPane(QtWidgets.QFrame):
    """Drop-in widget that shows busy/progress state with accessible messaging."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        busy_text: str = "Working…",
        show_progress_bar: bool = True,
    ):
        super().__init__(parent)
        self.setObjectName("NDXProgressPane")
        self.setProperty("role", "feedback")
        self._default_busy_text = busy_text

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        self.message_label = QtWidgets.QLabel(busy_text, self)
        self.message_label.setWordWrap(True)
        self.message_label.setProperty("status", "info")
        self.message_label.setAccessibleDescription("Operation status message")
        layout.addWidget(self.message_label)

        self.progress_bar = None
        if show_progress_bar:
            self.progress_bar = QtWidgets.QProgressBar(self)
            self.progress_bar.setRange(0, 0)  # indeterminate by default
            self.progress_bar.setTextVisible(False)
            layout.addWidget(self.progress_bar)

        self.setVisible(False)

    # Public API ---------------------------------------------------------
    def start(self, message: Optional[str] = None, *, indeterminate: bool = True) -> None:
        """Show the pane and optionally set the message."""
        if message:
            self.set_message(message)
        else:
            self.set_message(self._default_busy_text)
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 0 if indeterminate else 100)
            if not indeterminate:
                self.progress_bar.setValue(0)
        self.setVisible(True)

    def set_message(self, message: str, *, status: str = "info") -> None:
        """Update visible copy + semantic status (impacts styling)."""
        self.message_label.setText(message)
        self.message_label.setProperty("status", status)
        self.message_label.style().unpolish(self.message_label)
        self.message_label.style().polish(self.message_label)

    def set_progress(self, value: int) -> None:
        """Update determinate progress (auto-switches out of busy mode)."""
        if self.progress_bar is None:
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(value))

    def finish(self, message: Optional[str] = None, *, success: bool = True, delay_ms: int = 1200) -> None:
        """Show completion text then hide after a short delay."""
        if message:
            status = "success" if success else "warning"
            self.set_message(message, status=status)
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100 if success else 0)
        QtCore.QTimer.singleShot(delay_ms, self.hide)

    def fail(self, message: str) -> None:
        """Display an error state immediately."""
        self.set_message(message, status="error")
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        self.setVisible(True)


class FriendlyErrorPresenter:
    """Unifies information/warning/error dialogs with consistent copy."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        self.parent = parent

    def info(self, title: str, text: str, *, details: Optional[str] = None) -> None:
        self._show_box(QtWidgets.QMessageBox.Information, title, text, details)

    def warn(self, title: str, text: str, *, details: Optional[str] = None) -> None:
        self._show_box(QtWidgets.QMessageBox.Warning, title, text, details)

    def error(self, title: str, text: str, *, details: Optional[str] = None) -> None:
        self._show_box(QtWidgets.QMessageBox.Critical, title, text, details)

    def question(
        self,
        title: str,
        text: str,
        *,
        default: QtWidgets.QMessageBox.StandardButton = QtWidgets.QMessageBox.Yes,
        buttons: QtWidgets.QMessageBox.StandardButtons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        details: Optional[str] = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        msg = self._build_box(QtWidgets.QMessageBox.Question, title, text, details)
        msg.setStandardButtons(buttons)
        msg.setDefaultButton(default)
        return QtWidgets.QMessageBox.StandardButton(msg.exec_())

    # Internal helpers ---------------------------------------------------
    def _show_box(
        self,
        icon: QtWidgets.QMessageBox.Icon,
        title: str,
        text: str,
        details: Optional[str],
    ) -> None:
        msg = self._build_box(icon, title, text, details)
        msg.exec_()

    def _build_box(
        self,
        icon: QtWidgets.QMessageBox.Icon,
        title: str,
        text: str,
        details: Optional[str],
    ) -> QtWidgets.QMessageBox:
        box = QtWidgets.QMessageBox(icon, title, text, parent=self.parent)
        box.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        if details:
            box.setInformativeText(details)
        return box
