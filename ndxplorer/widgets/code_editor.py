from __future__ import annotations

import pathlib
import sys

from qtpy import QtCore, QtGui, QtWidgets

class TextEditor(QtWidgets.QPlainTextEdit):
    """Simplified text editor with basic functionality."""

    def __init__(
            self,
            parent=None,
            font_family="Courier",
            font_point_size=10.0,
            **kwargs
    ):
        """
        Initialize the text editor.

        :param parent: Parent widget
        :param font_family: Font family to use
        :param font_point_size: Font size
        :param kwargs: Additional keyword arguments
        """
        super().__init__(parent)

        # Set the default font
        font = QtGui.QFont()
        font.setFamily(font_family)
        font.setPointSize(int(font_point_size))
        self.setFont(font)

        # Set up line numbers
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)

        # Set up colors
        paper_color = kwargs.get("paper_color", "#FFFFFF")
        default_color = kwargs.get("default_color", "#000000")
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(paper_color))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(default_color))
        self.setPalette(palette)

        # Set minimum size
        self.setMinimumSize(400, 200)

    def line_number_area_width(self):
        """Calculate the width of the line number area."""
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1

        space = 3 + self.fontMetrics().width('9') * digits
        return space

    def update_line_number_area_width(self, _):
        """Update the width of the line number area."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        """Update the line number area when the editor's viewport is scrolled."""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        """Handle resize events to adjust the line number area."""
        super().resizeEvent(event)

        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        """Paint the line number area."""
        painter = QtGui.QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QtGui.QColor("#F0F0F0"))  # Light gray background

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtCore.Qt.darkGray)
                rect = QtCore.QRect(0, int(top), self.line_number_area.width(), self.fontMetrics().height())
                painter.drawText(rect, QtCore.Qt.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    def text(self):
        """Get the text content of the editor."""
        return self.toPlainText()

    def setText(self, text):
        """Set the text content of the editor."""
        self.setPlainText(text)


class LineNumberArea(QtWidgets.QWidget):
    """Widget for displaying line numbers."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QtCore.QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class CodeEditor(QtWidgets.QWidget):
    """Widget that combines a text editor with load and save buttons."""

    def load_file_event(self, event=None, filename=None, **kwargs):
        self.load_file(filename)

    def load_file(self, filename=None, **kwargs):
        """Load a file into the editor."""
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File")
        if not filename:
            return
        try:
            with open(filename, encoding="utf-8") as file:
                self.editor.setText(file.read())
            self.line_edit.setText(str(filename))
            self.filename = filename
        except IOError as e:
            print(f"Error loading file {filename}: {e}")

    def save_text(self, event=None):
        """Save the current text to a file."""
        if not self.filename:
            self.filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File")
            if not self.filename:
                return
        try:
            with open(self.filename, mode='w', encoding="utf-8") as file:
                file.write(self.editor.text())
            self.line_edit.setText(str(self.filename))
            if callable(self.save_callback):
                self.save_callback()
        except IOError as e:
            print(f"Error saving file {self.filename}: {e}")

    def text(self):
        """Get the text content of the editor."""
        return self.editor.text()

    def __init__(
        self,
        *args,
        filename=None,
        language="Python",
        can_load=True,
        save_callback=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_callback = save_callback

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.filename = None
        self.setLayout(layout)
        self.line_edit = QtWidgets.QLineEdit()
        self.editor = TextEditor(
            parent=self
        )
        layout.addWidget(self.editor)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load")
        self.save_button = QtWidgets.QPushButton("Save")
        self.run_button = QtWidgets.QPushButton("Run")

        button_layout.addWidget(self.line_edit)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.run_button)
        layout.addLayout(button_layout)

        # Connect buttons to actions
        self.save_button.clicked.connect(self.save_text)
        self.load_button.clicked.connect(self.load_file_event)

        # Handle initial file loading
        if filename and pathlib.Path(filename).is_file():
            self.load_file(filename=filename)

        if language.lower() != "python":
            self.run_button.hide()

        if not can_load:
            self.load_button.hide()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    editor = CodeEditor()
    editor.show()
    app.exec_()