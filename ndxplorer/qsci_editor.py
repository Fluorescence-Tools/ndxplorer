import sys

from qtpy import QtGui, QtWidgets
try:
    from PyQt4.Qsci import QsciScintilla
except ImportError:
    from PyQt5.Qsci import QsciScintilla


class SimpleCodeEditor(QsciScintilla):

    ARROW_MARKER_NUM = 8

    def __init__(
            self,
            parent=None,
            font_family=None,
            font_point_size=8,
            margins_background_color=None,
            marker_background_color=None,
            caret_line_background_color=None,
            caret_line_visible=True
    ):
        """

        :param parent:
        :param font_family:
        :param font_point_size:
        :param margins_background_color:
        :param marker_background_color:
        :param caret_line_background_color:
        :param caret_line_visible:
        :param language: a string that is set to select the lexer of the
        editor (either Python or JSON) the default lexer is a YAML lexer
        :param kwargs:
        """
        super(SimpleCodeEditor, self).__init__(parent)
        # Set the default font
        font = QtGui.QFont()
        if font_family:
            font.setFamily(font_family)
        else:
            font.setFamily("Courier")
        font.setFixedPitch(True)
        font.setPointSize(font_point_size)

        self.setFont(font)
        self.setMarginsFont(font)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)

        # Margin 0 is used for line numbers
        fontmetrics = QtGui.QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, fontmetrics.width("0000") + 6)
        self.setMarginLineNumbers(0, True)
        if marker_background_color:
            self.setMarginsBackgroundColor(
                QtGui.QColor(margins_background_color)
            )

        self.markerDefine(QsciScintilla.RightArrow, self.ARROW_MARKER_NUM)

        if marker_background_color:
            self.setMarkerBackgroundColor(
                QtGui.QColor(marker_background_color),
                self.ARROW_MARKER_NUM
            )

        # Brace matching: enable for a brace immediately before or after
        # the current position
        #
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Current line visible with special background color
        self.setCaretLineVisible(caret_line_visible)
        if caret_line_background_color:
            self.setCaretLineBackgroundColor(
                QtGui.QColor(caret_line_background_color)
            )

        text = bytearray(str.encode("Courier New"))
        # 32, "Courier New"
        self.SendScintilla(QsciScintilla.SCI_STYLESETFONT, 1, text)

        # Don't want to see the horizontal scrollbar at all
        # Use raw message to Scintilla here (all messages are documented
        # here: http://www.scintilla.org/ScintillaDoc.html)
        self.SendScintilla(QsciScintilla.SCI_SETHSCROLLBAR, 0)

        # not too small
        self.setMinimumSize(900, 300)

    def on_margin_clicked(self, nmargin, nline, modifiers):
        # Toggle marker for the line the margin was clicked on
        if self.markersAtLine(nline) != 0:
            self.markerDelete(nline, self.ARROW_MARKER_NUM)
        else:
            self.markerAdd(nline, self.ARROW_MARKER_NUM)


class CodeEditor(QtWidgets.QWidget):

    def text(self):
        #  type: ()->str
        return self.editor.text()

    def load_file(
            self,
            filename=None
    ):
        if filename is None:
            filename = QtGui.QFileDialog.getOpenFileName()
        self.filename = filename
        try:
            print('loading filename: ', filename)
            with open(filename) as fp:
                text = fp.read()
            self.editor.setText(text)
        except IOError:
            print("Not a valid filename.")

    def save_text(self):
        if self.filename is None or self.filename == '':
            self.filename = QtGui.QFileDialog.getSaveFileName()
        with open(self.filename, mode='w') as fp:
            text = str(self.editor.text())
            fp.write(text)
        if callable(self.save_callback):
            self.save_callback()

    def __init__(
            self,
            filename=None,
            save_callback=None,
            **kwargs
    ):
        super(CodeEditor, self).__init__(**kwargs)
        self.save_callback = save_callback

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.filename = None
        self.setLayout(layout)
        self.editor = SimpleCodeEditor(
            parent=self
        )
        layout.addWidget(self.editor)
        h_layout = QtWidgets.QHBoxLayout()

        save_button = QtWidgets.QPushButton('save')

        h_layout.addWidget(save_button)
        layout.addLayout(h_layout)

        save_button.clicked.connect(self.save_text)

        # Load the file
        if filename is not None and filename is not '':
            self.load_file(filename=filename)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    editor = CodeEditor()
    editor.show()
    app.exec_()
