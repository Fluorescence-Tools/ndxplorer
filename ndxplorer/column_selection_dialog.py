"""
Dialog for selecting columns from a list.
"""
from .logging_config import logging

try:
    from chisurf.gui import QtGui, QtCore, QtWidgets
except ImportError:
    from qtpy import QtCore
    from qtpy import QtGui, QtWidgets


class ColumnSelectionDialog(QtWidgets.QDialog):
    """
    Dialog for selecting columns to use in clustering.
    Supports keyboard navigation:
    - Up/Down arrow keys to navigate through columns
    - Ctrl+Space to toggle selection of the focused column
    """
    def __init__(self, parent=None, column_names=None, selected_columns=None):
        logging.log(0, f"Initializing ColumnSelectionDialog with {len(column_names) if column_names else 0} columns")
        super(ColumnSelectionDialog, self).__init__(parent)
        self.setWindowTitle("Select Columns for Clustering")
        self.setMinimumWidth(300)

        # Store column names and selected columns
        self.column_names = column_names or []
        self.selected_columns = selected_columns or set()

        # Track the currently focused checkbox
        self.current_focus_index = -1
        self.visible_checkboxes = []

        # Create layout
        layout = QtWidgets.QVBoxLayout()

        # Add label
        label = QtWidgets.QLabel("Select columns to use for clustering:")
        layout.addWidget(label)

        # Add help text for keyboard navigation
        help_label = QtWidgets.QLabel("Navigation: Use Up/Down keys to move, Ctrl+Space to toggle selection")
        help_label.setStyleSheet("color: #666666; font-size: 10pt;")
        layout.addWidget(help_label)

        # Add filter line edit
        filter_layout = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Filter:")
        self.filter_line_edit = QtWidgets.QLineEdit()
        self.filter_line_edit.setPlaceholderText("Enter text to filter columns")
        self.filter_line_edit.textChanged.connect(self.filter_columns)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_line_edit)
        layout.addLayout(filter_layout)

        # Create scroll area for checkboxes
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        # Set alignment to top
        self.scroll_layout.setAlignment(QtCore.Qt.AlignTop)

        # Add checkboxes for each column
        self.checkboxes = {}
        for column in self.column_names:
            checkbox = QtWidgets.QCheckBox(column)
            checkbox.setChecked(column in self.selected_columns)
            self.checkboxes[column] = checkbox
            self.scroll_layout.addWidget(checkbox)
            self.visible_checkboxes.append(checkbox)

        # Set focus on the first checkbox if any exist
        if self.visible_checkboxes:
            self.current_focus_index = 0
            self.update_focus()

        # Add select all / deselect all buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        select_all_button = QtWidgets.QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        deselect_all_button = QtWidgets.QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self.deselect_all)
        buttons_layout.addWidget(select_all_button)
        buttons_layout.addWidget(deselect_all_button)

        # Add scroll area to layout
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)
        layout.addLayout(buttons_layout)

        # Add OK/Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def select_all(self):
        """Select all columns"""
        logging.log(0, f"Selecting all {len(self.checkboxes)} columns")
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
        # Maintain focus after selection
        self.update_focus()

    def deselect_all(self):
        """Deselect all columns"""
        logging.log(0, f"Deselecting all {len(self.checkboxes)} columns")
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
        # Maintain focus after deselection
        self.update_focus()

    def filter_columns(self, text):
        """Filter the checkboxes based on the text entered in the line edit"""
        logging.log(0, f"Filtering columns with text: '{text}'")
        filter_text = text.lower()
        self.visible_checkboxes = []

        for column, checkbox in self.checkboxes.items():
            # Show checkbox if column name contains filter text (case-insensitive)
            # or if filter text is empty
            is_visible = not filter_text or filter_text in column.lower()
            checkbox.setVisible(is_visible)
            if is_visible:
                self.visible_checkboxes.append(checkbox)

        # Reset focus index if no checkboxes are visible
        if not self.visible_checkboxes:
            self.current_focus_index = -1
        # Adjust focus index if it's out of bounds
        elif self.current_focus_index >= len(self.visible_checkboxes):
            self.current_focus_index = len(self.visible_checkboxes) - 1

        # Update visual focus
        self.update_focus()

    def update_focus(self):
        """Update the visual focus indicator for the currently focused checkbox"""
        # Reset all checkboxes to normal style
        for checkbox in self.visible_checkboxes:
            checkbox.setStyleSheet("")

        # Set style for the focused checkbox
        if 0 <= self.current_focus_index < len(self.visible_checkboxes):
            focused_checkbox = self.visible_checkboxes[self.current_focus_index]
            focused_checkbox.setStyleSheet("QCheckBox { background-color: lightblue; }")

            # Ensure the focused checkbox is visible in the scroll area
            self.scroll_area.ensureWidgetVisible(focused_checkbox)

    def keyPressEvent(self, event):
        """Handle keyboard navigation"""
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key_Up:
            # Move focus up
            if self.visible_checkboxes and self.current_focus_index > 0:
                self.current_focus_index -= 1
                self.update_focus()
            event.accept()
        elif key == QtCore.Qt.Key_Down:
            # Move focus down
            if self.visible_checkboxes and self.current_focus_index < len(self.visible_checkboxes) - 1:
                self.current_focus_index += 1
                self.update_focus()
            event.accept()
        elif key == QtCore.Qt.Key_Space and modifiers == QtCore.Qt.ControlModifier:
            # Toggle checkbox state with Ctrl+Space
            if 0 <= self.current_focus_index < len(self.visible_checkboxes):
                checkbox = self.visible_checkboxes[self.current_focus_index]
                checkbox.setChecked(not checkbox.isChecked())
            event.accept()
        else:
            # Pass other keys to parent class
            super(ColumnSelectionDialog, self).keyPressEvent(event)

    def get_selected_columns(self):
        """Get the set of selected column names"""
        selected = {column for column, checkbox in self.checkboxes.items() if checkbox.isChecked()}
        logging.log(0, f"Getting selected columns: {len(selected)} columns selected")
        return selected
