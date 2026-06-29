"""
Axis Control Dialog for ndxplorer.

This module provides a dialog for enabling and disabling axes in the ndxplorer plots.
It also allows controlling the visibility of axis labels and saving these settings to a file.
"""

import os
import yaml
import pathlib
from qtpy import QtCore, QtGui, QtWidgets

from ..logging_config import logging

# Import settings functions
from ..settings import get_settings_path
from .feedback import FriendlyErrorPresenter


class AxisControlDialog(QtWidgets.QDialog):
    """
    Dialog for controlling the visibility of axes and axis labels in ndxplorer plots.
    
    This dialog provides checkboxes for enabling and disabling individual axes
    for each plot in the ndxplorer application. It also allows controlling the
    visibility of axis labels and saving these settings to a file.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the axis control dialog.
        
        Args:
            parent: The parent widget (typically the main window)
        """
        super(AxisControlDialog, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("Axis Control")
        self.setMinimumWidth(420)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self.error_presenter = FriendlyErrorPresenter(self)
        self.setup_ui()
        self.load_current_state()

    @staticmethod
    def _set_accessibility(widget, *, name=None, description=None):
        """Utility to apply accessible metadata to widgets."""
        if name:
            widget.setAccessibleName(name)
        if description:
            widget.setAccessibleDescription(description)
        
    def setup_ui(self):
        """Set up the user interface for the axis control dialog."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hint_label = QtWidgets.QLabel(
            "Tip: You can resize this dialog on smaller screens. "
            "Use the mouse wheel or PgUp/PgDn keys to scroll through controls."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #bbbbbb; font-size: 10pt;")
        self._set_accessibility(
            hint_label,
            name="Axis control hint",
            description="Explains how to resize the dialog and scroll through the axis toggles."
        )
        layout.addWidget(hint_label)
        
        # Create a scroll area to handle many checkboxes
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(280)
        self._set_accessibility(
            scroll_area,
            name="Axis toggles",
            description="Scrollable region containing toggle checkboxes for each plot axis."
        )
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        
        # X Plot group
        x_plot_group = QtWidgets.QGroupBox("X Plot")
        self._set_accessibility(
            x_plot_group,
            name="X plot axes",
            description="Options for showing or hiding the X plot axes."
        )
        x_plot_layout = QtWidgets.QVBoxLayout()
        self.x_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.x_plot_top = QtWidgets.QCheckBox("Top Axis")
        self.x_plot_left = QtWidgets.QCheckBox("Left Axis")
        self.x_plot_right = QtWidgets.QCheckBox("Right Axis")
        
        self.x_plot_bottom.setToolTip("Show/hide the bottom axis of the X plot")
        self.x_plot_top.setToolTip("Show/hide the top axis of the X plot")
        self.x_plot_left.setToolTip("Show/hide the left axis of the X plot")
        self.x_plot_right.setToolTip("Show/hide the right axis of the X plot")
        
        for checkbox, label in (
            (self.x_plot_bottom, "Bottom axis of the X plot"),
            (self.x_plot_top, "Top axis of the X plot"),
            (self.x_plot_left, "Left axis frame of the X plot"),
            (self.x_plot_right, "Right axis frame of the X plot"),
        ):
            self._set_accessibility(
                checkbox,
                name=label,
                description=f"Toggle visibility for the {label.lower()}."
            )
            x_plot_layout.addWidget(checkbox)
        x_plot_group.setLayout(x_plot_layout)
        scroll_layout.addWidget(x_plot_group)
        
        # Y Plot group
        y_plot_group = QtWidgets.QGroupBox("Y Plot")
        self._set_accessibility(
            y_plot_group,
            name="Y plot axes",
            description="Options for showing or hiding the Y plot axes."
        )
        y_plot_layout = QtWidgets.QVBoxLayout()
        self.y_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.y_plot_top = QtWidgets.QCheckBox("Top Axis")
        self.y_plot_left = QtWidgets.QCheckBox("Left Axis")
        self.y_plot_right = QtWidgets.QCheckBox("Right Axis")
        
        self.y_plot_bottom.setToolTip("Show/hide the bottom axis of the Y plot")
        self.y_plot_top.setToolTip("Show/hide the top axis of the Y plot")
        self.y_plot_left.setToolTip("Show/hide the left axis of the Y plot")
        self.y_plot_right.setToolTip("Show/hide the right axis of the Y plot")
        
        for checkbox, label in (
            (self.y_plot_bottom, "Bottom axis of the Y plot"),
            (self.y_plot_top, "Top axis of the Y plot"),
            (self.y_plot_left, "Left axis frame of the Y plot"),
            (self.y_plot_right, "Right axis frame of the Y plot"),
        ):
            self._set_accessibility(
                checkbox,
                name=label,
                description=f"Toggle visibility for the {label.lower()}."
            )
            y_plot_layout.addWidget(checkbox)
        y_plot_group.setLayout(y_plot_layout)
        scroll_layout.addWidget(y_plot_group)
        
        # Z Plot group
        z_plot_group = QtWidgets.QGroupBox("Z Plot")
        self._set_accessibility(
            z_plot_group,
            name="Z plot axes",
            description="Options for enabling or disabling the Z plot and its axes."
        )
        z_plot_layout = QtWidgets.QVBoxLayout()
        self.z_plot_enable = QtWidgets.QCheckBox("Enable Z Plot")
        self.z_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.z_plot_left = QtWidgets.QCheckBox("Left Axis")
        
        self.z_plot_enable.setToolTip("Enable/disable the Z plot")
        self.z_plot_bottom.setToolTip("Show/hide the bottom axis of the Z plot")
        self.z_plot_left.setToolTip("Show/hide the left axis of the Z plot")
        
        for checkbox, label in (
            (self.z_plot_enable, "Enable Z plot"),
            (self.z_plot_bottom, "Bottom axis of the Z plot"),
            (self.z_plot_left, "Left axis of the Z plot"),
        ):
            self._set_accessibility(
                checkbox,
                name=label,
                description=f"{label} toggle."
            )
            z_plot_layout.addWidget(checkbox)
        z_plot_group.setLayout(z_plot_layout)
        scroll_layout.addWidget(z_plot_group)
        
        # 2D Plot group
        plot_2d_group = QtWidgets.QGroupBox("2D Plot")
        self._set_accessibility(
            plot_2d_group,
            name="2D plot axes",
            description="Options for showing or hiding axes on the 2D projection."
        )
        plot_2d_layout = QtWidgets.QVBoxLayout()
        self.plot_2d_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.plot_2d_top = QtWidgets.QCheckBox("Top Axis")
        self.plot_2d_left = QtWidgets.QCheckBox("Left Axis")
        self.plot_2d_right = QtWidgets.QCheckBox("Right Axis")
        
        self.plot_2d_bottom.setToolTip("Show/hide the bottom axis of the 2D plot")
        self.plot_2d_top.setToolTip("Show/hide the top axis of the 2D plot")
        self.plot_2d_left.setToolTip("Show/hide the left axis of the 2D plot")
        self.plot_2d_right.setToolTip("Show/hide the right axis of the 2D plot")
        
        for checkbox, label in (
            (self.plot_2d_bottom, "Bottom axis of the 2D plot"),
            (self.plot_2d_top, "Top axis of the 2D plot"),
            (self.plot_2d_left, "Left axis of the 2D plot"),
            (self.plot_2d_right, "Right axis of the 2D plot"),
        ):
            self._set_accessibility(
                checkbox,
                name=label,
                description=f"Toggle visibility for the {label.lower()}."
            )
            plot_2d_layout.addWidget(checkbox)
        plot_2d_group.setLayout(plot_2d_layout)
        scroll_layout.addWidget(plot_2d_group)
        
        # Overlay Plot group
        overlay_plot_group = QtWidgets.QGroupBox("Overlay Plot")
        self._set_accessibility(
            overlay_plot_group,
            name="Overlay plot axes",
            description="Options for showing or hiding axes on the overlay plot."
        )
        overlay_plot_layout = QtWidgets.QVBoxLayout()
        self.overlay_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.overlay_plot_top = QtWidgets.QCheckBox("Top Axis")
        self.overlay_plot_left = QtWidgets.QCheckBox("Left Axis")
        self.overlay_plot_right = QtWidgets.QCheckBox("Right Axis")
        
        self.overlay_plot_bottom.setToolTip("Show/hide the bottom axis of the overlay plot")
        self.overlay_plot_top.setToolTip("Show/hide the top axis of the overlay plot")
        self.overlay_plot_left.setToolTip("Show/hide the left axis of the overlay plot")
        self.overlay_plot_right.setToolTip("Show/hide the right axis of the overlay plot")
        
        for checkbox, label in (
            (self.overlay_plot_bottom, "Bottom axis of the overlay plot"),
            (self.overlay_plot_top, "Top axis of the overlay plot"),
            (self.overlay_plot_left, "Left axis of the overlay plot"),
            (self.overlay_plot_right, "Right axis of the overlay plot"),
        ):
            self._set_accessibility(
                checkbox,
                name=label,
                description=f"Toggle visibility for the {label.lower()}."
            )
            overlay_plot_layout.addWidget(checkbox)
        overlay_plot_group.setLayout(overlay_plot_layout)
        scroll_layout.addWidget(overlay_plot_group)
        
        # Axis Label Settings group
        label_settings_group = QtWidgets.QGroupBox("Axis Label Settings")
        self._set_accessibility(
            label_settings_group,
            name="Axis label settings",
            description="Controls for showing labels and configuring fonts."
        )
        label_settings_layout = QtWidgets.QVBoxLayout()
        
        # Global enable/disable checkbox
        self.enable_all_labels = QtWidgets.QCheckBox("Enable All Labels")
        self.enable_all_labels.setToolTip("Enable or disable all axis labels")
        self._set_accessibility(
            self.enable_all_labels,
            name="Enable all labels",
            description="Toggle to show or hide every axis label at once."
        )
        label_settings_layout.addWidget(self.enable_all_labels)
        
        # Y Plot Labels group
        y_plot_labels_group = QtWidgets.QGroupBox("Y Plot Labels")
        y_plot_labels_layout = QtWidgets.QVBoxLayout()
        self.y_plot_label_top = QtWidgets.QCheckBox("Top Axis Label")
        self.y_plot_label_right = QtWidgets.QCheckBox("Right Axis Label")
        
        self.y_plot_label_top.setToolTip("Show/hide the label for the top axis of the Y plot")
        self.y_plot_label_right.setToolTip("Show/hide the label for the right axis of the Y plot")
        
        y_plot_labels_layout.addWidget(self.y_plot_label_top)
        y_plot_labels_layout.addWidget(self.y_plot_label_right)
        y_plot_labels_group.setLayout(y_plot_labels_layout)
        label_settings_layout.addWidget(y_plot_labels_group)
        
        # X Plot Labels group
        x_plot_labels_group = QtWidgets.QGroupBox("X Plot Labels")
        x_plot_labels_layout = QtWidgets.QVBoxLayout()
        self.x_plot_label_top = QtWidgets.QCheckBox("Top Axis Label")
        
        self.x_plot_label_top.setToolTip("Show/hide the label for the top axis of the X plot")
        
        x_plot_labels_layout.addWidget(self.x_plot_label_top)
        x_plot_labels_group.setLayout(x_plot_labels_layout)
        label_settings_layout.addWidget(x_plot_labels_group)
        
        # Z Plot Labels group
        z_plot_labels_group = QtWidgets.QGroupBox("Z Plot Labels")
        z_plot_labels_layout = QtWidgets.QVBoxLayout()
        self.z_plot_label_bottom = QtWidgets.QCheckBox("Bottom Axis Label")
        self.z_plot_label_left = QtWidgets.QCheckBox("Left Axis Label")
        
        self.z_plot_label_bottom.setToolTip("Show/hide the label for the bottom axis of the Z plot")
        self.z_plot_label_left.setToolTip("Show/hide the label for the left axis of the Z plot")
        
        z_plot_labels_layout.addWidget(self.z_plot_label_bottom)
        z_plot_labels_layout.addWidget(self.z_plot_label_left)
        z_plot_labels_group.setLayout(z_plot_labels_layout)
        label_settings_layout.addWidget(z_plot_labels_group)
        
        # Finish label settings group setup
        label_settings_group.setLayout(label_settings_layout)
        scroll_layout.addWidget(label_settings_group)
        
        # Font Settings group
        font_group = QtWidgets.QGroupBox("Font Settings")
        font_layout = QtWidgets.QFormLayout()
        # Tick size
        self.font_tick_size = QtWidgets.QSpinBox()
        self.font_tick_size.setRange(6, 48)
        self.font_tick_size.setValue(8)
        # Title size
        self.font_title_size = QtWidgets.QSpinBox()
        self.font_title_size.setRange(6, 64)
        self.font_title_size.setValue(10)
        # Title weight (bold)
        self.font_title_bold = QtWidgets.QCheckBox("Bold titles")
        self.font_title_bold.setChecked(True)
        # Title color picker
        self.font_title_color_btn = QtWidgets.QPushButton("Pick Title Color")
        self._font_title_color = "#000000"
        def _update_color_btn():
            try:
                self.font_title_color_btn.setStyleSheet(f"background-color: {self._font_title_color}; color: white")
                self.font_title_color_btn.setText(self._font_title_color)
            except Exception:
                pass
        def _pick_color():
            col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._font_title_color), self, "Select Title Color")
            if col.isValid():
                self._font_title_color = col.name()
                _update_color_btn()
        self.font_title_color_btn.clicked.connect(_pick_color)
        _update_color_btn()
        # Layout rows
        font_layout.addRow("Tick size (pt)", self.font_tick_size)
        font_layout.addRow("Title size (pt)", self.font_title_size)
        font_layout.addRow(self.font_title_bold)
        font_layout.addRow("Title color", self.font_title_color_btn)
        font_group.setLayout(font_layout)
        scroll_layout.addWidget(font_group)
        
        # Finish scroll area setup
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Connect signals
        self.z_plot_enable.stateChanged.connect(self.on_z_plot_enable_changed)
        self.enable_all_labels.stateChanged.connect(self.on_enable_all_labels_changed)
        
        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Apply
        )
        # Add Save button
        save_button = button_box.addButton("Save", QtWidgets.QDialogButtonBox.ActionRole)
        save_button.setToolTip("Save axis label settings to the settings folder")
        save_button.clicked.connect(self.save_axis_label_settings)
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(button_box)
        
    def load_current_state(self):
        """
        Load the current state of axis visibility and axis label settings from the parent.
        
        This method loads both the axis visibility settings and the axis label settings
        from the parent and updates the checkboxes accordingly.
        """
        if not self.parent:
            return

        try:
            # X Plot
            self.x_plot_bottom.setChecked(self.parent.g_xplot.axisEnabled("bottom"))
            self.x_plot_top.setChecked(self.parent.g_xplot.axisEnabled("top"))
            self.x_plot_left.setChecked(self.parent.g_xplot.axisEnabled("left"))
            self.x_plot_right.setChecked(self.parent.g_xplot.axisEnabled("right"))

            # Y Plot
            self.y_plot_bottom.setChecked(self.parent.g_yplot.axisEnabled("bottom"))
            self.y_plot_top.setChecked(self.parent.g_yplot.axisEnabled("top"))
            self.y_plot_left.setChecked(self.parent.g_yplot.axisEnabled("left"))
            self.y_plot_right.setChecked(self.parent.g_yplot.axisEnabled("right"))

            # Z Plot
            if hasattr(self.parent, 'groupBox_3'):
                self.z_plot_enable.setChecked(self.parent.groupBox_3.isChecked())

            if hasattr(self.parent, 'g_zplot'):
                self.z_plot_bottom.setChecked(self.parent.g_zplot.axisEnabled("bottom"))
                self.z_plot_left.setChecked(self.parent.g_zplot.axisEnabled("left"))

                # Enable/disable Z plot axis checkboxes based on Z plot visibility
                self.z_plot_bottom.setEnabled(self.z_plot_enable.isChecked())
                self.z_plot_left.setEnabled(self.z_plot_enable.isChecked())

            # 2D Plot
            self.plot_2d_bottom.setChecked(self.parent.g_2dplot.axis_enabled('xBottom'))
            self.plot_2d_top.setChecked(self.parent.g_2dplot.axis_enabled('xTop'))
            self.plot_2d_left.setChecked(self.parent.g_2dplot.axis_enabled('yLeft'))
            self.plot_2d_right.setChecked(self.parent.g_2dplot.axis_enabled('yRight'))

            # Overlay Plot
            if hasattr(self.parent, 'overlay_plot'):
                self.overlay_plot_bottom.setChecked(self.parent.overlay_plot.axisEnabled("bottom"))
                self.overlay_plot_top.setChecked(self.parent.overlay_plot.axisEnabled("top"))
                self.overlay_plot_left.setChecked(self.parent.overlay_plot.axisEnabled("left"))
                self.overlay_plot_right.setChecked(self.parent.overlay_plot.axisEnabled("right"))

            # Load axis label settings
            if hasattr(self.parent, 'axis_label_settings'):
                settings = self.parent.axis_label_settings
                enable_all_labels = settings.get('enable_all_labels', True)
                self.enable_all_labels.setChecked(enable_all_labels)

                axis_labels = settings.get('axis_labels', {})

                y_plot_settings = axis_labels.get('y_plot', {})
                self.y_plot_label_top.setChecked(y_plot_settings.get('top', True))
                self.y_plot_label_right.setChecked(y_plot_settings.get('right', True))

                x_plot_settings = axis_labels.get('x_plot', {})
                self.x_plot_label_top.setChecked(x_plot_settings.get('top', True))

                z_plot_settings = axis_labels.get('z_plot', {})
                self.z_plot_label_bottom.setChecked(z_plot_settings.get('bottom', True))
                self.z_plot_label_left.setChecked(z_plot_settings.get('left', True))

                self.on_enable_all_labels_changed(enable_all_labels)

                try:
                    fonts = settings.get('fonts', {})
                    if hasattr(self.parent, 'font_settings') and self.parent.font_settings:
                        fonts = {**fonts, **self.parent.font_settings}
                    self.font_tick_size.setValue(int(fonts.get('tick_size_pt', 8)))
                    self.font_title_size.setValue(int(fonts.get('title_size_pt', 10)))
                    title_weight = int(fonts.get('title_weight', 700))
                    self.font_title_bold.setChecked(title_weight >= 600)
                    self._font_title_color = str(fonts.get('color', '#000000'))
                    self.font_title_color_btn.setStyleSheet(
                        f"background-color: {self._font_title_color}; color: white"
                    )
                    self.font_title_color_btn.setText(self._font_title_color)
                except Exception:
                    pass
            else:
                self.enable_all_labels.setChecked(True)
                self.y_plot_label_top.setChecked(True)
                self.y_plot_label_right.setChecked(True)
                self.x_plot_label_top.setChecked(True)
                self.z_plot_label_bottom.setChecked(True)
                self.z_plot_label_left.setChecked(True)
                self.on_enable_all_labels_changed(True)
                try:
                    self.font_tick_size.setValue(8)
                    self.font_title_size.setValue(10)
                    self.font_title_bold.setChecked(True)
                    self._font_title_color = '#000000'
                    self.font_title_color_btn.setStyleSheet("background-color: #000000; color: white")
                    self.font_title_color_btn.setText('#000000')
                except Exception:
                    pass
        except Exception as exc:
            logging.exception("Failed to load axis control state")
            self.error_presenter.error(
                "Axis Control",
                f"Unable to load current axis settings.\nDetails: {exc}"
            )
    
    def on_z_plot_enable_changed(self, state):
        """
        Handle changes to the Z plot enable checkbox.
        
        Args:
            state: The new state of the checkbox
        """
        # Enable/disable Z plot axis checkboxes based on Z plot visibility
        self.z_plot_bottom.setEnabled(bool(state))
        self.z_plot_left.setEnabled(bool(state))
        
    def on_enable_all_labels_changed(self, state):
        """
        Handle changes to the Enable All Labels checkbox.
        
        Args:
            state: The new state of the checkbox
        """
        # Enable/disable individual label checkboxes based on the global setting
        # If the global setting is checked, individual settings can be overridden
        # If the global setting is unchecked, individual settings determine visibility
        enabled = bool(state)
        
        # Update the enabled state of individual checkboxes
        # We don't change their checked state, just whether they're enabled
        self.y_plot_label_top.setEnabled(not enabled)
        self.y_plot_label_right.setEnabled(not enabled)
        self.x_plot_label_top.setEnabled(not enabled)
        self.z_plot_label_bottom.setEnabled(not enabled)
        self.z_plot_label_left.setEnabled(not enabled)
    
    def apply_changes(self):
        """
        Apply the changes to the parent's plots and axis label settings.
        
        This method applies both the axis visibility changes and the axis label
        settings changes to the parent.
        """
        if not self.parent:
            return

        try:
            self.parent.g_xplot.enableAxis("bottom", self.x_plot_bottom.isChecked())
            self.parent.g_xplot.enableAxis("top", self.x_plot_top.isChecked())
            self.parent.g_xplot.enableAxis("left", self.x_plot_left.isChecked())
            self.parent.g_xplot.enableAxis("right", self.x_plot_right.isChecked())

            self.parent.g_yplot.enableAxis("bottom", self.y_plot_bottom.isChecked())
            self.parent.g_yplot.enableAxis("top", self.y_plot_top.isChecked())
            self.parent.g_yplot.enableAxis("left", self.y_plot_left.isChecked())
            self.parent.g_yplot.enableAxis("right", self.y_plot_right.isChecked())

            if hasattr(self.parent, 'groupBox_3'):
                self.parent.groupBox_3.setChecked(self.z_plot_enable.isChecked())

            if hasattr(self.parent, 'g_zplot'):
                self.parent.g_zplot.enableAxis("bottom", self.z_plot_bottom.isChecked())
                self.parent.g_zplot.enableAxis("left", self.z_plot_left.isChecked())

            self.parent.g_2dplot.enable_axis('xBottom', self.plot_2d_bottom.isChecked())
            self.parent.g_2dplot.enable_axis('xTop', self.plot_2d_top.isChecked())
            self.parent.g_2dplot.enable_axis('yLeft', self.plot_2d_left.isChecked())
            self.parent.g_2dplot.enable_axis('yRight', self.plot_2d_right.isChecked())

            if hasattr(self.parent, 'overlay_plot'):
                self.parent.overlay_plot.enableAxis("bottom", self.overlay_plot_bottom.isChecked())
                self.parent.overlay_plot.enableAxis("top", self.overlay_plot_top.isChecked())
                self.parent.overlay_plot.enableAxis("left", self.overlay_plot_left.isChecked())
                self.parent.overlay_plot.enableAxis("right", self.overlay_plot_right.isChecked())

            if hasattr(self.parent, 'axis_label_settings'):
                settings = {
                    "enable_all_labels": self.enable_all_labels.isChecked(),
                    "axis_labels": {
                        "y_plot": {
                            "top": self.y_plot_label_top.isChecked(),
                            "right": self.y_plot_label_right.isChecked()
                        },
                        "x_plot": {
                            "top": self.x_plot_label_top.isChecked()
                        },
                        "z_plot": {
                            "bottom": self.z_plot_label_bottom.isChecked(),
                            "left": self.z_plot_label_left.isChecked()
                        }
                    },
                    "fonts": {
                        "tick_size_pt": int(self.font_tick_size.value()),
                        "title_size_pt": int(self.font_title_size.value()),
                        "title_weight": 700 if self.font_title_bold.isChecked() else 400,
                        "color": str(self._font_title_color)
                    }
                }
                self.parent.axis_label_settings.update(settings)
                try:
                    if hasattr(self.parent, 'font_settings'):
                        self.parent.font_settings.update(settings.get('fonts', {}))
                    else:
                        self.parent.font_settings = settings.get('fonts', {})
                    if hasattr(self.parent, 'apply_fonts'):
                        self.parent.apply_fonts()
                except Exception:
                    pass
                if hasattr(self.parent, 'update_parameter_names'):
                    self.parent.update_parameter_names()

            self.parent.g_xplot.replot()
            self.parent.g_yplot.replot()
            if hasattr(self.parent, 'g_zplot'):
                self.parent.g_zplot.replot()
            self.parent.g_2dplot.replot()
            if hasattr(self.parent, 'overlay_plot'):
                self.parent.overlay_plot.replot()

            logging.log(0, "Applied axis visibility and label settings changes")
        except Exception as exc:
            logging.exception("Failed to apply axis settings")
            self.error_presenter.error(
                "Axis Control",
                f"Unable to apply axis or label changes.\nDetails: {exc}"
            )
    
    def save_axis_label_settings(self):
        """
        Save the current axis label settings.
        
        This method gathers the current settings from the checkboxes and saves them
        to the user settings directory (~/.ndxplorer/).
        """
        if not self.parent:
            logging.warning("Cannot save axis label settings: parent is None")
            self.error_presenter.warn(
                "Axis Control",
                "Cannot save axis label settings because the plot parent is unavailable."
            )
            return

        # Gather current settings from checkboxes
        settings = {
            "enable_all_labels": self.enable_all_labels.isChecked(),
            "axis_labels": {
                "y_plot": {
                    "top": self.y_plot_label_top.isChecked(),
                    "right": self.y_plot_label_right.isChecked()
                },
                "x_plot": {
                    "top": self.x_plot_label_top.isChecked()
                },
                "z_plot": {
                    "bottom": self.z_plot_label_bottom.isChecked(),
                    "left": self.z_plot_label_left.isChecked()
                }
            },
            "fonts": {
                "tick_size_pt": int(self.font_tick_size.value()),
                "title_size_pt": int(self.font_title_size.value()),
                "title_weight": 700 if self.font_title_bold.isChecked() else 400,
                "color": str(self._font_title_color)
            }
        }

        try:
            settings_dir = get_settings_path()

            if hasattr(self.parent, 'settings') and "axis_labels" in self.parent.settings:
                fn_axis_labels = settings_dir / self.parent.settings["axis_labels"]
            else:
                fn_axis_labels = settings_dir / "axis_labels.yaml"

            settings_dir.mkdir(parents=True, exist_ok=True)

            with open(str(fn_axis_labels), "w") as fp:
                fp.write("# Configuration for axis labels and fonts in ndxplorer\n")
                fp.write("# axis_labels: visibility of labels; fonts: family and sizes\n\n")
                yaml.dump(settings, fp, default_flow_style=False, sort_keys=False)

            if hasattr(self.parent, 'axis_label_settings'):
                self.parent.axis_label_settings.update(settings)
            try:
                if hasattr(self.parent, 'font_settings'):
                    self.parent.font_settings.update(settings.get('fonts', {}))
                else:
                    self.parent.font_settings = settings.get('fonts', {})
                if hasattr(self.parent, 'apply_fonts'):
                    self.parent.apply_fonts()
            except Exception:
                pass

            try:
                if hasattr(self.parent, 'update_parameter_names'):
                    self.parent.update_parameter_names()
            except Exception:
                pass

            for plot_attr in ("g_xplot", "g_yplot", "g_zplot", "g_2dplot", "overlay_plot"):
                try:
                    plot = getattr(self.parent, plot_attr, None)
                    if plot:
                        plot.replot()
                except Exception:
                    pass

            logging.log(0, "Axis label and font settings saved and applied successfully")

            self.error_presenter.info(
                "Axis Control",
                "Axis label and font settings saved and applied successfully."
            )
            
        except Exception as exc:
            logging.exception("Error saving axis label settings")
            self.error_presenter.error(
                "Axis Control",
                f"Error saving axis label settings.\nDetails: {exc}"
            )
    
    def accept(self):
        """Apply changes and close the dialog."""
        self.apply_changes()
        super(AxisControlDialog, self).accept()