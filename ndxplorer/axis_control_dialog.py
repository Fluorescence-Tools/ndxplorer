"""
Axis Control Dialog for ndxplorer.

This module provides a dialog for enabling and disabling axes in the ndxplorer plots.
It also allows controlling the visibility of axis labels and saving these settings to a file.
"""

import os
import yaml
import pathlib
from qtpy import QtCore, QtGui, QtWidgets

from .logging_config import logging

from qwt.plot import QwtPlot

# Import settings functions
from .settings import get_settings_path


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
        self.setMinimumWidth(300)
        self.setup_ui()
        self.load_current_state()
        
    def setup_ui(self):
        """Set up the user interface for the axis control dialog."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create a scroll area to handle many checkboxes
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        
        # X Plot group
        x_plot_group = QtWidgets.QGroupBox("X Plot")
        x_plot_layout = QtWidgets.QVBoxLayout()
        self.x_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.x_plot_top = QtWidgets.QCheckBox("Top Axis")
        self.x_plot_left = QtWidgets.QCheckBox("Left Axis")
        self.x_plot_right = QtWidgets.QCheckBox("Right Axis")
        
        self.x_plot_bottom.setToolTip("Show/hide the bottom axis of the X plot")
        self.x_plot_top.setToolTip("Show/hide the top axis of the X plot")
        self.x_plot_left.setToolTip("Show/hide the left axis of the X plot")
        self.x_plot_right.setToolTip("Show/hide the right axis of the X plot")
        
        x_plot_layout.addWidget(self.x_plot_bottom)
        x_plot_layout.addWidget(self.x_plot_top)
        x_plot_layout.addWidget(self.x_plot_left)
        x_plot_layout.addWidget(self.x_plot_right)
        x_plot_group.setLayout(x_plot_layout)
        scroll_layout.addWidget(x_plot_group)
        
        # Y Plot group
        y_plot_group = QtWidgets.QGroupBox("Y Plot")
        y_plot_layout = QtWidgets.QVBoxLayout()
        self.y_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.y_plot_top = QtWidgets.QCheckBox("Top Axis")
        self.y_plot_left = QtWidgets.QCheckBox("Left Axis")
        self.y_plot_right = QtWidgets.QCheckBox("Right Axis")
        
        self.y_plot_bottom.setToolTip("Show/hide the bottom axis of the Y plot")
        self.y_plot_top.setToolTip("Show/hide the top axis of the Y plot")
        self.y_plot_left.setToolTip("Show/hide the left axis of the Y plot")
        self.y_plot_right.setToolTip("Show/hide the right axis of the Y plot")
        
        y_plot_layout.addWidget(self.y_plot_bottom)
        y_plot_layout.addWidget(self.y_plot_top)
        y_plot_layout.addWidget(self.y_plot_left)
        y_plot_layout.addWidget(self.y_plot_right)
        y_plot_group.setLayout(y_plot_layout)
        scroll_layout.addWidget(y_plot_group)
        
        # Z Plot group
        z_plot_group = QtWidgets.QGroupBox("Z Plot")
        z_plot_layout = QtWidgets.QVBoxLayout()
        self.z_plot_enable = QtWidgets.QCheckBox("Enable Z Plot")
        self.z_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.z_plot_left = QtWidgets.QCheckBox("Left Axis")
        
        self.z_plot_enable.setToolTip("Enable/disable the Z plot")
        self.z_plot_bottom.setToolTip("Show/hide the bottom axis of the Z plot")
        self.z_plot_left.setToolTip("Show/hide the left axis of the Z plot")
        
        z_plot_layout.addWidget(self.z_plot_enable)
        z_plot_layout.addWidget(self.z_plot_bottom)
        z_plot_layout.addWidget(self.z_plot_left)
        z_plot_group.setLayout(z_plot_layout)
        scroll_layout.addWidget(z_plot_group)
        
        # 2D Plot group
        plot_2d_group = QtWidgets.QGroupBox("2D Plot")
        plot_2d_layout = QtWidgets.QVBoxLayout()
        self.plot_2d_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.plot_2d_top = QtWidgets.QCheckBox("Top Axis")
        self.plot_2d_left = QtWidgets.QCheckBox("Left Axis")
        self.plot_2d_right = QtWidgets.QCheckBox("Right Axis")
        
        self.plot_2d_bottom.setToolTip("Show/hide the bottom axis of the 2D plot")
        self.plot_2d_top.setToolTip("Show/hide the top axis of the 2D plot")
        self.plot_2d_left.setToolTip("Show/hide the left axis of the 2D plot")
        self.plot_2d_right.setToolTip("Show/hide the right axis of the 2D plot")
        
        plot_2d_layout.addWidget(self.plot_2d_bottom)
        plot_2d_layout.addWidget(self.plot_2d_top)
        plot_2d_layout.addWidget(self.plot_2d_left)
        plot_2d_layout.addWidget(self.plot_2d_right)
        plot_2d_group.setLayout(plot_2d_layout)
        scroll_layout.addWidget(plot_2d_group)
        
        # Overlay Plot group
        overlay_plot_group = QtWidgets.QGroupBox("Overlay Plot")
        overlay_plot_layout = QtWidgets.QVBoxLayout()
        self.overlay_plot_bottom = QtWidgets.QCheckBox("Bottom Axis")
        self.overlay_plot_top = QtWidgets.QCheckBox("Top Axis")
        self.overlay_plot_left = QtWidgets.QCheckBox("Left Axis")
        self.overlay_plot_right = QtWidgets.QCheckBox("Right Axis")
        
        self.overlay_plot_bottom.setToolTip("Show/hide the bottom axis of the overlay plot")
        self.overlay_plot_top.setToolTip("Show/hide the top axis of the overlay plot")
        self.overlay_plot_left.setToolTip("Show/hide the left axis of the overlay plot")
        self.overlay_plot_right.setToolTip("Show/hide the right axis of the overlay plot")
        
        overlay_plot_layout.addWidget(self.overlay_plot_bottom)
        overlay_plot_layout.addWidget(self.overlay_plot_top)
        overlay_plot_layout.addWidget(self.overlay_plot_left)
        overlay_plot_layout.addWidget(self.overlay_plot_right)
        overlay_plot_group.setLayout(overlay_plot_layout)
        scroll_layout.addWidget(overlay_plot_group)
        
        # Axis Label Settings group
        label_settings_group = QtWidgets.QGroupBox("Axis Label Settings")
        label_settings_layout = QtWidgets.QVBoxLayout()
        
        # Global enable/disable checkbox
        self.enable_all_labels = QtWidgets.QCheckBox("Enable All Labels")
        self.enable_all_labels.setToolTip("Enable or disable all axis labels")
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
            
        # X Plot
        self.x_plot_bottom.setChecked(self.parent.g_xplot.axisEnabled(QwtPlot.xBottom))
        self.x_plot_top.setChecked(self.parent.g_xplot.axisEnabled(QwtPlot.xTop))
        self.x_plot_left.setChecked(self.parent.g_xplot.axisEnabled(QwtPlot.yLeft))
        self.x_plot_right.setChecked(self.parent.g_xplot.axisEnabled(QwtPlot.yRight))
        
        # Y Plot
        self.y_plot_bottom.setChecked(self.parent.g_yplot.axisEnabled(QwtPlot.xBottom))
        self.y_plot_top.setChecked(self.parent.g_yplot.axisEnabled(QwtPlot.xTop))
        self.y_plot_left.setChecked(self.parent.g_yplot.axisEnabled(QwtPlot.yLeft))
        self.y_plot_right.setChecked(self.parent.g_yplot.axisEnabled(QwtPlot.yRight))
        
        # Z Plot
        if hasattr(self.parent, 'checkBoxEnableZ'):
            self.z_plot_enable.setChecked(self.parent.checkBoxEnableZ.isChecked())
        
        if hasattr(self.parent, 'g_zplot'):
            self.z_plot_bottom.setChecked(self.parent.g_zplot.axisEnabled(QwtPlot.xBottom))
            self.z_plot_left.setChecked(self.parent.g_zplot.axisEnabled(QwtPlot.yLeft))
            
            # Enable/disable Z plot axis checkboxes based on Z plot visibility
            self.z_plot_bottom.setEnabled(self.z_plot_enable.isChecked())
            self.z_plot_left.setEnabled(self.z_plot_enable.isChecked())
        
        # 2D Plot
        self.plot_2d_bottom.setChecked(self.parent.g_2dplot.axisEnabled(QwtPlot.xBottom))
        self.plot_2d_top.setChecked(self.parent.g_2dplot.axisEnabled(QwtPlot.xTop))
        self.plot_2d_left.setChecked(self.parent.g_2dplot.axisEnabled(QwtPlot.yLeft))
        self.plot_2d_right.setChecked(self.parent.g_2dplot.axisEnabled(QwtPlot.yRight))
        
        # Overlay Plot
        if hasattr(self.parent, 'overlay_plot'):
            self.overlay_plot_bottom.setChecked(self.parent.overlay_plot.axisEnabled(QwtPlot.xBottom))
            self.overlay_plot_top.setChecked(self.parent.overlay_plot.axisEnabled(QwtPlot.xTop))
            self.overlay_plot_left.setChecked(self.parent.overlay_plot.axisEnabled(QwtPlot.yLeft))
            self.overlay_plot_right.setChecked(self.parent.overlay_plot.axisEnabled(QwtPlot.yRight))
            
        # Load axis label settings
        if hasattr(self.parent, 'axis_label_settings'):
            # Get the settings
            settings = self.parent.axis_label_settings
            
            # Global enable/disable setting
            enable_all_labels = settings.get('enable_all_labels', True)
            self.enable_all_labels.setChecked(enable_all_labels)
            
            # Individual axis label settings
            axis_labels = settings.get('axis_labels', {})
            
            # Y Plot Labels
            y_plot_settings = axis_labels.get('y_plot', {})
            self.y_plot_label_top.setChecked(y_plot_settings.get('top', True))
            self.y_plot_label_right.setChecked(y_plot_settings.get('right', True))
            
            # X Plot Labels
            x_plot_settings = axis_labels.get('x_plot', {})
            self.x_plot_label_top.setChecked(x_plot_settings.get('top', True))
            
            # Z Plot Labels
            z_plot_settings = axis_labels.get('z_plot', {})
            self.z_plot_label_bottom.setChecked(z_plot_settings.get('bottom', True))
            self.z_plot_label_left.setChecked(z_plot_settings.get('left', True))
            
            # Update enabled state of individual checkboxes based on global setting
            self.on_enable_all_labels_changed(enable_all_labels)
        else:
            # No axis label settings available, use defaults
            self.enable_all_labels.setChecked(True)
            self.y_plot_label_top.setChecked(True)
            self.y_plot_label_right.setChecked(True)
            self.x_plot_label_top.setChecked(True)
            self.z_plot_label_bottom.setChecked(True)
            self.z_plot_label_left.setChecked(True)
            
            # Update enabled state of individual checkboxes
            self.on_enable_all_labels_changed(True)
    
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
            
        # X Plot
        self.parent.g_xplot.enableAxis(QwtPlot.xBottom, self.x_plot_bottom.isChecked())
        self.parent.g_xplot.enableAxis(QwtPlot.xTop, self.x_plot_top.isChecked())
        self.parent.g_xplot.enableAxis(QwtPlot.yLeft, self.x_plot_left.isChecked())
        self.parent.g_xplot.enableAxis(QwtPlot.yRight, self.x_plot_right.isChecked())
        
        # Y Plot
        self.parent.g_yplot.enableAxis(QwtPlot.xBottom, self.y_plot_bottom.isChecked())
        self.parent.g_yplot.enableAxis(QwtPlot.xTop, self.y_plot_top.isChecked())
        self.parent.g_yplot.enableAxis(QwtPlot.yLeft, self.y_plot_left.isChecked())
        self.parent.g_yplot.enableAxis(QwtPlot.yRight, self.y_plot_right.isChecked())
        
        # Z Plot
        if hasattr(self.parent, 'checkBoxEnableZ'):
            self.parent.checkBoxEnableZ.setChecked(self.z_plot_enable.isChecked())
        
        if hasattr(self.parent, 'g_zplot'):
            self.parent.g_zplot.enableAxis(QwtPlot.xBottom, self.z_plot_bottom.isChecked())
            self.parent.g_zplot.enableAxis(QwtPlot.yLeft, self.z_plot_left.isChecked())
        
        # 2D Plot
        self.parent.g_2dplot.enableAxis(QwtPlot.xBottom, self.plot_2d_bottom.isChecked())
        self.parent.g_2dplot.enableAxis(QwtPlot.xTop, self.plot_2d_top.isChecked())
        self.parent.g_2dplot.enableAxis(QwtPlot.yLeft, self.plot_2d_left.isChecked())
        self.parent.g_2dplot.enableAxis(QwtPlot.yRight, self.plot_2d_right.isChecked())
        
        # Overlay Plot
        if hasattr(self.parent, 'overlay_plot'):
            self.parent.overlay_plot.enableAxis(QwtPlot.xBottom, self.overlay_plot_bottom.isChecked())
            self.parent.overlay_plot.enableAxis(QwtPlot.xTop, self.overlay_plot_top.isChecked())
            self.parent.overlay_plot.enableAxis(QwtPlot.yLeft, self.overlay_plot_left.isChecked())
            self.parent.overlay_plot.enableAxis(QwtPlot.yRight, self.overlay_plot_right.isChecked())
        
        # Apply axis label settings
        if hasattr(self.parent, 'axis_label_settings'):
            # Gather current settings from checkboxes
            settings = {
                # Global setting to enable/disable all axis labels
                "enable_all_labels": self.enable_all_labels.isChecked(),
                # Individual settings for each plot type and axis
                "axis_labels": {
                    # Y-plot axis labels (top and right axes)
                    "y_plot": {
                        "top": self.y_plot_label_top.isChecked(),
                        "right": self.y_plot_label_right.isChecked()
                    },
                    # X-plot axis labels (top axis)
                    "x_plot": {
                        "top": self.x_plot_label_top.isChecked()
                    },
                    # Z-plot axis labels (bottom and left axes)
                    "z_plot": {
                        "bottom": self.z_plot_label_bottom.isChecked(),
                        "left": self.z_plot_label_left.isChecked()
                    }
                }
            }
            
            # Update parent's axis_label_settings
            self.parent.axis_label_settings.update(settings)
            
            # Apply the changes by calling update_parameter_names
            if hasattr(self.parent, 'update_parameter_names'):
                self.parent.update_parameter_names()
        
        # Replot all plots to update the display
        self.parent.g_xplot.replot()
        self.parent.g_yplot.replot()
        if hasattr(self.parent, 'g_zplot'):
            self.parent.g_zplot.replot()
        self.parent.g_2dplot.replot()
        if hasattr(self.parent, 'overlay_plot'):
            self.parent.overlay_plot.replot()
        
        logging.log(0, "Applied axis visibility and label settings changes")
    
    def save_axis_label_settings(self):
        """
        Save the current axis label settings.
        
        This method gathers the current settings from the checkboxes and saves them
        to the appropriate settings directory (user settings folder if chisurf is installed,
        or default folder otherwise).
        """
        if not self.parent:
            logging.warning("Cannot save axis label settings: parent is None")
            return
            
        # Gather current settings from checkboxes
        settings = {
            # Global setting to enable/disable all axis labels
            "enable_all_labels": self.enable_all_labels.isChecked(),
            # Individual settings for each plot type and axis
            "axis_labels": {
                # Y-plot axis labels (top and right axes)
                "y_plot": {
                    "top": self.y_plot_label_top.isChecked(),
                    "right": self.y_plot_label_right.isChecked()
                },
                # X-plot axis labels (top axis)
                "x_plot": {
                    "top": self.x_plot_label_top.isChecked()
                },
                # Z-plot axis labels (bottom and left axes)
                "z_plot": {
                    "bottom": self.z_plot_label_bottom.isChecked(),
                    "left": self.z_plot_label_left.isChecked()
                }
            }
        }
        
        try:
            # Get the settings directory using get_settings_path
            settings_dir = get_settings_path()
            
            if hasattr(self.parent, 'settings') and "axis_labels" in self.parent.settings:
                # Use the filename from parent's settings
                fn_axis_labels = settings_dir / self.parent.settings["axis_labels"]
            else:
                # Fall back to default filename
                fn_axis_labels = settings_dir / "axis_labels.yaml"
            
            # The directory should already exist (created by get_settings_path)
            # but we'll ensure it just to be safe
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Save settings to file
            with open(str(fn_axis_labels), "w") as fp:
                # Add a header comment
                fp.write("# Configuration for axis labels in ndxplorer\n")
                fp.write("# This file controls whether axis labels are displayed or hidden\n\n")
                
                # Dump the settings as YAML
                yaml.dump(settings, fp, default_flow_style=False, sort_keys=False)
            
            # Update parent's axis_label_settings if it exists
            if hasattr(self.parent, 'axis_label_settings'):
                self.parent.axis_label_settings.update(settings)
            
            logging.log(0, "Axis label settings saved successfully")
            
            # Show a success message
            QtWidgets.QMessageBox.information(
                self,
                "Settings Saved",
                f"Axis label settings saved successfully"
            )
            
        except Exception as e:
            logging.error(f"Error saving axis label settings: {e}")
            
            # Show an error message
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error saving axis label settings: {e}"
            )
    
    def accept(self):
        """Apply changes and close the dialog."""
        self.apply_changes()
        super(AxisControlDialog, self).accept()