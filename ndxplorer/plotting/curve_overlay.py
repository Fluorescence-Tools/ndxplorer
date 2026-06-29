from ..logging_config import logging
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from pathlib import Path
import numpy as np
import re
import os
import csv
import yaml
import inspect
import textwrap

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt
from ..widgets import ParameterSlider


class CurveWidget(QtWidgets.QGroupBox):
    """
    A widget for a single curve equation with parameters and visibility control.
    """
    visibilityChanged = QtCore.Signal(bool)
    equationChanged = QtCore.Signal()
    deleteRequested = QtCore.Signal()
    colorChanged = QtCore.Signal()

    def __init__(self, name="Curve", equation_or_function="x", parent=None, is_function=False):
        super().__init__(name, parent)
        self.setCheckable(True)
        self.setChecked(True)

        self.parameters = {}  # Dictionary to store parameter widgets
        self.curve_color = "#ff0000"  # Default color is red
        self.use_sliders = True  # Whether to use sliders for parameters (default: True)
        self.is_function = is_function  # Whether the input is a function (True) or an equation (False)
        self.function = None  # Store the compiled function if is_function is True
        self.curve_evaluator = CurveEvaluator()  # Create a CurveEvaluator instance

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(0)  # Reduce spacing between elements
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins

        # Equation input
        eq_layout = QtWidgets.QHBoxLayout()
        eq_layout.setSpacing(0)  # Reduce spacing
        if self.is_function:
            eq_layout.addWidget(QtWidgets.QLabel("Function: "))
        else:
            eq_layout.addWidget(QtWidgets.QLabel("Equation: y = "))
        self.equation_edit = QtWidgets.QLineEdit(equation_or_function)
        eq_layout.addWidget(self.equation_edit)
        layout.addLayout(eq_layout)

        # Filled equation display (read-only)
        filled_eq_layout = QtWidgets.QHBoxLayout()
        filled_eq_layout.setSpacing(0)  # Reduce spacing
        if self.is_function:
            filled_eq_layout.addWidget(QtWidgets.QLabel("Filled Function: "))
        else:
            filled_eq_layout.addWidget(QtWidgets.QLabel("Filled: y = "))
        self.filled_equation_edit = QtWidgets.QLineEdit()
        self.filled_equation_edit.setReadOnly(True)  # Make it read-only for copy-paste
        filled_eq_layout.addWidget(self.filled_equation_edit)
        layout.addLayout(filled_eq_layout)

        # Color picker + Delete aligned right
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.setSpacing(0)
        color_layout.addWidget(QtWidgets.QLabel("Color:"))
        self.color_button = QtWidgets.QPushButton()
        self.color_button.setFixedSize(24, 24)
        self.update_color_button()
        self.color_button.clicked.connect(self._choose_color)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch(1)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        color_layout.addWidget(self.delete_button, 0, Qt.AlignRight)
        layout.addLayout(color_layout)

        # Parameters container (grid, 2 per row, unified label widths)
        self.param_container = QtWidgets.QWidget()
        self.param_layout = QtWidgets.QGridLayout(self.param_container)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout.setHorizontalSpacing(0)
        self.param_layout.setVerticalSpacing(0)
        layout.addWidget(self.param_container)

        # Connect signals
        self.toggled.connect(self.visibilityChanged)
        self.toggled.connect(self._on_toggled)
        self.equation_edit.editingFinished.connect(self._equation_changed)
        self.delete_button.clicked.connect(self.deleteRequested)

        # Parse initial equation and update filled equation
        self._parse_equation()
        self._update_filled_equation()
        # Ensure correct initial visibility state of content
        self._on_toggled(self.isChecked())

    def _equation_changed(self):
        self._parse_equation()
        self._update_filled_equation()
        self.equationChanged.emit()

    def _on_toggled(self, checked: bool):
        """Hide or show the contents of the groupbox when toggled."""
        # Iterate over direct child widgets and set their visibility
        for child in self.findChildren(QtWidgets.QWidget):
            if child is self:
                continue
            child.setVisible(checked)

    def _parse_equation(self):
        """
        Parse the equation or function to extract parameters and update the UI.
        """
        equation_or_function = self.equation_edit.text()
        params = set()

        if self.is_function:
            # For Python functions, use inspection to get parameter names
            try:
                # Print the function string for debugging
                print(f"Parsing function in CurveWidget:\n{equation_or_function}")

                # Compile the function if it's a string
                if isinstance(equation_or_function, str):
                    self.function = self.curve_evaluator.compile_function(equation_or_function)
                else:
                    self.function = equation_or_function

                # Get parameter names using inspection
                params = set(self.curve_evaluator.get_function_parameters(self.function))
                print(f"Function parameters: {params}")
            except Exception as e:
                print(f"Error parsing function: {e}")
                import traceback
                traceback.print_exc()
                # If there's an error, fall back to empty parameter set
                params = set()
        else:
            # For equations, use regex to find parameters
            # Find all parameters (variables that are not x or y)
            param_pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
            params = set(re.findall(param_pattern, equation_or_function))
            params.discard('x')
            params.discard('y')

        # Remove parameters that are no longer in the equation or function
        for param in list(self.parameters.keys()):
            if param not in params:
                self.param_layout.removeWidget(self.parameters[param])
                self.parameters[param].deleteLater()
                del self.parameters[param]

        # Add new parameters
        for param in sorted(params):
            if param not in self.parameters:
                # Use generic defaults for all parameters
                min_val, max_val, default_value = 0.1, 10.0, 1.0

                # If sliders are disabled, use ScientificSpinBox directly
                if not self.use_sliders:
                    from ..widgets import ScientificSpinBox
                    param_widget = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(param_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)

                    # Label
                    label = QtWidgets.QLabel(param)
                    # Add tooltip with full name and apply middle-ellipsis
                    label.setToolTip(param)
                    label.setMinimumWidth(50)
                    label.setMaximumWidth(160)
                    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    # Pre-elide text to fit max width
                    elided = label.fontMetrics().elidedText(param, Qt.ElideMiddle, label.maximumWidth())
                    label.setText(elided)
                    row_layout.addWidget(label)

                    # SpinBox
                    spinbox = ScientificSpinBox(format_str="%.3e", relative_step=0.01, decimals=3)
                    spinbox.setRange(min_val, max_val)
                    spinbox.setValue(default_value)
                    row_layout.addWidget(spinbox)

                    # Connect signal
                    spinbox.valueChanged.connect(lambda value, p=param: self._parameter_changed(value))

                    # Store the spinbox as an attribute for easy access
                    param_widget.spinbox = spinbox
                    param_widget.value = spinbox.value
                    param_widget.setValue = spinbox.setValue
                    param_widget.setRange = lambda min_val, max_val, sb=spinbox: sb.setRange(min_val, max_val)
                else:
                    param_widget = ParameterSlider(param, min_val, max_val, default_value)
                    # unify label width and alignment if available
                    if hasattr(param_widget, 'label'):
                        param_widget.label.setMinimumWidth(50)
                        param_widget.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    param_widget.valueChanged.connect(self._parameter_changed)

                # Place parameters two per row in grid
                current_count = self.param_layout.count()
                row = current_count // 2
                col = current_count % 2
                self.param_layout.addWidget(param_widget, row, col)
                self.parameters[param] = param_widget

    def _parameter_changed(self, value):
        """
        Called when a parameter value changes.
        Updates the filled equation and emits the equationChanged signal.
        """
        self._update_filled_equation()
        self.equationChanged.emit()

    def _update_filled_equation(self):
        """
        Updates the filled equation display by replacing parameter names with their values.
        """
        equation_or_function = self.equation_edit.text()

        # Build both raw and formatted parameter mappings
        raw_params = {}
        formatted_params = {}
        for name, widget in self.parameters.items():
            try:
                raw_val = widget.value()
            except Exception:
                # Fallback if widget has no value() method
                raw_val = None
            raw_params[name] = raw_val
            try:
                if hasattr(widget, 'spinbox') and widget.spinbox is not None:
                    # Use the spinbox's own text so decimals/format_str are respected
                    formatted_params[name] = widget.spinbox.text()
                else:
                    # Reasonable fallback formatting
                    formatted_params[name] = f"{raw_val:.6g}" if isinstance(raw_val, (int, float)) else str(raw_val)
            except Exception:
                formatted_params[name] = str(raw_val)

        if self.is_function:
            # For functions, just show the function name and parameter values (formatted)
            try:
                if self.function:
                    func_name = self.function.__name__
                    params_str = ", ".join([f"{name}={formatted_params.get(name, str(val))}" for name, val in raw_params.items()])
                    filled_equation = f"{func_name}({params_str})"
                else:
                    filled_equation = "Function not compiled"
            except Exception as e:
                filled_equation = f"Error: {str(e)}"
        else:
            # For equations, replace parameter names with their formatted values
            # If there's an equals sign, only use the right side
            if '=' in equation_or_function:
                equation_or_function = equation_or_function.split('=', 1)[1].strip()

            # Replace parameter names with their values
            filled_equation = equation_or_function
            for param_name, formatted_value in formatted_params.items():
                # Use word boundaries to ensure we only replace whole parameter names
                pattern = r'\b' + re.escape(param_name) + r'\b'
                filled_equation = re.sub(pattern, formatted_value, filled_equation)

        self.filled_equation_edit.setText(filled_equation)

    def get_equation(self):
        """
        Get the equation or function.

        Returns:
            str or Callable: The equation string or function object
        """
        if self.is_function and self.function:
            return self.function
        return self.equation_edit.text()

    def get_parameters(self):
        return {name: widget.value() for name, widget in self.parameters.items()}

    def set_parameters(self, parameters, ranges=None):
        """
        Set parameter values and ranges from dictionaries.

        Args:
            parameters (dict): Dictionary of parameter names and values
            ranges (dict, optional): Dictionary of parameter names and ranges [min, max]
        """
        # First parse the equation to ensure all parameters exist
        self._parse_equation()

        # Set values for existing parameters
        for name, value in parameters.items():
            if name in self.parameters:
                self.parameters[name].setValue(value)

        # Set ranges for existing parameters if provided
        if ranges:
            for name, range_values in ranges.items():
                if name in self.parameters and len(range_values) == 2:
                    min_val, max_val = range_values
                    self.parameters[name].setRange(min_val, max_val)

        # Update the filled equation with the new parameter values
        self._update_filled_equation()

    def get_color(self):
        return self.curve_color

    def is_visible(self):
        return self.isChecked()

    def update_color_button(self):
        """Update the color button appearance based on the current color."""
        self.color_button.setStyleSheet(f"background-color: {self.curve_color}; border: 1px solid #888;")

    def _choose_color(self):
        """Open a color dialog and set the selected color."""
        current_color = QtGui.QColor(self.curve_color)
        color = QtWidgets.QColorDialog.getColor(current_color, self)

        if color.isValid():
            hex_color = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
            self.curve_color = hex_color
            self.update_color_button()
            self.colorChanged.emit()


class CurveOverlayWidget(QtWidgets.QWidget):
    """
    Widget for managing curve overlays on the 2D histogram.
    """
    curvesChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.curves = []
        self.predefined_equations = []
        self._last_x_edges = None  # Cache latest x edges used for computation
        self._last_y_edges = None  # Cache latest y edges used for computation
        self._last_plot_control = None  # Cache latest plot control for scale info

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(0)  # Reduce spacing between elements
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins

        # Predefined equations dropdown with add button
        predefined_layout = QtWidgets.QHBoxLayout()
        predefined_layout.setSpacing(0)  # Reduce spacing
        predefined_layout.addWidget(QtWidgets.QLabel("Equation:"))
        self.predefined_combo = QtWidgets.QComboBox()
        self.predefined_combo.addItem("Custom Equation")  # Default option
        predefined_layout.addWidget(self.predefined_combo)
        self.add_button = QtWidgets.QPushButton("Add Curve")
        predefined_layout.addWidget(self.add_button)
        layout.addLayout(predefined_layout)

        # Number of points control
        points_layout = QtWidgets.QHBoxLayout()
        points_layout.setSpacing(0)  # Reduce spacing
        points_layout.addWidget(QtWidgets.QLabel("Number of points:"))
        self.points_spinbox = QtWidgets.QSpinBox()
        self.points_spinbox.setMinimum(10)
        self.points_spinbox.setMaximum(999)
        self.points_spinbox.setValue(500)  # Default to 500 points
        self.points_spinbox.valueChanged.connect(self.curvesChanged)
        points_layout.addWidget(self.points_spinbox)
        layout.addLayout(points_layout)

        # Action buttons (e.g., Save CSV)
        actions_layout = QtWidgets.QHBoxLayout()
        actions_layout.setSpacing(0)
        self.save_button = QtWidgets.QPushButton("Save CSV")
        self.save_button.setToolTip("Save visible overlay curves as CSV (curve, x, y)")
        self.save_button.clicked.connect(self._on_save_csv)
        actions_layout.addWidget(self.save_button)
        actions_layout.addStretch(1)
        layout.addLayout(actions_layout)

        # Scroll area for curves
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)  # Align widgets to the top
        self.scroll_layout.setSpacing(0)  # Reduce spacing between curve widgets
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

        # Connect signals
        self.add_button.clicked.connect(self.add_selected_curve)

        # Load predefined equations
        self.load_predefined_equations()

    def add_selected_curve(self):
        """
        Add a curve based on the selected item in the predefined_combo.
        If "Custom Equation" is selected, add a custom curve.
        Otherwise, add the selected predefined curve.
        """
        # Get the selected equation index
        index = self.predefined_combo.currentIndex()

        # If "Custom Equation" is selected (index 0), add a custom curve
        if index == 0:
            self.add_curve()
        else:
            # Otherwise, add the selected predefined curve
            self.add_predefined_curve()

    def load_predefined_equations(self):
        """
        Load predefined equations from the YAML file.
        """
        candidates: List[Path] = []
        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / "settings" / "curve_equations.yaml")
        candidates.append(Path(__file__).resolve().parents[1] / "settings" / "curve_equations.yaml")

        try:
            from ..settings import get_settings_path  # type: ignore
        except Exception:
            get_settings_path = None

        if callable(get_settings_path):
            try:
                candidates.append(get_settings_path() / "curve_equations.yaml")
            except Exception:
                pass

        file_path = next((path for path in candidates if path.exists()), None)

        if not file_path:
            logging.warning(
                "Curve overlay predefined equations not found. Tried: %s",
                ", ".join(str(path) for path in candidates),
            )
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.predefined_equations = yaml.safe_load(f) or []

            for equation in self.predefined_equations:
                if isinstance(equation, dict) and "name" in equation:
                    self.predefined_combo.addItem(equation["name"])
        except Exception as e:
            logging.error("Error loading predefined equations from %s: %s", file_path, e)

    def add_predefined_curve(self):
        """
        Add a curve with the selected predefined equation or function.
        """
        # Get the selected equation index (subtract 1 because the first item is "Custom Equation")
        index = self.predefined_combo.currentIndex() - 1

        # Get the selected equation data
        equation_data = self.predefined_equations[index]
        base_name = equation_data.get('name', 'Curve')

        # Check if it's a function or an equation
        if 'function' in equation_data:
            function_str = equation_data['function']
            print(f"Function string from YAML:\n{function_str}")
            curve_widget = self.add_curve(function_str, use_sliders=True, is_function=True, base_name=base_name)
        else:
            curve_widget = self.add_curve(equation_data['equation'], use_sliders=True, is_function=False, base_name=base_name)

        # Set the parameter values and ranges
        if 'parameters' in equation_data:
            ranges = equation_data.get('ranges', {})
            curve_widget.set_parameters(equation_data['parameters'], ranges)

        return curve_widget

    def add_curve(self, equation_or_function="x", use_sliders=True, is_function=False, base_name: str = None):
        """
        Add a new curve widget with the given equation or function.

        Args:
            equation_or_function (str): The equation or function to add
            use_sliders (bool): Whether to use sliders for parameters (default: True)
            is_function (bool): Whether the input is a function (True) or an equation (False)
            base_name (str): Base name for the groupbox title (e.g., "Static FRET line")
        """
        if base_name is None:
            base_name = "Custom Equation"
        # Count existing curves with the same base name
        existing = sum(1 for c in self.curves if str(c.title()).startswith(base_name))
        curve_title = f"{base_name} {existing + 1}"
        curve_widget = CurveWidget(curve_title, equation_or_function, is_function=is_function)
        curve_widget.use_sliders = use_sliders

        # Connect signals
        curve_widget.visibilityChanged.connect(self.curvesChanged)
        curve_widget.equationChanged.connect(self.curvesChanged)
        curve_widget.colorChanged.connect(self.curvesChanged)
        curve_widget.deleteRequested.connect(lambda: self.remove_curve(curve_widget))

        self.scroll_layout.addWidget(curve_widget)
        self.curves.append(curve_widget)
        self.curvesChanged.emit()
        return curve_widget

    def remove_curve(self, curve_widget):
        """
        Remove the specified curve widget.
        """
        if curve_widget in self.curves:
            self.curves.remove(curve_widget)
            self.scroll_layout.removeWidget(curve_widget)
            curve_widget.deleteLater()
            self.curvesChanged.emit()

    def clear_curves(self):
        """
        Remove all curve widgets from the list and layout.
        """
        # Make a copy of the list since we'll be modifying it during iteration
        curves_copy = self.curves.copy()

        # Remove each curve widget
        for curve_widget in curves_copy:
            # Remove from the layout
            self.scroll_layout.removeWidget(curve_widget)
            curve_widget.deleteLater()

        # Clear the list
        self.curves.clear()

        # Emit signal to update the plot
        self.curvesChanged.emit()

    def get_visible_curves(self):
        """
        Return a list of (equation, parameters, color) tuples for visible curves.
        """
        return [(curve.get_equation(), curve.get_parameters(), curve.get_color()) 
                for curve in self.curves if curve.is_visible()]

    def get_num_points(self):
        """
        Return the number of points to use for curve computation.
        """
        return self.points_spinbox.value()

    def update_curve_overlays(self, overlay_plot, histogram_data, plot_control, curve_evaluator, value_to_bin_func):
        """
        Update the curve overlays on the 2D histogram.

        Args:
            overlay_plot: The plot where curve items are added
            histogram_data: Tuple of (counts, x_edges, y_edges) for the 2D histogram
            plot_control: Widget for controlling plot settings
            curve_evaluator: Class for evaluating curve equations
            value_to_bin_func: Function to convert values to bin indices
        """
        overlay_plot.clear_curves()

        try:
            _, x_edges, y_edges = histogram_data
        except (ValueError, TypeError):
            return

        self._last_x_edges = x_edges
        self._last_y_edges = y_edges
        self._last_plot_control = plot_control

        visible_curves = self.get_visible_curves()
        num_points = self.get_num_points()

        overlay_plot.setAxisScale(0, 0, len(x_edges) - 1)
        overlay_plot.setAxisScale(1, 0, len(y_edges) - 1)

        for equation, parameters, color in visible_curves:
            # Create x values array with the specified number of points
            # Use the same scaling function (linear or logarithmic) that was used to create the bins
            x_min = x_edges[0]
            x_max = x_edges[-1]

            # Check if x-axis is using logarithmic scale
            if plot_control.scale_x == "log":
                if x_min <= 0:
                    x_min = 1e-6
                if x_max <= 0:
                    x_max = 1e-6
                x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_points)
            else:
                x_values = np.linspace(x_min, x_max, num_points)

            # Evaluate the equation or function
            result = curve_evaluator.evaluate(equation, x_values, parameters)
            if result is None:
                continue  # Skip if evaluation failed

            # Check if result is a tuple (parametric function) or array (equation)
            if isinstance(result, tuple) and len(result) == 2:
                # Parametric function - use both x and y values from the function
                x_values, y_values = result
            else:
                # Regular equation - use the generated x_values and the evaluated y_values
                y_values = result

            # Convert x and y values to bin coordinates for plotting
            # Note: The 2D histogram is rotated 90 degrees in the plot
            x_coords = []
            y_coords = []

            # Check if y-axis is using logarithmic scale and adjust y values accordingly
            if plot_control.scale_y == "log":
                # For logarithmic y-axis, we need to ensure y values are positive
                y_values = np.maximum(y_values, 1e-6)

            for i, (x, y) in enumerate(zip(x_values, y_values)):
                # Check if y is within the y range
                if y < y_edges[0] or y > y_edges[-1]:
                    continue

                # Convert to bin coordinates
                # Note: The 2D histogram is rotated 90 degrees in the plot
                # so we need to swap x and y coordinates
                y_bin = value_to_bin_func(y, y_edges)
                if y_bin is None:
                    continue

                # Convert x value to bin index
                x_bin = value_to_bin_func(x, x_edges)
                if x_bin is None:
                    continue

                # Add points to the curve
                # The y-coordinate is the bin index (not the value)
                # The x-coordinate is the bin index (not the value)
                x_coords.append(x_bin)  # x bin index
                y_coords.append(y_bin)  # y bin index

            if not x_coords:
                continue  # Skip if no valid points

            overlay_plot.add_curve(
                np.array(x_coords),
                np.array(y_coords),
                color=color,
                width=2
            )

        overlay_plot.replot()

    def _compute_curve_points(self, equation, parameters, num_points, x_edges, y_edges, plot_control, curve_evaluator):
        """
        Compute value-domain x,y points for a curve given current edges and scaling.
        Filters points outside the y range.
        Returns two numpy arrays (x_values_filtered, y_values_filtered).
        """
        # Determine x sampling based on axis scale
        x_min = x_edges[0]
        x_max = x_edges[-1]

        if plot_control.scale_x == "log":
            if x_min <= 0:
                x_min = 1e-6
            if x_max <= 0:
                x_max = 1e-6
            x_values = np.logspace(np.log10(x_min), np.log10(x_max), num_points)
        else:
            x_values = np.linspace(x_min, x_max, num_points)

        # Evaluate equation or function
        result = curve_evaluator.evaluate(equation, x_values, parameters)
        if result is None:
            return np.array([]), np.array([])

        # Unpack parametric vs standard equation
        if isinstance(result, tuple) and len(result) == 2:
            x_eval, y_eval = result
        else:
            x_eval, y_eval = x_values, result

        # For log y scale, clamp minimum positive
        if plot_control.scale_y == "log":
            y_eval = np.maximum(y_eval, 1e-6)

        # Filter to y range
        y_min, y_max = y_edges[0], y_edges[-1]
        mask = (y_eval >= y_min) & (y_eval <= y_max)
        return x_eval[mask], y_eval[mask]

    def _on_save_csv(self):
        """
        Save currently visible overlay curves as a CSV file with columns: curve, x, y.
        Uses the latest histogram edges and plot control cached during the last overlay update.
        """
        # Validate that we have context for computation
        if self._last_x_edges is None or self._last_y_edges is None or self._last_plot_control is None:
            QtWidgets.QMessageBox.warning(self, "Save Overlays", "No overlay data available yet. Create/update overlays first.")
            return

        # Gather visible curves
        visible_curves_widgets = [c for c in self.curves if c.is_visible()]
        if not visible_curves_widgets:
            QtWidgets.QMessageBox.information(self, "Save Overlays", "No visible curves to save.")
            return

        # Ask user for file path
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Overlay Curves", "overlays.csv", "CSV Files (*.csv)")
        if not filename:
            return

        # Compute and write CSV
        try:
            num_points = self.get_num_points()

            # Compute all curves first
            curves_data = []  # list of (name, x_vals, y_vals)
            max_len = 0
            for curve in visible_curves_widgets:
                name = str(curve.title())
                equation_or_function = curve.get_equation()
                parameters = curve.get_parameters()

                x_vals, y_vals = self._compute_curve_points(
                    equation_or_function,
                    parameters,
                    num_points,
                    self._last_x_edges,
                    self._last_y_edges,
                    self._last_plot_control,
                    curve.curve_evaluator  # evaluator from the curve widget ensures consistency
                )

                curves_data.append((name, x_vals, y_vals))
                if len(x_vals) > max_len:
                    max_len = len(x_vals)

            # Write the CSV with two header lines and horizontal stacking
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)

                # Header line 1: curve names repeated for x and y columns
                header1 = []
                for name, _, _ in curves_data:
                    header1.extend([name, name])
                writer.writerow(header1)

                # Header line 2: x,y under each curve
                header2 = []
                for _ in curves_data:
                    header2.extend(["x", "y"])
                writer.writerow(header2)

                # Data rows: pad with empty strings when a curve has fewer points
                for i in range(max_len):
                    row = []
                    for _, x_vals, y_vals in curves_data:
                        if i < len(x_vals):
                            row.extend([x_vals[i], y_vals[i]])
                        else:
                            row.extend(["", ""])  # pad
                    writer.writerow(row)

            QtWidgets.QMessageBox.information(self, "Save Overlays", f"Saved overlay curves to:\n{filename}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Overlays", f"Failed to save CSV:\n{e}")


class CurveEvaluator:
    """
    Class for evaluating curve equations and Python functions.
    """
    def __init__(self):
        self.last_error = None
        self.compiled_functions = {}  # Cache for compiled functions

    def compile_function(self, function_str: str) -> Callable:
        """
        Compile a Python function from a string.

        Args:
            function_str (str): String containing the function definition

        Returns:
            Callable: The compiled function
        """
        try:
            # Dedent the function string to handle indentation properly
            function_str = textwrap.dedent(function_str)

            # Ensure the function string has proper line breaks
            if '\n' not in function_str:
                # If there are no line breaks, try to split by indentation
                function_str = function_str.replace('    ', '\n    ')
                if '\n' not in function_str:
                    # If still no line breaks, this might be a one-line function definition
                    # which is not valid Python syntax, so we need to add proper formatting
                    parts = function_str.split(':', 1)
                    if len(parts) == 2:
                        function_header = parts[0].strip()
                        function_body = parts[1].strip()
                        function_str = f"{function_header}:\n    {function_body}"

            # Create a namespace for the function
            namespace = {
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'sqrt': np.sqrt,
                'pi': np.pi,
                'e': np.e
            }

            # Print the function string for debugging
            print(f"Compiling function:\n{function_str}")

            # Execute the function definition in the namespace
            exec(function_str, namespace)

            # Extract the function from the namespace
            # The function name is the first word after 'def ' in the function string
            function_name = function_str.split('def ')[1].split('(')[0].strip()
            return namespace[function_name]
        except Exception as e:
            print(f"Error compiling function: {e}")
            print(f"Function string: {function_str}")
            raise

    def get_function_parameters(self, function: Callable) -> List[str]:
        """
        Get the parameter names of a function using inspection.

        Args:
            function (Callable): The function to inspect

        Returns:
            List[str]: List of parameter names
        """
        return list(inspect.signature(function).parameters.keys())

    def evaluate(self, equation_or_function, x_values, parameters):
        """
        Evaluate the equation or function for the given x values and parameters.

        Args:
            equation_or_function (str or Callable): The equation to evaluate (e.g., "y = 1-x/tau0")
                                                   or a Python function that returns x, y pairs
            x_values (np.ndarray): Array of x values (used for equation evaluation)
            parameters (dict): Dictionary of parameter values

        Returns:
            tuple or np.ndarray: For parametric functions, returns (x_values, y_values) tuple.
                                For equations, returns array of y values, or None if evaluation failed
        """
        self.last_error = None

        # Check if equation_or_function is a callable (Python function)
        if isinstance(equation_or_function, Callable):
            # Call the function with parameters
            x_result, y_result = equation_or_function(**parameters)
            return (x_result, y_result)  # Return both x and y values

        # Check if equation_or_function is a function definition string
        elif isinstance(equation_or_function, str) and equation_or_function.strip().startswith("def "):
            # Compile the function if not already in cache
            if equation_or_function not in self.compiled_functions:
                self.compiled_functions[equation_or_function] = self.compile_function(equation_or_function)

            # Call the compiled function with parameters
            function = self.compiled_functions[equation_or_function]
            x_result, y_result = function(**parameters)
            return (x_result, y_result)  # Return both x and y values

        # Otherwise, treat as a mathematical expression
        else:
            # Create a safe local environment with only allowed functions and constants
            locals_dict = {
                'x': x_values,
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'sqrt': np.sqrt,
                'pi': np.pi,
                'e': np.e
            }

            # Add parameters to locals
            locals_dict.update(parameters)

            # Extract the right side of the equation (after '=')
            equation = equation_or_function
            if '=' in equation:
                equation = equation.split('=', 1)[1].strip()

            # Evaluate the expression
            result = eval(equation, {"__builtins__": {}}, locals_dict)

            return result
