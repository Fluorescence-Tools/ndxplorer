from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import re
import os
import yaml

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

class ParameterSlider(QtWidgets.QWidget):
    """
    A widget that combines a label, slider, and spin box for parameter adjustment.
    """
    valueChanged = QtCore.Signal(float)

    def __init__(self, name, min_val=0.0, max_val=10.0, value=1.0, parent=None):
        super().__init__(parent)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        self.label = QtWidgets.QLabel(name)
        layout.addWidget(self.label)

        # Slider
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        layout.addWidget(self.slider)

        # Spin box
        self.spinbox = QtWidgets.QDoubleSpinBox()
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(value)
        self.spinbox.setSingleStep((max_val - min_val) / 100.0)
        layout.addWidget(self.spinbox)

        # Connect signals
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)

        # Initialize slider position
        self._update_slider()

    def _slider_changed(self, value):
        # Convert slider value (0-1000) to parameter value (min_val-max_val)
        param_value = self.min_val + (value / 1000.0) * (self.max_val - self.min_val)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(param_value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(param_value)

    def _spinbox_changed(self, value):
        self._update_slider()
        self.valueChanged.emit(value)

    def _update_slider(self):
        # Convert parameter value to slider value
        value = self.spinbox.value()
        slider_value = int(((value - self.min_val) / (self.max_val - self.min_val)) * 1000)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)

    def value(self):
        return self.spinbox.value()

    def setValue(self, value):
        self.spinbox.setValue(value)

    def setRange(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setSingleStep((max_val - min_val) / 100.0)
        self._update_slider()


class CurveWidget(QtWidgets.QGroupBox):
    """
    A widget for a single curve equation with parameters and visibility control.
    """
    visibilityChanged = QtCore.Signal(bool)
    equationChanged = QtCore.Signal()
    deleteRequested = QtCore.Signal()
    colorChanged = QtCore.Signal()

    def __init__(self, name="Curve", equation="x", parent=None):
        super().__init__(name, parent)
        self.setCheckable(True)
        self.setChecked(True)

        self.parameters = {}  # Dictionary to store parameter widgets
        self.curve_color = "#ff0000"  # Default color is red

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(0)  # Reduce spacing between elements
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins

        # Equation input
        eq_layout = QtWidgets.QHBoxLayout()
        eq_layout.setSpacing(0)  # Reduce spacing
        eq_layout.addWidget(QtWidgets.QLabel("Equation: y = "))
        self.equation_edit = QtWidgets.QLineEdit(equation)
        eq_layout.addWidget(self.equation_edit)
        layout.addLayout(eq_layout)

        # Color picker
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.setSpacing(0)  # Reduce spacing
        color_layout.addWidget(QtWidgets.QLabel("Color:"))
        self.color_button = QtWidgets.QPushButton()
        self.color_button.setFixedSize(24, 24)
        self.update_color_button()
        self.color_button.clicked.connect(self._choose_color)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch(1)
        layout.addLayout(color_layout)

        # Parameters container
        self.param_container = QtWidgets.QWidget()
        self.param_layout = QtWidgets.QVBoxLayout(self.param_container)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.param_container)

        # Delete button
        self.delete_button = QtWidgets.QPushButton("Delete")
        layout.addWidget(self.delete_button)

        # Connect signals
        self.toggled.connect(self.visibilityChanged)
        self.equation_edit.editingFinished.connect(self._equation_changed)
        self.delete_button.clicked.connect(self.deleteRequested)

        # Parse initial equation
        self._parse_equation()

    def _equation_changed(self):
        self._parse_equation()
        self.equationChanged.emit()

    def _parse_equation(self):
        """
        Parse the equation to extract parameters and update the UI.
        """
        equation = self.equation_edit.text()

        # Find all parameters (variables that are not x or y)
        param_pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
        params = set(re.findall(param_pattern, equation))
        params.discard('x')
        params.discard('y')

        # Remove parameters that are no longer in the equation
        for param in list(self.parameters.keys()):
            if param not in params:
                self.param_layout.removeWidget(self.parameters[param])
                self.parameters[param].deleteLater()
                del self.parameters[param]

        # Add new parameters
        for param in params:
            if param not in self.parameters:
                param_widget = ParameterSlider(param, 0.1, 10.0, 1.0)
                param_widget.valueChanged.connect(self.equationChanged)
                self.param_layout.addWidget(param_widget)
                self.parameters[param] = param_widget

    def get_equation(self):
        return self.equation_edit.text()

    def get_parameters(self):
        return {name: widget.value() for name, widget in self.parameters.items()}

    def set_parameters(self, parameters):
        """
        Set parameter values from a dictionary.

        Args:
            parameters (dict): Dictionary of parameter names and values
        """
        # First parse the equation to ensure all parameters exist
        self._parse_equation()

        # Set values for existing parameters
        for name, value in parameters.items():
            if name in self.parameters:
                self.parameters[name].setValue(value)

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
        self.curves = []  # List to store curve widgets
        self.predefined_equations = []  # List to store predefined equations

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
        try:
            # Get the path to the curve_equations.yaml file
            file_path = os.path.join(os.path.dirname(__file__), "settings", "curve_equations.yaml")

            # Load the YAML file
            with open(file_path, 'r') as f:
                self.predefined_equations = yaml.safe_load(f)

            # Populate the dropdown with equation names
            for equation in self.predefined_equations:
                self.predefined_combo.addItem(equation['name'])

        except Exception as e:
            print(f"Error loading predefined equations: {e}")

    def add_predefined_curve(self):
        """
        Add a curve with the selected predefined equation.
        """
        # Get the selected equation index (subtract 1 because the first item is "Custom Equation")
        index = self.predefined_combo.currentIndex() - 1

        # Get the selected equation
        equation_data = self.predefined_equations[index]

        # Create a new curve with the equation
        curve_widget = self.add_curve(equation_data['equation'])

        # Set the parameter values
        if 'parameters' in equation_data:
            curve_widget.set_parameters(equation_data['parameters'])

        return curve_widget

    def add_curve(self, equation="x"):
        """
        Add a new curve widget with the given equation.
        """
        curve_name = f"Curve {len(self.curves) + 1}"
        curve_widget = CurveWidget(curve_name, equation)

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


class CurveEvaluator:
    """
    Class for evaluating curve equations.
    """
    def __init__(self):
        self.last_error = None

    def evaluate(self, equation, x_values, parameters):
        """
        Evaluate the equation for the given x values and parameters.

        Args:
            equation (str): The equation to evaluate (e.g., "y = 1-x/tau0")
            x_values (np.ndarray): Array of x values
            parameters (dict): Dictionary of parameter values

        Returns:
            np.ndarray: Array of y values, or None if evaluation failed
        """
        self.last_error = None

        try:
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
            if '=' in equation:
                equation = equation.split('=', 1)[1].strip()

            # Evaluate the expression
            result = eval(equation, {"__builtins__": {}}, locals_dict)

            return result

        except Exception as e:
            self.last_error = str(e)
            return None
