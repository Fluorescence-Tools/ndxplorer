from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from .scientific_spinbox import ScientificSpinBox


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
        # Store full text for tooltip and elision
        self._full_label_text = name
        self.label.setToolTip(self._full_label_text)
        # Constrain width so elision can take effect consistently
        self.label.setMinimumWidth(50)
        self.label.setMaximumWidth(160)
        layout.addWidget(self.label)

        # Slider
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        layout.addWidget(self.slider)

        # Spin box - using ScientificSpinBox for better precision display
        self.spinbox = ScientificSpinBox(format_str="%.4e", relative_step=0.1)
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(value)
        layout.addWidget(self.spinbox)

        # Connect signals
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)

        # Initialize slider position
        self._update_slider()
        # Apply initial elision after layout
        QtCore.QTimer.singleShot(0, self._apply_elision)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self):
        # Use label's font metrics to elide in the middle according to available width
        width = self.label.width() if self.label.width() > 0 else self.label.maximumWidth()
        elided = self.label.fontMetrics().elidedText(self._full_label_text, Qt.ElideMiddle, width)
        # Avoid recursive updates if unchanged
        if self.label.text() != elided:
            self.label.setText(elided)

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
        self._update_slider()