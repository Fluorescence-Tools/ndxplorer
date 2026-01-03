"""
Histogram control mixin for bins and normalization settings.

Extracts histogram configuration functionality from SurfacePlotWidget.
"""

import logging


class HistogramControlMixin:
    """
    Mixin providing histogram control properties.
    
    Manages:
    - Histogram normalization (X/Y/Z)
    - Weight enable/disable
    - Weight parameter selection
    - Checkbox state management
    """
    
    def _get_checkbox_state(self, checkbox):
        """Helper to get boolean state from checkbox."""
        try:
            if checkbox is None:
                logging.warning(f"Checkbox is None, returning False")
                return False
            state = bool(checkbox.isChecked())
            logging.debug(f"Checkbox {checkbox.objectName() if hasattr(checkbox, 'objectName') else 'unknown'} state: {state}")
            return state
        except Exception as e:
            logging.error(f"Error getting checkbox state: {e}")
            return False

    def _set_checkbox_with_signal_control(self, checkbox, value, block_signals=False):
        """Helper to set checkbox state with optional signal blocking."""
        was_blocked = checkbox.signalsBlocked()
        if block_signals and not was_blocked:
            checkbox.blockSignals(True)
        try:
            checkbox.setChecked(bool(value))
        finally:
            if block_signals and not was_blocked:
                checkbox.blockSignals(was_blocked)

    @property
    def normed_hist_x(self):
        """Get X histogram normalization state."""
        try:
            checkbox = getattr(self, 'checkBoxNormX', None)
            logging.debug(f"normed_hist_x: checkBoxNormX exists: {checkbox is not None}")
            return self._get_checkbox_state(checkbox)
        except Exception as e:
            logging.error(f"Error in normed_hist_x: {e}")
            return False

    @property
    def normed_hist_y(self):
        """Get Y histogram normalization state."""
        try:
            checkbox = getattr(self, 'checkBoxNormY', None)
            logging.debug(f"normed_hist_y: checkBoxNormY exists: {checkbox is not None}")
            return self._get_checkbox_state(checkbox)
        except Exception as e:
            logging.error(f"Error in normed_hist_y: {e}")
            return False

    @property
    def normed_hist_z(self):
        """Get Z histogram normalization state."""
        try:
            checkbox = getattr(self, 'checkBoxNormZ', None)
            logging.debug(f"normed_hist_z: checkBoxNormZ exists: {checkbox is not None}")
            return self._get_checkbox_state(checkbox)
        except Exception as e:
            logging.error(f"Error in normed_hist_z: {e}")
            return False
        
    @property
    def weight_enabled(self):
        """Returns whether weighting is enabled."""
        return bool(self.checkBoxWeight.isChecked())

    @weight_enabled.setter
    def weight_enabled(self, value, block_signals=False):
        """
        Set whether weighting is enabled.
        
        Args:
            value: Boolean indicating whether weighting should be enabled
            block_signals: If True, signals will be blocked during the change
        """
        self._set_checkbox_with_signal_control(self.checkBoxWeight, value, block_signals)
                
    @property
    def weight_parameter(self):
        """Returns the currently selected weight parameter."""
        return str(self.comboBoxWeight.currentText())
        
    @weight_parameter.setter
    def weight_parameter(self, value, block_signals=False):
        """
        Set the weight parameter.
        
        Args:
            value: Parameter name (str) to use for weighting or index (int)
            block_signals: If True, signals will be blocked during the change
        """
        # Reuse combobox control from AxisControlMixin
        if hasattr(self, '_set_combobox_with_signal_control'):
            self._set_combobox_with_signal_control(self.comboBoxWeight, value, block_signals)
