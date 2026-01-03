"""
Axis control mixin for X/Y/Z/Weight parameter selection.

Extracts axis selection functionality from SurfacePlotWidget.
"""

try:
    from qtpy import QtCore
except ImportError:
    from chisurf.gui import QtCore


class AxisControlMixin:
    """
    Mixin providing axis parameter selection.
    
    Manages:
    - X/Y/Z axis parameter selection
    - Weight parameter selection
    - Signal blocking during updates
    - Combobox state management
    """
    
    def _set_combobox_with_signal_control(self, combobox, value, block_signals=False):
        """Helper to set combobox value (by index or text) with optional signal blocking."""
        was_blocked = combobox.signalsBlocked()
        if block_signals and not was_blocked:
            combobox.blockSignals(True)
        try:
            if isinstance(value, str):
                index = combobox.findText(value)
                if index >= 0:
                    combobox.setCurrentIndex(index)
            elif isinstance(value, int) and value >= 0:
                combobox.setCurrentIndex(value)
        finally:
            if block_signals and not was_blocked:
                combobox.blockSignals(was_blocked)

    def _set_combobox_from_tuple_or_value(self, combobox, value, block_signals=False):
        """Helper to set combobox from tuple (idx, name) or direct value with signal control."""
        was_blocked = combobox.signalsBlocked()
        if block_signals and not was_blocked:
            combobox.blockSignals(True)
        try:
            if isinstance(value, tuple) and len(value) == 2:
                idx, name = value
                if isinstance(idx, int) and idx >= 0:
                    combobox.setCurrentIndex(idx)
                elif isinstance(name, str):
                    index = combobox.findText(name)
                    if index >= 0:
                        combobox.setCurrentIndex(index)
            elif isinstance(value, int) and value >= 0:
                combobox.setCurrentIndex(value)
            elif isinstance(value, str):
                index = combobox.findText(value)
                if index >= 0:
                    combobox.setCurrentIndex(index)
        finally:
            if block_signals and not was_blocked:
                combobox.blockSignals(was_blocked)

    @property
    def p1(self):
        """Get X axis parameter (index, name) tuple."""
        idx = self.comboBoxSelX.currentIndex()
        name = self.comboBoxSelX.currentText()
        return idx, str(name)
        
    @p1.setter
    def p1(self, value, block_signals=False):
        """
        Set the X axis parameter.
        
        Args:
            value: Either an index (int) or parameter name (str) or tuple (idx, name)
            block_signals: If True, signals will be blocked during the change
        """
        self._set_combobox_from_tuple_or_value(self.comboBoxSelX, value, block_signals)

    @property
    def p2(self):
        """Get Y axis parameter (index, name) tuple."""
        idx = self.comboBoxSelY.currentIndex()
        name = self.comboBoxSelY.currentText()
        return idx, str(name)
        
    @p2.setter
    def p2(self, value, block_signals=False):
        """
        Set the Y axis parameter.
        
        Args:
            value: Either an index (int) or parameter name (str) or tuple (idx, name)
            block_signals: If True, signals will be blocked during the change
        """
        self._set_combobox_from_tuple_or_value(self.comboBoxSelY, value, block_signals)

    @property
    def p3(self):
        """Get Z axis parameter (index, name) tuple."""
        idx = self.comboBoxSelZ.currentIndex()
        name = self.comboBoxSelZ.currentText()
        return idx, str(name)
        
    @property
    def x_label(self):
        """Get the X axis label."""
        return str(self.comboBoxSelX.currentText())
        
    @property
    def y_label(self):
        """Get the Y axis label."""
        return str(self.comboBoxSelY.currentText())
        
    @property
    def z_label(self):
        """Get the Z axis label."""
        return str(self.comboBoxSelZ.currentText())
        
    @p3.setter
    def p3(self, value, block_signals=False):
        """
        Set the Z axis parameter.
        
        Args:
            value: Either an index (int) or parameter name (str) or tuple (idx, name)
            block_signals: If True, signals will be blocked during the change
        """
        self._set_combobox_from_tuple_or_value(self.comboBoxSelZ, value, block_signals)
                
    def set_axis_by_name(self, axis, name, match_contains=True, block_signals=False):
        """
        Set an axis by parameter name with optional substring matching.
        
        Args:
            axis: String indicating which axis to set ('x', 'y', 'z', or 'weight')
            name: Parameter name to set
            match_contains: If True, use Qt.MatchContains to find partial matches
            block_signals: If True, signals will be blocked during the change
            
        Returns:
            bool: True if the axis was successfully set, False otherwise
        """
        if not isinstance(name, str) or not name:
            return False
            
        # Determine which combo box to use based on the axis
        combo_box = None
        if axis.lower() == 'x':
            combo_box = self.comboBoxSelX
        elif axis.lower() == 'y':
            combo_box = self.comboBoxSelY
        elif axis.lower() == 'z':
            combo_box = self.comboBoxSelZ
        elif axis.lower() == 'weight':
            combo_box = self.comboBoxWeight
        else:
            return False
            
        # Find the index of the parameter name (case-insensitive)
        # Qt.MatchFlag requires NOT including MatchCaseSensitive for case-insensitive search
        base_flag = QtCore.Qt.MatchContains if match_contains else QtCore.Qt.MatchExactly
        # Try case-sensitive first
        index = combo_box.findText(name, base_flag)
        
        # If not found, try case-insensitive by searching manually
        if index < 0:
            name_lower = name.lower()
            for i in range(combo_box.count()):
                item_text = combo_box.itemText(i)
                if match_contains:
                    if name_lower in item_text.lower():
                        index = i
                        break
                else:
                    if name_lower == item_text.lower():
                        index = i
                        break
        
        if index < 0:
            return False
            
        # Block signals if requested
        was_blocked = combo_box.signalsBlocked()
        if block_signals and not was_blocked:
            combo_box.blockSignals(True)
            
        try:
            combo_box.setCurrentIndex(index)
            return True
        finally:
            if block_signals and not was_blocked:
                combo_box.blockSignals(was_blocked)
